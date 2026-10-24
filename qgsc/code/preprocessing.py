# -*- coding: utf-8 -*-
"""ReQISC 输入适配、模板综合、层次综合和 canonical rebase。"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import importlib.util
import os
import pickle
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


from .runtime import settings as _settings, ISING_ROOT as _ISING_ROOT
import numpy as np
from . import models as _mod_models
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import common as _mod_common



def initialize_reqisc_optimizer() -> _mod_models.ReqiscOptimizeContext:
    """初始化 ReQISC 逻辑综合前端；整个文件夹只初始化一次。"""
    root = os.path.abspath(_settings.REQISC_ROOT)

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"REQISC_ROOT 不存在：{root}"
        )

    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        import bqskit
        from regulus import Circuit as RegulusCircuit
        from regulus.transforms.template import template_based_synthesize
        from regulus.transforms.hierarchy import (
            hierarchical_synthesize,
            rebase_to_canonical_by_bqskit,
        )
    except Exception as exc:
        raise RuntimeError(
            "ReQISC/Regulus 导入失败。请确认当前虚拟环境已经安装 "
            "ReQISC requirements 与 bqskit。\n"
            f"REQISC_ROOT={root}\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    template_file = os.path.join(
        root,
        "artifact",
        "scripts",
        "regulus_template_library.pkl",
    )
    if not os.path.isfile(template_file):
        raise FileNotFoundError(
            f"找不到 ReQISC template library：{template_file}"
        )

    with open(template_file, "rb") as f:
        structures_unique = pickle.load(f)

    compiler = bqskit.compiler.Compiler()

    print("=== ReQISC 前端综合初始化 ===")
    print(f"REQISC_ROOT={root}")
    print(
        "pipeline=ReQISC template/hierarchical/canonical "
        "-> 本文 G(theta)/polytope/routing"
    )
    print(
        f"template={_settings.REQISC_TEMPLATE_SYNTHESIS} | "
        f"full={_settings.REQISC_FULL_SYNTHESIS} | "
        f"approx_trials={_settings.REQISC_NUM_APPROX_SYN_TRIALS}"
    )
    print("ReQISC routing=False | ReQISC amplitude mirror=False")

    return _mod_models.ReqiscOptimizeContext(
        compiler=compiler,
        Circuit=RegulusCircuit,
        template_based_synthesize=template_based_synthesize,
        hierarchical_synthesize=hierarchical_synthesize,
        rebase_to_canonical_by_bqskit=rebase_to_canonical_by_bqskit,
        structures_unique=structures_unique,
        root=root,
    )


def close_reqisc_optimizer(
    ctx: Optional[_mod_models.ReqiscOptimizeContext],
) -> None:
    if ctx is None:
        return
    try:
        ctx.compiler.close()
    except Exception:
        pass


def _qiskit_instruction_name(item) -> str:
    if hasattr(item, "operation"):
        return str(item.operation.name)
    return str(item[0].name)


def _qiskit_instruction_has_condition(item) -> bool:
    op = item.operation if hasattr(item, "operation") else item[0]
    return getattr(op, "condition", None) is not None


def flatten_qiskit_qubit_registers(qc):
    """把 QASMBench 的多个 qreg 展平成单一 qreg。

    使用 qc.find_bit(q).index 的全局 qubit 编号，避免 Regulus 把
    不同寄存器的局部 index 混为同一 qubit。
    """
    from qiskit import QuantumCircuit

    flat = QuantumCircuit(
        int(qc.num_qubits),
        name=(qc.name or "reqisc_input_flat"),
    )

    try:
        flat.global_phase = qc.global_phase
    except Exception:
        pass

    for item in qc.data:
        if hasattr(item, "operation"):
            op = item.operation
            qargs = item.qubits
            cargs = item.clbits
        else:
            op, qargs, cargs = item

        if cargs:
            raise RuntimeError(
                "ReQISC 输入展平前仍存在 classical arguments："
                f"gate={getattr(op, 'name', type(op).__name__)}"
            )

        mapped_indices = [
            int(qc.find_bit(q).index)
            for q in qargs
        ]

        if len(mapped_indices) != len(set(mapped_indices)):
            raise RuntimeError(
                "输入线路自身存在重复 qubit operand："
                f"gate={getattr(op, 'name', type(op).__name__)}, "
                f"qubits={mapped_indices}"
            )

        flat.append(
            op,
            [flat.qubits[i] for i in mapped_indices],
            [],
        )

    return flat


def _strip_trailing_measurement_block_from_qasm_text(
    qasm_file: str,
) -> Tuple[str, int]:
    """只删除 QASM 文件末尾连续的 measurement 块。

    这里严格只删除“文件末尾的 measurement 块”：
      - 中间 measurement 不删除；
      - reset/条件控制不删除；
      - 量子门、门顺序、qreg 声明均不修改；
      - 原始 QASM 文件本身不修改，只生成临时清洗文本。
    """
    path = os.path.abspath(str(qasm_file))
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    remove_indices: Set[int] = set()
    found_measure = False

    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip()
        lowered = stripped.lower()

        # 允许文件尾部或 measurement 之间有空行/单行注释。
        if not stripped or lowered.startswith("//"):
            continue

        if lowered.startswith("measure "):
            remove_indices.add(idx)
            found_measure = True
            continue

        # 一旦越过末尾 measurement 块，就停止，绝不删除线路中间测量。
        if found_measure:
            break
        break

    if not remove_indices:
        return "".join(lines), 0

    cleaned = "".join(
        line
        for idx, line in enumerate(lines)
        if idx not in remove_indices
    )
    return cleaned, int(len(remove_indices))


def _load_qasm_with_trailing_measurement_fallback(
    qasm_file: str,
):
    """正常解析优先；失败时只对末尾 measurement 做一次兼容重试。

    返回：
        (QuantumCircuit, meta)

    正常 QASM 完全不改变；只有类似 hhl_n10/hhl_n14 这类因为文件末尾
    未定义 q/c measurement 导致 QASM2ParseError 的线路才触发 fallback。
    """
    from qiskit import QuantumCircuit

    source_path = os.path.abspath(str(qasm_file))

    try:
        qc = QuantumCircuit.from_qasm_file(source_path)
        return qc, {
            "qasm_parse_fallback_used": False,
            "preparse_trailing_measurements_removed": 0,
        }
    except Exception as first_exc:
        cleaned_text, removed = (
            _strip_trailing_measurement_block_from_qasm_text(source_path)
        )

        if removed <= 0:
            raise RuntimeError(
                "QASM_PARSE_ERROR | "
                f"{type(first_exc).__name__}: {first_exc}"
            ) from first_exc

        fd, temp_path = tempfile.mkstemp(
            prefix="reqisc_hhl_parser_clean_",
            suffix=".qasm",
            text=True,
        )
        os.close(fd)

        try:
            with open(temp_path, "w", encoding="utf-8", newline="") as f:
                f.write(cleaned_text)

            try:
                qc = QuantumCircuit.from_qasm_file(temp_path)
            except Exception as second_exc:
                raise RuntimeError(
                    "QASM_PARSE_ERROR_AFTER_TRAILING_MEASUREMENT_CLEANUP | "
                    f"source={source_path} | removed={removed} | "
                    f"first={type(first_exc).__name__}: {first_exc} | "
                    f"retry={type(second_exc).__name__}: {second_exc}"
                ) from second_exc
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        print(
            "      [QASM parser兼容] "
            f"正常解析失败，已仅删除文件末尾 measurement={removed} 条后重试成功 | "
            f"source={os.path.basename(source_path)}",
            flush=True,
        )

        return qc, {
            "qasm_parse_fallback_used": True,
            "preparse_trailing_measurements_removed": int(removed),
        }


def preprocess_qasm_for_reqisc(
    qasm_file: str,
    save_path: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """QASM -> ReQISC 可接受的 unitary 输入。

    只做输入兼容：
      1. 删除最终 measurement；
      2. 删除 barrier；
      3. 将 majority/ctu/cH 等展开到 u/cx/ccx；
      4. 保留 ccx，使 ReQISC template synthesis 仍可触发；
      5. 展平多个 qreg；
      6. optimization_level=0，不做额外 Qiskit 优化。
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers

    qc, qasm_parse_meta = _load_qasm_with_trailing_measurement_fallback(
        qasm_file
    )

    declared_qubits = int(qc.num_qubits)
    original_total_gates = int(qc.size())
    original_twoq_gates = int(
        sum(
            1
            for item in qc.data
            if len(item.qubits if hasattr(item, "qubits") else item[1]) == 2
        )
    )
    original_threeq_gates = int(
        sum(
            1
            for item in qc.data
            if len(item.qubits if hasattr(item, "qubits") else item[1]) == 3
        )
    )
    original_ops = {
        str(k): int(v)
        for k, v in qc.count_ops().items()
    }

    try:
        qc.remove_final_measurements(inplace=True)
    except Exception:
        pass

    qc = PassManager([RemoveBarriers()]).run(qc)

    remaining_measure = sum(
        1
        for item in qc.data
        if _qiskit_instruction_name(item) == "measure"
    )
    remaining_reset = sum(
        1
        for item in qc.data
        if _qiskit_instruction_name(item) == "reset"
    )
    remaining_condition = sum(
        1
        for item in qc.data
        if _qiskit_instruction_has_condition(item)
    )

    if (
        remaining_measure > 0
        or remaining_reset > 0
        or remaining_condition > 0
    ):
        raise RuntimeError(
            "DYNAMIC_CIRCUIT_UNSUPPORTED | "
            f"measure={remaining_measure}, "
            f"reset={remaining_reset}, "
            f"conditioned_ops={remaining_condition}"
        )

    try:
        qc_basis = transpile(
            qc,
            basis_gates=list(_settings.REQISC_QISKIT_BASIS),
            optimization_level=0,
        )
    except Exception as exc:
        raise RuntimeError(
            "QASM_BASIS_TRANSLATION_ERROR | "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    qc_basis = PassManager([RemoveBarriers()]).run(qc_basis)
    qreg_count_before_flatten = int(len(qc_basis.qregs))
    qc_basis = flatten_qiskit_qubit_registers(qc_basis)

    prep_ops = {
        str(k): int(v)
        for k, v in qc_basis.count_ops().items()
    }

    allowed = set(_settings.REQISC_QISKIT_BASIS)
    unexpected = sorted(
        name
        for name in prep_ops
        if name not in allowed
    )
    if unexpected:
        raise RuntimeError(
            "ReQISC 输入适配后仍存在不支持门："
            f"{unexpected}"
        )

    saved_path = ""
    if _settings.REQISC_SAVE_PREPROCESSED_QASM and save_path:
        out_abs = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        try:
            from qiskit import qasm2
            qasm2.dump(qc_basis, out_abs)
            saved_path = out_abs
        except Exception:
            try:
                with open(out_abs, "w", encoding="utf-8") as f:
                    f.write(qc_basis.qasm())
                saved_path = out_abs
            except Exception as exc:
                print(
                    f"[ReQISC输入] WARNING 保存预处理QASM失败：{exc}",
                    flush=True,
                )

    return qc_basis, {
        "declared_qubits": declared_qubits,
        "original_total_gates": original_total_gates,
        "original_twoq_gates": original_twoq_gates,
        "original_threeq_gates": original_threeq_gates,
        "original_ops": original_ops,
        "preprocessed_total_gates": int(qc_basis.size()),
        "preprocessed_twoq_gates": int(
            sum(1 for item in qc_basis.data if len(item.qubits) == 2)
        ),
        "preprocessed_threeq_gates": int(
            sum(1 for item in qc_basis.data if len(item.qubits) == 3)
        ),
        "preprocessed_ops": prep_ops,
        "qreg_count_before_flatten": qreg_count_before_flatten,
        "qreg_count_after_flatten": 1,
        "qasm_parse_fallback_used": bool(
            qasm_parse_meta.get("qasm_parse_fallback_used", False)
        ),
        "preparse_trailing_measurements_removed": int(
            qasm_parse_meta.get("preparse_trailing_measurements_removed", 0)
        ),
        "preprocessed_qasm": saved_path,
    }


def extract_reqisc_program_context(
    circ_opt,
) -> _mod_models.ReqiscProgramContext:
    """从 ReQISC canonical 完整线路中保存所有 1Q 门及其相对 2Q 位置。

    本文路由器仍然只对 2Q DAG 做路由；但为了计算真正的 program fidelity，
    不能丢掉 ReQISC 综合后已有的 1Q 门。

    做法：
      - 扫描完整 canonical 线路；
      - 每个逻辑 qubit 上，积累尚未遇到下一次 2Q 交互的 1Q 门；
      - 当遇到第 i 个 2Q 门时，把它两个逻辑 qubit 上积累的 1Q 门
        绑定到 before_twoq[i]；
      - 最后剩余的 1Q 门放入 trailing。

    这样在路由后重建物理线路时，只要按 routed block 重放映射，
    就可以把这些 1Q 门放到正确的“逻辑 wire 时序位置”。
    """
    pending: Dict[int, List[_mod_models.OneQProgramOp]] = {}
    before_twoq: Dict[int, List[_mod_models.OneQProgramOp]] = {}
    twoq_matrices: Dict[int, np.ndarray] = {}

    twoq_idx = 0
    oneq_count = 0

    for program_index, g in enumerate(circ_opt):
        nq = int(g.num_qregs)
        qregs = [int(q) for q in g.qregs]

        if nq == 1:
            q = int(qregs[0])
            mat = np.asarray(g.data, dtype=complex)
            if mat.shape != (2, 2):
                raise RuntimeError(
                    "ReQISC 1Q gate matrix 维度异常 | "
                    f"gate={g} | shape={mat.shape}"
                )

            pending.setdefault(q, []).append(
                _mod_models.OneQProgramOp(
                    program_index=int(program_index),
                    logical_q=q,
                    name=str(
                        getattr(
                            g,
                            "name",
                            type(g).__name__,
                        )
                    ),
                    matrix=mat.copy(),
                )
            )
            oneq_count += 1
            continue

        if nq == 2:
            mat2 = np.asarray(
                g.data,
                dtype=complex,
            )
            if mat2.shape != (4, 4):
                raise RuntimeError(
                    "ReQISC 2Q gate matrix 维度异常 | "
                    f"gate={g} | shape={mat2.shape}"
                )
            twoq_matrices[int(twoq_idx)] = (
                mat2.copy()
            )

            attached: List[_mod_models.OneQProgramOp] = []

            for q in qregs:
                attached.extend(
                    pending.pop(int(q), [])
                )

            # 不同 qubit 上的 1Q 门彼此可交换；按原 program_index 排序，
            # 保留最直观的原始线路次序。
            attached.sort(
                key=lambda op: int(op.program_index)
            )

            if attached:
                before_twoq[int(twoq_idx)] = attached

            twoq_idx += 1
            continue

        raise RuntimeError(
            "ReQISC canonical rebase 后仍发现 >2Q 操作，"
            "无法构造完整 program fidelity | "
            f"program_index={program_index} | "
            f"gate={g} | num_qregs={nq}"
        )

    trailing: List[_mod_models.OneQProgramOp] = []
    for ops in pending.values():
        trailing.extend(ops)

    trailing.sort(
        key=lambda op: int(op.program_index)
    )

    return _mod_models.ReqiscProgramContext(
        before_twoq=before_twoq,
        trailing=trailing,
        twoq_matrices=twoq_matrices,
        num_oneq=int(oneq_count),
        num_twoq=int(twoq_idx),
    )


def reqisc_circuit_to_twoq_gates(
    circ_opt,
) -> List["TwoQGate"]:
    """ReQISC canonical Circuit -> 本文 TwoQGate，直接传 Cartan 坐标。"""
    from . import common as _mod_common
    result: List[_mod_models.TwoQGate] = []

    for idx, g in enumerate(circ_opt):
        if int(g.num_qregs) != 2:
            continue

        qregs = list(g.qregs)
        if len(qregs) != 2:
            raise RuntimeError(
                f"ReQISC 2Q门 qregs 异常：gate={g}, qregs={qregs}"
            )

        try:
            x, y, z = [
                float(v)
                for v in g.kak_coefficients
            ]
        except Exception as exc:
            raise RuntimeError(
                "无法从 ReQISC canonical 2Q 门提取 "
                f"kak_coefficients：gate={g}"
            ) from exc

        x, y, z = _mod_common.normalize_cartan_to_balanced(
            (x, y, z)
        )

        result.append(
            _mod_models.TwoQGate(
                name=(
                    f"REQISC_"
                    f"{getattr(g, 'name', type(g).__name__)}_{idx}"
                ),
                q0=int(qregs[0]),
                q1=int(qregs[1]),
                x=float(x),
                y=float(y),
                z=float(z),
            )
        )

    return result


def save_reqisc_canonical_csv(
    csv_path: str,
    target_gates: List["TwoQGate"],
    declared_qubits: int,
) -> None:
    os.makedirs(
        os.path.dirname(os.path.abspath(csv_path)),
        exist_ok=True,
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gate_id",
                "name",
                "q0",
                "q1",
                "x",
                "y",
                "z",
                "declared_qubits",
            ],
        )
        writer.writeheader()

        for gate_id, g in enumerate(target_gates):
            writer.writerow({
                "gate_id": int(gate_id),
                "name": str(g.name),
                "q0": int(g.q0),
                "q1": int(g.q1),
                "x": f"{float(g.x):.15f}",
                "y": f"{float(g.y):.15f}",
                "z": f"{float(g.z):.15f}",
                "declared_qubits": int(declared_qubits),
            })


def run_reqisc_logical_optimization(
    qc_basis,
    ctx: _mod_models.ReqiscOptimizeContext,
) -> Tuple[List["TwoQGate"], Dict[str, Any], _mod_models.ReqiscProgramContext]:
    """只执行 ReQISC 的逻辑综合优化，不执行 ReQISC 路由。

    ReQISC 在这里结束：
      template synthesis
        -> hierarchical synthesis
        -> canonical rebase
        -> Can(x,y,z)

    下一步直接进入本文框架。
    """
    from . import common as _mod_common
    t0 = time.perf_counter()

    try:
        circ = ctx.Circuit.from_qiskit(qc_basis)
    except Exception as exc:
        raise RuntimeError(
            "REGULUS_CONVERSION_ERROR | "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    before_gates = int(circ.num_gates)
    before_2q = int(circ.num_nonlocal_gates)
    before_depth = int(circ.depth)
    before_depth_2q = int(circ.depth_nonlocal)

    circ_opt = circ
    template_used = False

    with _mod_common.reqisc_print_compat_context():
        if (
            _settings.REQISC_TEMPLATE_SYNTHESIS
            and int(circ_opt.max_gate_weight) == 3
        ):
            template_used = True
            print(
                "[ReQISC 1/3] template synthesis 开始",
                flush=True,
            )
            circ_opt = ctx.template_based_synthesize(
                circ_opt,
                num_trials=int(_settings.REQISC_TEMPLATE_NUM_TRIALS),
                bqskit_compiler=ctx.compiler,
                structures_unique=ctx.structures_unique,
            )
            print(
                "[ReQISC 1/3] template synthesis 完成",
                flush=True,
            )
        else:
            print(
                "[ReQISC 1/3] template synthesis 跳过 | "
                f"max_gate_weight={int(circ_opt.max_gate_weight)}",
                flush=True,
            )

        if _settings.REQISC_FULL_SYNTHESIS:
            print(
                "[ReQISC 2/3] hierarchical synthesis 开始",
                flush=True,
            )
            circ_opt = ctx.hierarchical_synthesize(
                circ_opt,
                bqskit_compiler=ctx.compiler,
                num_approx_syn_trials=max(
                    1,
                    int(_settings.REQISC_NUM_APPROX_SYN_TRIALS),
                ),
            )
            print(
                "[ReQISC 2/3] hierarchical synthesis 完成",
                flush=True,
            )

        print(
            "[ReQISC 3/3] canonical rebase 开始",
            flush=True,
        )
        circ_opt = ctx.rebase_to_canonical_by_bqskit(
            circ_opt,
            ctx.compiler,
        )
        print(
            "[ReQISC 3/3] canonical rebase 完成",
            flush=True,
        )

    # 完整 program fidelity 需要保留 ReQISC canonical 线路中的 1Q 门。
    program_ctx = extract_reqisc_program_context(
        circ_opt
    )

    target_gates = reqisc_circuit_to_twoq_gates(
        circ_opt
    )

    if int(program_ctx.num_twoq) != int(len(target_gates)):
        raise RuntimeError(
            "ReQISC 完整线路 2Q 计数与 target_gates 不一致 | "
            f"program_ctx={program_ctx.num_twoq} | "
            f"target_gates={len(target_gates)}"
        )

    after_gates = int(circ_opt.num_gates)
    after_2q = int(circ_opt.num_nonlocal_gates)
    after_depth = int(circ_opt.depth)
    after_depth_2q = int(circ_opt.depth_nonlocal)

    elapsed = time.perf_counter() - t0

    return target_gates, {
        "template_used": bool(template_used),
        "reqisc_after_oneq": int(program_ctx.num_oneq),
        "reqisc_before_gates": before_gates,
        "reqisc_before_2q": before_2q,
        "reqisc_before_depth": before_depth,
        "reqisc_before_depth_2q": before_depth_2q,
        "reqisc_after_gates": after_gates,
        "reqisc_after_2q": after_2q,
        "reqisc_after_depth": after_depth,
        "reqisc_after_depth_2q": after_depth_2q,
        "reqisc_elapsed_seconds": float(elapsed),
    }, program_ctx


def qc_qubit_index(qc, qobj) -> int:
    try:
        return qc.find_bit(qobj).index
    except Exception:
        pass

    try:
        return qc.qubits.index(qobj)
    except Exception:
        pass

    if hasattr(qobj, "_index"):
        return int(qobj._index)

    raise ValueError(f"无法获取量子比特索引，qobj={qobj}")


def extract_cartan_from_unitary(U):
    try:
        from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
        dec = TwoQubitWeylDecomposition(U)
        return float(dec.a), float(dec.b), float(dec.c)
    except Exception as exc:
        raise ValueError(f"无法从 2Q unitary 提取 Cartan 坐标，{exc}")


def qc_to_twoq_gates(qc) -> List[_mod_models.TwoQGate]:
    from . import common as _mod_common
    gates: List[_mod_models.TwoQGate] = []
    for idx, item in enumerate(qc.data):
        if hasattr(item, "operation"):
            op = item.operation
            qargs = item.qubits
            cargs = item.clbits
        else:
            op, qargs, cargs = item

        if len(qargs) != 2:
            continue

        q0 = qc_qubit_index(qc, qargs[0])
        q1 = qc_qubit_index(qc, qargs[1])
        name = getattr(op, "name", f"g{idx}")
        params = list(getattr(op, "params", []))
        lower = str(name).lower()

        if lower in {"cp", "cphase", "cu1"} and len(params) >= 1:
            theta = float(params[0])
            x, y, z = theta / 4.0, 0.0, 0.0
        elif lower in {"cx", "cnot"}:
            x, y, z = math.pi / 4, 0.0, 0.0
        elif lower == "swap":
            x, y, z = math.pi / 4, math.pi / 4, math.pi / 4
        elif lower == "iswap":
            x, y, z = math.pi / 4, math.pi / 4, 0.0
        else:
            try:
                from qiskit.quantum_info import Operator
                U = Operator(op).data
            except Exception:
                U = op.to_matrix()
            x, y, z = extract_cartan_from_unitary(U)

        x, y, z = _mod_common.normalize_cartan_to_balanced((x, y, z))
        gates.append(_mod_models.TwoQGate(f"{name}_{idx}", q0, q1, x, y, z))

    return gates


def load_qasm_twoq_gates(qasm_file: str) -> Tuple[List[_mod_models.TwoQGate], int, int]:
    """读取一个 QASM 文件，并提取其中所有两比特门。"""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit.from_qasm_file(str(qasm_file))
    return qc_to_twoq_gates(qc), int(qc.num_qubits), int(qc.size())

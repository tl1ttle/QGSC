

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
from scipy.optimize import minimize
from . import models as _mod_models
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import common as _mod_common
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import decomposition_routing as _mod_decomposition_routing

_PDBLS_MODULE = None


def resolve_support_file(
    configured_path: str,
    filename: str,
) -> str:
    
    script_dir = str(_ISING_ROOT)
    parent_dir = os.path.dirname(script_dir)

    candidates = [
        os.path.abspath(configured_path),
        os.path.join(script_dir, filename),
        os.path.join(parent_dir, filename),
        os.path.join(os.getcwd(), filename),
    ]

    seen: Set[str] = set()
    checked: List[str] = []

    for path in candidates:
        path = os.path.abspath(path)
        key = os.path.normcase(
            os.path.normpath(path)
        )
        if key in seen:
            continue
        seen.add(key)
        checked.append(path)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"找不到 {filename}。已检查："
        + " | ".join(checked)
    )


def load_pdbls_module():
    
    global _PDBLS_MODULE

    if _PDBLS_MODULE is not None:
        return _PDBLS_MODULE

    path = resolve_support_file(
        _settings.PDBLS_PY,
        "core/pdbls.py",
    )

    spec = importlib.util.spec_from_file_location(
        "user_pdbls_full_program",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法加载 core/pdbls.py：{path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[
        "user_pdbls_full_program"
    ] = module
    spec.loader.exec_module(module)

    for attr in (
        "canonical_gate",
        "compose_template_unitary",
        "process_fidelity",
        "u3_unitary",
    ):
        if not hasattr(module, attr):
            raise RuntimeError(
                "core/pdbls.py 缺少必要对象："
                f"{attr}"
            )

    print(
        f"      [Fidelity依赖] core/pdbls.py={path}",
        flush=True,
    )

    _PDBLS_MODULE = module
    return module


def _clip_pdbls_x(
    x: np.ndarray,
    bound: float,
    eps: float = 1e-8,
) -> np.ndarray:
    
    B = abs(float(bound))
    if (not np.isfinite(B)) or B <= 0.0:
        B = 2.0 * math.pi

    lo = -B + eps
    hi = B - eps

    arr = np.asarray(
        x,
        dtype=float,
    ).reshape(-1).copy()

    arr[~np.isfinite(arr)] = 0.0
    np.clip(
        arr,
        lo,
        hi,
        out=arr,
    )
    return arr


def solve_pdbls_fixed_k(
    pdbls,
    target_cartan: Tuple[float, float, float],
    base_cartan: Tuple[float, float, float],
    k: int,
    *,
    target_matrix: Optional[np.ndarray] = None,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    
    k = int(k)
    if k < 0:
        raise ValueError(
            f"fixed-k 必须 >=0，当前 k={k}"
        )

    if target_matrix is None:
        U_target = np.asarray(
            pdbls.canonical_gate(
                *[float(v) for v in target_cartan]
            ),
            dtype=complex,
        )
    else:
        U_target = np.asarray(
            target_matrix,
            dtype=complex,
        )
        if U_target.shape != (4, 4):
            raise ValueError(
                "target_matrix 必须为 4x4 | "
                f"shape={U_target.shape}"
            )
    U_basis = np.asarray(
        pdbls.canonical_gate(
            *[float(v) for v in base_cartan]
        ),
        dtype=complex,
    )

    if k == 0:
        I4 = np.eye(
            4,
            dtype=complex,
        )
        pf = float(
            pdbls.process_fidelity(
                I4,
                U_target,
            )
        )
        return {
            "k": 0,
            "params": np.zeros(
                6,
                dtype=float,
            ),
            "pf_decomp": pf,
            "traceF": math.sqrt(
                max(0.0, pf)
            ),
            "elapsed_s": 0.0,
            "U_basis": U_basis,
        }

    basis_list = [
        U_basis
        for _ in range(k)
    ]

    n_params = 6 * (k + 1)
    t0 = time.perf_counter()

    def loss(x: np.ndarray) -> float:
        x = _clip_pdbls_x(
            x,
            _settings.FIDELITY_PDBLS_BOUNDS_ANGLE,
        )
        U_dec = pdbls.compose_template_unitary(
            x,
            basis_list,
        )
        pf = float(
            pdbls.process_fidelity(
                U_dec,
                U_target,
            )
        )
        return float(1.0 - pf)

    
    
    seed_payload = (
        int(_settings.FIDELITY_PDBLS_SEED)
        + int(seed_offset) * 1009
        + int(k) * 7919
        + int(
            round(
                1e6
                * sum(
                    abs(float(v))
                    for v in target_cartan
                )
            )
        )
    )
    rng = np.random.default_rng(
        seed_payload
    )

    starts: List[np.ndarray] = [
        np.zeros(
            n_params,
            dtype=float,
        )
    ]

    B = abs(
        float(
            _settings.FIDELITY_PDBLS_BOUNDS_ANGLE
        )
    )
    for _ in range(
        max(
            0,
            int(_settings.FIDELITY_PDBLS_RESTARTS) - 1,
        )
    ):
        starts.append(
            rng.uniform(
                -B + 1e-8,
                B - 1e-8,
                size=n_params,
            ).astype(float)
        )

    best_pf = -1.0
    best_params: Optional[np.ndarray] = None

    for ridx, x0 in enumerate(starts):
        x0 = _clip_pdbls_x(
            x0,
            B,
        )

        opt = minimize(
            loss,
            x0=x0,
            method="L-BFGS-B",
            options={
                "maxiter": int(
                    _settings.FIDELITY_PDBLS_MAXITER
                ),
                "ftol": 1e-15,
            },
        )

        x_best = _clip_pdbls_x(
            np.asarray(
                opt.x,
                dtype=float,
            ),
            B,
        )

        U_dec = pdbls.compose_template_unitary(
            x_best,
            basis_list,
        )
        pf = float(
            pdbls.process_fidelity(
                U_dec,
                U_target,
            )
        )

        if pf > best_pf:
            best_pf = pf
            best_params = x_best.copy()

        if pf >= float(
            _settings.FIDELITY_DECOMP_FD
        ):
            break

    if best_params is None:
        raise RuntimeError(
            "固定 k PDBLS 未返回有效参数 | "
            f"target={target_cartan} | k={k}"
        )

    return {
        "k": int(k),
        "params": best_params,
        "pf_decomp": float(best_pf),
        "traceF": math.sqrt(
            max(0.0, float(best_pf))
        ),
        "elapsed_s": float(
            time.perf_counter() - t0
        ),
        "U_basis": U_basis,
    }


def _complex_matrix_to_json(
    mat: np.ndarray,
) -> Dict[str, Any]:
    arr = np.asarray(
        mat,
        dtype=complex,
    )
    return {
        "real": arr.real.tolist(),
        "imag": arr.imag.tolist(),
    }


def build_actual_physical_program(
    best: Dict[str, Any],
    program_ctx: _mod_models.ReqiscProgramContext,
) -> Dict[str, Any]:
    
    from . import common as _mod_common
    from . import decomposition_routing as _mod_decomposition_routing
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
    except Exception as exc:
        raise RuntimeError(
            "完整 program fidelity 需要 qiskit。"
        ) from exc

    try:
        import cirq
    except Exception as exc:
        raise RuntimeError(
            "完整 program fidelity 需要 cirq。"
        ) from exc

    pdbls = load_pdbls_module()

    routed_state = best.get("routed_state")
    if routed_state is None:
        raise RuntimeError("best 中没有 routed_state")

    n = int(len(routed_state.logical_to_physical))
    if n <= 0:
        raise RuntimeError("路由后物理 qubit 数量非法")

    base_cartan = tuple(float(v) for v in best["base_cartan"])
    tau_basis = float(best["tau_basis"])

    p_g = (
        float(_settings.FIDELITY_P0)
        * tau_basis
        / float(_settings.FIDELITY_TAU_CX)
    )
    p_g = min(1.0, max(0.0, float(p_g)))

    U_basis = np.asarray(
        pdbls.canonical_gate(*base_cartan),
        dtype=complex,
    )

    qc = QuantumCircuit(n, name="ours_actual_physical")
    cirq_circuit = cirq.Circuit()
    cirq_qubits = cirq.LineQubit.range(n)

    final_l2p = tuple(
        int(v)
        for v in routed_state.logical_to_physical
    )
    final_p2l = tuple(
        int(v)
        for v in routed_state.physical_to_logical
    )

    
    decomp_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    decomp_pf_values: List[float] = []
    low_pf_count = 0
    actual_g_count = 0
    actual_1q_count = 0
    actual_ops: List[Dict[str, Any]] = []
    consumed_source_ids: Set[int] = set()

    def _validate_mapping_snapshot(
        snapshot: Tuple[int, ...],
        *,
        label: str,
        block_id: int,
    ) -> Tuple[int, ...]:
        snap = tuple(int(v) for v in snapshot)
        if len(snap) != n:
            raise RuntimeError(
                f"{label} 长度错误 | block={block_id} | "
                f"len={len(snap)} expected={n}"
            )
        if sorted(snap) != list(range(n)):
            raise RuntimeError(
                f"{label} 不是 0..N-1 的一一映射 | "
                f"block={block_id} | mapping={snap}"
            )
        return snap

    def append_original_oneq(
        oneq: _mod_models.OneQProgramOp,
        mapping_snapshot: Tuple[int, ...],
        *,
        anchor_block_id: Optional[int],
        anchor: str,
    ) -> None:
        nonlocal actual_1q_count

        snap = _validate_mapping_snapshot(
            tuple(mapping_snapshot),
            label=f"{anchor}.mapping",
            block_id=(-1 if anchor_block_id is None else int(anchor_block_id)),
        )

        logical_q = int(oneq.logical_q)
        if not (0 <= logical_q < n):
            raise RuntimeError(
                f"ReQISC 1Q logical_q 越界 | logical_q={logical_q} n={n}"
            )

        physical_q = int(snap[logical_q])
        mat = np.asarray(oneq.matrix, dtype=complex)

        qc.append(
            UnitaryGate(mat, label="REQISC1Q:" + str(oneq.name)),
            [physical_q],
        )
        cirq_circuit.append(
            cirq.MatrixGate(mat).on(cirq_qubits[physical_q])
        )

        actual_1q_count += 1
        actual_ops.append({
            "type": "reqisc_1q",
            "program_index": int(oneq.program_index),
            "name": str(oneq.name),
            "logical_q": logical_q,
            "physical_q": physical_q,
            "mapping_anchor": str(anchor),
            "anchor_block_id": (
                None if anchor_block_id is None else int(anchor_block_id)
            ),
            "mapping_snapshot": [int(v) for v in snap],
            "matrix": _complex_matrix_to_json(mat),
        })

    def append_pdbls_block(
        blk: _mod_models.RoutedBlock,
        source_gate_idx: Optional[int],
    ) -> Dict[str, Any]:
        nonlocal actual_g_count
        nonlocal actual_1q_count
        nonlocal low_pf_count

        target_cartan = tuple(float(v) for v in blk.cartan)

        ratio = float(blk.duration) / float(tau_basis)
        k = int(round(ratio))
        if abs(ratio - float(k)) > 1e-7:
            raise RuntimeError(
                "无法从 routed block duration 恢复固定 k | "
                f"op_id={blk.op_id} | duration={blk.duration} | "
                f"tau_basis={tau_basis} | ratio={ratio}"
            )

        mapping_before = _validate_mapping_snapshot(
            tuple(blk.mapping_before),
            label="mapping_before",
            block_id=int(blk.op_id),
        )
        mapping_after = _validate_mapping_snapshot(
            tuple(blk.mapping_after),
            label="mapping_after",
            block_id=int(blk.op_id),
        )

        p0 = int(blk.physical_q0_ordered)
        p1 = int(blk.physical_q1_ordered)
        if not (0 <= p0 < n and 0 <= p1 < n and p0 != p1):
            raise RuntimeError(
                "routed block 有序物理操作数非法 | "
                f"op_id={blk.op_id} | ordered=({p0},{p1}) | n={n}"
            )

        if _mod_decomposition_routing.edge_key(p0, p1) != _mod_decomposition_routing.edge_key(int(blk.q0), int(blk.q1)):
            raise RuntimeError(
                "有序物理操作数与无向拓扑边不一致 | "
                f"op_id={blk.op_id} | ordered=({p0},{p1}) | "
                f"edge=({blk.q0},{blk.q1})"
            )

        
        
        if source_gate_idx is not None:
            if blk.logical_pair is None:
                raise RuntimeError(
                    "source 2Q block 缺少 logical_pair | "
                    f"op_id={blk.op_id} source={source_gate_idx}"
                )
            l0, l1 = [int(v) for v in blk.logical_pair]
            if (
                int(mapping_before[l0]) != p0
                or int(mapping_before[l1]) != p1
            ):
                raise RuntimeError(
                    "source 2Q 的 ordered physical pair 与 mapping_before 不一致 | "
                    f"op_id={blk.op_id} | logical=({l0},{l1}) | "
                    f"mapping_before->({mapping_before[l0]},{mapping_before[l1]}) | "
                    f"ordered=({p0},{p1})"
                )

        
        if blk.kind in {"swap", "swap_absorbed"}:
            expected_after = _mod_decomposition_routing.mapping_after_physical_swap_snapshot(
                mapping_before,
                (int(blk.q0), int(blk.q1)),
            )
            if mapping_after != expected_after:
                raise RuntimeError(
                    "SWAP block mapping_after 与局部 SWAP 语义不一致 | "
                    f"op_id={blk.op_id} | before={mapping_before} | "
                    f"after={mapping_after} | expected={expected_after}"
                )
        else:
            if mapping_after != mapping_before:
                raise RuntimeError(
                    "非 SWAP block 不应改变 mapping | "
                    f"op_id={blk.op_id} | before={mapping_before} | after={mapping_after}"
                )

        
        
        
        
        
        
        target_matrix: np.ndarray
        target_identity: Tuple[Any, ...]

        if blk.kind == "swap_absorbed":
            if source_gate_idx is None:
                raise RuntimeError(
                    "swap_absorbed 缺少 source_gate_idx | "
                    f"op_id={blk.op_id}"
                )

            U_original = np.asarray(
                program_ctx.twoq_matrices[int(source_gate_idx)],
                dtype=complex,
            )
            original_cartan = (
                tuple(float(v) for v in blk.absorbed_from_cartan)
                if blk.absorbed_from_cartan is not None
                else target_cartan
            )
            U_swap = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=complex,
            )
            target_matrix = U_swap @ U_original
            target_identity = (
                "swap_absorbed",
                int(source_gate_idx),
                _mod_common.cartan_key(*original_cartan),
                _mod_common.cartan_key(*target_cartan),
                int(k),
            )

        elif blk.kind == "swap":
            target_matrix = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=complex,
            )
            target_identity = ("swap", int(k))

        else:
            if source_gate_idx is not None:
                target_matrix = np.asarray(
                    program_ctx.twoq_matrices[int(source_gate_idx)],
                    dtype=complex,
                )
                
                
                target_identity = (
                    "source_unitary",
                    int(source_gate_idx),
                    int(k),
                )
            else:
                target_matrix = np.asarray(
                    pdbls.canonical_gate(*target_cartan),
                    dtype=complex,
                )
                target_identity = (
                    "canonical",
                    _mod_common.cartan_key(*target_cartan),
                    int(k),
                )

        if target_identity not in decomp_cache:
            decomp_cache[target_identity] = solve_pdbls_fixed_k(
                pdbls,
                target_cartan,
                base_cartan,
                k,
                target_matrix=target_matrix,
                seed_offset=int(len(decomp_cache)),
            )

        dec = decomp_cache[target_identity]
        pf = float(dec["pf_decomp"])
        decomp_pf_values.append(pf)
        if pf < float(_settings.FIDELITY_DECOMP_FD):
            low_pf_count += 1

        params = np.asarray(dec["params"], dtype=float)

        if k == 0:
            return {
                "block_id": int(blk.op_id),
                "kind": str(blk.kind),
                "k": 0,
                "pf_decomp": pf,
                "physical_pair_ordered": [p0, p1],
                "mapping_before": [int(v) for v in mapping_before],
                "mapping_after": [int(v) for v in mapping_after],
            }

        if params.size != 6 * (k + 1):
            raise RuntimeError(
                "PDBLS 参数长度与固定 k 不一致 | "
                f"k={k} | params={params.size}"
            )

        for layer in range(k + 1):
            off = 6 * layer
            t0, ph0, la0 = [float(v) for v in params[off : off + 3]]
            t1, ph1, la1 = [float(v) for v in params[off + 3 : off + 6]]

            
            
            qc.u(t0, ph0, la0, p0)
            qc.u(t1, ph1, la1, p1)

            U0 = np.asarray(pdbls.u3_unitary(t0, ph0, la0), dtype=complex)
            U1 = np.asarray(pdbls.u3_unitary(t1, ph1, la1), dtype=complex)
            cirq_circuit.append([
                cirq.MatrixGate(U0).on(cirq_qubits[p0]),
                cirq.MatrixGate(U1).on(cirq_qubits[p1]),
            ])

            actual_1q_count += 2
            actual_ops.extend([
                {
                    "type": "pdbls_1q",
                    "block_id": int(blk.op_id),
                    "layer": int(layer),
                    "pdbls_operand": "q0",
                    "physical_q": p0,
                    "theta": t0,
                    "phi": ph0,
                    "lambda": la0,
                },
                {
                    "type": "pdbls_1q",
                    "block_id": int(blk.op_id),
                    "layer": int(layer),
                    "pdbls_operand": "q1",
                    "physical_q": p1,
                    "theta": t1,
                    "phi": ph1,
                    "lambda": la1,
                },
            ])

            if layer >= k:
                continue

            qc.append(
                UnitaryGate(U_basis, label="G(theta)"),
                [p0, p1],
            )
            cirq_circuit.append(
                cirq.MatrixGate(U_basis).on(
                    cirq_qubits[p0],
                    cirq_qubits[p1],
                )
            )

            actual_g_count += 1
            actual_ops.append({
                "type": "G",
                "block_id": int(blk.op_id),
                "physical_q0_ordered": p0,
                "physical_q1_ordered": p1,
                "edge_q0": int(blk.q0),
                "edge_q1": int(blk.q1),
                "x": float(base_cartan[0]),
                "y": float(base_cartan[1]),
                "z": float(base_cartan[2]),
                "tau": float(tau_basis),
                "p_depolarizing": float(p_g),
            })

        return {
            "block_id": int(blk.op_id),
            "kind": str(blk.kind),
            "k": int(k),
            "pf_decomp": pf,
            "physical_pair_ordered": [p0, p1],
            "edge": [int(blk.q0), int(blk.q1)],
            "mapping_before": [int(v) for v in mapping_before],
            "mapping_after": [int(v) for v in mapping_after],
        }

    block_reports: List[Dict[str, Any]] = []

    for blk in routed_state.routed_blocks:
        source_gate_idx = (
            None
            if blk.source_gate_idx is None
            else int(blk.source_gate_idx)
        )

        
        
        if (
            source_gate_idx is not None
            and source_gate_idx not in consumed_source_ids
        ):
            source_mapping = _validate_mapping_snapshot(
                tuple(blk.mapping_before),
                label="source.mapping_before",
                block_id=int(blk.op_id),
            )
            for oneq in program_ctx.before_twoq.get(source_gate_idx, []):
                append_original_oneq(
                    oneq,
                    source_mapping,
                    anchor_block_id=int(blk.op_id),
                    anchor="before_source_2q",
                )
            consumed_source_ids.add(source_gate_idx)

        report = append_pdbls_block(blk, source_gate_idx)
        block_reports.append(report)

        
        

    expected_source_ids = set(range(int(program_ctx.num_twoq)))
    missing_source_ids = sorted(expected_source_ids - consumed_source_ids)
    if missing_source_ids:
        raise RuntimeError(
            "最终 routed_blocks 缺少 ReQISC source 2Q 门 | "
            f"missing={missing_source_ids[:20]}"
        )

    
    for oneq in program_ctx.trailing:
        append_original_oneq(
            oneq,
            final_l2p,
            anchor_block_id=None,
            anchor="final_route_mapping",
        )

    min_pf = min(decomp_pf_values) if decomp_pf_values else 1.0
    mean_pf = (
        float(np.mean(decomp_pf_values))
        if decomp_pf_values
        else 1.0
    )
    semantic_pass = bool(min_pf >= float(_settings.FIDELITY_DECOMP_FD))

    expected_g_count = int(
        (best.get("routing", {}) or {}).get(
            "routed_native_2q",
            actual_g_count,
        )
    )
    if int(actual_g_count) != int(expected_g_count):
        raise RuntimeError(
            "最终真实线路 G 数量与路由统计不一致 | "
            f"actual_G={actual_g_count} | routed_native_2q={expected_g_count}"
        )

    return {
        "qiskit_circuit": qc,
        "cirq_circuit": cirq_circuit,
        "actual_ops": actual_ops,
        "block_reports": block_reports,
        "logical_to_physical": [int(v) for v in final_l2p],
        "physical_to_logical": [int(v) for v in final_p2l],
        "actual_1q_count": int(actual_1q_count),
        "actual_g_count": int(actual_g_count),
        "min_block_pf_decomp": float(min_pf),
        "mean_block_pf_decomp": float(mean_pf),
        "low_pf_block_count": int(low_pf_count),
        "semantic_check_pass": semantic_pass,
        "p_g": float(p_g),
        "tau_basis": float(tau_basis),
    }


def save_actual_physical_program(
    built: Dict[str, Any],
    output_dir: str,
    safe_name: str,
) -> Dict[str, str]:
    
    try:
        from qiskit import qpy
    except Exception as exc:
        raise RuntimeError(
            "保存 QPY 需要 qiskit.qpy"
        ) from exc

    out_dir = os.path.join(
        os.path.abspath(output_dir),
        _settings.ACTUAL_CIRCUIT_SUBDIR,
    )
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    qpy_path = os.path.join(
        out_dir,
        safe_name + "_actual.qpy",
    )
    json_path = os.path.join(
        out_dir,
        safe_name + "_actual.json",
    )
    txt_path = os.path.join(
        out_dir,
        safe_name + "_actual.txt",
    )

    with open(
        qpy_path,
        "wb",
    ) as f:
        qpy.dump(
            built[
                "qiskit_circuit"
            ],
            f,
        )

    payload = {
        "actual_1q_count": int(
            built["actual_1q_count"]
        ),
        "actual_g_count": int(
            built["actual_g_count"]
        ),
        "min_block_pf_decomp": float(
            built[
                "min_block_pf_decomp"
            ]
        ),
        "mean_block_pf_decomp": float(
            built[
                "mean_block_pf_decomp"
            ]
        ),
        "low_pf_block_count": int(
            built[
                "low_pf_block_count"
            ]
        ),
        "semantic_check_pass": bool(
            built[
                "semantic_check_pass"
            ]
        ),
        "logical_to_physical": [
            int(v)
            for v in built[
                "logical_to_physical"
            ]
        ],
        "physical_to_logical": [
            int(v)
            for v in built[
                "physical_to_logical"
            ]
        ],
        "block_reports": built[
            "block_reports"
        ],
        "operations": built[
            "actual_ops"
        ],
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    
    
    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "actual_1q_count="
            f"{built['actual_1q_count']}\\n"
        )
        f.write(
            "actual_g_count="
            f"{built['actual_g_count']}\\n"
        )
        f.write(
            "min_block_pf_decomp="
            f"{built['min_block_pf_decomp']:.12f}\\n"
        )
        f.write(
            "semantic_check_pass="
            f"{built['semantic_check_pass']}\\n"
        )
        f.write("\\n=== operations ===\\n")
        for op in built["actual_ops"]:
            f.write(
                json.dumps(
                    op,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            f.write("\\n")

    return {
        "qpy": os.path.abspath(
            qpy_path
        ),
        "json": os.path.abspath(
            json_path
        ),
        "txt": os.path.abspath(
            txt_path
        ),
    }

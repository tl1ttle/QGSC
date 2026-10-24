

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
from . import models as _mod_models
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import common as _mod_common
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import decomposition_routing as _mod_decomposition_routing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import fidelity as _mod_fidelity
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import io_utils as _mod_io_utils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import preprocessing as _mod_preprocessing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import decomposition_routing as _mod_decomposition_routing



def run_folder_batch() -> List[Dict[str, Any]]:
    from . import common as _mod_common
    from . import decomposition_routing as _mod_decomposition_routing
    from . import fidelity as _mod_fidelity
    from . import io_utils as _mod_io_utils
    from . import preprocessing as _mod_preprocessing
    qasm_files = _mod_io_utils.discover_qasm_files(_settings.QASM_FOLDER)
    output_dir = os.path.abspath(_settings.BATCH_RESULT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "logs")
    if _settings.BATCH_SAVE_PER_CIRCUIT_LOGS:
        os.makedirs(log_dir, exist_ok=True)

    reqisc_canonical_dir = os.path.join(
        output_dir,
        _settings.REQISC_CANONICAL_SUBDIR,
    )
    if _settings.REQISC_PREOPTIMIZE and _settings.REQISC_SAVE_CANONICAL_CSV:
        os.makedirs(reqisc_canonical_dir, exist_ok=True)

    routed_blocks_dir = os.path.join(
        output_dir,
        _settings.ROUTED_BLOCKS_SUBDIR,
    )
    os.makedirs(
        routed_blocks_dir,
        exist_ok=True,
    )

    fidelity_output_dir = os.path.join(
        output_dir,
        _settings.FIDELITY_SUBDIR,
    )
    if _settings.AUTO_RUN_FIDELITY:
        os.makedirs(
            fidelity_output_dir,
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                fidelity_output_dir,
                _settings.ACTUAL_CIRCUIT_SUBDIR,
            ),
            exist_ok=True,
        )

    reqisc_input_dir = os.path.join(
        output_dir,
        _settings.REQISC_PREPROCESSED_SUBDIR,
    )
    if (
        _settings.REQISC_PREOPTIMIZE
        and _settings.REQISC_SAVE_PREPROCESSED_QASM
    ):
        os.makedirs(reqisc_input_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        _settings.BATCH_RESULT_CSV,
    )

    print("=== 文件夹批量运行 ===")
    print(f"QASM_FOLDER={os.path.abspath(_settings.QASM_FOLDER)}")
    print(f"files={len(qasm_files)} | CSV={csv_path}")
    print(
        "pipeline=QASM -> ReQISC综合优化 "
        "-> canonical Can(x,y,z) "
        "-> 本文G(theta)搜参/单值性多面体拟分解/本文路由"
    )
    print(
        "ReQISC阶段="
        "template + hierarchical + canonical rebase；"
        "不执行ReQISC mirroring-SABRE"
    )

    if _settings.TOPOLOGY == "grid":
        print(
            "本文Grid来源="
            "ReQISC官方 "
            "regulus.utils.arch.gene_grid_2d_graph(N)"
        )

    print(
        "完整Program Fidelity="
        f"{_settings.AUTO_RUN_FIDELITY} | "
        "ReQISC noisy_simulation/QSim | "
        "保留ReQISC原1Q + 固定k PDBLS求1Q + G(theta) | "
        f"p0={_settings.FIDELITY_P0} | "
        f"tau_CX={_settings.FIDELITY_TAU_CX:.12f} | "
        f"shots={_settings.FIDELITY_SHOTS} | "
        f"max_qubits={_settings.FIDELITY_MAX_QUBITS}"
    )

    
    _mod_decomposition_routing.prewarm_fixed_polytope_caches()

    rows: List[Dict[str, Any]] = []

    reqisc_ctx: Optional[_mod_models.ReqiscOptimizeContext] = None
    if _settings.REQISC_PREOPTIMIZE:
        reqisc_ctx = _mod_preprocessing.initialize_reqisc_optimizer()

    try:
        for idx, qasm_file in enumerate(
            qasm_files,
            start=1,
        ):
            rel = os.path.relpath(
                qasm_file,
                os.path.abspath(_settings.QASM_FOLDER),
            )

            log_file = ""
            if _settings.BATCH_SAVE_PER_CIRCUIT_LOGS:
                log_file = os.path.join(
                    log_dir,
                    _mod_io_utils.safe_log_name(
                        qasm_file,
                        os.path.abspath(_settings.QASM_FOLDER),
                    ) + ".log",
                )

            print()
            print("=" * 88)
            print(
                f"[{idx}/{len(qasm_files)}] START | {qasm_file}",
                flush=True,
            )
            print("=" * 88)

            t0 = time.perf_counter()
            prep_meta: Dict[str, Any] = {}
            reqisc_meta: Dict[str, Any] = {}
            program_ctx: Optional[
                _mod_models.ReqiscProgramContext
            ] = None

            try:
                
                
                
                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "1/8 ReQISC输入适配",
                    os.path.basename(qasm_file),
                )

                safe_name = _mod_io_utils.safe_log_name(
                    qasm_file,
                    os.path.abspath(_settings.QASM_FOLDER),
                )

                prep_qasm_path = os.path.join(
                    reqisc_input_dir,
                    safe_name + "_reqisc_input.qasm",
                )

                qc_basis, prep_meta = (
                    _mod_preprocessing.preprocess_qasm_for_reqisc(
                        qasm_file,
                        save_path=prep_qasm_path,
                    )
                )

                declared_qubits = int(
                    prep_meta["declared_qubits"]
                )

                
                
                
                
                
                
                
                
                
                
                
                reqisc_input_2q = int(
                    prep_meta.get("preprocessed_twoq_gates", 0)
                )

                if (
                    _settings.BATCH_MAX_2Q_GATES is not None
                    and reqisc_input_2q > int(_settings.BATCH_MAX_2Q_GATES)
                ):
                    elapsed = time.perf_counter() - t0

                    row = {
                        "index": idx,
                        "circuit": os.path.splitext(
                            os.path.basename(qasm_file)
                        )[0],
                        "file": os.path.abspath(qasm_file),
                        "status": "skipped_2q_limit",
                        "error": (
                            f"ReQISC综合优化前2Q门数量 "
                            f"{reqisc_input_2q} 超过限制 "
                            f"{int(_settings.BATCH_MAX_2Q_GATES)}，已跳过"
                        ),
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "declared_qubits": declared_qubits,
                        "input_total_gates": prep_meta.get(
                            "original_total_gates", ""
                        ),
                        "input_twoq_gates": prep_meta.get(
                            "original_twoq_gates", ""
                        ),
                        "input_threeq_gates": prep_meta.get(
                            "original_threeq_gates", ""
                        ),
                        "reqisc_input_total_gates": prep_meta.get(
                            "preprocessed_total_gates", ""
                        ),
                        "reqisc_input_2q": reqisc_input_2q,
                        "reqisc_input_3q": prep_meta.get(
                            "preprocessed_threeq_gates", ""
                        ),
                        "twoq_gates": reqisc_input_2q,
                        "reqisc_preprocessed_qasm": prep_meta.get(
                            "preprocessed_qasm", ""
                        ),
                        "qasm_parse_fallback_used": prep_meta.get(
                            "qasm_parse_fallback_used", False
                        ),
                        "preparse_trailing_measurements_removed": prep_meta.get(
                            "preparse_trailing_measurements_removed", 0
                        ),
                        "log_file": (
                            os.path.abspath(log_file)
                            if log_file
                            else ""
                        ),
                    }

                    rows.append(row)
                    _mod_io_utils.write_batch_csv(csv_path, rows)

                    _mod_common.progress_step(
                        idx,
                        len(qasm_files),
                        "SKIP",
                        (
                            f"ReQISC综合前 2Q={reqisc_input_2q} > "
                            f"limit={int(_settings.BATCH_MAX_2Q_GATES)}，"
                            "直接下一条"
                        ),
                    )
                    continue

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "2/8 ReQISC综合优化",
                    (
                        f"input2Q={prep_meta['preprocessed_twoq_gates']} | "
                        f"input3Q={prep_meta['preprocessed_threeq_gates']} | "
                        f"qubits={declared_qubits}"
                    ),
                )

                if reqisc_ctx is None:
                    raise RuntimeError(
                        "REQISC_PREOPTIMIZE=True "
                        "但 ReQISC context 未初始化"
                    )

                
                
                
                
                (
                    target_gates,
                    reqisc_meta,
                    program_ctx,
                ) = _mod_preprocessing.run_reqisc_logical_optimization(
                    qc_basis,
                    reqisc_ctx,
                )

                canonical_csv = ""
                if _settings.REQISC_SAVE_CANONICAL_CSV:
                    canonical_csv = os.path.join(
                        reqisc_canonical_dir,
                        safe_name + "_reqisc_canonical.csv",
                    )
                    _mod_preprocessing.save_reqisc_canonical_csv(
                        canonical_csv,
                        target_gates,
                        declared_qubits,
                    )

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "3/8 提取ReQISC Can门",
                    (
                        f"ReQISC_1Q={program_ctx.num_oneq if program_ctx is not None else ''} | "
                        f"ReQISC_2Q={len(target_gates)} | "
                        f"unique="
                        f"{len({_mod_common.cartan_key(g.x,g.y,g.z) for g in target_gates})}"
                    ),
                )

                unique_count = len({
                    _mod_common.cartan_key(g.x, g.y, g.z)
                    for g in target_gates
                })

                if not target_gates:
                    row = {
                        "index": idx,
                        "circuit": os.path.splitext(
                            os.path.basename(qasm_file)
                        )[0],
                        "file": os.path.abspath(qasm_file),
                        "status": "empty_2q_after_reqisc",
                        "error": "ReQISC综合后没有2Q Can门",
                        "elapsed_seconds": (
                            f"{time.perf_counter()-t0:.6f}"
                        ),
                        "declared_qubits": declared_qubits,
                        "input_total_gates": prep_meta.get(
                            "original_total_gates", ""
                        ),
                        "input_twoq_gates": prep_meta.get(
                            "original_twoq_gates", ""
                        ),
                        "input_threeq_gates": prep_meta.get(
                            "original_threeq_gates", ""
                        ),
                        "reqisc_after_2q": reqisc_meta.get(
                            "reqisc_after_2q", ""
                        ),
                        "reqisc_canonical_csv": (
                            os.path.abspath(canonical_csv)
                            if canonical_csv
                            else ""
                        ),
                        "log_file": (
                            os.path.abspath(log_file)
                            if log_file
                            else ""
                        ),
                    }
                    rows.append(row)
                    _mod_io_utils.write_batch_csv(csv_path, rows)
                    continue

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "4/8 本文Native搜参",
                    (
                        f"ReQISC_2Q={len(target_gates)} | "
                        f"maxiter={_settings.NATIVE_COBYLA_MAXITER}"
                    ),
                )

                
                
                
                if _settings.BATCH_SAVE_PER_CIRCUIT_LOGS:
                    import sys as _sys

                    with open(
                        log_file,
                        "w",
                        encoding="utf-8",
                    ) as lf:
                        tee = _mod_common.TeeOutput(
                            _sys.stdout,
                            lf,
                        )
                        with redirect_stdout(tee):
                            best = _mod_decomposition_routing.search_best_base_gate(
                                target_gates,
                                rel,
                                declared_num_qubits=(
                                    declared_qubits
                                ),
                            )
                else:
                    best = _mod_decomposition_routing.search_best_base_gate(
                        target_gates,
                        rel,
                        declared_num_qubits=declared_qubits,
                    )

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "5/8 本文搜参与拟分解完成",
                    (
                        f"best={_mod_common._fmt_cartan(best['base_cartan'])} | "
                        f"critical={best['critical']:.12f}"
                    ),
                )

                elapsed = time.perf_counter() - t0

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "6/8 汇总ReQISC前端信息",
                )

                
                
                row = _mod_io_utils.result_to_batch_row(
                    idx,
                    qasm_file,
                    best,
                    elapsed,
                    declared_qubits,
                    int(reqisc_meta.get(
                        "reqisc_after_gates",
                        len(target_gates),
                    )),
                    len(target_gates),
                    unique_count,
                    log_file,
                )

                row.update({
                    "input_total_gates": prep_meta.get(
                        "original_total_gates", ""
                    ),
                    "input_twoq_gates": prep_meta.get(
                        "original_twoq_gates", ""
                    ),
                    "input_threeq_gates": prep_meta.get(
                        "original_threeq_gates", ""
                    ),
                    "reqisc_input_total_gates": prep_meta.get(
                        "preprocessed_total_gates", ""
                    ),
                    "reqisc_input_2q": prep_meta.get(
                        "preprocessed_twoq_gates", ""
                    ),
                    "reqisc_input_3q": prep_meta.get(
                        "preprocessed_threeq_gates", ""
                    ),
                    "reqisc_template_used": reqisc_meta.get(
                        "template_used", ""
                    ),
                    "reqisc_before_gates": reqisc_meta.get(
                        "reqisc_before_gates", ""
                    ),
                    "reqisc_before_2q": reqisc_meta.get(
                        "reqisc_before_2q", ""
                    ),
                    "reqisc_before_depth": reqisc_meta.get(
                        "reqisc_before_depth", ""
                    ),
                    "reqisc_before_depth_2q": reqisc_meta.get(
                        "reqisc_before_depth_2q", ""
                    ),
                    "reqisc_after_gates": reqisc_meta.get(
                        "reqisc_after_gates", ""
                    ),
                    "reqisc_after_oneq": reqisc_meta.get(
                        "reqisc_after_oneq", ""
                    ),
                    "reqisc_after_2q": reqisc_meta.get(
                        "reqisc_after_2q", ""
                    ),
                    "reqisc_after_depth": reqisc_meta.get(
                        "reqisc_after_depth", ""
                    ),
                    "reqisc_after_depth_2q": reqisc_meta.get(
                        "reqisc_after_depth_2q", ""
                    ),
                    "reqisc_elapsed_seconds": _mod_io_utils._fmt_float(
                        reqisc_meta.get(
                            "reqisc_elapsed_seconds"
                        )
                    ),
                    "reqisc_canonical_csv": (
                        os.path.abspath(canonical_csv)
                        if canonical_csv
                        else ""
                    ),
                    "reqisc_preprocessed_qasm": prep_meta.get(
                        "preprocessed_qasm", ""
                    ),
                    "qasm_parse_fallback_used": prep_meta.get(
                        "qasm_parse_fallback_used", False
                    ),
                    "preparse_trailing_measurements_removed": prep_meta.get(
                        "preparse_trailing_measurements_removed", 0
                    ),
                })

                
                
                
                routed_json_path = os.path.join(
                    routed_blocks_dir,
                    safe_name + "_routed_blocks.json",
                )

                routed_json_path = _mod_decomposition_routing.save_routed_blocks_json(
                    best,
                    routed_json_path,
                    qasm_file=qasm_file,
                )

                row["routed_blocks_json"] = (
                    os.path.abspath(routed_json_path)
                )
                best["routed_blocks_json"] = (
                    os.path.abspath(routed_json_path)
                )

                print(
                    f"      routed_blocks_json="
                    f"{row['routed_blocks_json']}",
                    flush=True,
                )

                
                
                
                if _settings.AUTO_RUN_FIDELITY:
                    _mod_common.progress_step(
                        idx,
                        len(qasm_files),
                        "7/8 路由后Fidelity",
                        (
                            "full 1Q+G circuit + fixed-k PDBLS + QSim noisy Hellinger"
                        ),
                    )

                    if program_ctx is None:
                        raise RuntimeError(
                            "program_ctx 未生成，无法计算完整 program fidelity"
                        )

                    fidelity_result = (
                        _mod_fidelity.calculate_our_program_fidelity(
                            qasm_file=qasm_file,
                            best=best,
                            program_ctx=program_ctx,
                            output_dir=(
                                fidelity_output_dir
                            ),
                            safe_name=safe_name,
                        )
                    )

                    row.update(
                        fidelity_result
                    )

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "7/8 写入汇总结果",
                    _settings.BATCH_RESULT_CSV,
                )

                rows.append(row)
                _mod_io_utils.write_batch_csv(
                    csv_path,
                    rows,
                )

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "8/8 完成",
                    f"elapsed={elapsed:.3f}s",
                )

                print(
                    f"[{idx}/{len(qasm_files)}] OK | "
                    f"input2Q={prep_meta.get('original_twoq_gates','')} | "
                    f"ReQISC2Q={len(target_gates)} | "
                    f"unique={unique_count} | "
                    f"search={row['routed_critical']} | "
                    f"CX={row.get('cx_routed_critical', '')} | "
                    f"SQISW={row.get('sqisw_routed_critical', '')} | "
                    f"ISWAP={row.get('iswap_routed_critical', '')} | "
                    f"BGATE={row.get('bgate_routed_critical', '')} | "
                    f"best=Can("
                    f"{row['best_x']}, "
                    f"{row['best_y']}, "
                    f"{row['best_z']}) | "
                    f"F={row.get('hellinger_fidelity', '')} | "
                    f"F_status={row.get('fidelity_status', '')} | "
                    f"actual1Q={row.get('actual_1q_count', '')} | "
                    f"actualG={row.get('actual_g_count', '')} | "
                    f"elapsed={elapsed:.3f}s",
                    flush=True,
                )

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                err = f"{type(exc).__name__}: {exc}"

                if (
                    _settings.BATCH_SAVE_PER_CIRCUIT_LOGS
                    and log_file
                ):
                    with open(
                        log_file,
                        "a",
                        encoding="utf-8",
                    ) as lf:
                        lf.write(
                            "\n=== ERROR ===\n"
                            + traceback.format_exc()
                        )

                error_row = {
                    "index": idx,
                    "circuit": os.path.splitext(
                        os.path.basename(qasm_file)
                    )[0],
                    "file": os.path.abspath(qasm_file),
                    "status": "error",
                    "error": err,
                    "elapsed_seconds": f"{elapsed:.6f}",
                    "log_file": (
                        os.path.abspath(log_file)
                        if log_file
                        else ""
                    ),
                }

                
                if prep_meta:
                    error_row.update({
                        "declared_qubits": prep_meta.get(
                            "declared_qubits", ""
                        ),
                        "input_total_gates": prep_meta.get(
                            "original_total_gates", ""
                        ),
                        "input_twoq_gates": prep_meta.get(
                            "original_twoq_gates", ""
                        ),
                        "input_threeq_gates": prep_meta.get(
                            "original_threeq_gates", ""
                        ),
                        "reqisc_input_total_gates": prep_meta.get(
                            "preprocessed_total_gates", ""
                        ),
                        "reqisc_input_2q": prep_meta.get(
                            "preprocessed_twoq_gates", ""
                        ),
                        "reqisc_input_3q": prep_meta.get(
                            "preprocessed_threeq_gates", ""
                        ),
                        "reqisc_preprocessed_qasm": prep_meta.get(
                            "preprocessed_qasm", ""
                        ),
                        "qasm_parse_fallback_used": prep_meta.get(
                            "qasm_parse_fallback_used", False
                        ),
                        "preparse_trailing_measurements_removed": prep_meta.get(
                            "preparse_trailing_measurements_removed", 0
                        ),
                    })

                if reqisc_meta:
                    error_row.update({
                        "reqisc_template_used": reqisc_meta.get(
                            "template_used", ""
                        ),
                        "reqisc_before_gates": reqisc_meta.get(
                            "reqisc_before_gates", ""
                        ),
                        "reqisc_before_2q": reqisc_meta.get(
                            "reqisc_before_2q", ""
                        ),
                        "reqisc_after_gates": reqisc_meta.get(
                            "reqisc_after_gates", ""
                        ),
                        "reqisc_after_2q": reqisc_meta.get(
                            "reqisc_after_2q", ""
                        ),
                        "reqisc_elapsed_seconds": _mod_io_utils._fmt_float(
                            reqisc_meta.get(
                                "reqisc_elapsed_seconds"
                            )
                        ),
                    })

                rows.append(error_row)
                _mod_io_utils.write_batch_csv(
                    csv_path,
                    rows,
                )

                _mod_common.progress_step(
                    idx,
                    len(qasm_files),
                    "ERROR",
                    err,
                )

    finally:
        _mod_preprocessing.close_reqisc_optimizer(reqisc_ctx)

    print(f"\n完成 | CSV={csv_path}")
    return rows

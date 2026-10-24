# -*- coding: utf-8 -*-
"""调用实际分解模块构建线路，使用 QSim 进行理想/含噪采样与 Hellinger 保真度评估。"""
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
    from . import io_utils as _mod_io_utils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import parameter_solver as _mod_parameter_solver

_QSIM_SIMULATOR = None


def _get_qsim_simulator():
    """严格使用 ReQISC noisy_simulation 同样的 QSimSimulator。"""
    global _QSIM_SIMULATOR

    if _QSIM_SIMULATOR is not None:
        return _QSIM_SIMULATOR

    try:
        import qsimcirq
    except Exception as exc:
        raise RuntimeError(
            "ReQISC noisy_simulation 口径需要 qsimcirq。"
        ) from exc

    _QSIM_SIMULATOR = (
        qsimcirq.QSimSimulator()
    )
    return _QSIM_SIMULATOR


def reqisc_style_ideal_counts(
    circ_cirq,
    shots: int,
) -> Dict[Any, int]:
    """复现 ReQISC：同一最终线路，末尾测量全部已使用 qubit。"""
    try:
        import cirq
    except Exception as exc:
        raise RuntimeError(
            "需要 cirq"
        ) from exc

    c = circ_cirq.copy()
    all_qubits = sorted(
        c.all_qubits(),
        key=lambda q: str(q),
    )

    if not all_qubits:
        return {
            0: int(shots)
        }

    c.append(
        cirq.measure(
            *all_qubits,
            key="z",
        )
    )

    return dict(
        _get_qsim_simulator().run(
            c,
            repetitions=int(shots),
        ).histogram(
            key="z"
        )
    )


def reqisc_style_noisy_counts(
    circ_cirq,
    *,
    p_g: float,
    shots: int,
) -> Dict[Any, int]:
    """复现 ReQISC noisy_simulation 的逐 2Q 门 depolarizing 逻辑。

    最终真实线路中唯一的 2Q 门就是 G(theta)，因此每遇到一个 2Q op：
        G(theta)
        -> cirq.depolarize(p_g, 2)
    1Q 门全部保留，但不加噪声。
    """
    try:
        import cirq
    except Exception as exc:
        raise RuntimeError(
            "需要 cirq"
        ) from exc

    noisy = cirq.Circuit()

    p = min(
        1.0,
        max(0.0, float(p_g)),
    )

    for op in circ_cirq.all_operations():
        noisy.append(op)

        if cirq.num_qubits(op) == 2:
            noisy.append(
                cirq.depolarize(
                    p,
                    2,
                ).on(
                    *op.qubits
                )
            )

    all_qubits = sorted(
        noisy.all_qubits(),
        key=lambda q: str(q),
    )

    if not all_qubits:
        return {
            0: int(shots)
        }

    noisy.append(
        cirq.measure(
            *all_qubits,
            key="z",
        )
    )

    return dict(
        _get_qsim_simulator().run(
            noisy,
            repetitions=int(shots),
        ).histogram(
            key="z"
        )
    )


def calculate_our_program_fidelity(
    qasm_file: str,
    best: Dict[str, Any],
    program_ctx: _mod_models.ReqiscProgramContext,
    output_dir: str,
    safe_name: str,
) -> Dict[str, Any]:
    """最终真实 1Q+G 线路 -> ReQISC QSim noisy Hellinger program fidelity。"""
    from . import io_utils as _mod_io_utils
    from . import parameter_solver as _mod_parameter_solver
    t0 = time.perf_counter()

    try:
        try:
            import qiskit.quantum_info as qi
        except Exception as exc:
            raise RuntimeError(
                "需要 qiskit.quantum_info"
            ) from exc

        num_qubits = int(
            best.get(
                "num_qubits",
                0,
            )
        )

        if (
            _settings.FIDELITY_MAX_QUBITS
            is not None
            and num_qubits
            > int(
                _settings.FIDELITY_MAX_QUBITS
            )
        ):
            return {
                "fidelity_status": "skipped_qubit_limit",
                "fidelity_error": (
                    f"num_qubits={num_qubits} > "
                    f"FIDELITY_MAX_QUBITS={_settings.FIDELITY_MAX_QUBITS}"
                ),
                "hellinger_fidelity": "",
                "hellinger_error": "",
                "semantic_check_pass": "",
                "fidelity_p_g": "",
                "fidelity_p0": _mod_io_utils._fmt_float(
                    _settings.FIDELITY_P0
                ),
                "fidelity_tau_cx": _mod_io_utils._fmt_float(
                    _settings.FIDELITY_TAU_CX
                ),
                "fidelity_shots": int(
                    _settings.FIDELITY_SHOTS
                ),
                "min_block_pf_decomp": "",
                "mean_block_pf_decomp": "",
                "low_pf_block_count": "",
                "fidelity_elapsed_seconds": _mod_io_utils._fmt_float(
                    time.perf_counter()
                    - t0
                ),
                "actual_circuit_qpy": "",
                "actual_circuit_json": "",
                "actual_circuit_txt": "",
                "reqisc_program_oneq": int(
                    program_ctx.num_oneq
                ),
                "actual_1q_count": "",
                "actual_g_count": "",
            }

        print(
            "      [FullProgram 1/4] "
            "固定k PDBLS 求实际1Q参数并重建最终物理线路 ...",
            flush=True,
        )

        built = (
            _mod_parameter_solver.build_actual_physical_program(
                best,
                program_ctx,
            )
        )

        print(
            "      [FullProgram 1/4] DONE | "
            f"ReQISC原1Q={program_ctx.num_oneq} | "
            f"actual1Q={built['actual_1q_count']} | "
            f"G={built['actual_g_count']} | "
            f"minPF={built['min_block_pf_decomp']:.12f}",
            flush=True,
        )

        print(
            "      [FullProgram 2/4] "
            "保存 actual QPY/JSON/TXT ...",
            flush=True,
        )

        paths = _mod_parameter_solver.save_actual_physical_program(
            built,
            output_dir,
            safe_name,
        )

        print(
            "      [FullProgram 3/4] "
            "ReQISC-style QSim ideal sampling ...",
            flush=True,
        )

        ideal_counts = (
            reqisc_style_ideal_counts(
                built["cirq_circuit"],
                int(
                    _settings.FIDELITY_SHOTS
                ),
            )
        )

        print(
            "      [FullProgram 4/4] "
            "ReQISC-style QSim noisy sampling "
            "+ Hellinger fidelity ...",
            flush=True,
        )

        noisy_counts = (
            reqisc_style_noisy_counts(
                built["cirq_circuit"],
                p_g=float(
                    built["p_g"]
                ),
                shots=int(
                    _settings.FIDELITY_SHOTS
                ),
            )
        )

        hf = float(
            qi.hellinger_fidelity(
                ideal_counts,
                noisy_counts,
            )
        )

        out_counts = os.path.join(
            os.path.dirname(
                paths["json"]
            ),
            safe_name
            + "_counts.json",
        )
        with open(
            out_counts,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "ideal_counts": {
                        str(k): int(v)
                        for k, v
                        in ideal_counts.items()
                    },
                    "noisy_counts": {
                        str(k): int(v)
                        for k, v
                        in noisy_counts.items()
                    },
                    "hellinger_fidelity": hf,
                    "hellinger_error": (
                        1.0 - hf
                    ),
                    "shots": int(
                        _settings.FIDELITY_SHOTS
                    ),
                    "p0": float(
                        _settings.FIDELITY_P0
                    ),
                    "tau_cx": float(
                        _settings.FIDELITY_TAU_CX
                    ),
                    "tau_g": float(
                        built["tau_basis"]
                    ),
                    "p_g": float(
                        built["p_g"]
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        elapsed = (
            time.perf_counter()
            - t0
        )

        print(
            "      [Fidelity] DONE | "
            f"Hellinger={hf:.12f} | "
            f"1-F={1.0-hf:.12f} | "
            f"p_G={built['p_g']:.12e} | "
            f"semantic_pass={built['semantic_check_pass']} | "
            f"elapsed={elapsed:.3f}s",
            flush=True,
        )

        return {
            "fidelity_status": (
                "ok"
                if bool(
                    built[
                        "semantic_check_pass"
                    ]
                )
                else "low_pf"
            ),
            "fidelity_error": (
                ""
                if bool(
                    built[
                        "semantic_check_pass"
                    ]
                )
                else (
                    "PDBLS 分解存在低于阈值的 block；"
                    "Hellinger 已计算，但不建议作为最终论文结果。"
                )
            ),
            "hellinger_fidelity": _mod_io_utils._fmt_float(
                hf
            ),
            "hellinger_error": _mod_io_utils._fmt_float(
                1.0 - hf
            ),
            "semantic_check_pass": bool(
                built[
                    "semantic_check_pass"
                ]
            ),
            "fidelity_p_g": _mod_io_utils._fmt_float(
                built["p_g"]
            ),
            "fidelity_p0": _mod_io_utils._fmt_float(
                _settings.FIDELITY_P0
            ),
            "fidelity_tau_cx": _mod_io_utils._fmt_float(
                _settings.FIDELITY_TAU_CX
            ),
            "fidelity_shots": int(
                _settings.FIDELITY_SHOTS
            ),
            "min_block_pf_decomp": _mod_io_utils._fmt_float(
                built[
                    "min_block_pf_decomp"
                ]
            ),
            "mean_block_pf_decomp": _mod_io_utils._fmt_float(
                built[
                    "mean_block_pf_decomp"
                ]
            ),
            "low_pf_block_count": int(
                built[
                    "low_pf_block_count"
                ]
            ),
            "fidelity_elapsed_seconds": _mod_io_utils._fmt_float(
                elapsed
            ),
            "actual_circuit_qpy": paths[
                "qpy"
            ],
            "actual_circuit_json": paths[
                "json"
            ],
            "actual_circuit_txt": paths[
                "txt"
            ],
            "reqisc_program_oneq": int(
                program_ctx.num_oneq
            ),
            "actual_1q_count": int(
                built[
                    "actual_1q_count"
                ]
            ),
            "actual_g_count": int(
                built[
                    "actual_g_count"
                ]
            ),
        }

    except Exception as exc:
        err = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "      [Fidelity] ERROR | "
            f"{err}",
            flush=True,
        )

        if _settings.FIDELITY_PRINT_TRACEBACK:
            traceback.print_exc()

        if not _settings.FIDELITY_CONTINUE_ON_ERROR:
            raise

        return {
            "fidelity_status": "error",
            "fidelity_error": err,
            "hellinger_fidelity": "",
            "hellinger_error": "",
            "semantic_check_pass": "",
            "fidelity_p_g": "",
            "fidelity_p0": _mod_io_utils._fmt_float(
                _settings.FIDELITY_P0
            ),
            "fidelity_tau_cx": _mod_io_utils._fmt_float(
                _settings.FIDELITY_TAU_CX
            ),
            "fidelity_shots": int(
                _settings.FIDELITY_SHOTS
            ),
            "min_block_pf_decomp": "",
            "mean_block_pf_decomp": "",
            "low_pf_block_count": "",
            "fidelity_elapsed_seconds": _mod_io_utils._fmt_float(
                time.perf_counter()
                - t0
            ),
            "actual_circuit_qpy": "",
            "actual_circuit_json": "",
            "actual_circuit_txt": "",
            "reqisc_program_oneq": int(
                program_ctx.num_oneq
            ),
            "actual_1q_count": "",
            "actual_g_count": "",
        }

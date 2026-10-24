# -*- coding: utf-8 -*-
"""线路发现和实验汇总的输入输出。"""
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



def discover_qasm_files(folder: str) -> List[str]:
    """查找文件夹中的 QASM 文件。"""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"QASM_FOLDER 不存在或不是文件夹，QASM_FOLDER={folder}")

    suffixes = tuple(str(s).lower() for s in _settings.QASM_FILE_SUFFIXES)
    qasm_files: List[str] = []
    if _settings.QASM_RECURSIVE:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(suffixes):
                    qasm_files.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(suffixes):
                qasm_files.append(path)

    qasm_files.sort()
    if _settings.BATCH_MAX_FILES is not None:
        qasm_files = qasm_files[:int(_settings.BATCH_MAX_FILES)]
    return qasm_files


def safe_log_name(path: str, root: str) -> str:
    """把相对路径转换成可作为日志文件名的字符串。"""
    try:
        rel = os.path.relpath(path, root)
    except Exception:
        rel = os.path.basename(path)
    rel = rel.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in rel)


def _fmt_float(value: Any) -> str:
    try:
        if value is None:
            return ""
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.12f}"
    except Exception:
        return ""


def write_batch_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    preferred = [
        "index", "circuit", "file", "status", "error", "elapsed_seconds",
        "declared_qubits", "active_qubits",
        "input_total_gates", "input_twoq_gates", "input_threeq_gates",
        "reqisc_input_total_gates", "reqisc_input_2q", "reqisc_input_3q",
        "reqisc_template_used",
        "reqisc_before_gates", "reqisc_before_2q", "reqisc_before_depth", "reqisc_before_depth_2q",
        "reqisc_after_gates", "reqisc_after_oneq", "reqisc_after_2q", "reqisc_after_depth", "reqisc_after_depth_2q",
        "reqisc_elapsed_seconds", "reqisc_canonical_csv", "reqisc_preprocessed_qasm",
        "qasm_parse_fallback_used", "preparse_trailing_measurements_removed",
        "total_gates", "twoq_gates", "unique_target_types",
        "best_x", "best_y", "best_z", "tau_basis", "routed_critical",
        "routed_native_2q", "routed_2q_depth",
        "routing_topology", "routing_topology_source",
        "routing_edge_count",
        "routing_blocks", "explicit_swaps", "swap_absorbed",
        "routed_blocks_json",
        "actual_circuit_qpy", "actual_circuit_json", "actual_circuit_txt",
        "reqisc_program_oneq", "actual_1q_count", "actual_g_count",
        "fidelity_status", "fidelity_error",
        "hellinger_fidelity", "hellinger_error",
        "semantic_check_pass",
        "fidelity_p_g", "fidelity_p0", "fidelity_tau_cx",
        "fidelity_shots",
        "min_block_pf_decomp", "mean_block_pf_decomp",
        "low_pf_block_count", "fidelity_elapsed_seconds",
        "cx_routed_critical", "cx_routed_native_2q", "cx_routed_2q_depth",
        "sqisw_routed_critical", "sqisw_routed_native_2q", "sqisw_routed_2q_depth",
        "iswap_routed_critical", "iswap_routed_native_2q", "iswap_routed_2q_depth",
        "bgate_routed_critical", "bgate_routed_native_2q", "bgate_routed_2q_depth",
        "optimizer_nfev", "polytope_build_seconds", "route_estimate_seconds",
        "route_seconds", "candidate_total_seconds", "log_file",
    ]
    extras = sorted({k for row in rows for k in row.keys()} - set(preferred))
    fieldnames = preferred + extras
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def result_to_batch_row(
    index: int,
    qasm_file: str,
    best: Dict[str, Any],
    elapsed_seconds: float,
    declared_qubits: int,
    total_gates: int,
    twoq_gates: int,
    unique_target_types: int,
    log_file: str,
) -> Dict[str, Any]:
    base = best["base_cartan"]
    timing = best.get("timing", {})
    routing = best.get("routing", {})
    optimizer = best.get("optimizer", {})
    baselines = best.get("baselines", {}) or {}

    def _baseline_value(name: str, field: str):
        res = baselines.get(name)
        if not res:
            return ""
        if field == "critical":
            return _fmt_float(res.get("critical"))
        route_info = res.get("routing", {}) or {}
        return route_info.get(field, "")

    return {
        "index": int(index),
        "circuit": os.path.splitext(os.path.basename(qasm_file))[0],
        "file": os.path.abspath(qasm_file),
        "status": "ok",
        "error": "",
        "elapsed_seconds": f"{float(elapsed_seconds):.6f}",
        "declared_qubits": int(declared_qubits),
        "active_qubits": int(best.get("num_qubits", 0)),
        "total_gates": int(total_gates),
        "twoq_gates": int(twoq_gates),
        "unique_target_types": int(unique_target_types),
        "best_x": _fmt_float(base[0]),
        "best_y": _fmt_float(base[1]),
        "best_z": _fmt_float(base[2]),
        "tau_basis": _fmt_float(best.get("tau_basis")),
        "routed_critical": _fmt_float(best.get("critical")),
        "routed_native_2q": routing.get("routed_native_2q", ""),
        "routed_2q_depth": routing.get("routed_2q_depth", ""),
        "routing_topology": routing.get("topology", ""),
        "routing_topology_source": routing.get(
            "topology_source",
            "",
        ),
        "routing_edge_count": routing.get(
            "coupling_edge_count",
            "",
        ),
        "routing_blocks": routing.get("num_blocks", ""),
        "explicit_swaps": routing.get("explicit_swaps", ""),
        "swap_absorbed": routing.get("swap_absorbed", ""),
        "routed_blocks_json": best.get("routed_blocks_json", ""),
        "actual_circuit_qpy": best.get("actual_circuit_qpy", ""),
        "actual_circuit_json": best.get("actual_circuit_json", ""),
        "actual_circuit_txt": best.get("actual_circuit_txt", ""),
        "reqisc_program_oneq": best.get("reqisc_program_oneq", ""),
        "actual_1q_count": best.get("actual_1q_count", ""),
        "actual_g_count": best.get("actual_g_count", ""),

        # 完整 program fidelity 字段。
        "fidelity_status": best.get("fidelity_status", ""),
        "fidelity_error": best.get("fidelity_error", ""),
        "hellinger_fidelity": best.get("hellinger_fidelity", ""),
        "hellinger_error": best.get("hellinger_error", ""),
        "semantic_check_pass": best.get("semantic_check_pass", ""),
        "fidelity_p_g": best.get("fidelity_p_g", ""),
        "fidelity_p0": best.get("fidelity_p0", ""),
        "fidelity_tau_cx": best.get("fidelity_tau_cx", ""),
        "fidelity_shots": best.get("fidelity_shots", ""),
        "min_block_pf_decomp": best.get("min_block_pf_decomp", ""),
        "mean_block_pf_decomp": best.get("mean_block_pf_decomp", ""),
        "low_pf_block_count": best.get("low_pf_block_count", ""),
        "fidelity_elapsed_seconds": best.get("fidelity_elapsed_seconds", ""),

        "cx_routed_critical": _baseline_value("CX", "critical"),
        "cx_routed_native_2q": _baseline_value("CX", "routed_native_2q"),
        "cx_routed_2q_depth": _baseline_value("CX", "routed_2q_depth"),

        "sqisw_routed_critical": _baseline_value("SQISW", "critical"),
        "sqisw_routed_native_2q": _baseline_value("SQISW", "routed_native_2q"),
        "sqisw_routed_2q_depth": _baseline_value("SQISW", "routed_2q_depth"),

        "iswap_routed_critical": _baseline_value("ISWAP", "critical"),
        "iswap_routed_native_2q": _baseline_value("ISWAP", "routed_native_2q"),
        "iswap_routed_2q_depth": _baseline_value("ISWAP", "routed_2q_depth"),

        "bgate_routed_critical": _baseline_value("BGATE", "critical"),
        "bgate_routed_native_2q": _baseline_value("BGATE", "routed_native_2q"),
        "bgate_routed_2q_depth": _baseline_value("BGATE", "routed_2q_depth"),

        "optimizer_nfev": optimizer.get("nfev", ""),
        "polytope_build_seconds": _fmt_float(timing.get("polytope_build_seconds")),
        "route_estimate_seconds": _fmt_float(timing.get("route_estimate_seconds")),
        "route_seconds": _fmt_float(timing.get("route_seconds")),
        "candidate_total_seconds": _fmt_float(timing.get("candidate_total_seconds")),
        "log_file": os.path.abspath(log_file) if log_file else "",
    }

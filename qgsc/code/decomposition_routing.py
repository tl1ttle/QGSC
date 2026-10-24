# -*- coding: utf-8 -*-
"""基门参数搜索、单值性多面体拟分解和严格有效推进路由。"""
from __future__ import annotations

import builtins
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
import numpy as np
from scipy.optimize import minimize
from .runtime import settings as _settings, ISING_ROOT as _ISING_ROOT
import gate_count as poly_est
from . import models as _mod_models

_GLOBAL_FIXED_POLYTOPE_CACHE = {}
_ROUTE_STATIC_CACHE = {}


def fixed_baseline_name(
    cartan: Tuple[float, float, float],
) -> Optional[str]:
    """若 cartan 对应 CX/SQiSW/iSWAP/BGate，返回其名称。"""
    from . import common as _mod_common
    key = _mod_common.cartan_key(*cartan)
    for name, baseline_cartan in _settings.FIXED_BASELINE_GATES.items():
        if key == _mod_common.cartan_key(*baseline_cartan):
            return str(name)
    return None


def fixed_polytope_cache_key(
    cartan: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], int, int]:
    from . import common as _mod_common
    return (
        _mod_common.cartan_key(*cartan),
        int(_settings.MAX_GATE_COUNT),
        int(_settings.MAX_DEN),
    )


def get_or_build_fixed_polytope_cache(
    base_cartan: Tuple[float, float, float],
) -> Tuple[_mod_models.CandidatePolytopeCache, bool]:
    """获取固定基门的全局 depth_polys。

    返回：
        (cache, cache_hit)

    cache_hit=True：
        当前基门此前已构造，直接跨线路复用。

    cache_hit=False：
        当前进程中第一次遇到该固定基门，本次负责构造并存入全局缓存。
    """
    from . import common as _mod_common
    name = fixed_baseline_name(base_cartan)
    if name is None:
        raise ValueError(
            f"get_or_build_fixed_polytope_cache 只用于固定基门，"
            f"收到 {_mod_common._fmt_cartan(base_cartan)}"
        )

    key = fixed_polytope_cache_key(base_cartan)

    cached = _GLOBAL_FIXED_POLYTOPE_CACHE.get(key)
    if cached is not None:
        return cached, True

    t0 = time.perf_counter()
    print(
        f"[固定门全局缓存] BUILD | {name:6s} | "
        f"base={_mod_common._fmt_cartan(base_cartan)} | "
        f"MAX_GATE_COUNT={_settings.MAX_GATE_COUNT} | MAX_DEN={_settings.MAX_DEN}"
    )

    cached = build_candidate_polytope_cache(
        base_cartan=base_cartan,
        max_n=_settings.MAX_GATE_COUNT,
        max_den=_settings.MAX_DEN,
    )
    _GLOBAL_FIXED_POLYTOPE_CACHE[key] = cached

    print(
        f"[固定门全局缓存] DONE  | {name:6s} | "
        f"elapsed={time.perf_counter() - t0:.3f}s"
    )

    return cached, False


def prewarm_fixed_polytope_caches() -> None:
    """在文件夹批处理开始前一次性构造四个固定门的 depth_polys。"""
    from . import common as _mod_common
    print("=== 固定门 depth_polys 全局缓存 ===")

    before = len(_GLOBAL_FIXED_POLYTOPE_CACHE)

    for name, cartan in _settings.FIXED_BASELINE_GATES.items():
        cache, hit = get_or_build_fixed_polytope_cache(
            tuple(float(v) for v in cartan)
        )
        if hit:
            print(
                f"[固定门全局缓存] HIT   | {name:6s} | "
                f"base={_mod_common._fmt_cartan(cartan)}"
            )

    after = len(_GLOBAL_FIXED_POLYTOPE_CACHE)

    print(
        f"固定门缓存准备完成 | "
        f"new={after - before} | total={after}"
    )
    print()


def build_candidate_polytope_cache(
    base_cartan: Tuple[float, float, float],
    max_n: int,
    max_den: int,
) -> _mod_models.CandidatePolytopeCache:
    """
    和 ising_preconstruct_per_candidate_weyl.py 一样，
    对单个候选基门先预构造 depth_polys[0..max_n]，后续只做 membership test。
    """
    from . import common as _mod_common
    build_t0 = time.perf_counter()

    with _mod_common.lrs_restart_file_context():
        gate_polytope_t0 = time.perf_counter()
        base_gate, _base_mono = poly_est.gate_polytope_from_cartan(
            *base_cartan,
            name="G",
            max_den=int(max_den),
        )
        gate_polytope_seconds = time.perf_counter() - gate_polytope_t0

        current = poly_est.CircuitPolytope(
            operations=[],
            cost=0.0,
            convex_subpolytopes=poly_est.identity_polytope.convex_subpolytopes,
        )
        depth_polys: Dict[int, Any] = {0: current}
        depth_build_seconds: Dict[int, float] = {}

        for depth in range(1, int(max_n) + 1):
            depth_t0 = time.perf_counter()
            current = poly_est.compose_once(current, base_gate)
            depth_build_seconds[depth] = time.perf_counter() - depth_t0
            depth_polys[depth] = current

    build_seconds = time.perf_counter() - build_t0

    return _mod_models.CandidatePolytopeCache(
        base_cartan=tuple(float(x) for x in base_cartan),
        depth_polys=depth_polys,
        max_n=int(max_n),
        max_den=int(max_den),
        build_seconds=float(build_seconds),
        gate_polytope_seconds=float(gate_polytope_seconds),
        depth_build_seconds=depth_build_seconds,
    )


def cartan_to_target_point(
    target_cartan: Tuple[float, float, float],
    max_den: int,
):
    from . import common as _mod_common
    with _mod_common.lrs_restart_file_context():
        return poly_est.cartan_to_monodromy_point(
            *target_cartan,
            max_den=int(max_den),
        )


def estimate_gate_count_from_candidate_cache(
    cache: _mod_models.CandidatePolytopeCache,
    target_point,
) -> Optional[int]:
    for depth in range(0, int(cache.max_n) + 1):
        if cache.depth_polys[depth].has_element(target_point):
            return int(depth)
    return None


def compress_target_gates(target_gates: List[_mod_models.TwoQGate]) -> Dict[str, Any]:
    from . import common as _mod_common
    unique_targets: List[Tuple[float, float, float]] = []
    type_index_map: Dict[Tuple[float, float, float], int] = {}
    gate_instances: List[Dict[str, Any]] = []
    for gate in target_gates:
        cartan = _mod_common.normalize_cartan_to_balanced((gate.x, gate.y, gate.z))
        key = _mod_common.cartan_key(*cartan)
        if key not in type_index_map:
            type_index_map[key] = len(unique_targets)
            unique_targets.append(cartan)
        gate_instances.append({
            "name": gate.name,
            "q0": int(gate.q0),
            "q1": int(gate.q1),
            "cartan": cartan,
            "type_idx": type_index_map[key],
        })
    return {"unique_targets": unique_targets, "gate_instances": gate_instances}


def project_to_legal_region(x: float, y: float, z: float) -> Tuple[float, float, float]:
    x = min(max(float(x), _settings.WEYL_X_MIN), math.pi / 4)
    y = min(max(float(y), 0.0), x)
    z = min(max(float(z), -y), y)
    return float(x), float(y), float(z)


def _native_coupling_ratios(a: float, b: float, c: float) -> Optional[Tuple[float, float]]:
    from . import common as _mod_common
    if abs(a) <= _settings.NATIVE_MIN_A:
        return None
    yr = float(b) / float(a)
    zr = float(c) / float(a)
    if yr < -_mod_common.EPS or yr > 1.0 + _mod_common.EPS or abs(zr) > yr + _mod_common.EPS:
        return None
    return float(min(1.0, max(0.0, yr))), float(zr)


def evaluate_base_gate(
    base_cartan: Tuple[float, float, float],
    gate_instances: List[Dict[str, Any]],
    num_qubits: int,
    a: float,
    b: float,
    c: float,
    candidate_polytope_cache: Dict[Tuple[float, float, float], _mod_models.CandidatePolytopeCache],
    candidate_label: str = "",
) -> Optional[Dict[str, Any]]:
    from . import common as _mod_common
    eval_t0 = time.perf_counter()
    xb, yb, zb = base_cartan

    label_txt = f" | {candidate_label}" if candidate_label else ""
    print(
        f"    [候选评估开始]{label_txt} | "
        f"base={_mod_common._fmt_cartan((xb, yb, zb))}",
        flush=True,
    )

    if not (math.pi / 4 >= xb >= yb >= abs(zb)):
        print(
            f"    [候选评估失败]{label_txt} | reason=不在Weyl合法区域",
            flush=True,
        )
        return None

    print(f"      [1/4] 计算候选基门演化时间 ...", flush=True)
    pulse_t0 = time.perf_counter()
    try:
        tau_basis, pulse_info = _mod_common.gate_pulse_duration(a, b, c, xb, yb, zb)
    except Exception as exc:
        print(
            f"      [1/4] FAILED | {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    pulse_seconds = time.perf_counter() - pulse_t0
    print(
        f"      [1/4] DONE | tau={tau_basis:.12f} | "
        f"elapsed={pulse_seconds:.3f}s",
        flush=True,
    )

    print(
        f"      [2/4] 获取/构造 depth_polys[0..{_settings.MAX_GATE_COUNT}] ...",
        flush=True,
    )
    poly_stage_t0 = time.perf_counter()

    base_key = _mod_common.cartan_key(xb, yb, zb)
    baseline_name = fixed_baseline_name((xb, yb, zb))

    if baseline_name is not None:
        # --------------------------------------------------------------
        # CX / SQiSW / iSWAP / BGate：
        # 使用跨整个文件夹批处理共享的全局 depth_polys。
        # --------------------------------------------------------------
        poly_cache, cache_hit = get_or_build_fixed_polytope_cache(
            (xb, yb, zb)
        )

        # 固定门即使本线路第一次使用，只要在批处理启动阶段已经 prewarm，
        # 这里通常都会是 HIT，不再重新构造。
        polytope_build_seconds = (
            0.0
            if cache_hit
            else float(poly_cache.build_seconds)
        )

    else:
        # --------------------------------------------------------------
        # COBYLA 搜索产生的普通 G(x)：
        # 仍只在当前线路内部缓存，避免全局缓存无限增长。
        # --------------------------------------------------------------
        cache_hit = base_key in candidate_polytope_cache

        if not cache_hit:
            candidate_polytope_cache[base_key] = (
                build_candidate_polytope_cache(
                    base_cartan=(xb, yb, zb),
                    max_n=_settings.MAX_GATE_COUNT,
                    max_den=_settings.MAX_DEN,
                )
            )

        poly_cache = candidate_polytope_cache[base_key]
        polytope_build_seconds = (
            0.0
            if cache_hit
            else float(poly_cache.build_seconds)
        )

    print(
        f"      [2/4] DONE | "
        f"cache={'HIT' if cache_hit else 'BUILD'} | "
        f"scope={'global_fixed' if baseline_name is not None else 'per_circuit_search'} | "
        f"elapsed={time.perf_counter() - poly_stage_t0:.3f}s",
        flush=True,
    )

    target_gates = [
        _mod_models.TwoQGate(
            name=str(item["name"]),
            q0=int(item["q0"]),
            q1=int(item["q1"]),
            x=float(item["cartan"][0]),
            y=float(item["cartan"][1]),
            z=float(item["cartan"][2]),
        )
        for item in gate_instances
    ]

    try:
        static_ctx = get_or_build_route_static_context(
            target_gates,
            num_qubits,
        )
    except Exception as exc:
        print(
            f"      [拓扑构造失败] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None

    ctx = _mod_models.BaseRouteContext(
        base_cartan=(xb, yb, zb),
        tau_basis=float(tau_basis),
        poly_cache=poly_cache,
        point_cache=static_ctx.point_cache,
        est_cache={},
    )

    print(
        f"      [3/4] 检查 {len(static_ctx.route_cartan_types)} 种原始逻辑门可分解性 ...",
        flush=True,
    )
    estimate_t0 = time.perf_counter()
    try:
        # 候选基门的前置可行性只检查原始逻辑 2Q 门。
        #
        # 不提前检查：
        #   1. mirror(U)=SWAP·U
        #   2. 显式 SWAP
        #
        # 它们都属于“路由时才可能用到”的操作：
        # - 显式 SWAP：在 evaluate_candidate_swap_variants() 中按需判断；
        # - mirror(U)：只有计算 SWAP 吸收候选代价时才按需判断。
        #
        # 因此某个 mirror 门不可分解，不会在路由开始前错误淘汰整个候选基门。
        precompute_route_estimates(ctx, static_ctx.route_cartan_types)
    except _mod_models.CandidateInvalidError as exc:
        print(
            f"      [3/4] FAILED | {exc}",
            flush=True,
        )
        return None
    route_estimate_seconds = time.perf_counter() - estimate_t0
    print(
        f"      [3/4] DONE | elapsed={route_estimate_seconds:.3f}s",
        flush=True,
    )

    print(
        f"      [4/4] 执行拓扑路由/关键路径计算 | topology={_settings.TOPOLOGY} ...",
        flush=True,
    )
    route_t0 = time.perf_counter()
    try:
        routed_state = route_with_min_swap_cost_vb(
            gates=target_gates,
            edges=static_ctx.edges,
            adj=static_ctx.adj,
            dist=static_ctx.dist,
            ctx=ctx,
            prebuilt_dag=(static_ctx.succs, static_ctx.required_predecessors, static_ctx.front),
        )
    except (_mod_models.CandidateInvalidError, RuntimeError, ValueError) as exc:
        print(
            f"      [4/4] FAILED | {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    route_seconds = time.perf_counter() - route_t0
    print(
        f"      [4/4] DONE | critical={routed_state.current_critical:.12f} | "
        f"elapsed={route_seconds:.3f}s",
        flush=True,
    )

    n_swap = sum(1 for blk in routed_state.routed_blocks if blk.kind == "swap")
    n_absorb = sum(1 for blk in routed_state.routed_blocks if blk.kind == "swap_absorbed")
    routed_native_2q, routed_2q_depth = routed_native_2q_metrics(
        routed_state,
        tau_basis=float(tau_basis),
    )
    total_eval_seconds = time.perf_counter() - eval_t0

    if _settings.PRINT_WORKFLOW and _settings.PRINT_TIMING:
        print(
            f"[候选] Can({xb:.12f},{yb:.12f},{zb:.12f}) | "
            f"tau={tau_basis:.12f} | branch={pulse_info['branch']} | "
            f"poly={polytope_build_seconds:.3f}s | est={route_estimate_seconds:.3f}s | "
            f"route={route_seconds:.3f}s | routed_critical={routed_state.current_critical:.12f}"
        )

    print(
        f"    [候选评估完成]{label_txt} | "
        f"critical={routed_state.current_critical:.12f} | "
        f"total={total_eval_seconds:.3f}s",
        flush=True,
    )

    return {
        "base_cartan": (xb, yb, zb),
        "tau_basis": float(tau_basis),
        "critical": float(routed_state.current_critical),
        "routed_state": routed_state,
        "num_qubits": int(num_qubits),
        "routing": {
            "topology": str(_settings.TOPOLOGY),
            "topology_source": (
                "ReQISC:regulus.utils.arch."
                "gene_grid_2d_graph"
                if _settings.TOPOLOGY == "grid"
                else str(_settings.TOPOLOGY)
            ),
            "coupling_edge_count": int(
                len(static_ctx.edges)
            ),
            "edges": [
                (int(u), int(v))
                for u, v in static_ctx.edges
            ],
            "num_blocks": int(len(routed_state.routed_blocks)),
            "explicit_swaps": int(n_swap),
            "swap_absorbed": int(n_absorb),
            "routed_native_2q": int(routed_native_2q),
            "routed_2q_depth": int(routed_2q_depth),
        },
        "timing": {
            "pulse_seconds": float(pulse_seconds),
            "polytope_build_seconds": float(polytope_build_seconds),
            "polytope_cache_hit": bool(cache_hit),
            "polytope_cache_scope": (
                "global_fixed"
                if baseline_name is not None
                else "per_circuit_search"
            ),
            "fixed_baseline_name": (
                baseline_name
                if baseline_name is not None
                else ""
            ),
            "route_estimate_seconds": float(route_estimate_seconds),
            "route_seconds": float(route_seconds),
            "candidate_total_seconds": float(total_eval_seconds),
        },
    }


def _constraint_violation_sum(values: List[float]) -> float:
    return float(sum(max(0.0, -float(v)) for v in values))


def _native_constraints(v: np.ndarray, y_ratio: float, z_ratio: float) -> List[float]:
    x = float(v[0])
    y = x * y_ratio
    z = x * z_ratio
    return [
        x - _settings.WEYL_X_MIN,
        math.pi / 4 - x,
        y,
        x - y,
        y + z,
        y - z,
    ]


def _native_constraint_defs(y_ratio: float, z_ratio: float):
    return [
        {"type": "ineq", "fun": lambda v, i=i: _native_constraints(v, y_ratio, z_ratio)[i]}
        for i in range(6)
    ]


def _candidate_from_native_vector(v: np.ndarray, y_ratio: float, z_ratio: float) -> Tuple[float, float, float]:
    x = float(v[0])
    return project_to_legal_region(x, x * y_ratio, x * z_ratio)


def optimize_native_with_cobyla(
    gate_instances: List[Dict[str, Any]],
    num_qubits: int,
    a: float,
    b: float,
    c: float,
    candidate_polytope_cache: Dict[Tuple[float, float, float], _mod_models.CandidatePolytopeCache],
) -> Dict[str, Any]:
    from . import common as _mod_common
    ratios = _native_coupling_ratios(a, b, c)
    if ratios is None:
        raise RuntimeError("硬件耦合不满足 a>0 且 a>=b>=|c|")
    y_ratio, z_ratio = ratios
    if _settings.NATIVE_COBYLA_X0 is None:
        raise RuntimeError("NATIVE_COBYLA_X0 不能为 None")

    x0 = np.array([
        min(max(float(_settings.NATIVE_COBYLA_X0), _settings.WEYL_X_MIN), math.pi / 4)
    ], dtype=float)
    eval_cache: Dict[Tuple[float, float, float], Optional[Dict[str, Any]]] = {}
    best_holder: Dict[str, Optional[Dict[str, Any]]] = {"best": None}
    eval_counter = {"n": 0}

    def evaluate(cand: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
        key = _mod_common.cartan_key(*cand)
        if key in eval_cache:
            print(
                f"  [Native-COBYLA] CACHE HIT | "
                f"Can({cand[0]:.12f}, {cand[1]:.12f}, {cand[2]:.12f})",
                flush=True,
            )
            return eval_cache[key]

        eval_counter["n"] += 1
        eval_no = eval_counter["n"]
        print(
            f"  [Native-COBYLA] 候选 {eval_no}/{_settings.NATIVE_COBYLA_MAXITER} | "
            f"Can({cand[0]:.12f}, {cand[1]:.12f}, {cand[2]:.12f})",
            flush=True,
        )

        # 现在外层批处理使用 TeeOutput，候选内部阶段也直接实时打印。
        res = evaluate_base_gate(
            cand,
            gate_instances,
            num_qubits,
            a,
            b,
            c,
            candidate_polytope_cache,
            candidate_label=f"Native-{eval_no}",
        )

        eval_cache[key] = res
        return res

    def objective(v: np.ndarray) -> float:
        violations = _constraint_violation_sum(_native_constraints(v, y_ratio, z_ratio))
        if violations > 0:
            return _settings.OPT_INVALID_PENALTY + _settings.OPT_CONSTRAINT_PENALTY_SCALE * violations
        cand = _candidate_from_native_vector(v, y_ratio, z_ratio)
        res = evaluate(cand)
        if res is None:
            return _settings.OPT_INVALID_PENALTY
        current = best_holder["best"]
        if current is None or res["critical"] < current["critical"] - _mod_common.EPS:
            best_holder["best"] = res
            print(
                f"[Native-COBYLA] 更新最优 | Can({cand[0]:.12f}, {cand[1]:.12f}, {cand[2]:.12f}) | "
                f"routed_critical={res['critical']:.12f}"
            )
        return float(res["critical"])

    print("=== Native COBYLA ===")
    print(f"x0={x0[0]:.12f} | y/x={y_ratio:.12f} | z/x={z_ratio:.12f} | maxiter={_settings.NATIVE_COBYLA_MAXITER}")
    opt = minimize(
        objective,
        x0=x0,
        method="COBYLA",
        constraints=_native_constraint_defs(y_ratio, z_ratio),
        options={
            "maxiter": int(_settings.NATIVE_COBYLA_MAXITER),
            "rhobeg": float(_settings.NATIVE_COBYLA_RHOBEG),
            "tol": float(_settings.COBYLA_TOL),
            "catol": float(_settings.COBYLA_CATOL),
            "disp": False,
        },
    )

    final_cand = _candidate_from_native_vector(opt.x, y_ratio, z_ratio)
    final_res = evaluate(final_cand)
    if final_res is not None:
        current = best_holder["best"]
        if current is None or final_res["critical"] < current["critical"] - _mod_common.EPS:
            best_holder["best"] = final_res

    best = best_holder["best"]
    if best is None:
        raise RuntimeError(
            "Native COBYLA 未找到可行候选；请调整 NATIVE_COBYLA_X0、MAXITER、MAX_GATE_COUNT 或 MAX_DEN。"
        )
    best["optimizer"] = {
        "method": "COBYLA",
        "nfev": int(getattr(opt, "nfev", -1)),
        "success": bool(getattr(opt, "success", False)),
        "message": str(getattr(opt, "message", "")),
    }
    return best


def search_best_base_gate(
    target_gates: List[_mod_models.TwoQGate],
    circuit_label: Optional[str] = None,
    declared_num_qubits: Optional[int] = None,
) -> Dict[str, Any]:
    from . import common as _mod_common
    raw_a = _settings.G_COUPLING / 2.0 if _settings.A_COEFF is None else float(_settings.A_COEFF)
    raw_b = _settings.G_COUPLING / 2.0 if _settings.B_COEFF is None else float(_settings.B_COEFF)
    raw_c = 0.0 if _settings.C_COEFF is None else float(_settings.C_COEFF)
    if _settings.NORMALIZE_HARDWARE_COUPLING:
        a, b, c = _mod_common.normal_form_diagonal_coupling(raw_a, raw_b, raw_c)
    else:
        a, b, c = raw_a, raw_b, raw_c

    target_gates = [_mod_common.normalize_twoq_gate(g) for g in target_gates]
    if not target_gates:
        raise ValueError("目标线路为空")
    active_num_qubits = (
        1 + max(max(g.q0, g.q1) for g in target_gates)
    )
    num_qubits = max(
        active_num_qubits,
        int(declared_num_qubits)
        if declared_num_qubits is not None
        else active_num_qubits,
    )
    compressed = compress_target_gates(target_gates)
    unique_targets = compressed["unique_targets"]
    gate_instances = compressed["gate_instances"]

    if _settings.PRINT_WORKFLOW:
        print("=== 线路 ===")
        if circuit_label:
            print(f"file={circuit_label}")
        print(
            f"2Q={len(gate_instances)} | unique={len(unique_targets)} | "
            f"active_qubits={active_num_qubits} | declared_qubits={num_qubits}"
        )
        if _settings.TOPOLOGY == "grid":
            print(
                f"hardware=({a:.12f},{b:.12f},{c:.12f}) | "
                "topology=grid | "
                "grid_source=ReQISC:"
                "arch.gene_grid_2d_graph | "
                f"N={num_qubits} | MAX_DEN={_settings.MAX_DEN}"
            )
        else:
            print(
                f"hardware=({a:.12f},{b:.12f},{c:.12f}) | "
                f"topology={_settings.TOPOLOGY} | MAX_DEN={_settings.MAX_DEN}"
            )

    candidate_polytope_cache: Dict[Tuple[float, float, float], _mod_models.CandidatePolytopeCache] = {}

    print("=== 阶段 A：Native 参数搜索 ===", flush=True)
    best = optimize_native_with_cobyla(
        gate_instances=gate_instances,
        num_qubits=num_qubits,
        a=a,
        b=b,
        c=c,
        candidate_polytope_cache=candidate_polytope_cache,
    )

    print(
        "=== 阶段 B：CX / SQiSW / iSWAP / BGate 对比 ===",
        flush=True,
    )
    baseline_results: Dict[str, Optional[Dict[str, Any]]] = {}
    total_baselines = len(_settings.FIXED_BASELINE_GATES)

    for baseline_idx, (baseline_name, baseline_cartan) in enumerate(
        _settings.FIXED_BASELINE_GATES.items(),
        start=1,
    ):
        print(
            f"=== 固定门集评估 [{baseline_idx}/{total_baselines}] "
            f"{baseline_name} ===",
            flush=True,
        )

        if _mod_common.cartan_key(*best["base_cartan"]) == _mod_common.cartan_key(*baseline_cartan):
            print(
                "    与 Native 最优基门相同，直接复用 Native 结果。",
                flush=True,
            )
            res = best
        else:
            res = evaluate_base_gate(
                baseline_cartan,
                gate_instances,
                num_qubits,
                a,
                b,
                c,
                candidate_polytope_cache,
                candidate_label=f"Baseline-{baseline_name}",
            )

        baseline_results[baseline_name] = res

        if res is None:
            print(
                f"[固定门集] {baseline_name:6s} | base={_mod_common._fmt_cartan(baseline_cartan)} | INVALID"
            )
        else:
            rinfo = res.get("routing", {})
            print(
                f"[固定门集] {baseline_name:6s} | base={_mod_common._fmt_cartan(baseline_cartan)} | "
                f"routed={res['critical']:.12f} | "
                f"2Q={rinfo.get('routed_native_2q', '')} | "
                f"2Qdepth={rinfo.get('routed_2q_depth', '')} | "
                f"SWAP={rinfo.get('explicit_swaps', '')} | "
                f"absorb={rinfo.get('swap_absorbed', '')}"
            )

    best["baselines"] = baseline_results
    best["hardware_raw"] = (raw_a, raw_b, raw_c)
    best["hardware_canonical"] = (a, b, c)
    return best


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def mirror_cartan(cartan: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """计算 SWAP·U 的 canonical mirror 坐标。"""
    from . import common as _mod_common
    in_cartan = tuple(float(t) for t in cartan)
    x, y, z = _mod_common.normalize_cartan_to_balanced(in_cartan)

    if z >= 0.0:
        out = (math.pi / 4.0 - z, math.pi / 4.0 - y, x - math.pi / 4.0)
    else:
        out = (math.pi / 4.0 + z, math.pi / 4.0 - y, math.pi / 4.0 - x)

    out = _mod_common.normalize_cartan_to_balanced(out)

    if not _mod_common.is_balanced_cartan(out):
        raise ValueError(
            f"mirror_cartan 结果不在平衡 canonical 区域内: "
            f"in={in_cartan}, normalized={(x, y, z)}, out={out}"
        )

    return tuple(float(v) for v in out)


def route_state_signature(state: _mod_models.RouteState) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """路由状态签名同时包含执行进度与映射，避免误删不同进度的同布局状态。"""
    return (
        tuple(sorted(int(x) for x in state.mapped_gate_ids)),
        tuple(int(x) for x in state.logical_to_physical),
    )


def _extract_reqisc_graph_edges(
    graph_obj: Any,
    num_qubits: int,
) -> List[Tuple[int, int]]:
    """严格按 ReQISC 官方 rustworkx.PyGraph 接口提取 Grid 耦合边。

    已根据 ReQISC 源码 regulus/utils/arch.py 验证：

        def gene_grid_2d_graph(total_number):
            n = int(np.sqrt(total_number))
            m = int(np.ceil(total_number / n))
            g = rx.generators.grid_graph(n, m)
            return g.subgraph(range(total_number))

    因而其返回值为 rustworkx.PyGraph。

    官方代码在其他函数中也是使用：
        device.node_indices()
        device.edge_list()

    所以这里不再使用：
        graph.edges()
        graph.nodes()

    避免把 rustworkx 的 edge payload（通常是 None）
    错误当成 (u, v) 端点。
    """
    n = int(num_qubits)

    if n <= 0:
        raise ValueError(
            f"num_qubits 必须 > 0，实际为 {n}"
        )

    if graph_obj is None:
        raise RuntimeError(
            "ReQISC gene_grid_2d_graph() 返回 None | "
            f"num_qubits={n}"
        )

    # --------------------------------------------------------
    # ReQISC 官方 arch.py 明确使用 rustworkx.PyGraph。
    # --------------------------------------------------------
    if not hasattr(graph_obj, "node_indices"):
        raise TypeError(
            "ReQISC Grid 对象缺少 node_indices() | "
            f"type={type(graph_obj).__module__}."
            f"{type(graph_obj).__name__}"
        )

    if not hasattr(graph_obj, "edge_list"):
        raise TypeError(
            "ReQISC Grid 对象缺少 edge_list() | "
            f"type={type(graph_obj).__module__}."
            f"{type(graph_obj).__name__}"
        )

    try:
        node_indices = [
            int(q)
            for q in graph_obj.node_indices()
        ]
    except Exception as exc:
        raise RuntimeError(
            "读取 ReQISC Grid node_indices() 失败 | "
            f"type={type(graph_obj).__module__}."
            f"{type(graph_obj).__name__}"
        ) from exc

    try:
        raw_edges = list(
            graph_obj.edge_list()
        )
    except Exception as exc:
        raise RuntimeError(
            "读取 ReQISC Grid edge_list() 失败 | "
            f"type={type(graph_obj).__module__}."
            f"{type(graph_obj).__name__}"
        ) from exc

    # --------------------------------------------------------
    # ReQISC:
    #     return g.subgraph(range(total_number))
    #
    # 因此理论上实际物理节点就应为 0..N-1。
    # 为保证本文与 ReQISC 使用完全相同的物理节点编号，
    # 这里不做重编号，只做严格检查。
    # --------------------------------------------------------
    expected_nodes = list(range(n))
    actual_nodes = sorted(node_indices)

    if actual_nodes != expected_nodes:
        raise RuntimeError(
            "ReQISC Grid 物理节点编号异常 | "
            f"expected={expected_nodes} | "
            f"actual={actual_nodes}"
        )

    edges: Set[Tuple[int, int]] = set()

    for raw_edge in raw_edges:
        try:
            u = int(raw_edge[0])
            v = int(raw_edge[1])
        except Exception as exc:
            raise RuntimeError(
                "无法解析 ReQISC edge_list() 返回的边 | "
                f"edge={raw_edge!r}"
            ) from exc

        if u == v:
            # 理论上 ReQISC grid_graph 不会产生 self-loop。
            continue

        if (
            u < 0
            or v < 0
            or u >= n
            or v >= n
        ):
            raise RuntimeError(
                "ReQISC Grid 边端点越界 | "
                f"edge=({u},{v}) | "
                f"num_qubits={n}"
            )

        edges.add(
            edge_key(u, v)
        )

    result = sorted(edges)

    if n > 1 and not result:
        raise RuntimeError(
            "ReQISC Grid 没有有效耦合边 | "
            f"num_qubits={n}"
        )

    # 官方 Grid 的 n、m 仅用于日志核查；
    # 实际拓扑仍来自 gene_grid_2d_graph() 返回的 PyGraph，
    # 不是本文重新生成。
    grid_n = int(math.sqrt(n))
    grid_m = int(math.ceil(n / grid_n))

    print(
        "      [ReQISC Grid对象] "
        "type=rustworkx.PyGraph | "
        "extract=node_indices()+edge_list()",
        flush=True,
    )
    print(
        f"      [ReQISC Grid形状] "
        f"N={n} | n={grid_n} | m={grid_m} | "
        f"nodes={len(actual_nodes)} | "
        f"edges={len(result)}",
        flush=True,
    )

    return result


def build_coupling_edges(
    num_qubits: int,
) -> List[Tuple[int, int]]:
    """构造本文路由器使用的硬件耦合边。

    特别地：
      TOPOLOGY == "grid"
    时不再自行生成 Grid，而是直接复用 ReQISC 官方
    regulus.utils.arch.gene_grid_2d_graph(num_qubits)。
    """
    n = int(num_qubits)

    if n <= 0:
        raise ValueError(
            f"num_qubits 必须 > 0，实际为 {n}"
        )

    if _settings.TOPOLOGY == "alltoall":
        return [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
        ]

    if _settings.TOPOLOGY == "chain":
        return [
            (i, i + 1)
            for i in range(n - 1)
        ]

    if _settings.TOPOLOGY == "grid":
        try:
            from regulus.utils import arch as reqisc_arch
        except Exception as exc:
            raise RuntimeError(
                "无法导入 ReQISC regulus.utils.arch；"
                f"REQISC_ROOT={_settings.REQISC_ROOT}"
            ) from exc

        try:
            reqisc_grid = (
                reqisc_arch.gene_grid_2d_graph(n)
            )
        except Exception as exc:
            raise RuntimeError(
                "ReQISC 官方 gene_grid_2d_graph "
                f"构造失败：num_qubits={n} | "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        edges = _extract_reqisc_graph_edges(
            reqisc_grid,
            n,
        )

        print(
            "      [Grid拓扑] "
            "source=ReQISC "
            "regulus.utils.arch.gene_grid_2d_graph(N) "
            f"| N={n} | edges={len(edges)}",
            flush=True,
        )

        return edges

    if _settings.TOPOLOGY == "custom":
        result = sorted({
            edge_key(int(u), int(v))
            for u, v in _settings.CUSTOM_COUPLING_EDGES
        })

        for u, v in result:
            if (
                u < 0
                or v < 0
                or u >= n
                or v >= n
                or u == v
            ):
                raise ValueError(
                    "CUSTOM_COUPLING_EDGES 包含非法边："
                    f"({u},{v}), num_qubits={n}"
                )

        return result

    raise ValueError(
        f"不支持的 TOPOLOGY: {_settings.TOPOLOGY}"
    )


def build_adjacency(num_qubits: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(num_qubits)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def all_pairs_shortest_paths(num_qubits: int, adj: List[List[int]]) -> List[List[int]]:
    dist = [[10 ** 9] * num_qubits for _ in range(num_qubits)]
    for s in range(num_qubits):
        dist[s][s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[s][v] > dist[s][u] + 1:
                    dist[s][v] = dist[s][u] + 1
                    queue.append(v)
    return dist


def route_cartan_to_target_point(point_cache: Dict[Tuple[float, float, float], Any], target_cartan: Tuple[float, float, float]):
    from . import common as _mod_common
    key = _mod_common.cartan_key(*target_cartan)
    if key not in point_cache:
        point_cache[key] = cartan_to_target_point(target_cartan, max_den=_settings.MAX_DEN)
    return point_cache[key]


def estimate_duration_for_target(ctx: _mod_models.BaseRouteContext, target_cartan: Tuple[float, float, float]) -> _mod_models.EstimatedDecompInfo:
    from . import common as _mod_common
    key = _mod_common.cartan_key(*target_cartan)
    if key in ctx.est_cache:
        info = ctx.est_cache[key]
        if _settings.PRINT_PRECOMPUTE_DETAIL:
            _mod_common.trace_event(
                "门型预估",
                f"cache=HIT | target={_mod_common._fmt_cartan(target_cartan)} | estimated_k={info.estimated_k} | duration={info.duration:.12f}"
            )
        return info

    est_t0 = time.perf_counter()
    if _settings.PRINT_PRECOMPUTE_DETAIL:
        _mod_common.trace_event("门型预估", f"开始 membership test | target={_mod_common._fmt_cartan(target_cartan)}")
    target_point_t0 = time.perf_counter()
    target_point = route_cartan_to_target_point(ctx.point_cache, target_cartan)
    target_point_seconds = time.perf_counter() - target_point_t0

    membership_t0 = time.perf_counter()
    est_k = estimate_gate_count_from_candidate_cache(ctx.poly_cache, target_point)
    membership_seconds = time.perf_counter() - membership_t0

    if est_k is None:
        _mod_common.trace_event(
            "门型预估",
            f"失败 | target={_mod_common._fmt_cartan(target_cartan)} | MAX_GATE_COUNT={_settings.MAX_GATE_COUNT} | "
            f"target_point_seconds={target_point_seconds:.6f}s | membership_seconds={membership_seconds:.6f}s"
        )
        raise _mod_models.CandidateInvalidError(
            f"当前候选基门无法在 MAX_GATE_COUNT={_settings.MAX_GATE_COUNT} 内覆盖目标门 {target_cartan}"
        )

    info = _mod_models.EstimatedDecompInfo(
        cartan=tuple(float(x) for x in target_cartan),
        estimated_k=int(est_k),
        duration=float(est_k) * float(ctx.tau_basis),
    )
    ctx.est_cache[key] = info
    if _settings.PRINT_PRECOMPUTE_DETAIL:
        _mod_common.trace_event(
            "门型预估",
            f"完成 | target={_mod_common._fmt_cartan(target_cartan)} | estimated_k={int(est_k)} | "
            f"duration={info.duration:.12f} | target_point_seconds={target_point_seconds:.6f}s | "
            f"membership_seconds={membership_seconds:.6f}s | total={time.perf_counter() - est_t0:.6f}s"
        )
    return info


def collect_route_cartan_types(target_gates: List[_mod_models.TwoQGate]) -> List[Tuple[float, float, float]]:
    """收集候选基门评估开始前必须可分解的门型。

    这里只加入原始逻辑 2Q 门，不再提前加入 mirror(U)=SWAP·U。

    mirror 门的可分解性改为延迟判断：
    只有路由过程中真正评估某个 SWAP 吸收候选时，
    才在 evaluate_candidate_swap_variants() 中计算 mirror_cartan，
    并调用 estimate_duration_for_target() 判断其是否可由当前候选基门分解。

    这样：
      1. all-to-all 拓扑不会因为根本不会使用的 mirror 门而淘汰候选基门；
      2. chain/grid 中 mirror 不可分解时，只禁用对应 absorb 方案，
         显式 SWAP 或其他 SWAP 候选仍然可以继续参与比较。
    """
    from . import common as _mod_common
    uniq: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}

    def add(cartan: Tuple[float, float, float]) -> None:
        c = _mod_common.normalize_cartan_to_balanced(cartan)
        uniq[_mod_common.cartan_key(*c)] = tuple(float(x) for x in c)

    for g in target_gates:
        c = _mod_common.normalize_cartan_to_balanced((g.x, g.y, g.z))
        add(c)

    return list(uniq.values())


def precompute_route_estimates(ctx: _mod_models.BaseRouteContext, route_cartan_types: List[Tuple[float, float, float]]) -> None:
    from . import common as _mod_common
    with _mod_common.timed_trace("路由门型预估", f"route_cartan_type_count={len(route_cartan_types)}"):
        for idx, cartan in enumerate(route_cartan_types):
            with _mod_common.timed_trace("路由门型预估", f"type_idx={idx:02d} | target={_mod_common._fmt_cartan(cartan)}"):
                estimate_duration_for_target(ctx, cartan)


def build_2q_dag(gates: List[_mod_models.TwoQGate]) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, int], List[int]]:
    preds: Dict[int, Set[int]] = {i: set() for i in range(len(gates))}
    succs: Dict[int, Set[int]] = {i: set() for i in range(len(gates))}
    last_on_qubit: Dict[int, int] = {}
    for idx, g in enumerate(gates):
        for q in (g.q0, g.q1):
            if q in last_on_qubit:
                p = last_on_qubit[q]
                preds[idx].add(p)
                succs[p].add(idx)
            last_on_qubit[q] = idx
    req = {i: len(preds[i]) for i in range(len(gates))}
    front = [i for i in range(len(gates)) if req[i] == 0]
    return preds, succs, req, front


def init_route_state(num_qubits: int, front_layer: List[int], required_predecessors: Dict[int, int]) -> _mod_models.RouteState:
    return _mod_models.RouteState(
        logical_to_physical=list(range(num_qubits)),
        physical_to_logical=list(range(num_qubits)),
        qubit_ready=[0.0] * num_qubits,
        current_critical=0.0,
        routed_blocks=[],
        mapped_gate_ids=set(),
        front_layer=list(front_layer),
        last_mapped_layer=[],
        required_predecessors=dict(required_predecessors),
        last_edge_gate={},
        last_touch_op=[None] * num_qubits,
    )


def append_block(
    state: _mod_models.RouteState,
    source_gate_idx: Optional[int],
    name: str,
    kind: str,
    p0: int,
    p1: int,
    logical_pair: Optional[Tuple[int, int]],
    duration: float,
    cartan: Tuple[float, float, float],
    detail: str = "",
) -> int:
    from . import common as _mod_common
    start = max(state.qubit_ready[p0], state.qubit_ready[p1])
    end = start + float(duration)
    state.qubit_ready[p0] = end
    state.qubit_ready[p1] = end
    state.current_critical = max(state.current_critical, end)
    op_id = len(state.routed_blocks)
    edge = edge_key(p0, p1)
    mapping_snapshot = tuple(
        int(v)
        for v in state.logical_to_physical
    )

    state.routed_blocks.append(
        _mod_models.RoutedBlock(
            op_id=op_id,
            source_gate_idx=source_gate_idx,
            name=name,
            kind=kind,
            q0=min(p0, p1),
            q1=max(p0, p1),
            physical_q0_ordered=int(p0),
            physical_q1_ordered=int(p1),
            logical_pair=logical_pair,
            start=float(start),
            end=float(end),
            duration=float(duration),
            cartan=tuple(float(x) for x in cartan),
            mapping_before=mapping_snapshot,
            mapping_after=mapping_snapshot,
            detail=detail,
        )
    )
    state.last_edge_gate[edge] = _mod_models.LastEdgeGateInfo(
        op_id=op_id,
        edge=edge,
        cartan=tuple(float(x) for x in cartan),
        duration=float(duration),
        start=float(start),
        end=float(end),
        name=name,
        kind=kind,
    )
    state.last_touch_op[p0] = op_id
    state.last_touch_op[p1] = op_id
    if _settings.PRINT_ROUTE_BLOCK_DETAIL:
        _mod_common.trace_event(
            "追加路由块",
            f"op_id={op_id:04d} | name={name} | kind={kind} | phys=({min(p0, p1)},{max(p0, p1)}) | "
            f"logical={logical_pair} | duration={float(duration):.12f} | start={start:.12f} | "
            f"end={end:.12f} | current_critical={state.current_critical:.12f} | detail={detail}"
        )
    return op_id


def remap_successors_into_front_layer(state: _mod_models.RouteState, gate_idx: int, succs: Dict[int, Set[int]]) -> None:
    for s in succs[gate_idx]:
        state.required_predecessors[s] -= 1
        if state.required_predecessors[s] == 0:
            state.front_layer.append(s)


def update_last_mapped_layer(state: _mod_models.RouteState, succs: Dict[int, Set[int]]) -> None:
    """更新论文中的最后映射层 L。

    L 包含已完成映射、且当前不存在已映射后继双量子比特门的逻辑门。
    """
    state.last_mapped_layer = sorted(
        gate_idx
        for gate_idx in state.mapped_gate_ids
        if not any(s in state.mapped_gate_ids for s in succs[gate_idx])
    )


def execute_logical_gate(
    state: _mod_models.RouteState,
    gate_idx: int,
    gates: List[_mod_models.TwoQGate],
    succs: Dict[int, Set[int]],
    ctx: _mod_models.BaseRouteContext,
) -> int:
    from . import common as _mod_common
    g = gates[gate_idx]
    p0 = state.logical_to_physical[g.q0]
    p1 = state.logical_to_physical[g.q1]
    _mod_common.trace_event(
        "执行逻辑门",
        f"gate_idx={gate_idx:04d} | name={g.name} | logical=({g.q0},{g.q1}) | "
        f"physical=({p0},{p1}) | target={_mod_common._fmt_cartan((g.x, g.y, g.z))}"
    )
    est = estimate_duration_for_target(ctx, (g.x, g.y, g.z))
    _mod_common.trace_event(
        "执行逻辑门",
        f"gate_idx={gate_idx:04d} | estimated_k={est.estimated_k} | duration={est.duration:.12f}"
    )
    append_block(
        state=state,
        source_gate_idx=gate_idx,
        name=g.name,
        kind="logical_2q",
        p0=p0,
        p1=p1,
        logical_pair=(g.q0, g.q1),
        duration=est.duration,
        cartan=(g.x, g.y, g.z),
        detail=f"estimated_k={est.estimated_k}",
    )
    state.mapped_gate_ids.add(gate_idx)
    if gate_idx in state.front_layer:
        state.front_layer.remove(gate_idx)
    remap_successors_into_front_layer(state, gate_idx, succs)
    update_last_mapped_layer(state, succs)
    return 1


def peel_executable_gates(
    state: _mod_models.RouteState,
    gates: List[_mod_models.TwoQGate],
    succs: Dict[int, Set[int]],
    edge_set: Set[Tuple[int, int]],
    ctx: _mod_models.BaseRouteContext,
) -> Tuple[int, List[str], List[int]]:
    """剥离当前布局下所有可直接执行的 front-layer 门。

    返回执行数量、执行门名和执行门下标。
    执行门下标用于区分候选是否真正执行了当前 blocked gate，避免只执行
    其他 front-layer 门却被误判为对当前阻塞门有推进。
    """
    count = 0
    executed_names: List[str] = []
    executed_gate_ids: List[int] = []
    changed = True
    while changed:
        changed = False
        ready_now = list(state.front_layer)
        for gate_idx in ready_now:
            g = gates[gate_idx]
            p0 = state.logical_to_physical[g.q0]
            p1 = state.logical_to_physical[g.q1]
            if edge_key(p0, p1) in edge_set:
                execute_logical_gate(state, gate_idx, gates, succs, ctx)
                count += 1
                executed_names.append(g.name)
                executed_gate_ids.append(int(gate_idx))
                changed = True
    return count, executed_names, executed_gate_ids


def apply_swap_to_mapping(state: _mod_models.RouteState, edge: Tuple[int, int]) -> None:
    from . import common as _mod_common
    u, v = edge
    lu = state.physical_to_logical[u]
    lv = state.physical_to_logical[v]
    _mod_common.trace_event("映射交换", f"edge=({u},{v}) | before p{u}->l{lu}, p{v}->l{lv}")
    state.physical_to_logical[u], state.physical_to_logical[v] = lv, lu
    state.logical_to_physical[lu], state.logical_to_physical[lv] = v, u
    _mod_common.trace_event("映射交换", f"edge=({u},{v}) | after p{u}->l{state.physical_to_logical[u]}, p{v}->l{state.physical_to_logical[v]}")


def mapping_after_physical_swap_snapshot(
    mapping_before: Tuple[int, ...],
    edge: Tuple[int, int],
) -> Tuple[int, ...]:
    """只对给定 mapping 快照施加一次物理 SWAP，返回新的 l2p 快照。

    该函数不读取当前 RouteState，因此特别适合 swap_absorbed：
    被吸收的 SWAP 会把一个较早的 logical_2q block 改写成 SWAP·U，
    我们需要记录“该 block 自身前后”的局部 mapping 变化，而不是把
    之后其他不相关 SWAP 已经造成的全局 mapping 也混进来。
    """
    l2p = [int(v) for v in mapping_before]
    n = len(l2p)
    u, v = int(edge[0]), int(edge[1])

    if not (0 <= u < n and 0 <= v < n and u != v):
        raise ValueError(
            f"非法 physical SWAP edge={edge} | n={n}"
        )

    p2l = [-1] * n
    for logical_q, physical_q in enumerate(l2p):
        if not (0 <= physical_q < n):
            raise ValueError(
                "mapping_before 含越界物理比特 | "
                f"logical={logical_q} physical={physical_q} n={n}"
            )
        if p2l[physical_q] != -1:
            raise ValueError(
                "mapping_before 不是一一映射 | "
                f"physical={physical_q}"
            )
        p2l[physical_q] = int(logical_q)

    lu = int(p2l[u])
    lv = int(p2l[v])
    if lu < 0 or lv < 0:
        raise ValueError(
            f"mapping_before 无法反解 edge={edge}"
        )

    l2p[lu] = v
    l2p[lv] = u
    return tuple(l2p)


def build_extended_set(front_layer: List[int], succs: Dict[int, Set[int]], mapped: Set[int], required_predecessors: Dict[int, int], limit: int) -> List[int]:
    if limit <= 0:
        return []
    extended: List[int] = []
    tmp = list(front_layer)
    tmp_req = dict(required_predecessors)
    seen = set(front_layer)
    while tmp and len(extended) < limit:
        nxt: List[int] = []
        for node in tmp:
            for s in succs[node]:
                if s in mapped:
                    continue
                tmp_req[s] -= 1
                if tmp_req[s] == 0 and s not in seen:
                    seen.add(s)
                    extended.append(s)
                    nxt.append(s)
                    if len(extended) >= limit:
                        break
            if len(extended) >= limit:
                break
        tmp = nxt
    return extended[:limit]


def candidate_swaps(front_layer: List[int], gates: List[_mod_models.TwoQGate], layout: List[int], adj: List[List[int]]) -> List[Tuple[int, int]]:
    cands = set()
    for gate_idx in front_layer[:max(1, _settings.FRONT_LAYER_SIZE)]:
        g = gates[gate_idx]
        for p in (layout[g.q0], layout[g.q1]):
            for q in adj[p]:
                cands.add(edge_key(p, q))
    return sorted(cands)


def average_distance_for_gates(front: List[int], gates: List[_mod_models.TwoQGate], layout: List[int], dist: List[List[int]]) -> float:
    if not front:
        return 0.0
    total = 0.0
    for idx in front:
        g = gates[idx]
        p0 = layout[g.q0]
        p1 = layout[g.q1]
        total += float(dist[p0][p1])
    return total / float(len(front))


def clone_route_state(state: _mod_models.RouteState) -> _mod_models.RouteState:
    """轻量复制路由状态；历史 routed_blocks 只复制列表，不深拷贝每个块。"""
    return _mod_models.RouteState(
        logical_to_physical=list(state.logical_to_physical),
        physical_to_logical=list(state.physical_to_logical),
        qubit_ready=list(state.qubit_ready),
        current_critical=float(state.current_critical),
        routed_blocks=list(state.routed_blocks),
        mapped_gate_ids=set(state.mapped_gate_ids),
        front_layer=list(state.front_layer),
        last_mapped_layer=list(state.last_mapped_layer),
        required_predecessors=dict(state.required_predecessors),
        last_edge_gate=dict(state.last_edge_gate),
        last_touch_op=list(state.last_touch_op),
        consecutive_zero_advance=int(state.consecutive_zero_advance),
        last_swap_edge=state.last_swap_edge,
        last_blocked_gate_idx=state.last_blocked_gate_idx,
    )


def evaluate_candidate_swap_variants(
    state: _mod_models.RouteState,
    swap_edge: Tuple[int, int],
    gates: List[_mod_models.TwoQGate],
    succs: Dict[int, Set[int]],
    edge_set: Set[Tuple[int, int]],
    dist: List[List[int]],
    ctx: _mod_models.BaseRouteContext,
    blocked_gate_idx: int,
) -> List[Dict[str, Any]]:
    from . import common as _mod_common
    variant_eval_t0 = time.perf_counter()
    candidates: List[Dict[str, Any]] = []
    u, v = swap_edge

    blocked_gate = gates[blocked_gate_idx]
    _mod_common.trace_event(
        "SWAP候选展开",
        f"开始 | edge=({u},{v}) | blocked_gate_idx={blocked_gate_idx:04d} | blocked={blocked_gate.name}"
    )
    before_p0 = state.logical_to_physical[blocked_gate.q0]
    before_p1 = state.logical_to_physical[blocked_gate.q1]
    before_dist = dist[before_p0][before_p1]

    # 论文中的 ΔD_F 与 ΔD_E 必须在同一组 F/E 门上比较，不能在执行候选
    # 后换成新的集合，否则得到的不是“该 SWAP 引起的距离变化”。
    front_before = list(state.front_layer)
    ext_before = build_extended_set(
        front_before,
        succs,
        state.mapped_gate_ids,
        state.required_predecessors,
        _settings.EXTENDED_LAYER_SIZE,
    )
    avg_front_before = average_distance_for_gates(
        front_before, gates, state.logical_to_physical, dist
    )
    avg_ext_before = average_distance_for_gates(
        ext_before, gates, state.logical_to_physical, dist
    )

    def finalize_candidate(
        new_state: _mod_models.RouteState,
        variant: str,
        combo_cartan: Optional[Tuple[float, float, float]],
        base_duration: float,
        base_n2q: int,
        detail_tag: str,
        explicit_swap_added: int,
    ) -> Dict[str, Any]:
        # 此时 new_state 已完成候选 SWAP 对映射和线路块的试探性修改，但尚未
        # 剥离新变得可执行的门。先在原 F/E 上计算纯粹由映射变化产生的距离增量。
        avg_front_after_same_set = average_distance_for_gates(
            front_before, gates, new_state.logical_to_physical, dist
        )
        avg_ext_after_same_set = average_distance_for_gates(
            ext_before, gates, new_state.logical_to_physical, dist
        )
        delta_front = float(avg_front_after_same_set - avg_front_before)
        delta_ext = float(avg_ext_after_same_set - avg_ext_before)

        advanced_count, executed_names, executed_gate_ids = peel_executable_gates(new_state, gates, succs, edge_set, ctx)
        blocked_executed = int(blocked_gate_idx) in executed_gate_ids

        after_p0 = new_state.logical_to_physical[blocked_gate.q0]
        after_p1 = new_state.logical_to_physical[blocked_gate.q1]
        after_dist = dist[after_p0][after_p1]

        delta_crit = float(new_state.current_critical - state.current_critical)
        route_impact = float(delta_front + _settings.ROUTING_K_E * delta_ext)
        swap_cost = float(delta_crit + _settings.ROUTING_LAMBDA * route_impact)
        front_after = list(new_state.front_layer)
        ext_after = build_extended_set(front_after, succs, new_state.mapped_gate_ids, new_state.required_predecessors, _settings.EXTENDED_LAYER_SIZE)
        avg_front_after = average_distance_for_gates(front_after, gates, new_state.logical_to_physical, dist)
        avg_ext_after = average_distance_for_gates(ext_after, gates, new_state.logical_to_physical, dist)

        new_state.last_swap_edge = swap_edge
        new_state.last_blocked_gate_idx = blocked_gate_idx

        # 这里的 zero_advance 只表示
        #   既没有执行任何 front-layer 门，也没有缩短当前 blocked 门距离。
        # 之前的版本把 blocked_dist 13→12、12→11 这类必要搬运也累计到
        # consecutive_zero_advance，导致远距离门在 chain 拓扑上还没靠近就被中断。
        # 对路由来说，只要当前 blocked 门的物理距离变小，就是有效进展。
        has_real_progress = bool(advanced_count > 0 or after_dist < before_dist)
        if has_real_progress:
            new_state.consecutive_zero_advance = 0
        else:
            new_state.consecutive_zero_advance = state.consecutive_zero_advance + 1

        sig = route_state_signature(new_state)
        return {
            "swap_edge": swap_edge,
            "variant": variant,
            "state": new_state,
            "delta_crit": delta_crit,
            "delta_front": delta_front,
            "delta_ext": delta_ext,
            "route_impact": route_impact,
            "swap_cost": swap_cost,
            "advanced_count": int(advanced_count),
            "executed_names": executed_names,
            "executed_gate_ids": executed_gate_ids,
            "blocked_executed": bool(blocked_executed),
            "blocked_before": int(before_dist),
            "blocked_after": int(after_dist),
            "blocked_reduced": bool(after_dist < before_dist),
            "avg_front_after": float(avg_front_after),
            "avg_ext_after": float(avg_ext_after),
            "base_duration": float(base_duration),
            "base_n2q": int(base_n2q),
            "combo_cartan": combo_cartan,
            "detail_tag": detail_tag,
            "explicit_swap_added": int(explicit_swap_added),
            "state_signature": sig,
        }

    if not (
        _settings.FORBID_IMMEDIATE_ZERO_ADVANCE_BACKTRACK
        and state.consecutive_zero_advance > 0
        and state.last_swap_edge == swap_edge
        and state.last_blocked_gate_idx == blocked_gate_idx
    ):
        try:
            # 显式 SWAP 与普通逻辑门使用相同的单值性多面体拟分解口径。
            swap_est = estimate_duration_for_target(ctx, _settings.SWAP_CARTAN)
        except _mod_models.CandidateInvalidError:
            swap_est = None
            _mod_common.trace_event(
                "SWAP候选展开",
                f"显式SWAP不可用 | edge=({u},{v}) | 当前基本门无法在 MAX_GATE_COUNT 内拟分解 SWAP"
            )

        if swap_est is not None:
            swap_duration = float(swap_est.duration)
            new_state = clone_route_state(state)
            append_block(
                state=new_state,
                source_gate_idx=None,
                name="SWAP",
                kind="swap",
                p0=u,
                p1=v,
                logical_pair=None,
                duration=swap_duration,
                cartan=_settings.SWAP_CARTAN,
                detail=f"explicit_swap | estimated_k={swap_est.estimated_k}",
            )
            apply_swap_to_mapping(new_state, swap_edge)

            # 这个显式 SWAP 就发生在当前时刻，因此 mapping_after 直接记录
            # 当前 RouteState 的真实结果，不需要 fidelity 阶段再次推导。
            explicit_blk = new_state.routed_blocks[-1]
            explicit_blk.mapping_after = tuple(
                int(v)
                for v in new_state.logical_to_physical
            )

            explicit_info = finalize_candidate(
                new_state=new_state,
                variant="swap",
                combo_cartan=None,
                base_duration=swap_duration,
                base_n2q=int(swap_est.estimated_k),
                detail_tag="explicit_swap_quasi_decomposition",
                explicit_swap_added=1,
            )
            _mod_common.trace_event(
                "SWAP候选展开",
                f"显式SWAP候选 | edge=({u},{v}) | estimated_k={swap_est.estimated_k} | "
                f"base_duration={swap_duration:.12f} | advanced={explicit_info['advanced_count']} | "
                f"delta_crit={explicit_info['delta_crit']:.12f} | H_s={explicit_info['swap_cost']:.12f}"
            )
            candidates.append(explicit_info)

    last_info = state.last_edge_gate.get(edge_key(u, v))
    last_block = None if last_info is None else state.routed_blocks[last_info.op_id]
    if (
        last_info is not None
        and last_block is not None
        and last_block.source_gate_idx in state.last_mapped_layer
        and state.last_touch_op[u] == last_info.op_id
        and state.last_touch_op[v] == last_info.op_id
        and last_info.kind not in {"swap", "swap_absorbed"}
    ):
        combo_cartan = mirror_cartan(last_info.cartan)
        _mod_common.trace_event(
            "SWAP吸收候选",
            f"尝试 | edge=({u},{v}) | last_op={last_info.op_id} | "
            f"last_cartan={_mod_common._fmt_cartan(last_info.cartan)} | combo={_mod_common._fmt_cartan(combo_cartan)}"
        )

        # --------------------------------------------------------------
        # 只有真正计算这个 SWAP 吸收候选的代价时，
        # 才判断 mirror(U)=SWAP·U 是否能由当前候选基门拟分解。
        #
        # 如果不能在 MAX_GATE_COUNT 内分解：
        #   - 只禁用当前 absorb variant；
        #   - 不淘汰整个候选基门；
        #   - 不影响同一 edge 的显式 SWAP variant；
        #   - 不影响其他 SWAP 候选。
        # --------------------------------------------------------------
        try:
            combo_est = estimate_duration_for_target(ctx, combo_cartan)
        except _mod_models.CandidateInvalidError:
            combo_est = None
            _mod_common.trace_event(
                "SWAP吸收候选",
                f"不可用 | edge=({u},{v}) | "
                f"combo={_mod_common._fmt_cartan(combo_cartan)} | "
                f"reason=mirror_not_decomposable_within_MAX_GATE_COUNT={_settings.MAX_GATE_COUNT}"
            )
            if _settings.PRINT_CANDIDATE_EVAL:
                print(
                    f"    [过滤] edge={swap_edge} | variant=absorb | "
                    f"combo={_mod_common._fmt_cartan(combo_cartan)} | "
                    f"reason=mirror_not_decomposable_within_MAX_GATE_COUNT"
                )

        if combo_est is not None:
            _mod_common.trace_event(
                "SWAP吸收候选",
                f"combo_estimated_k={combo_est.estimated_k} | "
                f"combo_duration={combo_est.duration:.12f}"
            )

            new_state = clone_route_state(state)

            blk = copy.copy(new_state.routed_blocks[last_info.op_id])
            new_state.routed_blocks[last_info.op_id] = blk
            old_start = float(blk.start)
            old_duration = float(blk.duration)
            new_duration = float(combo_est.duration)
            new_end = old_start + new_duration

            other_max = 0.0
            for q, t in enumerate(new_state.qubit_ready):
                if q not in (u, v):
                    other_max = max(other_max, t)

            blk.name = f"SWAP·{blk.name}"
            blk.kind = "swap_absorbed"
            blk.duration = new_duration
            blk.end = new_end
            blk.absorbed_from_cartan = tuple(
                float(v)
                for v in last_info.cartan
            )
            blk.cartan = tuple(combo_cartan)
            blk.detail = (
                f"swap_absorbed_on_edge={swap_edge} | "
                f"combo_estimated_k={combo_est.estimated_k} | "
                f"old_dur={old_duration:.12f}"
            )

            new_state.qubit_ready[u] = new_end
            new_state.qubit_ready[v] = new_end
            new_state.current_critical = max(other_max, new_end)
            new_state.last_edge_gate[edge_key(u, v)] = _mod_models.LastEdgeGateInfo(
                op_id=blk.op_id,
                edge=edge_key(u, v),
                cartan=tuple(combo_cartan),
                duration=new_duration,
                start=old_start,
                end=new_end,
                name=blk.name,
                kind=blk.kind,
            )
            apply_swap_to_mapping(new_state, swap_edge)

            # 注意：swap_absorbed 是把较早的 logical_2q block 改写为 SWAP·U。
            # 该 block 的 mapping_after 应描述“它自身”造成的 mapping 变化：
            # mapping_before + 当前 edge SWAP。不能直接写 new_state 当前完整 mapping，
            # 因为那里面可能已经包含后来发生在其他物理比特上的 SWAP。
            blk.mapping_after = mapping_after_physical_swap_snapshot(
                tuple(blk.mapping_before),
                swap_edge,
            )

            absorb_info = finalize_candidate(
                new_state=new_state,
                variant="absorb",
                combo_cartan=tuple(combo_cartan),
                base_duration=float(combo_est.duration),
                base_n2q=int(combo_est.estimated_k),
                detail_tag=f"swap_absorbed_from_op={last_info.op_id}",
                explicit_swap_added=0,
            )
            _mod_common.trace_event(
                "SWAP吸收候选",
                f"完成 | edge=({u},{v}) | "
                f"advanced={absorb_info['advanced_count']} | "
                f"delta_crit={absorb_info['delta_crit']:.12f}"
            )
            candidates.append(absorb_info)

    _mod_common.trace_event(
        "SWAP候选展开",
        f"结束 | edge=({u},{v}) | variants={len(candidates)} | elapsed={time.perf_counter() - variant_eval_t0:.6f}s"
    )
    return candidates


def route_with_min_swap_cost_vb(
    gates: List[_mod_models.TwoQGate],
    edges: List[Tuple[int, int]],
    adj: List[List[int]],
    dist: List[List[int]],
    ctx: _mod_models.BaseRouteContext,
    prebuilt_dag: Optional[Tuple[Dict[int, Set[int]], Dict[int, int], List[int]]] = None,
) -> _mod_models.RouteState:
    from . import common as _mod_common
    route_total_t0 = time.perf_counter()
    edge_set = set(edges)
    if prebuilt_dag is None:
        _, succs, required_predecessors, front = build_2q_dag(gates)
    else:
        succs, required_predecessors0, front0 = prebuilt_dag
        required_predecessors = dict(required_predecessors0)
        front = list(front0)
    # 使用拓扑上下文中的完整物理比特数，而不是只看活跃 2Q 门。
    # 这样 ReQISC 综合后即使存在暂时未参与 2Q 门的逻辑比特，也不会丢失。
    num_qubits = len(adj)
    _mod_common.trace_event(
        "路由开始",
        f"num_gates={len(gates)} | num_qubits={num_qubits} | edges={edges} | initial_front={front}"
    )
    state = init_route_state(num_qubits, front, required_predecessors)

    initial_peel_t0 = time.perf_counter()
    initial_count, initial_names, initial_gate_ids = peel_executable_gates(state, gates, succs, edge_set, ctx)
    _mod_common.trace_event("初始门剥离", f"count={initial_count} | elapsed={time.perf_counter() - initial_peel_t0:.6f}s | executed={initial_names} | executed_ids={initial_gate_ids}")
    if _settings.PRINT_WORKFLOW:
        print("=== 初始可执行门剥离 ===")
        print(f"初始执行数量 = {initial_count} | executed = {initial_names}")

    visited_signatures: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
    init_sig = route_state_signature(state)
    visited_signatures[init_sig] = 1

    steps = 0
    while len(state.mapped_gate_ids) < len(gates):
        step_t0 = time.perf_counter()
        steps += 1
        if steps > _settings.MAX_ROUTE_STEPS:
            raise RuntimeError("路由超过 MAX_ROUTE_STEPS，停止")
        if not state.front_layer:
            raise RuntimeError("front_layer 为空但仍未完成路由")

        front_now = list(state.front_layer)
        ext_now = build_extended_set(front_now, succs, state.mapped_gate_ids, state.required_predecessors, _settings.EXTENDED_LAYER_SIZE)
        cands = candidate_swaps(front_now, gates, state.logical_to_physical, adj)

        blocked_gate_idx = front_now[0]
        blocked_gate = gates[blocked_gate_idx]
        blocked_p0 = state.logical_to_physical[blocked_gate.q0]
        blocked_p1 = state.logical_to_physical[blocked_gate.q1]
        blocked_dist = dist[blocked_p0][blocked_p1]
        avg_front = average_distance_for_gates(front_now, gates, state.logical_to_physical, dist)
        avg_ext = average_distance_for_gates(ext_now, gates, state.logical_to_physical, dist)

        if _settings.PRINT_ROUTE_WORKFLOW:
            print()
            print(f"[路由步] step={steps:03d} | blocked={blocked_gate.name} | logical=({blocked_gate.q0},{blocked_gate.q1}) | physical=({blocked_p0},{blocked_p1}) | blocked_dist={blocked_dist}")
            print(f"         front_layer={[gates[i].name for i in front_now]} | extended_set={[gates[i].name for i in ext_now]}")
            print(f"         last_mapped_layer={[gates[i].name for i in state.last_mapped_layer]}")
            print(f"         avg_front={avg_front:.6f} | avg_ext={avg_ext:.6f} | candidates={cands} | zero_streak={state.consecutive_zero_advance}")

        if not cands:
            raise RuntimeError("当前没有候选 SWAP，无法继续路由")

        eval_swaps_t0 = time.perf_counter()
        evaluated: List[Dict[str, Any]] = []
        for sw in cands:
            infos = evaluate_candidate_swap_variants(state, sw, gates, succs, edge_set, dist, ctx, blocked_gate_idx)
            if not infos and _settings.PRINT_CANDIDATE_EVAL:
                print(f"    [过滤] edge={sw} | reason=no_valid_variant")
            for info in infos:
                visits = visited_signatures.get(info["state_signature"], 0)
                info["state_visits"] = int(visits)

                if _settings.PRINT_CANDIDATE_EVAL:
                    combo_txt = "" if info["combo_cartan"] is None else f" | combo={info['combo_cartan']}"
                    print(
                        f"    [候选] edge={info['swap_edge']} | variant={info['variant']} | base_dur={info['base_duration']:.12f} | "
                        f"base_n2q={info['base_n2q']} | ΔTcrit={info['delta_crit']:.12f} | advanced={info['advanced_count']} | "
                        f"ΔD_F={info['delta_front']:.6f} | ΔD_E={info['delta_ext']:.6f} | "
                        f"impact={info['route_impact']:.6f} | H_s={info['swap_cost']:.12f} | "
                        f"exec={info['executed_names']} | blocked_exec={info['blocked_executed']} | "
                        f"blocked_dist={info['blocked_before']}→{info['blocked_after']} | "
                        f"avg_front_after={info['avg_front_after']:.6f} | avg_ext_after={info['avg_ext_after']:.6f} | "
                        f"explicit_swap_added={info['explicit_swap_added']} | visits={info['state_visits']}{combo_txt}"
                    )

                if visits >= _settings.MAX_STATE_VISITS:
                    if _settings.PRINT_CANDIDATE_EVAL:
                        print(
                            f"      [过滤] edge={sw} | "
                            f"variant={info['variant']} | "
                            f"reason=state_visited_too_many_times"
                        )
                    continue

                # ------------------------------------------------------
                # 必须产生真实推进：
                #   1) 候选 SWAP 后至少执行了一个 front-layer 门；或
                #   2) 当前 blocked gate 的物理距离严格缩短。
                #
                # 不满足上述任一条件的候选，即使 H_s 更小，也不能进入
                # 最终排序。这样避免路由器在不同 mapping 之间长期游走，
                # 产生大量无效 SWAP。
                # ------------------------------------------------------
                has_real_progress = bool(
                    int(info["advanced_count"]) > 0
                    or bool(info["blocked_reduced"])
                )

                if _settings.REQUIRE_REAL_PROGRESS and not has_real_progress:
                    if _settings.PRINT_CANDIDATE_EVAL:
                        print(
                            f"      [过滤] edge={sw} | "
                            f"variant={info['variant']} | "
                            f"reason=no_real_progress | "
                            f"advanced={info['advanced_count']} | "
                            f"blocked_dist="
                            f"{info['blocked_before']}->{info['blocked_after']}"
                        )
                    continue

                evaluated.append(info)

        _mod_common.trace_event(
            "路由步候选评估",
            f"step={steps:03d} | raw_candidates={len(cands)} | evaluated={len(evaluated)} | "
            f"elapsed={time.perf_counter() - eval_swaps_t0:.6f}s"
        )
        if not evaluated:
            raise RuntimeError(
                "所有候选 SWAP 都被过滤，路由停止："
                "没有满足“执行至少一个 front-layer 门或缩短当前 blocked gate 距离”"
                "的真实推进候选"
            )

        # 先满足“必须真实推进”的硬约束，再严格按论文统一代价 H_s 选择。
        # 真实推进定义：
        #   advanced_count > 0
        #   或 blocked_after < blocked_before。
        # 在满足推进约束的候选中，不再把显式 SWAP 数量或分解门数作为排序条件。
        best = min(
            evaluated,
            key=lambda x: (
                x["swap_cost"],
                x["delta_crit"],
                x["delta_front"],
                x["delta_ext"],
                x["state_visits"],
                x["swap_edge"],
                x["variant"],
            ),
        )

        best_sig = best["state_signature"]
        visited_signatures[best_sig] = visited_signatures.get(best_sig, 0) + 1

        if _settings.PRINT_ROUTE_WORKFLOW:
            combo_txt = "" if best["combo_cartan"] is None else f" | combo={best['combo_cartan']}"
            print(
                f"[选中] edge={best['swap_edge']} | variant={best['variant']} | "
                f"H_s={best['swap_cost']:.12f} | ΔTcrit={best['delta_crit']:.12f} | "
                f"ΔD_F={best['delta_front']:.6f} | ΔD_E={best['delta_ext']:.6f} | "
                f"advanced={best['advanced_count']} | exec={best['executed_names']} | "
                f"blocked_exec={best['blocked_executed']} | "
                f"blocked_dist={best['blocked_before']}→{best['blocked_after']} | "
                f"explicit_swap_added={best['explicit_swap_added']} | zero_streak={best['state'].consecutive_zero_advance}{combo_txt}"
            )

        state = best["state"]
        _mod_common.trace_event(
            "路由步完成",
            f"step={steps:03d} | selected_edge={best['swap_edge']} | variant={best['variant']} | "
            f"mapped={len(state.mapped_gate_ids)}/{len(gates)} | current_critical={state.current_critical:.12f} | "
            f"elapsed={time.perf_counter() - step_t0:.6f}s"
        )

    _mod_common.trace_event(
        "路由结束",
        f"steps={steps} | blocks={len(state.routed_blocks)} | current_critical={state.current_critical:.12f} | "
        f"elapsed={time.perf_counter() - route_total_t0:.6f}s"
    )
    return state


def save_routed_blocks_json(
    best: Dict[str, Any],
    path: str,
    *,
    qasm_file: str = "",
) -> str:
    """保存本文最终路由后的二比特块序列。

    JSON 保存本文最终 routed 2Q block，供完整实际线路与实验核查使用：
      metadata
        - qasm_file
        - base_cartan
        - tau_basis
        - routing_topology
        - routing_edges
        - logical_to_physical
        - physical_to_logical
        - num_qubits
        - current_critical
        - explicit_swaps
        - swap_absorbed
      blocks
        - q0/q1: 无向拓扑边端点
        - physical_q0_ordered/physical_q1_ordered: 实际有序物理操作数
        - mapping_before/mapping_after: 路由阶段保存的映射快照
        - kind
        - cartan
        - duration
        - start/end
        - detail
    """
    routed_state = best.get("routed_state")
    if routed_state is None:
        raise RuntimeError(
            "best 中没有 routed_state，无法导出 routed_blocks.json"
        )

    routing = best.get("routing", {}) or {}

    block_rows: List[Dict[str, Any]] = []

    for blk in routed_state.routed_blocks:
        block_rows.append({
            "op_id": int(blk.op_id),
            "source_gate_idx": (
                None
                if blk.source_gate_idx is None
                else int(blk.source_gate_idx)
            ),
            "name": str(blk.name),
            "kind": str(blk.kind),
            # 无向拓扑边端点（用于路由核查）
            "q0": int(blk.q0),
            "q1": int(blk.q1),

            # 有序物理操作数（用于实际 1Q+G 线路重建）
            "physical_q0_ordered": int(
                blk.physical_q0_ordered
            ),
            "physical_q1_ordered": int(
                blk.physical_q1_ordered
            ),

            "mapping_before": [
                int(v)
                for v in blk.mapping_before
            ],
            "mapping_after": [
                int(v)
                for v in blk.mapping_after
            ],

            "logical_pair": (
                None
                if blk.logical_pair is None
                else [
                    int(blk.logical_pair[0]),
                    int(blk.logical_pair[1]),
                ]
            ),
            "start": float(blk.start),
            "end": float(blk.end),
            "duration": float(blk.duration),
            "cartan": [
                float(blk.cartan[0]),
                float(blk.cartan[1]),
                float(blk.cartan[2]),
            ],
            "absorbed_from_cartan": (
                None
                if blk.absorbed_from_cartan is None
                else [
                    float(blk.absorbed_from_cartan[0]),
                    float(blk.absorbed_from_cartan[1]),
                    float(blk.absorbed_from_cartan[2]),
                ]
            ),
            "detail": str(blk.detail),
        })

    routing_edges = [
        [int(u), int(v)]
        for u, v in (routing.get("edges", []) or [])
    ]

    # 保险：若 best 里没有保存边，再从当前官方拓扑生成器恢复。
    if not routing_edges:
        n = int(len(routed_state.logical_to_physical))
        routing_edges = [
            [int(u), int(v)]
            for u, v in build_coupling_edges(n)
        ]

    payload = {
        "metadata": {
            "qasm_file": (
                os.path.abspath(qasm_file)
                if qasm_file
                else ""
            ),
            "base_cartan": [
                float(v)
                for v in best.get(
                    "base_cartan",
                    (math.nan, math.nan, math.nan),
                )
            ],
            "tau_basis": (
                None
                if best.get("tau_basis") is None
                else float(best.get("tau_basis"))
            ),
            "routing_topology": str(
                routing.get("topology", _settings.TOPOLOGY)
            ),
            "routing_topology_source": str(
                routing.get("topology_source", "")
            ),
            "routing_edges": routing_edges,
            "logical_to_physical": [
                int(v)
                for v in routed_state.logical_to_physical
            ],
            "physical_to_logical": [
                int(v)
                for v in routed_state.physical_to_logical
            ],
            "num_qubits": int(
                len(routed_state.logical_to_physical)
            ),
            "current_critical": float(
                routed_state.current_critical
            ),
            "explicit_swaps": int(
                sum(
                    1
                    for blk in routed_state.routed_blocks
                    if blk.kind == "swap"
                )
            ),
            "swap_absorbed": int(
                sum(
                    1
                    for blk in routed_state.routed_blocks
                    if blk.kind == "swap_absorbed"
                )
            ),
        },
        "blocks": block_rows,
    }

    out_abs = os.path.abspath(path)
    os.makedirs(
        os.path.dirname(out_abs),
        exist_ok=True,
    )

    with open(
        out_abs,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_abs


def _route_static_cache_key(
    target_gates: List[_mod_models.TwoQGate],
    num_qubits: int,
) -> Tuple[Any, ...]:
    from . import common as _mod_common
    gate_sig = tuple(
        (
            int(g.q0),
            int(g.q1),
            _mod_common.cartan_key(g.x, g.y, g.z),
        )
        for g in target_gates
    )

    topo_sig = (
        str(_settings.TOPOLOGY),
        int(num_qubits),
        (
            "reqisc_arch.gene_grid_2d_graph"
            if _settings.TOPOLOGY == "grid"
            else ""
        ),
        tuple(
            sorted(
                edge_key(int(u), int(v))
                for u, v in _settings.CUSTOM_COUPLING_EDGES
            )
        ),
        int(_settings.MAX_DEN),
    )

    return topo_sig + (gate_sig,)


def get_or_build_route_static_context(
    target_gates: List[_mod_models.TwoQGate],
    num_qubits: int,
) -> _mod_models.RouteStaticContext:
    """构造一次并缓存与 θ 无关的拓扑、最短路、2Q DAG 和目标 monodromy 点。"""
    from . import common as _mod_common
    key = _route_static_cache_key(target_gates, num_qubits)
    cached = _ROUTE_STATIC_CACHE.get(key)
    if cached is not None:
        _mod_common.trace_event("静态路由缓存", f"cache=HIT | topology={_settings.TOPOLOGY} | num_qubits={num_qubits}")
        return cached

    t0 = time.perf_counter()
    edges = build_coupling_edges(
        num_qubits
    )
    adj = build_adjacency(
        num_qubits,
        edges,
    )
    dist = all_pairs_shortest_paths(
        num_qubits,
        adj,
    )

    # 确认官方 ReQISC Grid / 当前拓扑是连通的。
    if int(num_qubits) > 1:
        unreachable = [
            q
            for q in range(int(num_qubits))
            if dist[0][q] >= 10 ** 9
        ]
        if unreachable:
            raise RuntimeError(
                "硬件耦合图不是连通图："
                f"topology={_settings.TOPOLOGY}, "
                f"num_qubits={num_qubits}, "
                f"unreachable_from_q0={unreachable}"
            )

    _, succs, required_predecessors, front = (
        build_2q_dag(target_gates)
    )
    route_cartan_types = collect_route_cartan_types(target_gates)

    # 这里只提前缓存“原始逻辑门”的 monodromy point。
    # mirror(U)=SWAP·U 不再提前生成/检查：
    # 只有真正计算 SWAP 吸收代价时才按需生成并判断可分解性。
    shared_point_cache: Dict[Tuple[float, float, float], Any] = {}
    for cartan in route_cartan_types:
        route_cartan_to_target_point(shared_point_cache, cartan)

    # 显式 SWAP 的 point 可以提前缓存。
    # 注意：这里只缓存目标点，不判断当前候选基门是否能分解 SWAP；
    # SWAP 的可分解性仍在 evaluate_candidate_swap_variants() 中按需判断。
    route_cartan_to_target_point(shared_point_cache, _settings.SWAP_CARTAN)

    cached = _mod_models.RouteStaticContext(
        edges=edges,
        adj=adj,
        dist=dist,
        succs=succs,
        required_predecessors=required_predecessors,
        front=front,
        route_cartan_types=route_cartan_types,
        point_cache=shared_point_cache,
    )
    _ROUTE_STATIC_CACHE[key] = cached
    _mod_common.trace_event(
        "静态路由缓存",
        f"cache=MISS->BUILD | topology={_settings.TOPOLOGY} | num_qubits={num_qubits} | "
        f"route_types={len(route_cartan_types)} | points={len(shared_point_cache)} | "
        f"elapsed={time.perf_counter() - t0:.6f}s",
    )
    return cached


def routed_native_2q_metrics(
    routed_state: _mod_models.RouteState,
    tau_basis: float,
) -> Tuple[int, int]:
    """统计路由后真正展开为候选基门后的 2Q 门数量与 2Q 深度。"""
    from . import common as _mod_common
    tau = float(tau_basis)
    if tau <= _mod_common.EPS:
        raise ValueError("tau_basis 必须大于 0")

    n_phys = len(routed_state.physical_to_logical)
    depth_ready = [0] * n_phys
    total_native_2q = 0

    for blk in routed_state.routed_blocks:
        ratio = float(blk.duration) / tau
        k = int(round(ratio))
        if abs(ratio - k) > 1e-7:
            raise RuntimeError(
                f"无法从 RoutedBlock 恢复原生 2Q 门数："
                f"name={blk.name}, duration={blk.duration}, tau_basis={tau}, ratio={ratio}"
            )
        if k <= 0:
            continue

        q0, q1 = int(blk.q0), int(blk.q1)
        start_depth = max(depth_ready[q0], depth_ready[q1])
        end_depth = start_depth + k
        depth_ready[q0] = end_depth
        depth_ready[q1] = end_depth
        total_native_2q += k

    routed_2q_depth = max(depth_ready) if depth_ready else 0
    return int(total_native_2q), int(routed_2q_depth)

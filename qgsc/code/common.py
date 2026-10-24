# -*- coding: utf-8 -*-
"""日志、LRS 环境、Cartan 坐标与脉冲时间等公共工具。"""
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


from .runtime import settings as _settings, ISING_ROOT as _ISING_ROOT
from .models import TwoQGate
import numpy as np
import gate_count as poly_est
from . import models as _mod_models

_TRACE_RUN_T0 = time.perf_counter()
_TRACE_COUNTER = 0
EPS = 1e-12


class TeeOutput:
    """把 print 同时输出到终端和日志文件。"""

    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def reqisc_print_compat_context():
    """兼容 ReQISC/Regulus 中 print(..., style=...) 的日志写法。

    只丢弃 Rich 的显示参数，不修改 ReQISC 算法。
    """
    original_print = builtins.print

    def compat_print(*args, **kwargs):
        for key in (
            "style", "highlight", "markup", "emoji",
            "soft_wrap", "new_line_start", "crop",
            "overflow", "justify", "width", "height",
        ):
            kwargs.pop(key, None)
        return original_print(*args, **kwargs)

    builtins.print = compat_print
    try:
        yield
    finally:
        builtins.print = original_print


def progress_step(
    circuit_index: int,
    total_circuits: int,
    step: str,
    message: str = "",
) -> None:
    """统一打印每条线路当前运行阶段。"""
    suffix = f" | {message}" if message else ""
    print(
        f"[{circuit_index}/{total_circuits}] "
        f"[{step}]"
        f"{suffix}",
        flush=True,
    )


def _fmt_cartan(cartan: Tuple[float, float, float]) -> str:
    x, y, z = [float(v) for v in cartan]
    return f"Can({x:.12f}, {y:.12f}, {z:.12f})"


def trace_event(tag: str, message: str) -> None:
    """打印带全局相对时间的详细步骤日志。"""
    global _TRACE_COUNTER
    if not _settings.PRINT_DETAILED_STEP_TIMING:
        return
    _TRACE_COUNTER += 1
    elapsed = time.perf_counter() - _TRACE_RUN_T0
    print(f"[TRACE #{_TRACE_COUNTER:05d} | +{elapsed:10.6f}s | {tag}] {message}")


@contextmanager
def timed_trace(tag: str, message: str):
    """打印某一步开始与结束耗时。"""
    if _settings.PRINT_DETAILED_STEP_TIMING:
        trace_event(tag, f"开始 | {message}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if _settings.PRINT_DETAILED_STEP_TIMING:
            trace_event(tag, f"结束 | {message} | elapsed={dt:.6f}s")


def get_lrs_restart_dir() -> str:
    script_dir = str(_ISING_ROOT)
    if os.path.isabs(_settings.LRS_RESTART_SUBDIR):
        folder = os.path.abspath(_settings.LRS_RESTART_SUBDIR)
    else:
        folder = os.path.abspath(os.path.join(script_dir, _settings.LRS_RESTART_SUBDIR))
    os.makedirs(folder, exist_ok=True)
    return folder


def configure_lrs_restart_environment() -> str:
    """统一 Python、monodromy 与 lrs 的临时目录。"""
    folder = get_lrs_restart_dir()
    os.makedirs(folder, exist_ok=True)

    os.environ["LRS_PATH"] = poly_est.LRS_PATH
    os.environ["TMPDIR"] = folder
    os.environ["TEMP"] = folder
    os.environ["TMP"] = folder
    os.environ["LRS_TMPDIR"] = folder
    tempfile.tempdir = folder
    return folder


@contextmanager
def lrs_restart_file_context():
    folder = configure_lrs_restart_environment()
    if not _settings.LRS_FORCE_CWD_TO_RESTART_DIR:
        yield folder
        return

    old_cwd = os.getcwd()
    try:
        os.chdir(folder)
        yield folder
    finally:
        os.chdir(old_cwd)


def _fold_to_pi_over_4(value: float) -> float:
    """把单个 Cartan 分量折叠到 [-pi/4, pi/4] 区间。"""
    v = float(value)
    period = math.pi / 2.0
    v = (v + math.pi / 4.0) % period - math.pi / 4.0
    if abs(v) < 1e-12:
        v = 0.0
    return float(v)


def is_balanced_cartan(cartan: Tuple[float, float, float], tol: float = 1e-10) -> bool:
    """判断 Cartan 坐标是否位于平衡 canonical 区域 pi/4 >= x >= y >= |z|。"""
    x, y, z = [float(v) for v in cartan]
    return (
        x <= math.pi / 4.0 + tol
        and x >= y - tol
        and y >= -tol
        and y >= abs(z) - tol
    )


def normalize_cartan_to_balanced(cartan: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """将任意等价 Cartan 坐标归一化到平衡 canonical 区域。

    该函数主要用于批量读取 QASM 时处理负角度 CP/CU1 门。
    例如 cp(-pi/4) 直接得到 (-pi/16, 0, 0)，必须先折叠为 (pi/16, 0, 0)，
    否则后续 mirror_cartan 会收到非法输入并报错。
    """
    from itertools import permutations

    raw = tuple(float(v) for v in cartan)
    if is_balanced_cartan(raw):
        x, y, z = raw
        return (
            0.0 if abs(x) < 1e-12 else float(x),
            0.0 if abs(y) < 1e-12 else float(y),
            0.0 if abs(z) < 1e-12 else float(z),
        )

    folded = tuple(_fold_to_pi_over_4(v) for v in raw)

    # Weyl chamber 允许置换坐标，同时允许偶数个符号翻转。
    sign_patterns = (
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )

    candidates: List[Tuple[float, float, float]] = []
    for perm in set(permutations(folded, 3)):
        for signs in sign_patterns:
            candidate = tuple(float(perm[i] * signs[i]) for i in range(3))
            if is_balanced_cartan(candidate):
                x, y, z = candidate
                if abs(x) < 1e-12:
                    x = 0.0
                if abs(y) < 1e-12:
                    y = 0.0
                if abs(z) < 1e-12:
                    z = 0.0
                candidates.append((float(x), float(y), float(z)))

    if candidates:
        # 选择稳定的代表元，避免同一目标门在不同文件中因为浮点扰动出现不同 key。
        candidates.sort(
            key=lambda t: (round(t[0], 12), round(t[1], 12), round(t[2], 12)),
            reverse=True,
        )
        return candidates[0]

    # 理论上很少进入这里。作为保护，把绝对值排序后投影到 Weyl 区域。
    vals = sorted((abs(v) for v in folded), reverse=True)
    x = min(vals[0], math.pi / 4.0)
    y = min(vals[1], x)
    z = min(vals[2], y)
    if abs(x) < 1e-12:
        x = 0.0
    if abs(y) < 1e-12:
        y = 0.0
    if abs(z) < 1e-12:
        z = 0.0
    return (float(x), float(y), float(z))


def normalize_twoq_gate(g: "TwoQGate") -> "TwoQGate":
    x, y, z = normalize_cartan_to_balanced((g.x, g.y, g.z))
    return _mod_models.TwoQGate(str(g.name), int(g.q0), int(g.q1), float(x), float(y), float(z))


def normal_form_diagonal_coupling(
    raw_a: float,
    raw_b: float,
    raw_c: float,
) -> Tuple[float, float, float]:
    """
    将物理输入的对角两比特耦合

        H = raw_a XX + raw_b YY + raw_c ZZ

    转换为 ReQISC 后续公式使用的 Weyl/canonical normal form

        H_can = a XX + b YY + c ZZ,
        a >= b >= |c|.

    对角耦合情况下，这一步等价于对三个耦合强度的绝对值降序排列，
    并将局部 SO(3) 旋转不能消去的取向符号保留在第三个 canonical 系数上。

    典型例子
    --------
    1. 纯 ZZ 耦合 H = gamma ZZ
       NormalForm 后得到 (|gamma|, 0, 0)，即按 canonical 的 XX 主轴计算。

    2. XY + ZZ 耦合 H = g/2 XX + g/2 YY + h/2 ZZ 且 |h| <= g
       NormalForm 后得到 (g/2, g/2, h/2)，ZZ 作为第三 canonical 分量参与计算。
    """
    vals = [float(raw_a), float(raw_b), float(raw_c)]
    abs_vals = sorted((abs(v) for v in vals), reverse=True)

    a_norm = float(abs_vals[0])
    b_norm = float(abs_vals[1])
    c_abs = float(abs_vals[2])

    if a_norm <= EPS:
        raise ValueError("硬件没有有效两比特耦合，无法构造 canonical normal form")

    # rank < 3 时第三个 canonical 系数为 0，符号没有意义。
    if c_abs <= EPS:
        c_norm = 0.0
    else:
        det_sign = math.copysign(1.0, vals[0] * vals[1] * vals[2])
        c_norm = det_sign * c_abs

    return a_norm, b_norm, c_norm


def _validate_inputs(a: float, b: float, c: float, x: float, y: float, z: float) -> None:
    if not (a > 0):
        raise ValueError("要求 a > 0")
    if not (a >= b >= abs(c)):
        raise ValueError("要求满足 a >= b >= |c|")
    if not (math.pi / 4 >= x >= y >= abs(z)):
        raise ValueError("要求满足 pi/4 >= x >= y >= |z|")


def gate_pulse_duration(
    a: float, b: float, c: float,
    x: float, y: float, z: float,
) -> Tuple[float, Dict[str, Any]]:
    """只计算候选基门的最短演化时间。

    当前优化目标不再包含幅值约束，因此不再求解 Ω1、Ω2、δ、alpha、beta。
    这避免了每个候选参数额外执行 SLSQP/root 数值求解。
    """
    _validate_inputs(a, b, c, x, y, z)

    tau0 = x / a
    tau_plus = (x + y - z) / (a + b - c)
    tau_minus = (x + y + z) / (a + b + c)
    tau1 = max(tau0, tau_plus, tau_minus)

    tau0_p = (math.pi / 2 - x) / a
    tau_plus_p = (math.pi / 2 - x + y + z) / (a + b - c)
    tau_minus_p = (math.pi / 2 - x + y - z) / (a + b + c)
    tau2 = max(tau0_p, tau_plus_p, tau_minus_p)

    mirrored_dual = tau2 < tau1 - EPS
    tau = tau2 if mirrored_dual else tau1
    active = (tau0_p, tau_plus_p, tau_minus_p) if mirrored_dual else (tau0, tau_plus, tau_minus)
    labels = ("tau0", "tau_plus", "tau_minus")
    branch = labels[int(np.argmax(np.asarray(active, dtype=float)))]

    return float(tau), {
        "branch": branch,
        "mirrored_dual": bool(mirrored_dual),
        "tau1": float(tau1),
        "tau2": float(tau2),
    }


def cartan_key(x: float, y: float, z: float, digits: int = 12) -> Tuple[float, float, float]:
    return (round(float(x), digits), round(float(y), digits), round(float(z), digits))

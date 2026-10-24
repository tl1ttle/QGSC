# -*- coding: utf-8 -*-
"""参数入口。编辑 PARAMETERS，或用命令行覆盖参数后运行完整流程。

示例：python run.py --topology chain --input benchmarks/original
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "qgsc.code"

from .runtime import ISING_ROOT, configure, run_lock

PARAMETERS = {
    # 输入与输出
    'QASM_FOLDER': str(ISING_ROOT / 'benchmarks' / 'original'),
    'QASM_RECURSIVE': True,
    'QASM_FILE_SUFFIXES': ('.qasm', '.QASM'),
    'BATCH_RESULT_DIR': str(ISING_ROOT / 'outputs' / 'runs'),
    'BATCH_RESULT_CSV': 'test-near-chain.csv',
    'BATCH_SAVE_PER_CIRCUIT_LOGS': True,
    'BATCH_MAX_FILES': None,
    # ReQISC 前端综合
    'REQISC_PREOPTIMIZE': True,
    'REQISC_ROOT': os.environ.get('REQISC_ROOT', str(ISING_ROOT / 'external' / 'reqisc_asplos_v2')),
    'REQISC_QISKIT_BASIS': ('u', 'cx', 'ccx'),
    'REQISC_TEMPLATE_SYNTHESIS': True,
    'REQISC_TEMPLATE_NUM_TRIALS': 1,
    'REQISC_FULL_SYNTHESIS': True,
    'REQISC_NUM_APPROX_SYN_TRIALS': 3,
    'REQISC_SAVE_CANONICAL_CSV': True,
    'REQISC_CANONICAL_SUBDIR': 'reqisc_canonical',
    'REQISC_SAVE_PREPROCESSED_QASM': True,
    'REQISC_PREPROCESSED_SUBDIR': 'reqisc_input',
    'REQISC_SAVE_OPTIMIZED_QASM': False,
    'BATCH_MAX_2Q_GATES': 5000,
    # 实际分解与保真度
    'AUTO_RUN_FIDELITY': True,
    'PDBLS_PY': str(ISING_ROOT / 'qgsc' / 'core' / 'pdbls.py'),
    'ROUTED_BLOCKS_SUBDIR': 'routed_blocks',
    'FIDELITY_SUBDIR': 'fidelity',
    'ACTUAL_CIRCUIT_SUBDIR': 'actual_physical_circuit',
    'FIDELITY_P0': 0.001,
    'FIDELITY_TAU_CX': math.pi / math.sqrt(2.0),
    'FIDELITY_SHOTS': 4096,
    'FIDELITY_MAX_QUBITS': 24,
    'FIDELITY_PDBLS_RESTARTS': 8,
    'FIDELITY_PDBLS_MAXITER': 10000,
    'FIDELITY_DECOMP_FD': 1.0 - 1e-06,
    'FIDELITY_PDBLS_BOUNDS_ANGLE': 2.0 * math.pi,
    'FIDELITY_PDBLS_SEED': 2026,
    'FIDELITY_CONTINUE_ON_ERROR': True,
    'FIDELITY_PRINT_TRACEBACK': True,
    # 拟分解与硬件耦合
    'MAX_GATE_COUNT': 3,
    'MAX_DEN': 4096,
    'G_COUPLING': 1.0,
    'A_COEFF': 0.5,
    'B_COEFF': 0.5,
    'C_COEFF': 0.0,
    'NORMALIZE_HARDWARE_COUPLING': True,
    # 拓扑与路由
    'TOPOLOGY': 'chain',
    'CUSTOM_COUPLING_EDGES': [],
    'FRONT_LAYER_SIZE': 4,
    'EXTENDED_LAYER_SIZE': 12,
    'ROUTING_LAMBDA': 1.0,
    'ROUTING_K_E': 0.5,
    'MAX_ROUTE_STEPS': 100000,
    'MAX_STATE_VISITS': 1,
    'FORBID_IMMEDIATE_ZERO_ADVANCE_BACKTRACK': True,
    'REQUIRE_REAL_PROGRESS': True,
    'SWAP_CARTAN': (math.pi / 4.0, math.pi / 4.0, math.pi / 4.0),
    # 传统门集
    'FIXED_BASELINE_GATES': {'CX': (math.pi / 4.0, 0.0, 0.0), 'SQISW': (math.pi / 8.0, math.pi / 8.0, 0.0), 'ISWAP': (math.pi / 4.0, math.pi / 4.0, 0.0), 'BGATE': (math.pi / 4.0, math.pi / 8.0, 0.0)},
    # 基门参数搜索
    'WEYL_X_MIN': 0.001,
    'NATIVE_COBYLA_X0': math.pi / 8,
    'NATIVE_MIN_A': 1e-12,
    'NATIVE_COBYLA_MAXITER': 10,
    'NATIVE_COBYLA_RHOBEG': 0.12,
    'COBYLA_TOL': 0.0001,
    'COBYLA_CATOL': 1e-05,
    'OPT_INVALID_PENALTY': 1000000000.0,
    'OPT_CONSTRAINT_PENALTY_SCALE': 1000000.0,
    # LRS 临时目录
    'LRS_RESTART_SUBDIR': str(ISING_ROOT / 'outputs' / 'lrs_tmp'),
    'LRS_FORCE_CWD_TO_RESTART_DIR': True,
    # 日志
    'PRINT_WORKFLOW': True,
    'PRINT_TIMING': True,
    'PRINT_ROUTE_WORKFLOW': False,
    'PRINT_ROUTE_CANDIDATE_EVAL': False,
    'PRINT_CANDIDATE_EVAL': False,
    'PRINT_DETAILED_STEP_TIMING': False,
    'PRINT_ROUTE_BLOCK_DETAIL': False,
    'PRINT_PRECOMPUTE_DETAIL': False,
    'COBYLA_VERBOSE_EVAL': False,
    'PRINT_EVAL_CAPTURED_LOG_ON_INVALID': False,
}


def _parameters(overrides=None):
    values = copy.deepcopy(PARAMETERS)
    overrides = {} if overrides is None else dict(overrides)
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError("未知参数：" + ", ".join(sorted(unknown)))
    values.update(overrides)
    if values["TOPOLOGY"] not in ("chain", "grid", "alltoall", "custom"):
        raise ValueError("TOPOLOGY 必须是 chain、grid、alltoall 或 custom")
    return values


def run(parameters=None, **overrides):
    """一次完成预处理、基门搜索/拟分解、路由、实际分解及保真度评估。

    parameters 使用 PARAMETERS 中的同名键；返回原程序的批处理结果列表。
    """
    merged = {} if parameters is None else dict(parameters)
    merged.update(overrides)
    values = _parameters(merged)
    if not run_lock.acquire(blocking=False):
        raise RuntimeError("同一进程已有实验正在运行，请等待其完成。")
    try:
        configure(values)
        from . import common, decomposition_routing, parameter_solver, fidelity, pipeline
        common._TRACE_RUN_T0 = common.time.perf_counter()
        common._TRACE_COUNTER = 0
        decomposition_routing._GLOBAL_FIXED_POLYTOPE_CACHE.clear()
        decomposition_routing._ROUTE_STATIC_CACHE.clear()
        parameter_solver._PDBLS_MODULE = None
        fidelity._QSIM_SIMULATOR = None
        common.configure_lrs_restart_environment()
        return pipeline.run_folder_batch()
    finally:
        run_lock.release()


def _parse_value(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="包含 PARAMETERS 同名键的 JSON 文件")
    parser.add_argument("--input", "--qasm-folder", dest="QASM_FOLDER", help="原始 QASM 文件夹")
    parser.add_argument("--output", dest="BATCH_RESULT_DIR", help="实验输出文件夹")
    parser.add_argument("--topology", dest="TOPOLOGY", choices=["chain", "grid", "alltoall", "custom"])
    parser.add_argument("--reqisc-root", dest="REQISC_ROOT")
    parser.add_argument("--shots", dest="FIDELITY_SHOTS", type=int)
    parser.add_argument("--p0", dest="FIDELITY_P0", type=float)
    parser.add_argument("--maxiter", dest="NATIVE_COBYLA_MAXITER", type=int)
    parser.add_argument("--max-files", dest="BATCH_MAX_FILES", type=int)
    parser.add_argument("--max-2q", dest="BATCH_MAX_2Q_GATES", type=int)
    parser.add_argument("--a", dest="A_COEFF", type=float)
    parser.add_argument("--b", dest="B_COEFF", type=float)
    parser.add_argument("--c", dest="C_COEFF", type=float)
    parser.add_argument("--no-fidelity", action="store_true", help="仅运行综合、搜索与路由")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="覆盖任意其他参数，可重复使用")
    parser.add_argument("--show-config", action="store_true", help="显示实际参数后退出，不启动实验")
    args = parser.parse_args(argv)
    values = {}
    if args.config:
        with args.config.open(encoding="utf-8-sig") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            parser.error("配置文件必须是 JSON 对象")
    for name in PARAMETERS:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if args.no_fidelity:
        values["AUTO_RUN_FIDELITY"] = False
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error("--set 的格式为 NAME=VALUE")
        values[key.strip()] = _parse_value(value)
    try:
        selected = _parameters(values)
    except ValueError as exc:
        parser.error(str(exc))
    if args.show_config:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return None
    return run(selected)


if __name__ == "__main__":
    main()

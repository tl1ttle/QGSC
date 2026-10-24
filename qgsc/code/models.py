# -*- coding: utf-8 -*-
"""各阶段共享的数据结构，不包含算法或实验参数。"""
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



@dataclass
class TwoQGate:
    name: str
    q0: int
    q1: int
    x: float
    y: float
    z: float


@dataclass
class OneQProgramOp:
    """ReQISC canonical 完整线路中的一个原始 1Q 操作。"""
    program_index: int
    logical_q: int
    name: str
    matrix: np.ndarray


@dataclass
class ReqiscProgramContext:
    """为了在本文路由完成后恢复完整 program 所保留的 1Q 上下文。

    before_twoq[i]：
        原 ReQISC canonical 线路中，位于第 i 个 2Q 门之前、且尚未被更早
        2Q 门消耗的 1Q 操作。它们必须在该 source 2Q 门执行前，作用于
        当前映射下对应的物理 qubit。

    trailing：
        每条逻辑 wire 最后一个 2Q 门之后剩余的 1Q 操作。
    """
    before_twoq: Dict[int, List[OneQProgramOp]]
    trailing: List[OneQProgramOp]
    # ReQISC canonical 第 i 个 2Q 门的精确 4x4 unitary。
    # 用于 full-program PDBLS target，避免只靠 canonical 坐标重建产生约定偏差。
    twoq_matrices: Dict[int, np.ndarray]
    num_oneq: int
    num_twoq: int


@dataclass
class ReqiscOptimizeContext:
    """整个文件夹批处理期间复用 ReQISC/BQSKit 编译器。"""
    compiler: Any
    Circuit: Any
    template_based_synthesize: Any
    hierarchical_synthesize: Any
    rebase_to_canonical_by_bqskit: Any
    structures_unique: Any
    root: str


@dataclass
class CandidatePolytopeCache:
    base_cartan: Tuple[float, float, float]
    depth_polys: Dict[int, Any]
    max_n: int
    max_den: int
    build_seconds: float = 0.0
    gate_polytope_seconds: float = 0.0
    depth_build_seconds: Dict[int, float] = field(default_factory=dict)


@dataclass
class EstimatedDecompInfo:
    cartan: Tuple[float, float, float]
    estimated_k: int
    duration: float


@dataclass
class LastEdgeGateInfo:
    op_id: int
    edge: Tuple[int, int]
    cartan: Tuple[float, float, float]
    duration: float
    start: float
    end: float
    name: str
    kind: str


@dataclass
class RoutedBlock:
    op_id: int
    source_gate_idx: Optional[int]
    name: str
    kind: str

    # q0/q1 继续作为“无向耦合边端点”使用，始终满足 q0 < q1。
    # 它们只用于拓扑、距离、last_edge_gate 等路由逻辑。
    q0: int
    q1: int

    # 真正执行该 2Q unitary 时的有序物理操作数。
    # 不能用 min/max 排序，否则 PDBLS 求出的 q0/q1 局域 1Q 层可能被交换。
    physical_q0_ordered: int
    physical_q1_ordered: int

    logical_pair: Optional[Tuple[int, int]]
    start: float
    end: float
    duration: float
    cartan: Tuple[float, float, float]

    # 路由阶段直接保存 mapping 快照。
    # fidelity 阶段不再“重放 SWAP 推导 mapping”，而是直接使用快照。
    mapping_before: Tuple[int, ...]
    mapping_after: Tuple[int, ...]

    detail: str = ""
    # swap_absorbed 时，cartan 会被改写为 mirror canonical 坐标。
    # 为了最终 full-program PDBLS 求解真正的 SWAP·U 矩阵，
    # 必须保留吸收前原逻辑 Can 的 Cartan。
    absorbed_from_cartan: Optional[Tuple[float, float, float]] = None


@dataclass
class RouteState:
    logical_to_physical: List[int]
    physical_to_logical: List[int]
    qubit_ready: List[float]
    current_critical: float
    routed_blocks: List[RoutedBlock]
    mapped_gate_ids: Set[int]
    front_layer: List[int]
    last_mapped_layer: List[int]
    required_predecessors: Dict[int, int]
    last_edge_gate: Dict[Tuple[int, int], LastEdgeGateInfo]
    last_touch_op: List[Optional[int]]
    consecutive_zero_advance: int = 0
    last_swap_edge: Optional[Tuple[int, int]] = None
    last_blocked_gate_idx: Optional[int] = None


@dataclass
class BaseRouteContext:
    base_cartan: Tuple[float, float, float]
    tau_basis: float
    poly_cache: CandidatePolytopeCache
    point_cache: Dict[Tuple[float, float, float], Any]
    est_cache: Dict[Tuple[float, float, float], EstimatedDecompInfo]


@dataclass
class RouteStaticContext:
    """与候选参数 x 无关，在同一线路的整个 COBYLA 搜索期间复用。"""
    edges: List[Tuple[int, int]]
    adj: List[List[int]]
    dist: List[List[int]]
    succs: Dict[int, Set[int]]
    required_predecessors: Dict[int, int]
    front: List[int]
    route_cartan_types: List[Tuple[float, float, float]]
    point_cache: Dict[Tuple[float, float, float], Any]


class CandidateInvalidError(RuntimeError):
    pass

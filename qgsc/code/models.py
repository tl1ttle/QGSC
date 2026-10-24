

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
    
    program_index: int
    logical_q: int
    name: str
    matrix: np.ndarray


@dataclass
class ReqiscProgramContext:
    
    before_twoq: Dict[int, List[OneQProgramOp]]
    trailing: List[OneQProgramOp]
    
    
    twoq_matrices: Dict[int, np.ndarray]
    num_oneq: int
    num_twoq: int


@dataclass
class ReqiscOptimizeContext:
    
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

    
    
    q0: int
    q1: int

    
    
    physical_q0_ordered: int
    physical_q1_ordered: int

    logical_pair: Optional[Tuple[int, int]]
    start: float
    end: float
    duration: float
    cartan: Tuple[float, float, float]

    
    
    mapping_before: Tuple[int, ...]
    mapping_after: Tuple[int, ...]

    detail: str = ""
    
    
    
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

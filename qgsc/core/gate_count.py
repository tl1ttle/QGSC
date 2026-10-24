
# Resolve local dependencies when running scripts from their new folders.
import sys as _ising_sys
from pathlib import Path as _IsingPath
_ISING_ROOT = _IsingPath(__file__).resolve().parents[1]
for _ising_path in (str(_ISING_ROOT), str(_ISING_ROOT / "core")):
    if _ising_path not in _ising_sys.path:
        _ising_sys.path.insert(0, _ising_path)


import os
import tempfile
import shutil
from fractions import Fraction

import numpy as np

# ============================================================
# LRS 配置：所有临时文件统一放在同一个目录
# ============================================================
LRS_PATH = os.environ.get("LRS_PATH") or shutil.which("lrs") or "lrs"
LRS_TMP_DIR = os.environ.get("LRS_TMPDIR") or str(_ISING_ROOT.parent / "outputs" / "lrs_tmp")

# NamedTemporaryFile / monodromy / lrs 全部使用这一目录。
os.makedirs(LRS_TMP_DIR, exist_ok=True)
os.environ["LRS_PATH"] = LRS_PATH
os.environ["TEMP"] = LRS_TMP_DIR
os.environ["TMP"] = LRS_TMP_DIR
os.environ["TMPDIR"] = LRS_TMP_DIR
os.environ["LRS_TMPDIR"] = LRS_TMP_DIR
tempfile.tempdir = LRS_TMP_DIR

from monodromy.backend.lrs import LRSBackend
import monodromy.backend

monodromy.backend.backend = LRSBackend()

# Workaround: the gate-count workflow does not need expensive trim by exact volume.
import monodromy.polytopes as poly_mod
def _no_trim(trimmable_polytopes, fixed_polytopes=None):
    return trimmable_polytopes
poly_mod.trim_polytope_set = _no_trim

from monodromy.coverage import CircuitPolytope, deduce_qlr_consequences
from monodromy.static import exactly, identity_polytope, everything_polytope
from monodromy.static.matrices import canonical_matrix
from monodromy.coordinates import unitary_to_monodromy_coordinate

def frac(x, max_den=1000):
    return Fraction(str(x)).limit_denominator(max_den)

def cartan_to_monodromy_point(a, b, c, max_den=1000):
    U = canonical_matrix(a, b, c)
    mx, my, mz = unitary_to_monodromy_coordinate(U)[:3]
    return frac(mx, max_den), frac(my, max_den), frac(mz, max_den)

def gate_polytope_from_cartan(a, b, c, name="G", max_den=1000):
    mx, my, mz = cartan_to_monodromy_point(a, b, c, max_den=max_den)
    point_poly = exactly(mx, my, mz, name=name)
    G = CircuitPolytope(
        operations=[name],
        cost=1.0,
        convex_subpolytopes=point_poly.convex_subpolytopes,
    )
    return G, (mx, my, mz)

def compose_once(current_poly, base_gate):
    next_poly = deduce_qlr_consequences(
        target="c",
        a_polytope=current_poly,
        b_polytope=base_gate,
        c_polytope=everything_polytope,
    )
    return CircuitPolytope(
        operations=current_poly.operations + base_gate.operations,
        cost=current_poly.cost + base_gate.cost,
        convex_subpolytopes=next_poly.convex_subpolytopes,
    )

def min_gate_count_for_target(base_cartan, target_cartan, max_n=10, max_den=1000):
    Ga, Gb, Gc = base_cartan
    Ta, Tb, Tc = target_cartan

    base_gate, base_mono = gate_polytope_from_cartan(Ga, Gb, Gc, name="G", max_den=max_den)
    target_point = cartan_to_monodromy_point(Ta, Tb, Tc, max_den=max_den)

    print("基门 G 的 monodromy 坐标 =", base_mono)
    print("目标门 T 的 monodromy 坐标 =", target_point)

    current = CircuitPolytope(
        operations=[],
        cost=0.0,
        convex_subpolytopes=identity_polytope.convex_subpolytopes,
    )

    if current.has_element(target_point):
        return 0, current

    for n in range(1, max_n + 1):
        current = compose_once(current, base_gate)
        print(f"\n===== n = {n} =====")
        print("operations =", current.operations)
        print("vertices =", current.vertices)
        if current.has_element(target_point):
            return n, current

    return None, None

def random_base_cartan(min_a=0.05):
    a = np.random.uniform(min_a, np.pi/4)
    b = np.random.uniform(0, a)
    c = np.random.uniform(-b, b)
    return (a, b, c)

if __name__ == "__main__":
    base_cartan = random_base_cartan(min_a=0.05)
    print("随机 base_cartan =", base_cartan)

    # CNOT/CZ class:
    # base_cartan = (np.pi/8, 0.0, 0.0)
    # target_cartan=random_base_cartan(min_a=0.05)
    target_cartan = (np.pi/4, np.pi/4, np.pi/4)

    n, poly = min_gate_count_for_target(
        base_cartan=base_cartan,
        target_cartan=target_cartan,
        max_n=4,
        max_den=200,
    )

    print("\n====================")
    if n is None:
        print("在给定 max_n 范围内没有找到。")
    else:
        print(f"目标门最少需要 {n} 个基门。")
        print("对应多面体顶点 =", poly.vertices)

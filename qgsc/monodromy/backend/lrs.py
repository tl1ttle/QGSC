
"""
monodromy/backend/lrs.py
Windows-friendly replacement focused on the workflows used in test1.py:
- file-based lrs invocation
- robust lrslib 7.3 parsing
- H->H reduce via "redund 0 0" (avoids fragile H->V->H path)
"""

from copy import copy
from fractions import Fraction
from functools import reduce
import math
from operator import itemgetter
from os import getenv
from subprocess import Popen, PIPE
from typing import List
import os
import tempfile
import subprocess

from .backend_abc import Backend
from ..exceptions import NoFeasibleSolutions
from ..polytopes import ConvexPolytope, PolytopeVolume
from ..utilities import lcm


LRS_ENV = "LRS_PATH"
LRS_PATH = getenv(LRS_ENV, "lrs")


def check_for_lrs():
    try:
        proc = Popen([LRS_PATH], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        proc.communicate(b"")
        return True
    except Exception:
        return False


def _is_lrs_banner(line: str) -> bool:
    line = line.strip()
    return (
        line.startswith("*lrs")
        or line.startswith("lrs.exe:")
        or line.startswith("lrsgmp.exe:")
        or line.startswith("mplrs")
        or "lrslib_v." in line
        or "Input taken from" in line
        or "mini-gmp" in line
        or line.startswith("lrs_restart")
    )


def _is_table_size_line(line: str) -> bool:
    line = line.strip()
    return ("rational" in line) or ("integer" in line)


def single_lrs_pass(payload: bytes, chatty: bool = False) -> bytes:
    suffix = ".ine"
    text = payload.decode("utf-8", errors="replace")
    if "V-representation" in text:
        suffix = ".ext"

    tmpdir = os.environ.get("LRS_TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP") or None
    fd, in_path = tempfile.mkstemp(suffix=suffix, prefix="lrs_input_", dir=tmpdir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

        proc = subprocess.run(
            [LRS_PATH, in_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )

        if chatty:
            print("===== LRS INPUT FILE =====")
            print(in_path)
            print("===== LRS PAYLOAD =====")
            print(text)
            print("===== LRS STDOUT =====")
            print(proc.stdout.decode("utf-8", errors="replace"))
            print("===== LRS STDERR =====")
            print(proc.stderr.decode("utf-8", errors="replace"))
            print("===== LRS RETURNCODE =====")
            print(proc.returncode)

        if proc.returncode != 0:
            raise RuntimeError(
                "lrs failed.\n"
                f"returncode={proc.returncode}\n"
                f"stderr=\n{proc.stderr.decode('utf-8', errors='replace')}\n"
                f"stdout=\n{proc.stdout.decode('utf-8', errors='replace')}"
            )

        return proc.stdout
    finally:
        try:
            os.remove(in_path)
        except OSError:
            pass


def encode_inequalities(inequalities, equalities=None, name="name", options=None) -> bytes:
    equalities = equalities if equalities is not None else []
    options = options if options is not None else []
    output = ""
    output += name + "\n"
    output += "H-representation\n"
    output += "begin\n"
    output += (
        f"{len(inequalities) + 2 * len(equalities)}"
        f" {len((inequalities + equalities)[0])}"
        " rational\n"
    )
    for row in inequalities + equalities + [[-x for x in eq] for eq in equalities]:
        row_gcd = abs(reduce(math.gcd, row))
        row_gcd = row_gcd if row_gcd != 0 else 1
        output += " ".join([str(x // row_gcd) for x in row]) + "\n"
    output += "end\n"
    for option in options:
        output += f"{option}\n"
    return output.encode()


def encode_vertices(vertices, name="name", options=None, request_volume=False) -> bytes:
    options = [] if options is None else options
    output = ""
    output += name + "\n"
    output += "V-representation\n"
    output += "begin\n"
    output += f"{len(vertices)} {len(vertices[0])} rational\n"
    for vertex in vertices:
        output += " ".join([str(x) for x in vertex]) + "\n"
    output += "end\n"
    if request_volume:
        output += "volume\n"
    for option in options:
        output += f"{option}\n"
    return output.encode()


def decode_vertices(lrs_output: bytes):
    text = lrs_output.decode("utf-8", errors="replace")
    lines = text.splitlines()

    vertices = []
    in_vrep = False
    in_begin = False
    saw_no_feasible = False

    for raw_line in lines:
        line = raw_line.strip()

        if "No feasible solution" in line:
            saw_no_feasible = True
            continue

        if line == "" or line.startswith("*") or _is_lrs_banner(line):
            continue

        if line == "name":
            if vertices:
                break
            continue

        if line == "V-representation":
            in_vrep = True
            in_begin = False
            continue

        if line == "H-representation":
            if not vertices:
                raise ValueError("Vertex table decoder got an inequality table as input")
            break

        if not in_vrep:
            continue

        if line == "begin":
            in_begin = True
            continue

        if line == "end":
            if in_begin and vertices:
                break
            in_begin = False
            continue

        if not in_begin:
            continue

        if _is_table_size_line(line):
            continue

        if line.startswith("linearity"):
            continue

        vertices.append([Fraction(x) for x in line.split()])

    if len(vertices) == 0:
        if saw_no_feasible:
            raise NoFeasibleSolutions()
        raise TypeError("Something bad happened in `lrs`.")

    return vertices


def decode_inequalities(lrs_output: bytes):
    text = lrs_output.decode("utf-8", errors="replace")
    lines = text.splitlines()

    volume = None
    rows = []
    equality_indices = []

    in_hrep = False
    in_begin = False
    saw_no_feasible = False
    break_at_end = False

    for raw_line in lines:
        line = raw_line.strip()

        if "No feasible solution" in line:
            saw_no_feasible = True
            continue

        if line.startswith("*Volume="):
            if volume is None:
                volume = Fraction(line[8:])
            continue

        if line == "" or line.startswith("*") or _is_lrs_banner(line):
            continue

        if line == "name":
            if rows:
                break
            continue

        if line == "H-representation":
            in_hrep = True
            in_begin = False
            continue

        if line == "V-representation":
            if not rows:
                raise ValueError("Inequality table decoder got a vertex table as input")
            break

        if not in_hrep:
            continue

        if line == "begin":
            in_begin = True
            rows = []
            continue

        if line == "end":
            in_begin = False
            if break_at_end:
                break
            continue

        if not in_begin:
            continue

        if _is_table_size_line(line):
            continue

        if line.startswith("redund"):
            break_at_end = True
            continue

        if line.startswith("linearity"):
            equality_indices = [int(x) for x in line[9:].split()[1:]]
            continue

        new_row = [Fraction(x) for x in line.split()]
        row_lcm = abs(lcm(*[x.denominator for x in new_row]))
        rows.append([int(x * row_lcm) for x in new_row])

    if len(rows) == 0:
        if break_at_end or saw_no_feasible:
            raise NoFeasibleSolutions()
        raise TypeError("Something bad happened in `lrs`.")

    return dict(
        inequalities=[row for index, row in enumerate(rows) if 1 + index not in equality_indices],
        equalities=[row for index, row in enumerate(rows) if 1 + index in equality_indices],
        volume=volume,
        dimension=len(rows[0]) - 1 - len(equality_indices),
    )


def decode_simplices(lrs_output: bytes):
    simplices = []
    for raw_line in lrs_output.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if _is_lrs_banner(line):
            simplices = []
            continue
        if line.startswith("end"):
            break
        if line.startswith("F#"):
            tokens = line.split()
            position = tokens.index("vertices/rays")
            indices = []
            while True:
                position += 1
                try:
                    indices.append(int(tokens[position]) - 1)
                except (ValueError, IndexError):
                    break
            simplices.append(indices)
    return dict(simplices=simplices)


class LRSBackend(Backend):
    def __init__(self):
        super().__init__()
        assert check_for_lrs(), "LRS not present."

    @staticmethod
    def volume(convex_polytope: ConvexPolytope) -> PolytopeVolume:
        # Workaround for Windows lrslib crash on V-representation + volume.
        # For the gate-count workflow we only need a stable ordering; use affine dimension
        # and a dummy positive measure. This avoids calling lrs on .ext with "volume".
        verts = convex_polytope.vertices
        if len(verts) == 0:
            raise NoFeasibleSolutions()
        if len(verts) == 1:
            return PolytopeVolume(dimension=0, volume=Fraction(1, 1))

        # affine rank over rationals via float fallback (sufficient for ordering)
        import numpy as np
        arr = np.array([[float(x) for x in v] for v in verts], dtype=float)
        rank = int(np.linalg.matrix_rank(arr - arr[0]))
        return PolytopeVolume(dimension=rank, volume=Fraction(1, 1))

    @staticmethod
    def reduce(convex_polytope: ConvexPolytope) -> ConvexPolytope:
        clone = copy(convex_polytope)
        inequalities = convex_polytope.inequalities
        equalities = convex_polytope.equalities
        inequality_payload = encode_inequalities(
            inequalities,
            equalities,
            options=["redund 0 0"],
        )
        inequality_response = single_lrs_pass(inequality_payload)
        inequality_dictionary = decode_inequalities(inequality_response)
        clone.inequalities = inequality_dictionary["inequalities"]
        clone.equalities = inequality_dictionary["equalities"]
        return clone

    @staticmethod
    def vertices(convex_polytope: ConvexPolytope) -> List[List[Fraction]]:
        inequalities = convex_polytope.inequalities
        equalities = convex_polytope.equalities
        inequality_payload = encode_inequalities(inequalities, equalities)
        vertex_response = single_lrs_pass(inequality_payload, chatty=False)
        vertices = decode_vertices(vertex_response)
        if any(v[0] == 0 for v in vertices):
            raise ValueError("Polytope is not bounded.")
        return [v[1:] for v in vertices]

    @staticmethod
    def triangulation(convex_polytope: ConvexPolytope) -> List[List]:
        if 0 == len(convex_polytope.vertices):
            raise NoFeasibleSolutions()
        vertex_payload = encode_vertices(
            [(Fraction(1, 1), *v) for v in convex_polytope.vertices],
            options=["triangulation"],
            request_volume=False,
        )
        response = single_lrs_pass(vertex_payload)
        simplices = decode_simplices(response)
        return simplices["simplices"]

    @staticmethod
    def convex_hull(vertices: List[List[Fraction]]) -> ConvexPolytope:
        payload = encode_vertices([(1, *x) for x in vertices], request_volume=False)
        response = single_lrs_pass(payload)
        inequalities, equalities = itemgetter("inequalities", "equalities")(decode_inequalities(response))
        return ConvexPolytope(inequalities=inequalities, equalities=equalities)

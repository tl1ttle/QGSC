
















from __future__ import annotations


import sys as _ising_sys
from pathlib import Path as _IsingPath
_ISING_ROOT = _IsingPath(__file__).resolve().parents[1]
for _ising_path in (str(_ISING_ROOT), str(_ISING_ROOT / "core")):
    if _ising_path not in _ising_sys.path:
        _ising_sys.path.insert(0, _ising_path)


import math
import time
import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import random_unitary, Operator, process_fidelity as qk_process_fidelity

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False





def _to_mat(U) -> np.ndarray:
    
    if hasattr(U, "data"):
        U = U.data
    return np.asarray(U, dtype=complex)


def tensor_qiskit(*ops_q0_to_qn1):
    
    out = _to_mat(ops_q0_to_qn1[0])
    for op in ops_q0_to_qn1[1:]:
        out = np.kron(_to_mat(op), out)
    return out



def align_global_phase(U_ref: np.ndarray, U: np.ndarray) -> np.ndarray:
    
    U_ref = _to_mat(U_ref)
    U = _to_mat(U)
    tr = np.trace(U_ref.conj().T @ U)
    if abs(tr) < 1e-14:
        return U
    return np.exp(-1j * np.angle(tr)) * U





I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

I4 = tensor_qiskit(I2, I2)
XX = tensor_qiskit(X2, X2)
YY = tensor_qiskit(Y2, Y2)
ZZ = tensor_qiskit(Z2, Z2)


def _exp_i_tP(t: float, P: np.ndarray) -> np.ndarray:
    
    t = float(t)
    return np.cos(t) * I4 + 1j * np.sin(t) * _to_mat(P)


def cartan_unitary(eta_x: float, eta_y: float, eta_z: float) -> np.ndarray:
    
    return _exp_i_tP(eta_x, XX) @ _exp_i_tP(eta_y, YY) @ _exp_i_tP(eta_z, ZZ)


def dx_unitary(theta_x: float, theta_y: float, theta_z: float) -> np.ndarray:
    
    return cartan_unitary(theta_x, theta_y, theta_z)





def u3_unitary(theta: float, phi: float, lam: float) -> np.ndarray:
    
    th = float(theta)
    ph = float(phi)
    la = float(lam)
    c = math.cos(th / 2.0)
    s = math.sin(th / 2.0)
    return np.array(
        [
            [c, -np.exp(1j * la) * s],
            [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c],
        ],
        dtype=complex,
    )


def local_layer_u3(params6: np.ndarray) -> np.ndarray:
    
    t0, p0, l0, t1, p1, l1 = [float(x) for x in np.asarray(params6, dtype=float).tolist()]
    U0 = u3_unitary(t0, p0, l0)
    U1 = u3_unitary(t1, p1, l1)
    return tensor_qiskit(U0, U1)





def process_fidelity(Ud: np.ndarray, Ut: np.ndarray) -> float:
    
    Ud = _to_mat(Ud)
    Ut = _to_mat(Ut)
    return float(qk_process_fidelity(Operator(Ud), Operator(Ut)))


def trace_fidelity(Ud: np.ndarray, Ut: np.ndarray) -> float:
    
    pf = process_fidelity(Ud, Ut)
    return float(math.sqrt(max(0.0, pf)))


def avg_gate_fidelity_from_pf(pf: float, d: int) -> float:
    
    return float((d * float(pf) + 1.0) / (d + 1.0))


def process_fidelity_unitary(Ut: np.ndarray, Ud: np.ndarray) -> float:
    
    return process_fidelity(Ud, Ut)


def avg_gate_fidelity_from_unitary(Ut: np.ndarray, Ud: np.ndarray) -> float:
    Ut = _to_mat(Ut)
    return avg_gate_fidelity_from_pf(process_fidelity(Ud, Ut), Ut.shape[0])


def max_abs_err_up_to_phase(Ut: np.ndarray, Ud: np.ndarray) -> float:
    Ut = _to_mat(Ut)
    Ud = align_global_phase(Ut, Ud)
    return float(np.max(np.abs(Ut - Ud)))





def compose_template_unitary(
    params: np.ndarray,
    basis_list: Sequence[np.ndarray],
) -> np.ndarray:
    
    k = len(basis_list)
    params = np.asarray(params, dtype=float).reshape(-1)
    assert params.shape[0] == 6 * (k + 1)

    def layer(i: int) -> np.ndarray:
        return local_layer_u3(params[6 * i : 6 * i + 6])

    U = layer(0)
    for i in range(k):
        U = _to_mat(basis_list[i]) @ U
        U = layer(i + 1) @ U
    return U


def _enumerate_basis_patterns(U_basis: np.ndarray, k: int, allow_dagger: bool) -> List[List[np.ndarray]]:
    
    U_basis = _to_mat(U_basis)
    if not allow_dagger:
        return [[U_basis for _ in range(k)]]
    U_dag = U_basis.conj().T
    out: List[List[np.ndarray]] = []
    for bits in itertools.product([0, 1], repeat=k):
        out.append([U_dag if b else U_basis for b in bits])
    return out





@dataclass
class DecompResult:
    best_layers: int
    best_pf: float           
    best_trace_f: float      
    best_params: np.ndarray
    best_pattern_bits: Tuple[int, ...]
    elapsed_s: float





def decompose_2q_by_pdbls_lbfgs(
    U_target: np.ndarray,
    U_basis: np.ndarray,
    *,
    max_layers: int = 6,
    Fd: float = 1.0 - 1e-12,            
    allow_dagger: bool = True,          
    n_restarts: int = 8,                
    maxiter: int = 3000,                
    bounds_angle: float = 2 * math.pi,  
    hardware_fidelity: Optional[float] = None,  
    debug: bool = True,
    
    return_result: bool = False,        
    label_basis: str = "GB",
    qargs_mode: str = "auto",          
) -> Union["DecompResult", "QuantumCircuit"]:
    
    t0 = time.time()
    if not _HAVE_SCIPY:
        raise RuntimeError("未检测到 scipy，无法使用 scipy 的 L-BFGS-B。请安装 scipy。")

    U_target = _to_mat(U_target)
    U_basis = _to_mat(U_basis)
    assert U_target.shape == (4, 4) and U_basis.shape == (4, 4)

    
    Ut_op = Operator(U_target)

    def pf_qiskit(Ud_mat: np.ndarray) -> float:
        return float(qk_process_fidelity(Operator(_to_mat(Ud_mat)), Ut_op))

    best_global_pf_obj = 0.0
    best_global = None

    def make_bounds(n: int, epsB: float = 1e-8):
        
        B = float(bounds_angle)
        if (not np.isfinite(B)) or (B <= 0.0):
            B = 2.0 * math.pi
        lo = -B + epsB
        hi =  B - epsB
        if hi <= lo:
            lo, hi = -1.0, 1.0
        return [(lo, hi)] * n

    def _clip_x_to_bounds(x: np.ndarray, bounds, eps: float = 1e-8) -> np.ndarray:
        
        x = np.asarray(x, dtype=float).reshape(-1).copy()
        for i, (lb, ub) in enumerate(bounds):
            xi = x[i]
            lb = None if lb is None else float(lb)
            ub = None if ub is None else float(ub)

            if not np.isfinite(xi):
                
                if lb is None and ub is None:
                    xi = 0.0
                elif lb is None:
                    xi = ub - 1.0
                elif ub is None:
                    xi = lb + 1.0
                else:
                    xi = 0.5 * (lb + ub)

            
            if lb is not None and xi <= lb + eps:
                xi = lb + eps
            if ub is not None and xi >= ub - eps:
                xi = ub - eps

            
            if lb is not None and ub is not None and lb == ub:
                xi = lb
            x[i] = xi
        return x

    for k in range(1, int(max_layers) + 1):
        patterns = _enumerate_basis_patterns(U_basis, k, allow_dagger=allow_dagger)

        best_k_pf_obj = 0.0
        best_k_trace = 0.0
        best_k_params = None
        best_k_bits = None

        if debug:
            print(f"[PDBLS-LBFGS] layers={k}, patterns={len(patterns)}, restarts={n_restarts}")

        for patt_idx, basis_list in enumerate(patterns):
            
            if allow_dagger:
                U_dag = U_basis.conj().T
                bits = tuple(1 if np.allclose(b, U_dag) else 0 for b in basis_list)
            else:
                bits = tuple(0 for _ in range(k))

            n_params = 6 * (k + 1)
            bounds = make_bounds(n_params)

            def _loss_raw(x: np.ndarray) -> float:
                Ud = compose_template_unitary(np.asarray(x, dtype=float), basis_list)
                pf = pf_qiskit(Ud)
                if hardware_fidelity is not None:
                    pf = (float(hardware_fidelity) ** k) * pf
                return float(1.0 - pf)


            def loss(x: np.ndarray) -> float:
                
                
                x = _clip_x_to_bounds(np.asarray(x, dtype=float), bounds)
                return _loss_raw(x)

            
            x0_list = [np.zeros(n_params, dtype=float)]
            lb0, ub0 = float(bounds[0][0]), float(bounds[0][1])
            for _ in range(max(0, int(n_restarts) - 1)):
                x0_list.append(np.random.uniform(lb0, ub0, size=n_params).astype(float))

            for ridx, x0 in enumerate(x0_list):
                x0 = _clip_x_to_bounds(x0, bounds)
                opt = minimize(
                    loss,
                    x0=x0,
                    method="L-BFGS-B",
                    options=dict(maxiter=int(maxiter), ftol=1e-15),
                )
                x_best = _clip_x_to_bounds(np.asarray(opt.x, dtype=float), bounds)

                Ud = compose_template_unitary(x_best, basis_list)
                pf_decomp = pf_qiskit(Ud)
                tf = float(math.sqrt(max(0.0, pf_decomp)))

                pf_obj = pf_decomp
                if hardware_fidelity is not None:
                    pf_obj = (float(hardware_fidelity) ** k) * pf_decomp

                if pf_obj > best_k_pf_obj:
                    best_k_pf_obj = float(pf_obj)
                    best_k_trace = float(tf)
                    best_k_params = x_best.copy()
                    best_k_bits = bits

                if debug and (ridx == 0):
                    print(f"  patt#{patt_idx:03d} bits={bits}  pf={pf_obj:.12f}  iters={opt.nit}")

                
                if pf_decomp >= Fd:
                    break

            if best_k_params is not None and (best_k_trace * best_k_trace) >= Fd:
                break

        if debug:
            print(
                f"[PDBLS-LBFGS] best@k={k}: PF_obj={best_k_pf_obj:.12f} (traceF={best_k_trace:.12f}) bits={best_k_bits}"
            )

        if best_k_pf_obj > best_global_pf_obj and best_k_params is not None:
            best_global_pf_obj = best_k_pf_obj
            best_global = (k, best_k_pf_obj, best_k_trace, best_k_params, best_k_bits)

        if best_k_params is not None and (best_k_trace * best_k_trace) >= Fd:
            break

    elapsed = time.time() - t0
    if best_global is None:
        res = DecompResult(0, 0.0, 0.0, np.zeros(0), tuple(), float(elapsed))
        if return_result:
            return res
        from qiskit import QuantumCircuit
        qc0 = QuantumCircuit(2)
        qc0.metadata = {
            "layers": 0,
            "bits": tuple(),
            "elapsed_s": float(elapsed),
            "pf_obj": 0.0,
            "pf_decomp": 0.0,
            "traceF": 0.0,
        }
        return qc0

    k, pf_obj, tf, params, bits = best_global
    res = DecompResult(int(k), float(pf_obj), float(tf), np.asarray(params, dtype=float), tuple(bits), float(elapsed))

    if return_result:
        return res

    
    qc = build_qiskit_circuit_from_result(res, U_basis, label_basis=label_basis, qargs_mode=qargs_mode)

    
    pf_decomp = float(res.best_trace_f * res.best_trace_f)

    if getattr(qc, "metadata", None) is None:
        qc.metadata = {}
    qc.metadata.update(
        {
            "layers": int(res.best_layers),
            "bits": tuple(res.best_pattern_bits),
            "elapsed_s": float(res.elapsed_s),
            "pf_obj": float(res.best_pf),
            "pf_decomp": float(pf_decomp),
            "traceF": float(res.best_trace_f),
        }
    )
    return qc





def build_qiskit_circuit_from_result(
    res: DecompResult,
    U_basis: np.ndarray,
    *,
    label_basis: str = "GB",
    qargs_mode: str = "auto",  
):
    
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    k = int(res.best_layers)
    params = np.asarray(res.best_params, dtype=float).reshape(-1)
    assert params.shape[0] == 6 * (k + 1)

    U_basis = _to_mat(U_basis)
    U_dag = U_basis.conj().T

    basis_bits = res.best_pattern_bits
    if len(basis_bits) != k:
        basis_bits = tuple(0 for _ in range(k))

    
    basis_list = [(U_dag if b == 1 else U_basis) for b in basis_bits]
    Ud_internal = compose_template_unitary(params, basis_list)

    def _build(order: str) -> QuantumCircuit:
        qc = QuantumCircuit(2)

        def apply_layer(i: int):
            t0, p0, l0 = [float(x) for x in params[6 * i : 6 * i + 3]]
            t1, p1, l1 = [float(x) for x in params[6 * i + 3 : 6 * i + 6]]
            qc.u(t0, p0, l0, 0)  
            qc.u(t1, p1, l1, 1)  

        apply_layer(0)
        for i in range(k):
            U = U_dag if basis_bits[i] == 1 else U_basis
            lab = label_basis + ("^dag" if basis_bits[i] == 1 else "")

            if order == "01":
                qc.append(UnitaryGate(U, label=lab), [0, 1])
            elif order == "10":
                qc.append(UnitaryGate(U, label=lab), [1, 0])
            else:
                raise ValueError(f"unknown order={order}")

            apply_layer(i + 1)
        return qc

    if qargs_mode in ("01", "10"):
        return _build(qargs_mode)

    
    qc01 = _build("01")
    qc10 = _build("10")

    U01 = _to_mat(Operator(qc01).data)
    U10 = _to_mat(Operator(qc10).data)

    
    err01 = float(np.max(np.abs(Ud_internal - align_global_phase(Ud_internal, U01))))
    err10 = float(np.max(np.abs(Ud_internal - align_global_phase(Ud_internal, U10))))

    return qc01 if err01 <= err10 else qc10


import numpy as np
from scipy.linalg import expm

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)

def canonical_gate(x: float, y: float, z: float) -> np.ndarray:
    
    H = x * XX + y * YY + z * ZZ
    return expm(-1j * H)




if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    
    a=math.pi/4,
    b=math.pi/32,
    c=0
    U_target =canonical_gate(a, b, c)
    

    x, y, z = math.pi/16,math.pi/16,0

    U_basis =canonical_gate(x, y, z)


    qc = decompose_2q_by_pdbls_lbfgs(
        U_target=U_target,
        U_basis=U_basis,
        max_layers=6,
        Fd=1.0 - 1e-6,
        allow_dagger=False,
        n_restarts=6,
        maxiter=5000,
        debug=True,
        return_result=False,
        label_basis="GB",
        qargs_mode="auto",
    )

    print("\n=== QC ===")
    print("depth:", qc.depth(), "size:", qc.size())
    print("metadata:", qc.metadata)
    print(qc)


    Ud_qiskit = _to_mat(Operator(qc).data)
    pf_q = process_fidelity_unitary(U_target, Ud_qiskit)
    agf = avg_gate_fidelity_from_unitary(U_target, Ud_qiskit)
    err = max_abs_err_up_to_phase(U_target, Ud_qiskit)

    print(f"\nPF(qiskit)={pf_q:.12f}")
    print(f"AGF       ={agf:.12f}")
    print(f"max|Δ|    ={err:.3e}")
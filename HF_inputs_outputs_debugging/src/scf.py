# scf.py
import numpy as np
from integrals import (
    build_Hcore,
    build_S_inv_sqrt,
    compute_P,
    build_J,
    build_K,
    build_Fock,
)

def scf_iteration(P, Hcore, eri, S_inv_sqrt, nelec):
    J = build_J(P, eri)
    K = build_K(P, eri)
    F = build_Fock(Hcore, J, K)
    Fprime = S_inv_sqrt.T @ F @ S_inv_sqrt
    eps, Cprime = np.linalg.eigh(Fprime)
    C = S_inv_sqrt @ Cprime
    P_new = compute_P(C, nelec)
    return F, C, P_new, eps

def run_scf(S, T, V, eri, nelec, max_iter=50, tol=1e-6):
    nbasis = S.shape[0]

    # Matrices base
    Hcore = build_Hcore(T, V)
    S_inv_sqrt = build_S_inv_sqrt(S)

    # Densidad inicial
    P = np.zeros((nbasis, nbasis))

    for it in range(max_iter):
        F, C, P_new, eps = scf_iteration(P, Hcore, eri, S_inv_sqrt, nelec)

        deltaP = np.linalg.norm(P_new - P)
        print(f"Iter {it+1:2d}: ||ΔP|| = {deltaP:.6e}")

        if deltaP < tol:
            print("SCF converged.")
            return F, C, P_new, eps, Hcore

        P = P_new

    raise RuntimeError("SCF did not converge within max_iter")

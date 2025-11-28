# integrals.py

import numpy as np


def build_Hcore(T, V):
    """
    Construct the core Hamiltonian matrix:
        H_core = T + V
    where:
        T = kinetic energy matrix
        V = nuclear attraction matrix
    """
    return T + V

def build_S_inv_sqrt(S):
    """
    Compute the inverse square root of the overlap matrix S:
        S^{-1/2} = U * diag(1/sqrt(s_i)) * U^T
    using the symmetric eigenvalue decomposition S = U s U^T.
    """
    evals, evecs = np.linalg.eigh(S)

    # build the diagonal matrix s^{-1/2}
    inv_sqrt_evals = np.diag(1.0 / np.sqrt(evals))

    # reconstruct S^{-1/2}
    S_inv_sqrt = evecs @ inv_sqrt_evals @ evecs.T

    return S_inv_sqrt

def compute_P(C, nelec):
    nbasis = C.shape[0]
    nocc = nelec // 2

    P = np.zeros((nbasis, nbasis))

    for mu in range(nbasis):
        for nu in range(nbasis):
            total = 0.0
            for i in range(nocc):
                total += C[mu, i] * C[nu, i]
            P[mu, nu] = 2.0 * total

    return P

def build_J(P, eri):
    """
    Build the Coulomb matrix J from the density matrix P and
    the two-electron integrals eri:

        J_{μν} = sum_{λσ} P_{λσ} (μν | λσ)

    P  : (nbasis x nbasis) density matrix
    eri: (nbasis x nbasis x nbasis x nbasis) tensor of two-electron integrals
    """
    nbasis = P.shape[0]          # number of basis functions
    J = np.zeros((nbasis, nbasis))

    for mu in range(nbasis):
        for nu in range(nbasis):
            total = 0.0
            for lam in range(nbasis):
                for sig in range(nbasis):
                    total += P[lam, sig] * eri[mu, nu, lam, sig]
            J[mu, nu] = total

    return J

def build_K(P, eri):
    nbasis = P.shape[0]
    K = np.zeros((nbasis, nbasis))
    for mu in range(nbasis):
        for nu in range(nbasis):
            total = 0.0
            for lam in range(nbasis):
                for sig in range(nbasis):
                    total += P[lam, sig] * eri[mu, lam, nu, sig]
            K[mu, nu] = total
    return K

def build_Fock(Hcore, J, K):
    """
    F = H_core + J - 1/2 K
    """
    return Hcore + J - 0.5 * K


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

ANGSTROM_TO_BOHR = 1.8897259886


def nuclear_repulsion_energy(atoms, coords_in_angstrom=True):
    """
    E_nuc = sum_{A<B} Z_A Z_B / R_AB
    """
    E = 0.0
    n = len(atoms)

    for A in range(n):
        ZA = atoms[A].Z
        for B in range(A + 1, n):
            ZB = atoms[B].Z

            dx = atoms[A].x - atoms[B].x
            dy = atoms[A].y - atoms[B].y
            dz = atoms[A].z - atoms[B].z

            if coords_in_angstrom:
                dx *= ANGSTROM_TO_BOHR
                dy *= ANGSTROM_TO_BOHR
                dz *= ANGSTROM_TO_BOHR

            R = np.sqrt(dx * dx + dy * dy + dz * dz)
            E += ZA * ZB / R

    return E


def electronic_energy(P, Hcore, F):
    """
    RHF electronic energy:
        E_elec = 1/2 * sum_{μν} P_{μν} (H_{μν} + F_{μν})
    """
    return 0.5 * np.sum(P * (Hcore + F))


def scf_step(P, Hcore, eri, S_inv_sqrt, nelec):
    """
    One SCF update:
      P -> (J,K) -> F -> diagonalize -> C -> P_new
    """
    J = build_J(P, eri)
    K = build_K(P, eri)
    F = build_Fock(Hcore, J, K)

    # Orthonormalize: F' = X^T F X, with X = S^{-1/2}
    Fp = S_inv_sqrt.T @ F @ S_inv_sqrt
    eps, Cp = np.linalg.eigh(Fp)

    # Back-transform MO coeffs
    C = S_inv_sqrt @ Cp

    # New density
    P_new = compute_P(C, nelec)

    return F, C, eps, P_new


def run_scf(
    mol,
    S,
    T,
    V,
    eri,
    max_iter=50,
    tol=1e-6,
    coords_in_angstrom=True,
    verbose=True,
):
    """
    Minimal RHF SCF driver.

    Inputs:
      mol: molecule object with mol.atoms and mol.charge
      S,T,V,eri: integrals in AO basis
      max_iter: maximum SCF cycles
      tol: convergence threshold for max(|P_new - P_old|)
      coords_in_angstrom: for nuclear repulsion energy only
      verbose: print iteration table

    Returns:
      results dict with keys:
        P, F, C, eps, E_elec, E_tot, E_nuc, niter, converged
    """
    # electron count
    charge = mol.charge
    nelec = sum(a.Z for a in mol.atoms) - charge
    if nelec % 2 != 0:
        raise ValueError(f"RHF requires even nelec, got {nelec}")
    nocc = nelec // 2  # not used here directly, but good to know

    nbasis = S.shape[0]
    Hcore = build_Hcore(T, V)
    S_inv_sqrt = build_S_inv_sqrt(S)

    # energies
    E_nuc = nuclear_repulsion_energy(mol.atoms, coords_in_angstrom=coords_in_angstrom)

    # initial guess: P = 0 (simple and robust)
    P = np.zeros((nbasis, nbasis))

    if verbose:
        print(f"SCF start: charge={charge}, nelec={nelec}, nocc={nocc}, nbasis={nbasis}")
        print(f"E_nuc = {E_nuc:.10f}")
        print("iter  max|ΔP|          E_elec              E_tot")

    converged = False
    F = None
    C = None
    eps = None
    E_elec = None
    E_tot = None

    for it in range(1, max_iter + 1):
        F, C, eps, P_new = scf_step(P, Hcore, eri, S_inv_sqrt, nelec)

        # convergence metric: max element change in density
        deltaP = np.max(np.abs(P_new - P))

        # energies (use current density P_new and current F)
        E_elec = electronic_energy(P_new, Hcore, F)
        E_tot = E_elec + E_nuc

        if verbose:
            print(f"{it:4d}  {deltaP: .6e}  {E_elec: .12f}  {E_tot: .12f}")

        if deltaP < tol:
            converged = True
            P = P_new
            break

        P = P_new

    return {
        "P": P,
        "F": F,
        "C": C,
        "eps": eps,
        "E_elec": E_elec,
        "E_tot": E_tot,
        "E_nuc": E_nuc,
        "niter": it,
        "converged": converged,
    }


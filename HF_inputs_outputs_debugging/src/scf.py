# scf.py
import numpy as np
ANGSTROM_TO_BOHR = 1.8897261246257702

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

def nuclear_repulsion_energy(atoms):
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
    verbose=True,
):
    """
    Minimal RHF SCF driver.

    Inputs:
      mol: molecule object with mol.atoms and mol.charge
      S,T,V,eri: integrals in AO basis
      max_iter: maximum SCF cycles
      tol: convergence threshold for max(|P_new - P_old|)
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
    nocc = nelec // 2

    nbasis = S.shape[0]
    Hcore = build_Hcore(T, V)
    S_inv_sqrt = build_S_inv_sqrt(S)

    # energies
    E_nuc = nuclear_repulsion_energy(mol.atoms)

    # initial guess: core Hamiltonian diagonalization
    F_core = Hcore
    Fp_core = S_inv_sqrt.T @ F_core @ S_inv_sqrt
    eps_core, Cp_core = np.linalg.eigh(Fp_core)
    C_core = S_inv_sqrt @ Cp_core
    P = compute_P(C_core, nelec)

    if verbose:
        print(f"SCF start: charge={charge}, nelec={nelec}, nocc={nocc}, nbasis={nbasis}")
        print(f"E_nuc = {E_nuc:.10f}")
        print("iter  max|ΔP|          E_elec              E_tot")

    # Match legacy output: first line prints E_elec=0.0, E_tot=E_nuc,
    # then each subsequent line prints the energy from the previous cycle.
    E_elec_print = 0.0
    E_tot_print = E_nuc

    converged = False

    for it in range(1, max_iter + 1):
        F, C, eps, P_new = scf_step(P, Hcore, eri, S_inv_sqrt, nelec)

        # convergence metric: max element change in density
        deltaP = np.max(np.abs(P_new - P))

        # energies (use the current density with a matching Fock build)
        J_energy = build_J(P_new, eri)
        K_energy = build_K(P_new, eri)
        F_energy = build_Fock(Hcore, J_energy, K_energy)
        E_elec = electronic_energy(P_new, Hcore, F_energy)
        E_tot = E_elec + E_nuc

        if verbose:
            print(f"{it:4d}  {deltaP: .10f}  {E_elec_print: .10f}  {E_tot_print: .10f}")

        E_elec_print = E_elec
        E_tot_print = E_tot

        if deltaP < tol and it > 1:
            converged = True
            P = P_new
            break

        P = P_new

    # Recompute final F, C, eps, and energies using the converged density.
    # This avoids returning energies built from an older density matrix.
    F, C, eps, _ = scf_step(P, Hcore, eri, S_inv_sqrt, nelec)
    E_elec = electronic_energy(P, Hcore, F)
    E_tot = E_elec + E_nuc

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

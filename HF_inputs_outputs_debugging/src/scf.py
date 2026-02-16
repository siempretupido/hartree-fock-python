# scf.py
import numpy as np
ANGSTROM_TO_BOHR = 1.8897261246257702

def build_Hcore(T, V):
    """Build core Hamiltonian matrix Hcore = T + V."""
    return T + V


def build_S_inv_sqrt(S):
    """Compute inverse square root of overlap matrix S."""
    evals, evecs = np.linalg.eigh(S)

    # Build diag(s^-1/2).
    inv_sqrt_evals = np.diag(1.0 / np.sqrt(evals))

    # Reconstruct S^-1/2.
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
    """Build Coulomb matrix J from density matrix P."""
    nbasis = P.shape[0]          # Number of basis functions.
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
    """Build Fock matrix F = Hcore + J - 0.5*K."""
    return Hcore + J - 0.5 * K

def nuclear_repulsion_energy(atoms):
    """Compute nuclear repulsion energy."""
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
    """Compute RHF electronic energy."""
    return 0.5 * np.sum(P * (Hcore + F))


def scf_step(P, Hcore, eri, S_inv_sqrt, nelec):
    """Run one SCF update step."""
    J = build_J(P, eri)
    K = build_K(P, eri)
    F = build_Fock(Hcore, J, K)

    # Orthonormalize F.
    Fp = S_inv_sqrt.T @ F @ S_inv_sqrt
    eps, Cp = np.linalg.eigh(Fp)

    # Back-transform MO coefficients.
    C = S_inv_sqrt @ Cp

    # Build new density.
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
    """Run RHF SCF and return densities, orbitals and energies."""
    # Electron count.
    charge = mol.charge
    nelec = sum(a.Z for a in mol.atoms) - charge
    if nelec % 2 != 0:
        raise ValueError(f"RHF requires even nelec, got {nelec}")
    nocc = nelec // 2

    nbasis = S.shape[0]
    Hcore = build_Hcore(T, V)
    S_inv_sqrt = build_S_inv_sqrt(S)

    # Nuclear repulsion energy.
    E_nuc = nuclear_repulsion_energy(mol.atoms)

    # Initial guess from Hcore diagonalization.
    F_core = Hcore
    Fp_core = S_inv_sqrt.T @ F_core @ S_inv_sqrt
    eps_core, Cp_core = np.linalg.eigh(Fp_core)
    C_core = S_inv_sqrt @ Cp_core
    P = compute_P(C_core, nelec)

    if verbose:
        print(f"SCF start: charge={charge}, nelec={nelec}, nocc={nocc}, nbasis={nbasis}")
        print(f"E_nuc = {E_nuc:.10f}")
        print("iter  max|ΔP|          E_elec              E_tot")

    # Print format compatible with legacy output.
    E_elec_print = 0.0
    E_tot_print = E_nuc

    converged = False

    for it in range(1, max_iter + 1):
        F, C, eps, P_new = scf_step(P, Hcore, eri, S_inv_sqrt, nelec)

        # Convergence metric.
        deltaP = np.max(np.abs(P_new - P))

        # Energies for current density.
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

    # Recompute final quantities with converged density.
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

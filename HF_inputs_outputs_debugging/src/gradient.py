import numpy as np

from scf import ANGSTROM_TO_BOHR

def build_Q(C, eps, nelec):
    """
    Construct the energy-weighted density (aka Pulay matrix)
    W_{μν} = 2 * sum_i ε_i C_{μi} C_{νi}, over occupied orbitals.
    """
    if C is None or eps is None:
        raise ValueError("SCF coefficients/energies are required to build W.")

    nocc = nelec // 2
    C_mui = C[:, :nocc]
    eps_i = eps[:nocc]

    return 2.0 * C_mui @ np.diag(eps_i) @ C_mui.T


def _contract_density_deriv(dX, density):
    """
    Contract derivative tensor dX[A, mu, nu, 3] with a density matrix using loops.
    """
    natoms = dX.shape[0]
    nbasis = density.shape[0]
    grad = np.zeros((natoms, 3), dtype=float)

    for A in range(natoms):
        for mu in range(nbasis):
            for nu in range(nbasis):
                grad[A] += density[mu, nu] * dX[A, mu, nu]

    return grad


def nuclear_repulsion_gradient(atoms):
    """
    Gradient of the classical nuclear repulsion energy.

    Returns:
      grad[A, :] = ∂E_nuc / ∂R_A  (Hartree / Bohr)
    """
    natoms = len(atoms)
    grad = np.zeros((natoms, 3), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    charges = np.array([atom.Z for atom in atoms], dtype=float)

    for A in range(natoms):
        RA = coords[A]
        ZA = charges[A]
        for B in range(natoms):
            if A == B:
                continue
            RB = coords[B]
            ZB = charges[B]

            R = RA - RB
            dist = np.linalg.norm(R)
            grad[A] += -ZA * ZB * R / (dist ** 3)

    return grad


def compute_gradient(
    mol,
    scf_results,
    dS,
    dT,
    dV,
    dERI,
):
    """
    Compute the RHF energy gradient broken down by contribution.

    Returns a dict with keys:
      overlap, kinetic, nuclear_attraction, two_electron, nuclear_repulsion, total
    Each value is an array (natoms, 3) in Hartree/Bohr.
    """
    P = scf_results.get("P")
    C = scf_results.get("C")
    eps = scf_results.get("eps")

    nelec = sum(atom.Z for atom in mol.atoms) - mol.charge
    Q = build_Q(C, eps, nelec)

    grad_overlap = -_contract_density_deriv(dS, Q)
    grad_kinetic = _contract_density_deriv(dT, P)
    grad_nuclear_attr = _contract_density_deriv(dV, P)

    natoms = dERI.shape[0]
    nbasis = P.shape[0]
    grad_two_electron = np.zeros((natoms, 3), dtype=float)

    for A in range(natoms):
        for mu in range(nbasis):
            for nu in range(nbasis):
                for lam in range(nbasis):
                    for sig in range(nbasis):
                        pref = P[mu, nu] * P[lam, sig]
                        val1 = dERI[A, mu, nu, lam, sig]
                        val2 = dERI[A, mu, lam, nu, sig]
                        grad_two_electron[A, 0] += 0.5 * pref * val1[0]
                        grad_two_electron[A, 1] += 0.5 * pref * val1[1]
                        grad_two_electron[A, 2] += 0.5 * pref * val1[2]
                        grad_two_electron[A, 0] -= 0.25 * pref * val2[0]
                        grad_two_electron[A, 1] -= 0.25 * pref * val2[1]
                        grad_two_electron[A, 2] -= 0.25 * pref * val2[2]

    grad_nuclear_rep = nuclear_repulsion_gradient(mol.atoms)

    total = grad_overlap + grad_kinetic + grad_nuclear_attr + grad_two_electron + grad_nuclear_rep

    return {
        "overlap": grad_overlap,
        "kinetic": grad_kinetic,
        "nuclear_attraction": grad_nuclear_attr,
        "two_electron": grad_two_electron,
        "nuclear_repulsion": grad_nuclear_rep,
        "total": total,
    }

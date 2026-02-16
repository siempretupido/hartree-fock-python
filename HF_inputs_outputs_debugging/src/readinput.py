# readinput.py

import numpy as np


class Atom:
    def __init__(self, label, Z, x, y, z):
        self.label = label
        self.Z = Z
        self.x = x
        self.y = y
        self.z = z


class Molecule:
    def __init__(self, atoms, charge, nbasis, max_nc):
        # List of Atom objects.
        self.atoms = atoms
        self.charge = charge
        self.nbasis = nbasis
        self.max_nc = max_nc


def _find_index(lines, pattern):
    """Return index of first line starting with pattern (case-insensitive)."""
    pattern_lower = pattern.lower()
    for i, line in enumerate(lines):
        if line.lower().startswith(pattern_lower):
            return i
    raise ValueError("Could not find line starting with: " + pattern)


def read_basic_input(path):
    """Read molecule data (atoms, charge, nbasis, max_nc) from input file."""
    # Read non-empty lines.
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Number of atoms.
    idx_na = _find_index(lines, "number of atoms")
    natoms = int(lines[idx_na + 1])

    # Atoms block.
    idx_atoms_header = _find_index(lines, "Atom labels")
    first_atom_line = idx_atoms_header + 1

    atoms = []
    for j in range(natoms):
        parts = lines[first_atom_line + j].split()
        label = parts[0]
        Z = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        atom = Atom(label, Z, x, y, z)
        atoms.append(atom)

    # Total charge.
    idx_charge = _find_index(lines, "Overall charge")
    charge = int(lines[idx_charge + 1])

    # Number of basis functions.
    idx_nb = _find_index(lines, "Number of basis funcs")
    nbasis = int(lines[idx_nb + 1])

    # Maximum number of primitives.
    idx_maxnc = _find_index(lines, "Maximum number of primitives")
    max_nc = int(lines[idx_maxnc + 1])

    mol = Molecule(atoms, charge, nbasis, max_nc)
    return mol


def read_integrals(path, nbasis):
    """Read S, T, V and two-electron integrals from extended input file."""
    # Read non-empty lines.
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # A. Overlap integrals.
    idx_ov = _find_index(lines, "A. Overlap integrals")
    nS = int(lines[idx_ov + 1])

    S = np.zeros((nbasis, nbasis), dtype=float)

    # Next nS lines: mu nu value.
    for k in range(nS):
        parts = lines[idx_ov + 2 + k].split()
        mu = int(parts[0]) - 1   # 1-based to 0-based.
        nu = int(parts[1]) - 1
        val = float(parts[2])

        S[mu, nu] = val
        S[nu, mu] = val  # Symmetry.

    # B. Kinetic integrals.
    idx_kin = _find_index(lines, "B. Kinetic integrals")
    nT = int(lines[idx_kin + 1])

    T = np.zeros((nbasis, nbasis), dtype=float)

    for k in range(nT):
        parts = lines[idx_kin + 2 + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        val = float(parts[2])

        T[mu, nu] = val
        T[nu, mu] = val

    # C. Nuclear attraction integrals.
    idx_v = _find_index(lines, "C. Nuclear Attraction integrals")
    nV = int(lines[idx_v + 1])

    V = np.zeros((nbasis, nbasis), dtype=float)

    for k in range(nV):
        parts = lines[idx_v + 2 + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        val = float(parts[2])

        V[mu, nu] = val
        V[nu, mu] = val

    # D. Two-electron integrals.
    idx_eri = _find_index(lines, "D. Two-Electron integrals")
    nERI = int(lines[idx_eri + 1])

    # eri[mu, nu, lam, sig] = (mu nu | lam sig).
    eri = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=float)

    for k in range(nERI):
        parts = lines[idx_eri + 2 + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        lam = int(parts[2]) - 1
        sig = int(parts[3]) - 1
        val = float(parts[4])

        # Fill symmetry-related entries.
        eri[mu, nu, lam, sig] = val
        eri[mu, nu, sig, lam] = val
        eri[nu, mu, lam, sig] = val
        eri[nu, mu, sig, lam] = val
        eri[lam, sig, mu, nu] = val
        eri[lam, sig, nu, mu] = val
        eri[sig, lam, mu, nu] = val
        eri[sig, lam, nu, mu] = val

    return S, T, V, eri


def read_basis_block(path, nbasis):
    """Read basis set block and map each basis function to its atom."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = _find_index(lines, "Basis set:")
    idx += 1  # First line after header.

    basis = [None] * nbasis
    mu_to_atom = [None] * nbasis

    # Per basis function: header, nprim, then primitive lines.
    for k in range(nbasis):
        parts = lines[idx].split()
        mu = int(parts[0]) - 1          # 0-based.
        atom_index = int(parts[2]) - 1  # 0-based.
        mu_to_atom[mu] = atom_index

        idx += 1
        nprim = int(lines[idx])
        idx += 1

        primitives = []
        for i in range(nprim):
            zeta_str, coeff_str = lines[idx].split()
            primitives.append((float(zeta_str), float(coeff_str)))
            idx += 1

        basis[mu] = {
            "atom_index": atom_index,
            "primitives": primitives,
        }

    if any(x is None for x in mu_to_atom):
        raise ValueError("Failed to build mu_to_atom from Basis set block.")

    return basis, mu_to_atom



def read_derivatives(path, nbasis, natoms, mu_to_atom):
    """Read derivative sections E/F/G/H from extended input file."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # E. Overlap derivatives.
    idx = _find_index(lines, "E. Derivatives of overlap integrals")
    nvec = int(lines[idx + 1])
    start = idx + 2
    while start < len(lines) and not lines[start][0].isdigit():
        start += 1

    dS = np.zeros((natoms, nbasis, nbasis, 3), dtype=float)  # dS[A, mu, nu, xyz]

    for k in range(nvec):
        parts = lines[start + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        dx = float(parts[2])
        dy = float(parts[3])
        dz = float(parts[4])
        vec = (dx, dy, dz)

        atom_mu = mu_to_atom[mu]
        atom_nu = mu_to_atom[nu]

        dS[atom_mu, mu, nu, :] = vec
        dS[atom_mu, nu, mu, :] = vec

        if atom_nu != atom_mu:
            dS[atom_nu, mu, nu, :] = (-dx, -dy, -dz)
            dS[atom_nu, nu, mu, :] = (-dx, -dy, -dz)

    # F. Kinetic derivatives.
    idx = _find_index(lines, "F. Derivatives of kinetic energy integrals")
    nvec = int(lines[idx + 1])
    start = idx + 2
    while start < len(lines) and not lines[start][0].isdigit():
        start += 1

    dT = np.zeros((natoms, nbasis, nbasis, 3), dtype=float)  # dT[A, mu, nu, xyz]

    for k in range(nvec):
        parts = lines[start + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        dx = float(parts[2])
        dy = float(parts[3])
        dz = float(parts[4])
        vec = (dx, dy, dz)

        atom_mu = mu_to_atom[mu]
        atom_nu = mu_to_atom[nu]

        dT[atom_mu, mu, nu, :] = vec
        dT[atom_mu, nu, mu, :] = vec

        if atom_nu != atom_mu:
            dT[atom_nu, mu, nu, :] = (-dx, -dy, -dz)
            dT[atom_nu, nu, mu, :] = (-dx, -dy, -dz)

    # G. Nuclear attraction derivatives.
    idx = _find_index(lines, "G. Derivatives of Nucleus-electron energy integrals")
    nvec = int(lines[idx + 1])
    start = idx + 2
    while start < len(lines) and not lines[start][0].isdigit():
        start += 1

    dV = np.zeros((natoms, nbasis, nbasis, 3), dtype=float)  # dV[A, mu, nu, xyz]

    for k in range(nvec):
        parts = lines[start + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        atom = int(parts[2]) - 1
        dx = float(parts[3])
        dy = float(parts[4])
        dz = float(parts[5])

        dV[atom, mu, nu, 0] = dx
        dV[atom, mu, nu, 1] = dy
        dV[atom, mu, nu, 2] = dz

        dV[atom, nu, mu, 0] = dx
        dV[atom, nu, mu, 1] = dy
        dV[atom, nu, mu, 2] = dz

    # H. Two-electron derivatives.
    idx = _find_index(lines, "H. Derivatives of two-electron integrals")
    nvec = int(lines[idx + 1])
    start = idx + 2
    while start < len(lines) and not lines[start][0].isdigit():
        start += 1

    dERI = np.zeros((natoms, nbasis, nbasis, nbasis, nbasis, 3), dtype=float)  # dERI[A, mu, nu, lam, sig, xyz]

    for k in range(nvec):
        parts = lines[start + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        lam = int(parts[2]) - 1
        sig = int(parts[3]) - 1
        atom = int(parts[4]) - 1
        dx = float(parts[5])
        dy = float(parts[6])
        dz = float(parts[7])

        for (a, b, c, d) in [
            (mu, nu, lam, sig),
            (mu, nu, sig, lam),
            (nu, mu, lam, sig),
            (nu, mu, sig, lam),
            (lam, sig, mu, nu),
            (lam, sig, nu, mu),
            (sig, lam, mu, nu),
            (sig, lam, nu, mu),
        ]:
            dERI[atom, a, b, c, d, 0] = dx
            dERI[atom, a, b, c, d, 1] = dy
            dERI[atom, a, b, c, d, 2] = dz

    return dS, dT, dV, dERI


class Atom:
    def __init__(self, label, Z, x, y, z):
        self.label = label
        self.Z = Z
        self.x = x
        self.y = y
        self.z = z


class Molecule:
    def __init__(self, atoms, charge, nbasis, max_nc):
        # atoms will be a list of Atom objects
        self.atoms = atoms
        self.charge = charge
        self.nbasis = nbasis
        self.max_nc = max_nc


def _find_index(lines, tittle):
    """
    Searches the list 'lines' for the first line that starts with 'tittle',
    ignoring uppercase/lowercase differences.
    """
    tittle_lower = tittle.lower()

    for i, line in enumerate(lines):
        if line.lower().startswith(tittle_lower):
            return i

    raise ValueError("No line was found starting with: " + tittle)


def read_basic_input(path):
    """
    Reads the basic data from a Hartree–Fock input file:
      - number of atoms
      - list of atoms
      - total charge
      - number of basis functions
      - max_nc (maximum number of primitives)

    Assumes an input format of the form:

      Input for Hartree-Fock calculations:
      number of atoms
         2
      Atom labels, atom number Z, coords (Angstrom)
      H 1 ...
      H 1 ...
      Overall charge
         0
      Number of basis funcs
         4
      Maximum number of primitives
         3
    """

    # Read all non-empty lines from the file
    with open(path, "r") as f:
        lines = []
        for line in f:                   # iterate through all lines in the file
            stripped = line.strip()      # remove leading/trailing whitespace
            if stripped:                 # keep only non-empty lines
                lines.append(stripped)

    # --- number of atoms ---
    idx_na = _find_index(lines, "number of atoms")
    natoms = int(lines[idx_na + 1])

    # --- atom block ---
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

    # --- total molecular charge ---
    idx_charge = _find_index(lines, "Overall charge")
    charge = int(lines[idx_charge + 1])

    # --- number of basis functions ---
    idx_nb = _find_index(lines, "Number of basis funcs")
    nbasis = int(lines[idx_nb + 1])

    # --- maximum number of primitives ---
    idx_maxnc = _find_index(lines, "Maximum number of primitives")
    max_nc = int(lines[idx_maxnc + 1])

    # Construct and return the Molecule object
    mol = Molecule(atoms, charge, nbasis, max_nc)
    return mol

def read_integrals(path, nbasis):
    """
    Reads one- and two-electron integrals from an extended HF input file.

    Returns:
      S   : overlap matrix (nbasis x nbasis)
      T   : kinetic energy matrix (nbasis x nbasis)
      V   : nuclear attraction matrix (nbasis x nbasis)
      eri: two-electron integral tensor (nbasis x nbasis x nbasis x nbasis)
    """
    # Read all non-empty lines
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # --- A. Overlap integrals ---
    idx_ov = _find_index(lines, "A. Overlap integrals")
    nS = int(lines[idx_ov + 1])

    # Initialize S with zeros
    S = [[0.0 for _ in range(nbasis)] for _ in range(nbasis)]

    # Each of the next nS lines has: i  j  value
    for k in range(nS):
        parts = lines[idx_ov + 2 + k].split()
        i = int(parts[0]) - 1  # convert from 1-based to 0-based
        j = int(parts[1]) - 1
        val = float(parts[2])

        S[i][j] = val
        S[j][i] = val  # enforce symmetry S_ij = S_ji

    # --- B. Kinetic integrals ---
    idx_kin = _find_index(lines, "B. Kinetic integrals")
    nT = int(lines[idx_kin + 1])

    T = [[0.0 for _ in range(nbasis)] for _ in range(nbasis)]

    for k in range(nT):
        parts = lines[idx_kin + 2 + k].split()
        i = int(parts[0]) - 1
        j = int(parts[1]) - 1
        val = float(parts[2])

        T[i][j] = val
        T[j][i] = val  # T is also symmetric

    # --- C. Nuclear Attraction integrals ---
    idx_v = _find_index(lines, "C. Nuclear Attraction integrals")
    nV = int(lines[idx_v + 1])

    V = [[0.0 for _ in range(nbasis)] for _ in range(nbasis)]

    for k in range(nV):
        parts = lines[idx_v + 2 + k].split()
        i = int(parts[0]) - 1
        j = int(parts[1]) - 1
        val = float(parts[2])

        V[i][j] = val
        V[j][i] = val  # symmetric as well

    # --- D. Two-electron integrals ---
    idx_eri = _find_index(lines, "D. Two-Electron integrals")
    nERI = int(lines[idx_eri + 1])

    # 4D tensor for (mu nu | lambda sigma)
    eri = [[[[0.0 for _ in range(nbasis)]
                    for _ in range(nbasis)]
                    for _ in range(nbasis)]
                    for _ in range(nbasis)]

    for k in range(nERI):
        parts = lines[idx_eri + 2 + k].split()
        mu = int(parts[0]) - 1
        nu = int(parts[1]) - 1
        lam = int(parts[2]) - 1
        sig = int(parts[3]) - 1
        val = float(parts[4])

        # Store in all symmetry-related positions
        eri[mu][nu][lam][sig] = val
        eri[mu][nu][sig][lam] = val
        eri[nu][mu][lam][sig] = val
        eri[nu][mu][sig][lam] = val
        eri[lam][sig][mu][nu] = val
        eri[lam][sig][nu][mu] = val
        eri[sig][lam][mu][nu] = val
        eri[sig][lam][nu][mu] = val

    return S, T, V, eri


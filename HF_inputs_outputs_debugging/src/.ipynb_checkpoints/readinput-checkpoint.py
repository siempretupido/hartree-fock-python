# src/readinput.py

class Atom:
    def __init__(self, label, Z, x, y, z):
        self.label = label
        self.Z = Z
        self.x = x
        self.y = y
        self.z = z


class Molecule:
    def __init__(self, atoms, charge, nbasis, max_nc):
        # atoms será una lista de objetos Atom
        self.atoms = atoms
        self.charge = charge
        self.nbasis = nbasis
        self.max_nc = max_nc


def _find_index(lines, pattern):
    """
    Busca en la lista 'lines' la primera línea que empieza por 'tittle',
    ignorando mayúsculas/minúsculas.
    """
    pattern_lower = pattern.lower()

    for i, line in enumerate(lines):
        if line.lower().startswith(pattern_lower):
            return i

    raise ValueError("No se encontró ninguna línea que empiece por: " + pattern)


def read_basic_input(path):
    """
    Lee los datos básicos del .input:
      - número de átomos
      - lista de átomos
      - carga total
      - número de funciones de base
      - max_nc
    Asume el formato tipo:

      Input for Hartree-Fock calculations:
      number of atoms
         2
      Atom labels, atom number Z, coords (Angstrom)
      H 1  ...
      H 1  ...
      Overall charge
         0
      Number of basis funcs
         4
      Maximum number of primitives
         3
    """
    # Leemos todas las líneas no vacías
    with open(path, "r") as f:
        lines = []
        for line in f:  # recorre todas las líneas del archivo
            stripped = line.strip()  # quita espacios al principio y al final
            if stripped:  # si la línea NO está vacía
                lines.append(stripped)

    # --- número de átomos ---
    idx_na = _find_index(lines, "number of atoms")
    natoms = int(lines[idx_na + 1])

    # --- bloque de átomos ---
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

    # --- carga total ---
    idx_charge = _find_index(lines, "Overall charge")
    charge = int(lines[idx_charge + 1])

    # --- número de funciones de base ---
    idx_nb = _find_index(lines, "Number of basis funcs")
    nbasis = int(lines[idx_nb + 1])

    # --- máximo número de primitivas ---
    idx_maxnc = _find_index(lines, "Maximum number of primitives")
    max_nc = int(lines[idx_maxnc + 1])

    # Creamos el objeto Molecule
    mol = Molecule(atoms, charge, nbasis, max_nc)
    return mol

# main.py

import sys
import numpy as np

from readinput import read_basic_input, read_integrals
from integrals import (
    build_Hcore,
    build_S_inv_sqrt,
    compute_P,
    do_first_scf_iteration,
)


def compute_nelec(mol):
    """
    Compute total number of electrons for a closed-shell molecule:

        nelec = sum(Z_A) - charge

    where Z_A is the atomic number of atom A and 'charge' is the
    total molecular charge (from the input).
    """
    Z_total = sum(atom.Z for atom in mol.atoms)
    nelec = Z_total - mol.charge
    return nelec


def main():
    # Comprobar que se ha pasado la ruta del input
    if len(sys.argv) < 2:
        print("Usage: python -m main path_to_input")
        sys.exit(1)

    input_path = sys.argv[1]

    # 1) Leer datos básicos de la molécula
    mol = read_basic_input(input_path)

    # 2) Leer integrales (S, T, V, eri) del input extendido
    S, T, V, eri = read_integrals(input_path, mol.nbasis)

    # 3) Construir H_core = T + V
    Hcore = build_Hcore(T, V)

    # 4) Construir S^{-1/2}
    S_inv_sqrt = build_S_inv_sqrt(S)

    # 5) Calcular número de electrones (RHF → debe ser par)
    nelec = compute_nelec(mol)
    if nelec % 2 != 0:
        raise ValueError(
            f"RHF requires an even number of electrons, got nelec = {nelec}"
        )

    # 6) Densidad inicial P = 0
    P0 = np.zeros((mol.nbasis, mol.nbasis))

    # 7) Primera iteración SCF
    F, C, P1, eps = do_first_scf_iteration(
        P0, Hcore, eri, S_inv_sqrt, nelec
    )

    # 8) Imprimir resultados básicos para comprobar que todo tira
    print("Number of basis functions:", mol.nbasis)
    print("Total charge:", mol.charge)
    print("Number of electrons (nelec):", nelec)
    print()
    print("Core Hamiltonian H_core:")
    print(Hcore)
    print()
    print("Fock matrix after first SCF iteration:")
    print(F)
    print()
    print("Orbital energies (eps):")
    print(eps)
    print()
    print("Density matrix P after first SCF iteration:")
    print(P1)


if __name__ == "__main__":
    main()



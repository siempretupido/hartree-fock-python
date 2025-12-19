# main.py

import sys
from readinput import (
    read_basic_input,
    read_integrals,
    read_mu_to_atom,
    read_overlap_derivatives,
)
from scf import run_scf


def main():
    path = sys.argv[1]

    # --- basic molecular data (always needed) ---
    mol = read_basic_input(path)

    print("Number of atoms:", len(mol.atoms))
    print("Number of basis functions:", mol.nbasis)
    print("Total charge:", mol.charge)

    # --- extended input: integrals + SCF ---
    if "extended" in path.lower():
        print("Extended input detected.\n")

        # read integrals
        S, T, V, eri = read_integrals(path, mol.nbasis)

        # OPTIONAL: read overlap derivatives (for gradients later)
        mu_to_atom = read_mu_to_atom(path, mol.nbasis)
        dS = read_overlap_derivatives(
            path,
            mol.nbasis,
            len(mol.atoms),
            mu_to_atom,
        )

        # --- run SCF ---
        results = run_scf(
            mol,
            S,
            T,
            V,
            eri,
            max_iter=50,
            tol=1e-6,
            verbose=True,
        )

        print("\nSCF finished.")
        print("Converged:", results["converged"])
        print("Iterations:", results["niter"])
        print("Final total energy:", results["E_tot"])
        print("Final orbital energies:", results["eps"])

    else:
        print("Basic input detected (no integrals, no SCF).")


if __name__ == "__main__":
    main()


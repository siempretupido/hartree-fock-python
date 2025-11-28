# src/main.py
import sys
from readinput import read_basic_input, read_integrals


def print_matrix(name, M):
    """
    Pretty-print a square matrix M with its name.
    """
    n = len(M)
    print(f"\n=== {name} (size {n} x {n}) ===")
    for i in range(n):
        row_str = " ".join(f"{M[i][j]:12.6f}" for j in range(n))
        print(row_str)


def main():
    if len(sys.argv) < 2:
        print("Correct usage: python -m src.main path_to_input")
        return

    input_path = sys.argv[1]

    # 1) Read basic molecular information
    mol = read_basic_input(input_path)

    print("\n=== MOLECULE SUMMARY ===")
    print("Input file         :", input_path)
    print("Number of atoms    :", len(mol.atoms))
    print("Total charge       :", mol.charge)
    print("Number of basis fn :", mol.nbasis)
    print("max_nc             :", mol.max_nc)

    print("\nAtoms:")
    for i, a in enumerate(mol.atoms, start=1):
        print(
            f"{i:2d}. {a.label:2s}  Z={a.Z:2d}   "
            f"({a.x: .6f}, {a.y: .6f}, {a.z: .6f})"
        )

    # 2) Read integrals S, T, V, eri
    S, T, V, eri = read_integrals(input_path, mol.nbasis)

    # 3) Print the one-electron matrices
    print_matrix("Overlap matrix S", S)
    print_matrix("Kinetic matrix T", T)
    print_matrix("Nuclear attraction matrix V", V)



if __name__ == "__main__":
    main()


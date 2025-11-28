from readinput import read_basic_input, read_integrals
from scf import run_scf

def compute_nelec(mol):
    Z_total = sum(atom.Z for atom in mol.atoms)
    return Z_total - mol.charge

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m main path_to_input")
        raise SystemExit(1)

    input_path = sys.argv[1]
    mol = read_basic_input(input_path)
    S, T, V, eri = read_integrals(input_path, mol.nbasis)

    nelec = compute_nelec(mol)
    if nelec % 2 != 0:
        raise ValueError(f"RHF requires an even number of electrons, got {nelec}")

    F, C, P, eps, Hcore = run_scf(S, T, V, eri, nelec)

    print("\nFinal orbital energies:")
    print(eps)
    # (más adelante podrás calcular energía HF total, etc.)

if __name__ == "__main__":
    main()




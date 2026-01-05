import math
import numpy as np


ANGSTROM_TO_BOHR = 1.8897261246257702


def _overlap_ss(alpha, beta, RA, RB):
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = np.dot(RA - RB, RA - RB)
    return (np.pi / p) ** 1.5 * np.exp(-mu * rab2)


def _kinetic_ss(alpha, beta, RA, RB):
    p = alpha + beta
    mu = alpha * beta / p
    rab2 = np.dot(RA - RB, RA - RB)
    overlap = _overlap_ss(alpha, beta, RA, RB)
    return mu * (3.0 - 2.0 * mu * rab2) * overlap


def _boys0(t):
    if t < 1e-8:
        return 1.0
    return 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))


def _boys1(t):
    if t < 1e-8:
        return 1.0 / 3.0
    return (_boys0(t) - math.exp(-t)) / (2.0 * t)


def _nuclear_attraction_ss(alpha, beta, RA, RB, RC, ZC):
    p = alpha + beta
    RP = (alpha * RA + beta * RB) / p
    rpc2 = np.dot(RP - RC, RP - RC)
    overlap = _overlap_ss(alpha, beta, RA, RB)
    return  2.0 * -ZC * math.sqrt(p / np.pi) * overlap * _boys0(p * rpc2)


def _nuclear_attraction_ss1(alpha, beta, RA, RB, RC, ZC):
    p = alpha + beta
    RP = (alpha * RA + beta * RB) / p
    rpc2 = np.dot(RP - RC, RP - RC)
    overlap = _overlap_ss(alpha, beta, RA, RB)
    return  2.0 * -ZC * math.sqrt(p / np.pi) * overlap * _boys1(p * rpc2)

def _eri_ss(alpha, beta, gamma, delta, RA, RB, RC, RD):
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rab2 = np.dot(RA - RB, RA - RB)
    rcd2 = np.dot(RC - RD, RC - RD)
    RP = (alpha * RA + beta * RB) / p
    RQ = (gamma * RC + delta * RD) / q
    rpq2 = np.dot(RP - RQ, RP - RQ)
    K_ab = math.sqrt(2.0) * (math.pi ** (5.0 / 4.0) / p) * math.exp(-mu * rab2)
    K_cd = math.sqrt(2.0) * (math.pi ** (5.0 / 4.0) / q) * math.exp(-nu * rcd2)
    return (K_ab * K_cd / np.sqrt(p+q)) * _boys0((p * q / (p + q)) * rpq2)


def _eri_ss1(alpha, beta, gamma, delta, RA, RB, RC, RD):
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rab2 = np.dot(RA - RB, RA - RB)
    rcd2 = np.dot(RC - RD, RC - RD)
    RP = (alpha * RA + beta * RB) / p
    RQ = (gamma * RC + delta * RD) / q
    rpq2 = np.dot(RP - RQ, RP - RQ)
    K_ab = math.sqrt(2.0) * (math.pi ** (5.0 / 4.0) / p) * math.exp(-mu * rab2)
    K_cd = math.sqrt(2.0) * (math.pi ** (5.0 / 4.0) / q) * math.exp(-nu * rcd2)
    return (K_ab * K_cd / np.sqrt(p+q)) * _boys1((p * q / (p + q)) * rpq2)


def compute_overlap(mol, basis):
    nbasis = len(basis)
    S = np.zeros((nbasis, nbasis), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    for mu in range(nbasis):
        info_mu = basis[mu]
        RA = coords[info_mu["atom_index"]]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            RB = coords[info_nu["atom_index"]]
            prims_nu = info_nu["primitives"]

            total = 0.0
            for alpha, coeff_a in prims_mu:
                for beta, coeff_b in prims_nu:
                    total += coeff_a * coeff_b * _overlap_ss(alpha, beta, RA, RB)
            S[mu, nu] = total
            S[nu, mu] = total

    return S


def compute_kinetic(mol, basis):
    nbasis = len(basis)
    T = np.zeros((nbasis, nbasis), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    for mu in range(nbasis):
        info_mu = basis[mu]
        RA = coords[info_mu["atom_index"]]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            RB = coords[info_nu["atom_index"]]
            prims_nu = info_nu["primitives"]

            total = 0.0
            for alpha, coeff_a in prims_mu:
                for beta, coeff_b in prims_nu:
                    total += coeff_a * coeff_b * _kinetic_ss(alpha, beta, RA, RB)
            T[mu, nu] = total
            T[nu, mu] = total

    return T


def compute_nuclear_attraction(mol, basis):
    nbasis = len(basis)
    V = np.zeros((nbasis, nbasis), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    charges = np.array([atom.Z for atom in mol.atoms], dtype=float)

    for mu in range(nbasis):
        info_mu = basis[mu]
        RA = coords[info_mu["atom_index"]]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            RB = coords[info_nu["atom_index"]]
            prims_nu = info_nu["primitives"]

            total = 0.0
            for alpha, coeff_a in prims_mu:
                for beta, coeff_b in prims_nu:
                    for C, ZC in enumerate(charges):
                        RC = coords[C]
                        total += coeff_a * coeff_b * _nuclear_attraction_ss(
                            alpha, beta, RA, RB, RC, ZC
                        )
            V[mu, nu] = total
            V[nu, mu] = total

    return V


def compute_two_electron(mol, basis):
    nbasis = len(basis)
    eri = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    for mu in range(nbasis):
        info_mu = basis[mu]
        RA = coords[info_mu["atom_index"]]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            RB = coords[info_nu["atom_index"]]
            prims_nu = info_nu["primitives"]
            for lam in range(nbasis):
                info_lam = basis[lam]
                RC = coords[info_lam["atom_index"]]
                prims_lam = info_lam["primitives"]
                for sig in range(lam + 1):
                    info_sig = basis[sig]
                    RD = coords[info_sig["atom_index"]]
                    prims_sig = info_sig["primitives"]

                    total = 0.0
                    for alpha, coeff_a in prims_mu:
                        for beta, coeff_b in prims_nu:
                            for gamma, coeff_c in prims_lam:
                                for delta, coeff_d in prims_sig:
                                    total += (
                                        coeff_a
                                        * coeff_b
                                        * coeff_c
                                        * coeff_d
                                        * _eri_ss(
                                            alpha, beta, gamma, delta, RA, RB, RC, RD
                                        )
                                    )

                    eri[mu, nu, lam, sig] = total
                    eri[nu, mu, lam, sig] = total
                    eri[mu, nu, sig, lam] = total
                    eri[nu, mu, sig, lam] = total
                    eri[lam, sig, mu, nu] = total
                    eri[sig, lam, mu, nu] = total
                    eri[lam, sig, nu, mu] = total
                    eri[sig, lam, nu, mu] = total

    return eri


def compute_overlap_derivatives(mol, basis):
    natoms = len(mol.atoms)
    nbasis = len(basis)
    dS = np.zeros((natoms, nbasis, nbasis, 3), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    for mu in range(nbasis):
        info_mu = basis[mu]
        A = info_mu["atom_index"]
        RA = coords[A]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            B = info_nu["atom_index"]
            RB = coords[B]
            prims_nu = info_nu["primitives"]

            dA = np.zeros(3, dtype=float)
            dB = np.zeros(3, dtype=float)
            for alpha, coeff_a in prims_mu:
                for beta, coeff_b in prims_nu:
                    p = alpha + beta
                    RP = (alpha * RA + beta * RB) / p
                    S = _overlap_ss(alpha, beta, RA, RB)
                    pref = coeff_a * coeff_b
                    dA += pref * (2.0 * alpha * (RP - RA) * S)
                    dB += pref * (2.0 * beta * (RP - RB) * S)

            dS[A, mu, nu] += dA
            dS[B, mu, nu] += dB
            dS[A, nu, mu] = dS[A, mu, nu]
            dS[B, nu, mu] = dS[B, mu, nu]

    return dS


def compute_kinetic_derivatives(mol, basis):
    natoms = len(mol.atoms)
    nbasis = len(basis)
    dT = np.zeros((natoms, nbasis, nbasis, 3), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    for mu in range(nbasis):
        info_mu = basis[mu]
        A = info_mu["atom_index"]
        RA = coords[A]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            B = info_nu["atom_index"]
            RB = coords[B]
            prims_nu = info_nu["primitives"]

            dA = np.zeros(3, dtype=float)
            dB = np.zeros(3, dtype=float)
            for alpha, coeff_a in prims_mu:
                for beta, coeff_b in prims_nu:
                    p = alpha + beta
                    RP = (alpha * RA + beta * RB) / p
                    mu_exp = alpha * beta / p
                    S = _overlap_ss(alpha, beta, RA, RB)
                    T = _kinetic_ss(alpha, beta, RA, RB)
                    pref = coeff_a * coeff_b
                    dA += pref * (2.0 * alpha * (RP - RA) * (T + 2.0 * mu_exp * S))
                    dB += pref * (2.0 * beta * (RP - RB) * (T + 2.0 * mu_exp * S))

            dT[A, mu, nu] += dA
            dT[B, mu, nu] += dB
            dT[A, nu, mu] = dT[A, mu, nu]
            dT[B, nu, mu] = dT[B, mu, nu]

    return dT


def compute_nuclear_attraction_derivatives(mol, basis):
    natoms = len(mol.atoms)
    nbasis = len(basis)
    dV = np.zeros((natoms, nbasis, nbasis, 3), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )
    charges = np.array([atom.Z for atom in mol.atoms], dtype=float)

    for mu in range(nbasis):
        info_mu = basis[mu]
        A = info_mu["atom_index"]
        RA = coords[A]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            B = info_nu["atom_index"]
            RB = coords[B]
            prims_nu = info_nu["primitives"]

            dA = np.zeros(3, dtype=float)
            dB = np.zeros(3, dtype=float)
            dC = np.zeros((natoms, 3), dtype=float)

            for alpha, coeff_a in prims_mu:
                for beta, coeff_b in prims_nu:
                    p = alpha + beta
                    RP = (alpha * RA + beta * RB) / p
                    pref = coeff_a * coeff_b
                    for C, ZC in enumerate(charges):
                        RC = coords[C]
                        V = _nuclear_attraction_ss(alpha, beta, RA, RB, RC, ZC)
                        V1 = _nuclear_attraction_ss1(alpha, beta, RA, RB, RC, ZC)
                        dA += pref * (2.0 * alpha * ((RP - RA) * V - (RP - RC) * V1))
                        dB += pref * (2.0 * beta * ((RP - RB) * V - (RP - RC) * V1))
                        dC[C] += pref * (2.0 * p * (RP - RC) * V1)

            dV[A, mu, nu] += dA
            dV[B, mu, nu] += dB
            for C in range(natoms):
                dV[C, mu, nu] += dC[C]
            dV[:, nu, mu] = dV[:, mu, nu]

    return dV


def compute_two_electron_derivatives(mol, basis):
    natoms = len(mol.atoms)
    nbasis = len(basis)
    dERI = np.zeros((natoms, nbasis, nbasis, nbasis, nbasis, 3), dtype=float)

    coords = (
        np.array([[atom.x, atom.y, atom.z] for atom in mol.atoms], dtype=float)
        * ANGSTROM_TO_BOHR
    )

    for mu in range(nbasis):
        info_mu = basis[mu]
        A = info_mu["atom_index"]
        RA = coords[A]
        prims_mu = info_mu["primitives"]
        for nu in range(mu + 1):
            info_nu = basis[nu]
            B = info_nu["atom_index"]
            RB = coords[B]
            prims_nu = info_nu["primitives"]
            for lam in range(nbasis):
                info_lam = basis[lam]
                C = info_lam["atom_index"]
                RC = coords[C]
                prims_lam = info_lam["primitives"]
                for sig in range(lam + 1):
                    info_sig = basis[sig]
                    D = info_sig["atom_index"]
                    RD = coords[D]
                    prims_sig = info_sig["primitives"]

                    dA = np.zeros(3, dtype=float)
                    dB = np.zeros(3, dtype=float)
                    dC = np.zeros(3, dtype=float)
                    dD = np.zeros(3, dtype=float)

                    for alpha, coeff_a in prims_mu:
                        for beta, coeff_b in prims_nu:
                            p = alpha + beta
                            RP = (alpha * RA + beta * RB) / p
                            for gamma, coeff_c in prims_lam:
                                for delta, coeff_d in prims_sig:
                                    q = gamma + delta
                                    RQ = (gamma * RC + delta * RD) / q
                                    W = (p * RP + q * RQ) / (p + q)
                                    eri = _eri_ss(
                                        alpha, beta, gamma, delta, RA, RB, RC, RD
                                    )
                                    eri1 = _eri_ss1(
                                        alpha, beta, gamma, delta, RA, RB, RC, RD
                                    )
                                    pref = coeff_a * coeff_b * coeff_c * coeff_d
                                    dA += pref * (
                                        2.0 * alpha * ((RP - RA) * eri + (W - RP) * eri1)
                                    )
                                    dB += pref * (
                                        2.0 * beta * ((RP - RB) * eri + (W - RP) * eri1)
                                    )
                                    dC += pref * (
                                        2.0 * gamma * ((RQ - RC) * eri + (W - RQ) * eri1)
                                    )
                                    dD += pref * (
                                        2.0 * delta * ((RQ - RD) * eri + (W - RQ) * eri1)
                                    )

                    acc = {}
                    for idx, vec in ((A, dA), (B, dB), (C, dC), (D, dD)):
                        if idx in acc:
                            acc[idx] += vec
                        else:
                            acc[idx] = vec.copy()

                    for Aidx, vec in acc.items():
                        dERI[Aidx, mu, nu, lam, sig] = vec
                        dERI[Aidx, nu, mu, lam, sig] = vec
                        dERI[Aidx, mu, nu, sig, lam] = vec
                        dERI[Aidx, nu, mu, sig, lam] = vec
                        dERI[Aidx, lam, sig, mu, nu] = vec
                        dERI[Aidx, sig, lam, mu, nu] = vec
                        dERI[Aidx, lam, sig, nu, mu] = vec
                        dERI[Aidx, sig, lam, nu, mu] = vec

    return dERI

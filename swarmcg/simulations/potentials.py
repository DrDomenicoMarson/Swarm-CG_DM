import numpy as np


def gmx_bonds_func_1(x, a, b, c):
    """
    Gromacs potential function 1 for bonds.
    """
    return a / 2 * (x - b) ** 2 + c


def gmx_angles_func_1(x, a, b, c):
    """
    Gromacs potential function 1 for angles.
    """
    return gmx_bonds_func_1(x, a, b, c)


def gmx_angles_func_2(x, a, b, c):
    """
    Gromacs potential function 2 for angles.
    """
    return a / 2 * (np.cos(x) - np.cos(b)) ** 2 + c


def gmx_angles_func_10(x, a, b, c):
    """
    Gromacs angle function 10 (restricted bending / ReB).
    """
    sin_sq = np.sin(x) ** 2
    denom = np.where(sin_sq < 1e-8, 1e-8, sin_sq)
    return a / 2 * (np.cos(x) - np.cos(b)) ** 2 / denom + c


def gmx_dihedrals_func_1(mult):
    """
    Gromacs potential function 1 for angles -- generated on the fly with adjusted multiplicity
    """

    def mult_adjusted(x, a, b, c):
        return a * (1 + np.cos(mult * x - b)) + c

    return mult_adjusted


def gmx_dihedrals_func_2(x, a, b, c):
    """
    Gromacs potential function 2 for dihedrals -- the same as potential function 1 for angles
    """
    return gmx_bonds_func_1(x, a, b, c)  # it's actually the same


def gmx_dihedrals_func_3(x, c0, c1, c2, c3, c4, c5):
    """
    Gromacs dihedral function 3 (Ryckaert-Bellemans).
    """
    psi = x - np.pi
    cos_psi = np.cos(psi)
    return c0 + c1 * cos_psi + c2 * cos_psi ** 2 + c3 * cos_psi ** 3 + c4 * cos_psi ** 4 + c5 * cos_psi ** 5

# TODO: for dihedral function 9, this is the merging of several potentials of
# TODO: gmx_dihedrals_func_1 -- here one of mult=1 together with another of mult=2
# def f(x,a,b,c,d,e):
# 	return a * (1+np.cos(x-b)) + d * (1+np.cos(2*x-e)) + c

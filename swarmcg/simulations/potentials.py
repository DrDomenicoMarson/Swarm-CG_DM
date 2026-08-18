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
    """Evaluate GROMACS restricted-bending angle function 10 exactly.

    Args:
        x: Angle or array of angles in radians, restricted to 10--170 degrees.
        a: Finite force constant in kJ/mol.
        b: Finite equilibrium angle in radians, restricted to 10--170 degrees.
        c: Finite additive energy offset in kJ/mol.

    Returns:
        Restricted-bending energy in kJ/mol.

    Raises:
        ValueError: If an angle is outside the supported safety interval or
            any input is non-finite.
    """
    angles = np.asarray(x, dtype=float)
    equilibrium = float(b)
    if (
        not np.all(np.isfinite(angles))
        or not np.isfinite(a)
        or not np.isfinite(equilibrium)
        or not np.isfinite(c)
    ):
        raise ValueError("Restricted-bending inputs must be finite.")
    lower = np.deg2rad(10.0)
    upper = np.deg2rad(170.0)
    if np.any((angles < lower) | (angles > upper)) or not lower <= equilibrium <= upper:
        raise ValueError(
            "Restricted-bending angles and equilibrium must lie within 10--170 degrees."
        )
    return a / 2 * (np.cos(angles) - np.cos(equilibrium)) ** 2 / np.sin(angles) ** 2 + c


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


def gmx_dihedrals_func_11(theta_prev, theta_curr, phi, k_phi, a0, a1, a2, a3, a4):
    """
    Gromacs dihedral function 11 (combined bending-torsion potential).
    """
    sin_term = np.sin(theta_prev) ** 3 * np.sin(theta_curr) ** 3
    cos_phi = np.cos(phi)
    series = a0 + a1 * cos_phi + a2 * cos_phi ** 2 + a3 * cos_phi ** 3 + a4 * cos_phi ** 4
    return k_phi * sin_term * series

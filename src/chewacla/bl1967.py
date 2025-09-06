"""Method of Busing & Levy, 1967"""

import numpy as np
from hklpy2.misc import IDENTITY_MATRIX_3X3

from chewacla.shorthand import x_hat
from chewacla.shorthand import z_hat
from chewacla.utils import R_axis
from chewacla.utils import is_colinear
from chewacla.utils import matrix_from_2_vectors

rad = np.pi / 180.0
deg = 1 / rad


# TODO: generalize and move to utils
def sample_rotation_matrix(omega_deg, chi_deg, phi_deg):
    """Specific to FourC.  Only the sample motors are used here."""
    # TODO: generalize for any geometry

    R = IDENTITY_MATRIX_3X3
    # intrinsic order  # TODO: verify
    R @= R_axis(z_hat, omega_deg * rad)
    R @= R_axis(x_hat, chi_deg * rad)
    R @= R_axis(z_hat, phi_deg * rad)
    return R


# TODO: generalize and move to utils
def scattering_vector_lab(omega_deg, chi_deg, phi_deg, two_theta_deg, B, hkl):
    """
    Map hkl -> Cartesian reciprocal vector in lab/sample frame.
    Returns the vector (not necessarily normalized).
    """
    # TODO: generalize for any geometry

    # diffractometer geometry, sample stage only
    R_sample = sample_rotation_matrix(omega_deg, chi_deg, phi_deg)

    # crystal geometry
    hkl = np.asarray(hkl, dtype=float)
    B = np.asarray(B, dtype=float)
    R_crystal = B @ hkl  # crystal-frame reciprocal vector

    R_lab = R_sample @ R_crystal  # map to lab/frame after sample rotations
    return R_lab


def calcUB_BusingLevy(ref1, ref2, B, sample_axes):
    """
    ref1, ref2 : Reflection-like objects with 'pseudos' (h,k,l) and 'reals' (omega,chi,phi,two_theta)
    B : 3x3 lattice B matrix
    Returns UB (3x3 numpy array).
    Raises ValueError on colinear reflections.
    """
    # Extract hkl and motor angles
    hkl1 = np.asarray(list(ref1["pseudos"].values()))
    hkl2 = np.asarray(list(ref2["pseudos"].values()))
    # TODO: generalize, just the sample axes, needs its own Diffractometer class
    a1 = np.asarray(list(ref1["reals"].values()))  # for Fourc, expect attributes omega,chi,phi,two_theta
    a2 = np.asarray(list(ref2["reals"].values()))
    # a1 = np.asarray([ref1["reals"][k] for k in sample_axes])  # sample axes, in roder
    # a2 = np.asarray([ref2["reals"][k] for k in sample_axes])

    # Cartesian reciprocal vectors in the crystal frame via the lattice B matrix.
    # Describes the hkl vector in the lab frame.
    # TODO: just the sample motors here
    hkl1_cb = scattering_vector_lab(*a1, B, hkl1)
    hkl2_cb = scattering_vector_lab(*a2, B, hkl2)

    if is_colinear(np.asarray(hkl1), np.asarray(hkl2)):
        raise ValueError("Reflections are colinear; cannot compute UB")

    # Build Tc from B*hkl (Cartesian reciprocal-lattice vectors)
    r1_cart = B @ np.asarray(hkl1, dtype=float)
    r2_cart = B @ np.asarray(hkl2, dtype=float)
    Tc = matrix_from_2_vectors(r1_cart, r2_cart)

    # U from measured/sample-space directions (columns)
    U = matrix_from_2_vectors(hkl1_cb, hkl2_cb)

    UB = U @ Tc.T
    return UB

"""
Lattice support for chewacla.

.. autosummary::

    ~lattice_B
"""

import numpy as np


def lattice_B(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    tau: float = 2 * np.pi,
    tol: float = 1e-12,
) -> np.ndarray:
    """B matrix [BL67] computation from Hkl (Soleil).

    * BL67: Busing&Levy Acta Cyst. 22, 457 (1967)
    * https://repo.or.cz/hkl.git/blob/HEAD:/hkl/hkl-lattice.c

    Compute the reciprocal-lattice B matrix from unit-cell parameters.
    This matrix depends only on the crystal cell parameters.

    Input Units:

    * angles alpha, beta, gamma are given in degrees
    * a, b, c are in the same units as the wavelength

    Returns:
        B: (3,3) numpy.ndarray (float64)

    Raises:
        ValueError: if the cell is degenerate (invalid angles/parameters).

    Reference: standard crystallography formulas (see e.g. International Tables).
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("a,b,c must be > 0")

    # convert degrees -> radians once
    rad = np.pi / 180.0
    alpha_r = alpha * rad
    beta_r = beta * rad
    gamma_r = gamma * rad

    ca = np.cos(alpha_r)
    cb = np.cos(beta_r)
    cg = np.cos(gamma_r)
    sa = np.sin(alpha_r)
    sb = np.sin(beta_r)
    sg = np.sin(gamma_r)

    # metric determinant factor
    D_val = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    if D_val < tol:
        raise ValueError(f"Invalid unit cell (D <= {tol}): D={D_val}")
    inv_sqrtD = 1.0 / np.sqrt(D_val)

    # Following the standard expression for B such that G* = B * h (h =
    # Miller indices). Use consistent, standard formula (rows correspond
    # to reciprocal basis vectors in Cartesian).
    B = tau * np.vstack(
        [
            [
                sb * sg * inv_sqrtD / a,
                (cg * ca - cb) * inv_sqrtD / a,
                (cb * ca - cg) * inv_sqrtD / a,
            ],
            [0, sg * inv_sqrtD / b, (ca * cg - cb) * inv_sqrtD / b],
            [0, 0, sa * inv_sqrtD / c],
        ],
        dtype=np.float64,
    )

    return B

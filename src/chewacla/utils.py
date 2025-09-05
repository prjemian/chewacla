"""
Utility functions for Chewacla.

.. autosummary::

    ~is_colinear
    ~matrix_from_2_vectors
    ~normalize
    ~polar_decompose_rotation
    ~R_axis
"""

from typing import Iterable
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def is_colinear(v1: Iterable[float], v2: Iterable[float], *, tol: float = 1e-8) -> bool:
    """Return True if two vectors are colinear (parallel or anti-parallel).

    Two vectors are considered colinear if either is (near) zero or the norm of
    their cross product is <= tol * (norm(v1)*norm(v2)).

    Parameters
    ----------
    v1, v2
        Iterable of 3 numeric components (or array-like). Will be converted to
        1-D numpy arrays of dtype float.
    tol
        Absolute relative tolerance used to decide colinearity. The cross-product
        criterion is normalized by the product of vector norms to make the test
        scale-invariant.

    Returns
    -------
    bool
        True if vectors are colinear (including zero vectors), False otherwise.

    Raises
    ------
    ValueError
        If inputs cannot be converted to length-3 numeric arrays or contain
        non-finite values.
    """
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("v1 and v2 must be 3-component vectors")

    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("vectors must contain finite numbers")

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    # treat zero vectors as colinear by convention
    if na <= tol or nb <= tol:
        return True

    cross_norm = np.linalg.norm(np.cross(a, b))
    # scale-invariant comparison
    return bool(cross_norm <= tol * (na * nb))


def matrix_from_2_vectors(v1: Sequence[float], v2: Sequence[float], eps: float = 1e-12) -> NDArray[np.float64]:
    """
    Return the 3x3 matrix M with columns [x, y, z] where
      x = normalized(v1)
      z = normalized(v1 × v2)
      y = z × x
    """
    u1 = np.asarray(v1, dtype=float)
    u2 = np.asarray(v2, dtype=float)
    if not np.all(np.isfinite(u1)) or not np.all(np.isfinite(u2)):
        raise ValueError("input vectors must contain finite values")
    if u1.shape != (3,):
        raise ValueError(f"v1 must be array of length 3, received shape {u1.shape}")
    if u2.shape != (3,):
        raise ValueError(f"v2 must be array of length 3, received shape {u2.shape}")

    x = normalize(u1, tol=eps)

    # z should be proportional to v1 x v2; use original u2 (not normalized) or normalized is fine
    v1_cross_v2 = np.cross(x, u2)
    cross_norm = np.linalg.norm(v1_cross_v2)

    if cross_norm <= eps:
        raise ValueError(f"{v1=} and {v2=} are ~collinear")

    # cross products should be non-zero
    z = v1_cross_v2 / cross_norm
    z_cross_v1 = np.cross(z, x)

    cross_norm = np.linalg.norm(z_cross_v1)
    if cross_norm <= eps:
        raise ValueError("unexpected numerical failure when forming y")  # pragma: no cover
    y = z_cross_v1 / cross_norm

    return np.column_stack((x, y, z))


def normalize(v: Iterable[float], *, tol: float = 1e-12) -> np.ndarray:
    """Return the unit vector for `v`.

    Parameters
    ----------
    v
        Sequence or array-like of numeric values representing a vector.
    tol
        Tolerance below which the norm is considered zero (default 1e-12).

    Returns
    -------
    np.ndarray
        A float numpy array of the same shape as `v` with unit norm.

    Raises
    ------
    ValueError
        If the vector norm is below `tol` (treated as zero) or contains non-finite values.
    TypeError
        If input cannot be converted to a numeric array.
    """
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm):
        raise ValueError("vector contains non-finite values")
    if norm <= tol:
        raise ValueError(f"vector norm ({norm}) is below tolerance ({tol}); cannot normalize")
    return arr / norm


def polar_decompose_rotation(M):
    """
    Return R from the polar decomposition M = R @ S where R is a proper rotation
    (orthogonal with det +1) and S is symmetric positive-definite.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"M must be shape (3,3), got {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError("M must contain finite values")

    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # flip sign of last column of U without mutating original array in-place unexpectedly
        U2 = U.copy()
        U2[:, -1] *= -1.0
        R = U2 @ Vt
    return R


def R_axis(axis: Iterable[float], angle_rad: float, *, tol: float = 1e-12) -> np.ndarray:
    """Return the 3x3 rotation matrix for rotation about `axis` by `angle_rad`.

    Parameters
    ----------
    axis
        Iterable of 3 numeric components for the rotation axis.
    angle_rad
        Rotation angle in radians.
    tol
        Minimum allowed norm for `axis`. If the axis norm <= tol raises ValueError.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix (dtype float).

    Raises
    ------
    ValueError
        If the axis has zero/near-zero length or contains non-finite values.
    TypeError
        If inputs cannot be converted to numeric arrays.
    """
    a = np.asarray(axis, dtype=float)
    if a.shape != (3,):
        raise ValueError("axis must be an iterable of three numeric components")
    norm = np.linalg.norm(a)
    if not np.isfinite(norm):
        raise ValueError("axis contains non-finite values")
    if norm <= tol:
        raise ValueError(f"axis norm ({norm}) is at or below tolerance ({tol})")
    k = a / norm

    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=float,
    )
    I = np.eye(3, dtype=float)
    return I * c + (1.0 - c) * np.outer(k, k) + s * K

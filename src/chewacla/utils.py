"""Utility functions for Chewacla.

Copied from dev_stash utilities: normalize, is_colinear, matrix_from_2_vectors,
polar_decompose_rotation, and R_axis.
"""

from collections.abc import Mapping
from typing import Iterable
from typing import Sequence

import numpy as np

from chewacla.shorthand import DirectionVector

# TODO: use/substitute from dev_u5.py
# normalize, axes_rotation_matrix, rodrigues_rotation, compute_UB


def is_colinear(v1: Iterable[float], v2: Iterable[float], *, tol: float = 1e-8) -> bool:
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("v1 and v2 must be 3-component vectors")

    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("vectors must contain finite numbers")

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na <= tol or nb <= tol:
        return True

    cross_norm = np.linalg.norm(np.cross(a, b))
    return bool(cross_norm <= tol * (na * nb))


def matrix_from_2_vectors(v1: Sequence[float], v2: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    u1 = np.asarray(v1, dtype=float)
    u2 = np.asarray(v2, dtype=float)
    if not np.all(np.isfinite(u1)) or not np.all(np.isfinite(u2)):
        raise ValueError("input vectors must contain finite values")
    if u1.shape != (3,):
        raise ValueError(f"v1 must be array of length 3, received shape {u1.shape}")
    if u2.shape != (3,):
        raise ValueError(f"v2 must be array of length 3, received shape {u2.shape}")

    x = normalize(u1, tol=eps)

    v1_cross_v2 = np.cross(x, u2)
    cross_norm = np.linalg.norm(v1_cross_v2)

    if cross_norm <= eps:
        raise ValueError(f"{v1=} and {v2=} are ~collinear")

    z = v1_cross_v2 / cross_norm
    z_cross_v1 = np.cross(z, x)

    cross_norm = np.linalg.norm(z_cross_v1)
    if cross_norm <= eps:
        raise ValueError("unexpected numerical failure when forming y")
    y = z_cross_v1 / cross_norm

    return np.column_stack((x, y, z))


def normalize(v: Iterable[float], *, tol: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector to have unit norm.

    Parameters
    ----------
    v : Iterable[float]
        The input vector to normalize.
    tol : float, optional
        Tolerance below which the norm is considered too small to normalize (default is 1e-12).

    Returns
    -------
    np.ndarray
        The normalized vector as a NumPy array of floats.

    Raises
    ------
    ValueError
        If the vector contains non-finite values or if its norm is below the specified tolerance.

    Examples
    --------
    >>> normalize([3, 4])
    array([0.6, 0.8])
    """
    arr = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector contains non-finite values")
    norm = np.linalg.norm(arr)
    if norm <= tol:
        raise ValueError(
            f"vector norm ({norm}) is below tolerance ({tol});"
            #
            f" cannot normalize vector of shape {arr.shape}"
        )
    return arr / norm


def polar_decompose_rotation(M):
    """
    Compute the closest proper rotation matrix to a given 3x3 matrix using polar decomposition.

    This function takes a 3x3 matrix `M` and returns the nearest
    rotation matrix `R` (i.e., an orthogonal matrix with determinant +1)
    using the polar decomposition via Singular Value Decomposition
    (SVD).

    Parameters
    ----------
    M : array_like, shape (3, 3)
        Input matrix to decompose. Must be a 3x3 matrix with finite values.

    Returns
    -------
    R : ndarray, shape (3, 3)
        The closest proper rotation matrix to `M`.

    Raises
    ------
    ValueError
        If `M` is not of shape (3, 3) or contains non-finite values.

    Notes
    -----
    If the resulting matrix from SVD has a negative determinant, the
    function corrects it to ensure a proper rotation (determinant +1).
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"M must be shape (3,3), got {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError("M must contain finite values")

    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U2 = U.copy()
        U2[:, -1] *= -1.0
        R = U2 @ Vt
    return R


def R_axis(axis: Iterable[float], angle_rad: float, *, tol: float = 1e-12) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix for a rotation about an arbitrary axis.

    Parameters
    ----------
    axis : Iterable[float]
        An iterable of three numeric components representing the axis of rotation.
    angle_rad : float
        The rotation angle in radians.
    tol : float, optional
        Tolerance for the norm of the axis vector. If the norm is less than or equal to this value,
        a ValueError is raised. Default is 1e-12.

    Returns
    -------
    np.ndarray
        A 3x3 NumPy array representing the rotation matrix.

    Raises
    ------
    ValueError
        If the axis does not have exactly three components, contains non-finite values,
        or its norm is at or below the specified tolerance.

    Notes
    -----
    This function uses the Rodrigues' rotation formula to compute the rotation matrix.
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


def scattering_vector_lab(
    stage: Mapping[str, DirectionVector],
    angles: Mapping[str, float],
    B: np.ndarray,
    hkl: np.ndarray,
) -> np.ndarray:
    """
    Transform hkl to Cartesian reciprocal vector in lab/sample frame.

    Parameters
    ----------
    stage: Mapping[str, DirectionVector]
        The sample stage description: axis -> unit vector.
    angles: Mapping[str, float]
        The rotation angles for each stage axis: axis -> angle_degrees.
    B: np.ndarray
        The crystal's reciprocal lattice matrix.
    hkl: np.ndarray
        The Miller indices (h, k, l) to transform.

    Returns
    -------
    np.ndarray
        The transformed reciprocal vector in the lab frame.
    """
    B = np.asarray(B, dtype=float)
    hkl = np.asarray(hkl, dtype=float)
    if B.shape != (3, 3):
        raise ValueError(f"B must be shape (3,3), got {B.shape}")

    # diffractometer sample stage geometry
    R_sample = stage_rotation_matrix(stage, angles)

    R_crystal = B @ hkl  # orient hkl to Cartesian sample
    R_lab = R_sample @ R_crystal  # rotate sample to lab
    return R_lab


def stage_rotation_matrix(stage: Mapping[str, np.ndarray], axes: Mapping[str, float]) -> np.ndarray:
    """Build rotation matrix from a stage description and an axes->angle dict.

    Parameters
    ----------
    stage:
        Mapping of axis name -> unit vector (array-like length 3). Insertion
        order defines the intrinsic composition order.
    axes:
        Mapping of axis name -> angle in degrees.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix composing rotations about each axis in stage order.
    """
    if not isinstance(stage, Mapping):
        raise TypeError("stage must be a mapping of axis-name -> unit vector")
    if not isinstance(axes, Mapping):
        raise TypeError("axes must be a mapping of axis-name -> angle_degrees")

    defined = set(stage.keys())
    given = set(axes.keys())
    missing = sorted(list(defined - given))
    extra = sorted(list(given - defined))
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing stage axes: {missing}")
        if extra:
            parts.append(f"unexpected stage axes: {extra}")
        raise ValueError("; ".join(parts))

    R = np.eye(3, dtype=float)
    for axis, uvec in stage.items():
        degrees = axes[axis]
        try:
            radians = np.deg2rad(degrees)
        except Exception as exc:
            raise TypeError(f"angle for axis {axis!r} must be numeric") from exc
        R = R @ R_axis(uvec, radians)
    return R

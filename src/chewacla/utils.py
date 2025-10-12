"""Utility functions for Chewacla."""

from collections.abc import Mapping
from typing import Iterable
from typing import Sequence

import numpy as np

from chewacla.shorthand import DirectionVector


def axes_rotation_matrix(axes, angles):
    """
    Compute combined rotation matrix from crystal frame to lab frame given rotation axes and angles.

    Parameters
    ----------
    axes:
        list/array of shape (N,3) unit axis vectors ordered from ``lab->...->crystal``
        (outermost to innermost)
    angles:
        list/array of shape (N,) angles in radians for each axis at the known configuration

    Returns
    -------
    R_total (3x3) rotation matrix from crystal frame to lab frame.

    Notes
    -----
    * For axes ordered ``lab->...->crystal``,
      the combined rotation is ``R_total = R_outer @ ... @ R_inner``.
    * Each R_i rotates the coordinate frame about the i-th axis by angle_i;
      applying to a vector expressed in the crystal frame yields the vector in the lab frame.
    """
    R_total = np.eye(3, dtype=float)
    for axis, angle in zip(axes, angles):
        R = rodrigues_rotation(axis, angle)
        R_total = R @ R_total
    return R_total


def compute_UB(
    axes: Sequence[Sequence[float]],
    hkl1: Sequence[float],
    angles1: Sequence[float],
    hkl2: Sequence[float],
    angles2: Sequence[float],
    B: np.ndarray,
    *,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Method of Busing & Levy, 1967, for *ad hoc* rotation axes.

    Compute orientation matrix U and UB = U @ B from two known reflections and
    rotation axes/angles.

    Parameters
    ----------
    axes:
        ordered list (outermost -> innermost) of unit vectors for rotation axes,
        array-like (N,3) unit vectors
    hkl1, hkl2:
        Miller indices of reflection, length-3 arrays (integer or floats)
    angles1, angles2:
        rotation angles  at which corresponding hkl is on detector,
        array-like (N,) angles in degrees

        list must be same length and order as `axes`
    B:
        crystal lattice B matrix (maps hkl -> G_crystal = B @ hkl), array-like (3,3)
    tol : float, optional
        Tolerance below which the norm is considered too small to normalize (default is 1e-12).

    Raises
    -------
    ValueError
        When provided reflections are colinear and cannot be used to compute the
        UB matrix.

    Returns
    -------
    U (3x3), UB (3x3)

    References
    ----------

    * Busing, W. R. and Levy, H. A., 1967. "Orientation Matrix for a Crystal."
      Acta Crystallographica, 22(4), pp.457-464. doi:10.1107/S0365110X67001185.
    """
    from chewacla.utils import is_colinear

    axes = np.asarray(axes, dtype=float)
    B = np.asarray(B, dtype=float)

    hkl1 = np.asarray(hkl1, dtype=float)
    hkl2 = np.asarray(hkl2, dtype=float)

    if is_colinear(hkl1, hkl2, tol=tol):
        raise ValueError("Reflections are colinear; cannot compute UB")

    # Reciprocal vectors in crystal frame
    Gc1 = B @ hkl1
    Gc2 = B @ hkl2

    # Combined rotation from crystal -> lab at the known angles
    angles1 = np.deg2rad(angles1, dtype=float)
    angles2 = np.deg2rad(angles2, dtype=float)
    R_c1 = axes_rotation_matrix(axes, angles1)  # crystal -> lab
    R_c2 = axes_rotation_matrix(axes, angles2)  # crystal -> lab

    # Their images in lab frame via the known mechanical rotation:
    Glab1 = R_c1 @ Gc1
    Glab2 = R_c2 @ Gc2

    # We assume that the known configuration places hkl1 on the detector
    # and defines the mechanical R_c1; we solve for U such that U @ B
    # maps crystalline G vectors into the lab G vectors that the
    # experiment actually observes.
    #
    # Here we compute U by constructing orthonormal bases from the two
    # vectors in crystal and lab frames. This determines U up to a
    # possible reflection sign if the two vectors are nearly collinear;
    # for robust determination use a third non-collinear reference if
    # available.

    def orthonormal_basis(v1, v2) -> np.ndarray:
        """Orthonormal basis from Gc1, Gc2 in crystal frame."""
        e1 = normalize(v1)
        e2 = normalize(v2 - np.dot(e1, v2) * e1)
        e3 = np.cross(e1, e2)
        return np.column_stack((e1, e2, e3))

    # Construct bases
    Bc = orthonormal_basis(Gc1, Gc2)  # crystal basis columns
    Bl = orthonormal_basis(Glab1, Glab2)  # lab basis columns

    # U maps crystal vectors into lab vectors: Bl = U @ Bc  => U = Bl @ Bc^{-1}
    U = Bl @ np.linalg.inv(Bc)

    # Ensure U is a proper rotation (det=+1). If det(U) ~ -1, flip third column of Bl to enforce right-handedness.
    if np.linalg.det(U) < 0:
        Bl[:, 2] *= -1
        U = Bl @ np.linalg.inv(Bc)

    UB = U @ B
    return U, UB


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
    >>> normalize([3, 4, -5])
    array([0.424, 0.566, -0.707])
    """
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError("vector must be length-3")
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


def rodrigues_rotation(
    axis: Sequence[float],
    angle: float,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Compute 3x3 rotation matrix for rotation about arbitrary axis.

    Return 3x3 rotation matrix (using Rodrigues' rotation formula) for rotation
    by ``angle`` radians about ``axis`` (unit vector).

    Parameters
    ----------
    axis : Iterable[float]
        An iterable of three numeric components representing the unit vector of
        the rotation axis.
    angle : float
        The rotation angle in radians.
    tol : float, optional
        Tolerance for the norm of the axis vector. If the norm is less than or
        equal to this value, a ValueError is raised. Default is 1e-12.
    """
    arr = np.asarray(axis, dtype=float)
    if arr.shape != (3,):
        raise ValueError("axis must be an iterable of three numeric components")
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm):
        raise ValueError("axis contains non-finite values")
    if norm <= tol:
        raise ValueError(f"axis norm ({norm}) is at or below tolerance ({tol})")

    ux, uy, uz = normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array(
        [
            [c + ux * ux * C, ux * uy * C - uz * s, ux * uz * C + uy * s],
            [uy * ux * C + uz * s, c + uy * uy * C, uy * uz * C - ux * s],
            [uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C],
        ]
    )
    return R


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


def stage_rotation_matrix(
    stage: Mapping[str, np.ndarray],
    axes: Mapping[str, float],
) -> np.ndarray:
    """Build rotation matrix from *ad hoc* stage description and an axes->angle dict.

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
        R = R @ rodrigues_rotation(uvec, radians)
    return R

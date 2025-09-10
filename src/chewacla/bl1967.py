"""
Method of Busing & Levy, 1967, adapted for multiple rotation axes.

Prompt:

Consider an arbitrary number of rotation axes in a stack, each axis
rotating around a given unit vector.
The axes are provided in order from the lab reference frame to the
innermost.
A crystal of known B matrix is mounted on the innermost axis.
Two reflections (hkl1, hkl2) are known in the crystal frame.
For each reflection, the angles of each axis are known when hkl1 is on the detector.
Show the python code to compute the UB matrix for this stack.

Response:

Below is a concise Python implementation to compute the UB matrix for a stack of rotation axes. Assumptions and conventions used:

* Rotation axes are given as unit 3-vectors in order from lab frame (outermost) to crystal frame (innermost).
* Axis angles are given at the configuration where each reflection hkl is on the detector (i.e., the known rotation angles for each axis).
* The crystal has a known B matrix (3x3) mapping hkl to reciprocal-lattice vectors in the crystal frame: G_crystal = B @ hkl (column vector).
* Two reflections are provided as 3-vectors hkl1 and hkl2 (columns or length-3 arrays).
* We compute the combined rotation from crystal frame to lab frame at the known angles, then deduce the orientation matrix U that rotates crystal reciprocal vectors into lab reciprocal vectors, giving UB = U @ B.
* The code uses right-handed rotation convention: rotate a vector v by angle theta about unit axis n using Rodrigues' formula; rotation matrix R(n,theta) transforms vectors in the rotated frame into the original frame (i.e., active rotation). Check sign convention for your experiment and flip angles if needed.

Notes and cautions:

* Two non-collinear reflections determine rotation up to sign; using a third non-collinear reflection or known chirality removes ambiguity.
* Verify axis ordering and rotation sign conventions against your instrument; you may need to invert angles or reverse multiplication order depending on whether axes are mounted so rotations are applied in coordinate or body frame sense.
"""

from typing import Sequence

import numpy as np

np.set_printoptions(precision=4, floatmode="maxprec", suppress=True)


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
    R_total = np.eye(3)
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
    Compute orientation matrix U and UB = U @ B from two known reflections and rotation axes/angles.

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

    Returns
    -------
    U (3x3), UB (3x3)
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


def normalize(v: Sequence[float], tol: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector to have unit norm.

    Parameters
    ----------
    v : Sequence[float]
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
    a = np.asarray(v, dtype=float)
    if a.shape != (3,):
        raise ValueError("axis must be length-3")
    if not np.all(np.isfinite(a)):
        raise ValueError("axis contains non-finite values")
    n = np.linalg.norm(a)
    if n <= tol:
        raise ValueError(f"axis (shape: {a.shape}) norm {n} <= {tol}")
    return a / n


def rodrigues_rotation(
    axis: Sequence[float],
    angle: float,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix for a rotation about an arbitrary axis.

    Return 3x3 rotation matrix for rotation by `angle` radians about `axis` (unit vector).
    Uses Rodrigues' rotation formula.

    Parameters
    ----------
    axis : Iterable[float]
        An iterable of three numeric components representing the axis of rotation.
    angle : float
        The rotation angle in radians.
    tol : float, optional
        Tolerance for the norm of the axis vector. If the norm is less than or equal to this value,
        a ValueError is raised. Default is 1e-12.
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


# def demo():
#     """Demo with example axes, angles, B, and two reflections."""
#     # Example axes (outermost -> innermost), unit vectors
#     axes = [
#         [0, 0, 1],  # stage rotation about lab z
#         [0, 1, 0],  # next axis
#         [1, 0, 0],  # inner axis (closest to crystal)
#     ]
#     # Angles (degrees) at which hkl1 is on detector
#     angles1 = [5, 10, -15]
#     angles2 = [5, 100, -15]

#     # axes = [
#     #     [-1, 0, 0],  # -x
#     #     [0, 0, 1],  # +z
#     #     [0, 1, 0],  # y
#     # ]
#     # angles1 = [14.4, 0, 0]
#     # angles2 = [14.4, 90, 0]

#     # Example B matrix (3x3)
#     B = np.array([[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.9]])

#     # Two known reflections in crystal frame
#     hkl1 = [1, 0, 0]
#     hkl2 = [0, 1, 0]

#     U, UB = compute_UB(axes, angles1, angles2, B, hkl1, hkl2)
#     print("U =\n", U)
#     print("UB =\n", UB)

# if __name__ == "__main__":
#     demo()

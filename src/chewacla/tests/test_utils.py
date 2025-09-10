import math

import numpy as np
import pytest

from chewacla.utils import R_axis
from chewacla.utils import is_colinear
from chewacla.utils import matrix_from_2_vectors
from chewacla.utils import normalize
from chewacla.utils import polar_decompose_rotation


def test_R_axis_z_90_degrees():
    z = np.array([0.0, 0.0, 1.0])
    R = R_axis(z, math.pi / 2)
    # Rotate x_hat -> y_hat
    x = np.array([1.0, 0.0, 0.0])
    y = R @ x
    assert np.allclose(y, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_is_colinear_true_and_false():
    a = [1.0, 0.0, 0.0]
    b = [2.0, 0.0, 0.0]
    assert is_colinear(a, b)

    c = [0.0, 1.0, 0.0]
    assert not is_colinear(a, c)


def test_normalize_errors_and_success():
    with pytest.raises(ValueError):
        normalize([0.0, 0.0, 0.0])

    v = normalize([1.0, 1.0, 1.0])
    assert np.allclose(np.linalg.norm(v), 1.0)


def test_polar_decompose_rotation_identity_and_reflect():
    I = np.eye(3)
    R = polar_decompose_rotation(I)
    assert np.allclose(R, I)

    # a simple rotation matrix should return itself
    z = np.array([0.0, 0.0, 1.0])
    Rz = R_axis(z, math.pi / 3)
    R2 = polar_decompose_rotation(Rz)
    assert np.allclose(R2, Rz)


def test_matrix_from_2_vectors_basic():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    M = matrix_from_2_vectors(v1, v2)
    # Columns should be orthonormal basis (x,y,z)
    assert M.shape == (3, 3)
    assert np.allclose(M[:, 0], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(M[:, 1], np.array([0.0, 1.0, 0.0]))
    assert np.allclose(M[:, 2], np.array([0.0, 0.0, 1.0]))

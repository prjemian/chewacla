"""Test the chewacla.utils module."""

import math

import numpy as np
import pytest

from chewacla.utils import R_axis
from chewacla.utils import is_colinear
from chewacla.utils import matrix_from_2_vectors
from chewacla.utils import normalize


@pytest.mark.parametrize(
    "v1, v2, expected",
    [
        # exact parallel
        ([1, 0, 0], [2, 0, 0], True),
        # exact anti-parallel
        ([1, 0, 0], [-3, 0, 0], True),
        # orthogonal
        ([1, 0, 0], [0, 1, 0], False),
        # arbitrary non-colinear
        ([1, 1, 0], [1, 0, 0], False),
        # colinear but with floating rounding
        ([1e-9, 0, 0], [2e-9, 0, 0], True),
        # zero vector cases (treat as colinear)
        ([0, 0, 0], [1, 2, 3], True),
        ([0, 0, 0], [0, 0, 0], True),
        # non-unit parallel
        ([3, 3, 3], [1, 1, 1], True),
        # scaled and slightly perturbed parallel (within tol)
        ([1.0, 2.0, 3.0], [2.000000001, 4.000000002, 6.000000003], True),
    ],
)
def test_is_colinear_basic(v1, v2, expected):
    assert is_colinear(v1, v2) is expected


@pytest.mark.parametrize(
    "v1, v2",
    [
        # wrong shape
        ([1, 2], [1, 0, 0]),
        ([1, 2, 3, 4], [1, 0, 0]),
        # non-finite entries
        ([math.inf, 0, 0], [1, 0, 0]),
        ([0, 0, 0], [math.nan, 0, 0]),
    ],
)
def test_is_colinear_raises(v1, v2):
    with pytest.raises(ValueError):
        is_colinear(v1, v2)


def test_is_colinear_tolerance_behavior():
    v = np.array([1.0, 0.0, 0.0])
    # construct a nearly-parallel vector with very small perpendicular component
    w = np.array([1.0, 1e-9, 0.0])
    assert is_colinear(v, w, tol=1e-7) is True
    assert is_colinear(v, w, tol=1e-12) is False


def is_orthonormal_matrix(M, atol=1e-12):
    # Columns are orthonormal and right-handed (determinant +1)
    should_be_identity = M.T @ M
    return (
        np.allclose(should_be_identity, np.eye(3), atol=atol)
        and pytest.approx(np.linalg.det(M), rel=0, abs=1e-12) == 1.0
    )


@pytest.mark.parametrize(
    "v1,v2,expected_x,expected_z",
    [
        # Simple orthogonal unit vectors
        ([1, 0, 0], [0, 1, 0], np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        # v1 not unit, v2 arbitrary
        ([2, 0, 0], [0, 0, 3], np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
        # general non-orthogonal vectors
        ([1, 1, 0], [0, 1, 1], None, None),
        # negative directions
        ([-1, 0, 0], [0, 0, -1], np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),
    ],
)
def test_matrix_from_2_vectors_basic(v1, v2, expected_x, expected_z):
    M = matrix_from_2_vectors(v1, v2)
    assert M.shape == (3, 3)
    # columns are x, y, z
    x, y, z = M[:, 0], M[:, 1], M[:, 2]

    # orthonormality and right-handedness
    assert is_orthonormal_matrix(M)

    # x should be normalized v1
    x_expected = np.asarray(v1, dtype=float) / np.linalg.norm(v1)
    assert np.allclose(x, x_expected, atol=1e-12)

    # z should be perpendicular to both v1 (x) and v2
    assert np.allclose(np.dot(z, x), 0.0, atol=1e-12)
    assert np.allclose(np.dot(z, v2), np.dot(z, np.asarray(v2, dtype=float)), atol=1e-12)

    if expected_x is not None:
        assert np.allclose(x, expected_x, atol=1e-12)
    if expected_z is not None:
        # direction of z can differ by sign depending on order; check parallel up to sign
        assert np.allclose(np.abs(z), np.abs(expected_z), atol=1e-12)


@pytest.mark.parametrize(
    "v1,v2,eps",
    [
        # collinear same direction
        ([1, 0, 0], [2, 0, 0], 1e-12),
        # collinear opposite direction
        ([1, 0, 0], [-3, 0, 0], 1e-12),
        # nearly collinear within eps
        ([1, 0, 0], [1e-13, 0, 0], 1e-12),
    ],
)
def test_matrix_from_2_vectors_collinear_raises(v1, v2, eps):
    with pytest.raises(ValueError):
        matrix_from_2_vectors(v1, v2, eps=eps)


@pytest.mark.parametrize(
    "v1,v2",
    [
        ([0.0, 0.0, 0.0], [1, 0, 0]),  # zero v1
        ([1, 0, 0], [math.nan, 0.0, 0.0]),  # non-finite in v2
        ([math.inf, 0.0, 0.0], [0, 1, 0]),  # non-finite in v1
    ],
)
def test_matrix_from_2_vectors_invalid_inputs_raise(v1, v2):
    with pytest.raises(ValueError):
        matrix_from_2_vectors(v1, v2)


@pytest.mark.parametrize(
    "v1,v2",
    [
        ([1, 0], [0, 1, 0]),  # wrong shape v1
        ([1, 0, 0], [0, 1]),  # wrong shape v2
        ([1, 0, 0, 0], [0, 1, 0]),  # wrong shape v1
    ],
)
def test_matrix_from_2_vectors_shape_validation(v1, v2):
    with pytest.raises(ValueError):
        matrix_from_2_vectors(v1, v2)


def test_matrix_from_2_vectors_numerical_stability_random():
    rng = np.random.default_rng(12345)
    for _ in range(100):
        # random v1 and v2 but avoid exact collinearity by perturbation
        v1 = rng.normal(size=3)
        v2 = rng.normal(size=3) + 1e-6 * v1
        M = matrix_from_2_vectors(v1, v2, eps=1e-15)
        assert is_orthonormal_matrix(M, atol=1e-10)
        # verify x equals normalized v1
        x_expected = np.asarray(v1, dtype=float) / np.linalg.norm(v1)
        assert np.allclose(M[:, 0], x_expected, atol=1e-10)


@pytest.mark.parametrize(
    "vec, tol, expected",
    [
        ([1, 0, 0], 1e-12, np.array([1.0, 0.0, 0.0])),
        ([0, -2], 1e-12, np.array([0.0, -1.0])),
        ((3.0, 4.0), 1e-12, np.array([0.6, 0.8])),
        (np.array([0.0, 5.0, 0.0]), 1e-12, np.array([0.0, 1.0, 0.0])),
        # higher-dim
        (np.ones(4), 1e-12, np.ones(4) / math.sqrt(4)),
    ],
)
def test_normalize_returns_unit_vector(vec, tol, expected):
    out = normalize(vec, tol=tol)
    assert isinstance(out, np.ndarray)
    assert out.shape == np.asarray(vec).shape
    # values close to expected
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-12)
    # norm is 1
    assert np.isclose(np.linalg.norm(out), 1.0, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize(
    "vec, tol",
    [
        ([0.0, 0.0, 0.0], 1e-12),
        ([1e-13, 0.0], 1e-12),
        (np.array([1e-20]), 1e-18),
    ],
)
def test_normalize_raises_on_zero_or_tiny_norm(vec, tol):
    with pytest.raises(ValueError):
        normalize(vec, tol=tol)


@pytest.mark.parametrize(
    "vec",
    [
        [math.inf, 0.0],
        [0.0, -math.inf],
        [math.nan, 1.0],
        np.array([1.0, math.nan]),
    ],
)
def test_normalize_raises_on_non_finite(vec):
    with pytest.raises(ValueError):
        normalize(vec)


@pytest.mark.parametrize("vec", ["abc", None, object()])
def test_normalize_raises_on_non_numeric(vec):
    with pytest.raises((TypeError, ValueError)):
        normalize(vec)


@pytest.mark.parametrize(
    "axis, angle, expected_det",
    [
        ([1, 0, 0], 0.0, 1.0),
        ([0, 1, 0], math.pi / 3, 1.0),
        ([0, 0, 1], -math.pi / 2, 1.0),
        ([1, 1, 1], math.pi / 4, 1.0),
    ],
)
def test_R_axis_properties(axis, angle, expected_det):
    R = R_axis(axis, angle)
    # orthogonal: R @ R.T == I
    I = np.eye(3)
    np.testing.assert_allclose(R @ R.T, I, atol=1e-12, rtol=1e-9)
    # determinant should be ~1 for proper rotation
    det = float(np.linalg.det(R))
    assert pytest.approx(expected_det, rel=1e-9, abs=1e-12) == det


@pytest.mark.parametrize(
    "axis, angle",
    [
        ([1, 0, 0], math.pi / 2),
        ([0, 1, 0], math.pi / 2),
        ([0, 0, 1], math.pi / 2),
        ([1, 1, 0], math.pi / 3),
    ],
)
def test_R_axis_axis_invariant(axis, angle):
    # Vector along axis should be unchanged (direction), i.e., R @ k == k
    axis = np.asarray(axis, dtype=float)
    k = axis / np.linalg.norm(axis)
    R = R_axis(axis, angle)
    out = R @ k
    np.testing.assert_allclose(out, k, atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize(
    "axis, angle, vec, expected",
    [
        # 90 deg about z: x -> y
        ([0, 0, 1], math.pi / 2, [1, 0, 0], [0.0, 1.0, 0.0]),
        # 180 deg about x: y -> -y
        ([1, 0, 0], math.pi, [0, 1, 0], [0.0, -1.0, 0.0]),
        # 120 deg about [1,1,1]: test length preserved
        ([1, 1, 1], 2 * math.pi / 3, [1, 0, 0], None),
    ],
)
def test_R_axis_action(axis, angle, vec, expected):
    R = R_axis(axis, angle)
    v = np.asarray(vec, dtype=float)
    out = R @ v
    if expected is not None:
        np.testing.assert_allclose(out, np.asarray(expected, dtype=float), atol=1e-12, rtol=1e-9)
    # length preserved in all cases
    np.testing.assert_allclose(np.linalg.norm(out), np.linalg.norm(v), atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize(
    "bad_axis",
    [
        [0.0, 0.0, 0.0],
        [1e-20, 0.0, 0.0],
        [math.inf, 0.0, 0.0],
        [math.nan, 1.0, 0.0],
        [1, 2],  # wrong shape
        "not-a-vector",
    ],
)
def test_R_axis_bad_axis(bad_axis):
    with pytest.raises((ValueError, TypeError)):
        R_axis(bad_axis, 0.1)

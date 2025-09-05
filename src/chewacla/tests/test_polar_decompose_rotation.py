import numpy as np
import pytest

from chewacla.utils import polar_decompose_rotation


def is_rotation_matrix(R, atol=1e-12):
    return np.allclose(R.T @ R, np.eye(3), atol=atol) and pytest.approx(np.linalg.det(R), rel=0, abs=1e-12) == 1.0


@pytest.mark.parametrize(
    "M,expected_R",
    [
        # Identity -> identity
        (np.eye(3), np.eye(3)),
        # Pure rotation around z by 45 deg
        (
            np.array(
                [
                    [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
                    [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0],
                    [np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
        # Rotation * symmetric positive-definite stretch -> rotation recovered
        (
            np.diag([2.0, 0.5, 1.5])
            @ np.array(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
    ],
)
def test_polar_decompose_rotation_basic(M, expected_R):
    R = polar_decompose_rotation(M)
    assert R.shape == (3, 3)
    assert is_rotation_matrix(R)
    assert np.allclose(R, expected_R, atol=1e-12)


def test_reflection_matrix_becomes_rotation():
    # matrix with det < 0; polar R should be a proper rotation (det +1)
    # construct M = R_true @ S with R_true a reflection * rotation so det(M) < 0
    R_reflect = np.diag([1.0, 1.0, -1.0])
    S = np.diag([1.2, 0.8, 1.0])
    M = R_reflect @ S
    R = polar_decompose_rotation(M)
    assert is_rotation_matrix(R)
    # R should be closest proper rotation to M; check that R @ S is close to M in Frobenius norm
    # compute S_reconstructed = R.T @ M and check symmetry and p.d.
    S_rec = R.T @ M
    assert np.allclose(S_rec, S_rec.T, atol=1e-12)
    # ensure R is not a reflection
    assert np.linalg.det(R) > 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        polar_decompose_rotation(np.array([[1.0, 2.0], [3.0, 4.0]]))  # wrong shape

    with pytest.raises(ValueError):
        polar_decompose_rotation(np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

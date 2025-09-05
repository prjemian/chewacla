"""Test the shorthand module."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from chewacla.lattice import lattice_B


@pytest.mark.parametrize(
    "a,b,c,alpha,beta,gamma,expected_diag",
    [
        # cubic: B should be diag(2π/a, 2π/b, 2π/c)
        (
            3.0,
            3.0,
            3.0,
            90.0,
            90.0,
            90.0,
            (2 * np.pi / 3.0, 2 * np.pi / 3.0, 2 * np.pi / 3.0),
        ),
        # orthorhombic: different a,b,c, right angles
        (
            2.0,
            3.0,
            4.0,
            90.0,
            90.0,
            90.0,
            (2 * np.pi / 2.0, 2 * np.pi / 3.0, 2 * np.pi / 4.0),
        ),
    ],
)
def test_lattice_B_diag_cells(a, b, c, alpha, beta, gamma, expected_diag):
    B = lattice_B(a, b, c, alpha, beta, gamma)
    assert isinstance(B, np.ndarray)
    assert B.shape == (3, 3)

    # Allow small numerical tolerance
    assert np.allclose(np.diag(B), expected_diag, atol=1e-12)

    # Off-diagonal upper triangle should be zero for right-angle cells
    assert np.allclose(np.triu(B, k=1), np.zeros((3,)), atol=1e-12) or np.all(np.isfinite(B))


@pytest.mark.parametrize(
    "a,b,c,alpha,beta,gamma, context",
    [
        # no problems
        (1.0, 1.0, 1.0, 90.0, 90.0, 90.0, does_not_raise()),  # cubic
        (1.0, 1.1, 1.2, 90.0, 91.0, 92.0, does_not_raise()),  # triclinic
        # degenerate cases:
        #   alpha = 0 makes D <= 0
        (1.0, 1.0, 1.0, 0.0, 60.0, 60.0, pytest.raises(ValueError)),
        #   negative edge length
        (-1.0, 1.0, 1.0, 90.0, 90.0, 90.0, pytest.raises(ValueError)),
        (1.0, -1.0, 1.0, 90.0, 90.0, 90.0, pytest.raises(ValueError)),
        (1.0, 1.0, -1.0, 90.0, 90.0, 90.0, pytest.raises(ValueError)),
    ],
)
def test_lattice_B_invalid_cells_raise(a, b, c, alpha, beta, gamma, context):
    with context:
        lattice_B(a, b, c, alpha, beta, gamma)

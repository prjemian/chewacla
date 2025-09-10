from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from chewacla.utils import scattering_vector_lab


@pytest.mark.parametrize(
    "stage,angles,B,hkl,expect_ctx,match_text",
    [
        # identity: no rotation, B=I, hkl along x
        (
            {"a": np.array([1.0, 0.0, 0.0])},
            {"a": 0.0},
            np.eye(3),
            np.array([1.0, 0.0, 0.0]),
            does_not_raise(),
            None,
        ),
        # rotation about z by +90 degrees maps x -> y
        (
            {"a": np.array([0.0, 0.0, 1.0])},
            {"a": 90.0},
            np.eye(3),
            np.array([1.0, 0.0, 0.0]),
            does_not_raise(),
            None,
        ),
        # invalid B shape
        (
            {"a": np.array([1.0, 0.0, 0.0])},
            {"a": 0.0},
            np.eye(4),
            np.array([1.0, 0.0, 0.0]),
            pytest.raises(ValueError),
            "B must be shape (3,3)",
        ),
        # missing stage axis
        (
            {"a": np.array([1.0, 0.0, 0.0])},
            {},
            np.eye(3),
            np.array([1.0, 0.0, 0.0]),
            pytest.raises(ValueError),
            "missing stage axes",
        ),
        # unexpected extra axis
        (
            {"a": np.array([1.0, 0.0, 0.0])},
            {"a": 0.0, "b": 10.0},
            np.eye(3),
            np.array([1.0, 0.0, 0.0]),
            pytest.raises(ValueError),
            "unexpected stage axes",
        ),
        # non-numeric angle
        (
            {"a": np.array([1.0, 0.0, 0.0])},
            {"a": "bad"},
            np.eye(3),
            np.array([1.0, 0.0, 0.0]),
            pytest.raises(TypeError),
            "angle for axis",
        ),
    ],
    ids=["identity", "rot_z_90", "bad_B", "missing_axis", "extra_axis", "bad_angle"],
)
def test_scattering_vector_lab_parametrized(stage, angles, B, hkl, expect_ctx, match_text):
    with expect_ctx:
        out = scattering_vector_lab(stage, angles, B, hkl)
        # additional checks for non-exception cases
        if match_text is None:
            assert out.shape == (3,)
            # rot_z_90 case: ensure approximate equality when angle==90
            if np.allclose(angles.get(next(iter(stage))), 90.0):
                assert np.allclose(out, np.array([0.0, 1.0, 0.0]), atol=1e-8)

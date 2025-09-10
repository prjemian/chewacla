import numpy as np
import pytest

from chewacla.adhoc import Chewacla
from chewacla.utils import stage_rotation_matrix


@pytest.mark.parametrize(
    "sample_stage,axes",
    [
        ({"a": "x+"}, {"a": 30}),
        ({"a": "x+", "b": "y+"}, {"a": 10, "b": 20}),
    ],
    ids=["single-axis", "multi-axis"],
)
def test_sample_rotation_matrix_delegates(sample_stage, axes):
    """Ensure Chewacla._sample_rotation_matrix delegates to stage_rotation_matrix.

    This verifies equality for a simple single-axis stage and a two-axis stage.
    """
    c = Chewacla(sample_stage, {"d": "z+"})
    expected = stage_rotation_matrix(c.sample_stage, axes)
    actual = c._sample_rotation_matrix(axes)
    assert np.allclose(actual, expected)

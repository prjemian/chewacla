import math

import numpy as np
import pytest

from chewacla.utils import R_axis
from chewacla.utils import is_colinear
from chewacla.utils import normalize


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

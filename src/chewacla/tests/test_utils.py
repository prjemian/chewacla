import math

import numpy as np
import pytest

from chewacla.utils import colinear_vectors
from chewacla.utils import normalize
from chewacla.utils import rodrigues_rotation


def test_rodrigues_rotation_z_90_degrees():
    z = np.array([0.0, 0.0, 1.0])
    R = rodrigues_rotation(z, math.pi / 2)
    # Rotate x_hat -> y_hat
    x = np.array([1.0, 0.0, 0.0])
    y = R @ x
    assert np.allclose(y, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_colinear_vectors_true_and_false():
    a = [1.0, 0.0, 0.0]
    b = [2.0, 0.0, 0.0]
    assert colinear_vectors(a, b)

    c = [0.0, 1.0, 0.0]
    assert not colinear_vectors(a, c)


def test_normalize_errors_and_success():
    with pytest.raises(ValueError):
        normalize([0.0, 0.0, 0.0])

    v = normalize([1.0, 1.0, 1.0])
    assert np.allclose(np.linalg.norm(v), 1.0)

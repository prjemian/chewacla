"""Test the shorthand module."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from chewacla.shorthand import DirectionShorthand


@pytest.fixture
def ds():
    return DirectionShorthand()


@pytest.mark.parametrize(
    "sym,expected",
    [
        ("x+", (1, 0, 0)),
        ("+x", (1, 0, 0)),
        ("X-", (-1, 0, 0)),
        (" y+", (0, 1, 0)),
        ("z-", (0, 0, -1)),
    ],
)
def test_DirectionShorthand(ds, sym, expected):
    out = ds.vector(sym)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == list(expected)


@pytest.mark.parametrize(
    "expression, context",
    [
        ["xx", pytest.raises(ValueError)],  # two letters, no sign
        ["++", pytest.raises(ValueError)],  # no letter
        ["a+", pytest.raises(ValueError)],  # unknown axis
        ["x", pytest.raises(ValueError)],  # too short
        ["xyz", pytest.raises(ValueError)],  # too long
        [None, pytest.raises(TypeError)],  # not a string
    ],
)
def test_DirectionShorthand_invalid_inputs(expression, context):
    d = DirectionShorthand()
    with context:
        d.vector(expression)


@pytest.mark.parametrize(
    "vocabulary, context",
    [
        [None, does_not_raise()],  # default (xyz)
        [{"x": (1, 0, 0)}, does_not_raise()],  # acceptable
        [{"xx": (1, 0, 0)}, pytest.raises(ValueError)],  # multi-letter key
        [{"x": (1, 0)}, pytest.raises(ValueError)],  # wrong length vector
        [{"k": (0, np.cos(50), np.sin(50))}, does_not_raise()],  # kappa axis
    ],
)
def test_DirectionShorthand_vocabulary_validation(vocabulary, context):
    with context:
        DirectionShorthand(vocabulary=vocabulary)


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("x+", np.array([1, 0, 0], dtype=float)),
        ("+x", np.array([1, 0, 0], dtype=float)),
        ("x-", np.array([-1, 0, 0], dtype=float)),
        ("-x", np.array([-1, 0, 0], dtype=float)),
        (" y+", np.array([0, 1, 0], dtype=float)),  # leading space
        ("Z-", np.array([0, 0, -1], dtype=float)),  # uppercase
        ("\t+z", np.array([0, 0, 1], dtype=float)),  # tab then sign
    ],
)
def test_vector_valid(symbol, expected, ds):
    out = ds.vector(symbol)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert out.dtype == expected.dtype
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(
    "symbol,exc",
    [
        ("", ValueError),
        ("x", ValueError),  # missing sign
        ("++", ValueError),  # missing axis letter
        ("xx", ValueError),  # missing sign
        ("xy", ValueError),  # no sign
        ("xyz", ValueError),  # too long
        (42, TypeError),  # wrong type
        (None, TypeError),  # wrong type
    ],
)
def test_vector_invalid_format_raises(symbol, exc, ds):
    with pytest.raises(exc):
        ds.vector(symbol)


def test_vector_unknown_axis_raises(ds):
    with pytest.raises(ValueError):
        ds.vector("k+")


def test_vector_with_custom_single_letter_vocabulary():
    ds = DirectionShorthand(vocabulary={"k": (0, 1, 2)})
    assert np.array_equal(ds.vector("k+"), np.array([0, 1, 2], dtype=float))
    assert np.array_equal(ds.vector("-k"), np.array([0, -1, -2], dtype=float))


def test_vocabulary_rejects_multi_char_keys():
    with pytest.raises(ValueError):
        DirectionShorthand(vocabulary={"kx": (1, 0, 0)})

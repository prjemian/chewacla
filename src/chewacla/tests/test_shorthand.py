"""Test the shorthand module."""

import math
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from chewacla.shorthand import DirectionShorthand
from chewacla.shorthand import unit_vector


@pytest.fixture
def ds_default():
    return DirectionShorthand()


# ------------ unit_vector --------------------


@pytest.mark.parametrize(
    "inp, expect_ctx, expected_vec",
    [
        (np.array([1.0, 0.0, 0.0]), does_not_raise(), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 2.0, 0.0]), does_not_raise(), np.array([0.0, 1.0, 0.0])),
        (np.array([1.0, 1.0, 1.0]), does_not_raise(), np.ones(3) / math.sqrt(3)),
        ([3.0, 0.0, 4.0], does_not_raise(), np.array([0.6, 0.0, 0.8])),
        (np.array([-1.0, 0.0, 0.0]), does_not_raise(), np.array([-1.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, 0.0]), pytest.raises(ValueError), None),
        (np.array([np.nan, 0.0, 1.0]), pytest.raises(ValueError), None),
        (np.array([1.0, 2.0]), pytest.raises(ValueError), None),
        (np.array([[1.0, 0.0, 0.0]]), pytest.raises(ValueError), None),
        ("not-an-array", pytest.raises((TypeError, ValueError)), None),
        (None, pytest.raises((TypeError, ValueError)), None),
    ],
)
def test_unit_vector_parametrized(inp, expect_ctx, expected_vec):
    with expect_ctx:
        out = unit_vector(inp)
        # only check vector properties for non-raising cases
        if expected_vec is not None:
            assert out.shape == (3,)
            assert np.allclose(out, expected_vec, atol=1e-12)
            assert pytest.approx(np.linalg.norm(out), rel=1e-12) == 1.0


@pytest.mark.parametrize(
    "symbol,expected,ctx",
    [
        ("x+", np.array([1.0, 0.0, 0.0]), does_not_raise()),
        ("+x", np.array([1.0, 0.0, 0.0]), does_not_raise()),
        ("x-", np.array([-1.0, 0.0, 0.0]), does_not_raise()),
        ("-x", np.array([-1.0, 0.0, 0.0]), does_not_raise()),
        (" y+ ", np.array([0.0, 1.0, 0.0]), does_not_raise()),
        ("+Y", np.array([0.0, 1.0, 0.0]), does_not_raise()),
        ("z+", np.array([0.0, 0.0, 1.0]), does_not_raise()),
        ("xx", None, pytest.raises(ValueError)),
        ("++", None, pytest.raises(ValueError)),
        ("x", None, pytest.raises(ValueError)),
        ("", None, pytest.raises(ValueError)),
        (" xyz", None, pytest.raises(ValueError)),
        ("x++", None, pytest.raises(ValueError)),
        ("1+", None, pytest.raises(ValueError)),
        ("+1", None, pytest.raises(ValueError)),
        ("++x", None, pytest.raises(ValueError)),
    ],
)
def test_vector_symbols_various(ds_default, symbol, expected, ctx):
    with ctx:
        out = ds_default.vector(symbol)
        if expected is not None:
            assert isinstance(out, np.ndarray)
            assert out.shape == (3,)
            assert out.dtype.kind == "f"
            assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "input_value,ctx",
    [
        (123, pytest.raises(TypeError)),  # non-string symbol
        (None, pytest.raises(TypeError)),
        (b"x+", pytest.raises(TypeError)),
    ],
)
def test_vector_non_string_types_raise(ds_default, input_value, ctx):
    with ctx:
        ds_default.vector(input_value)


def test_vector_unknown_axis_raises(ds_default):
    with pytest.raises(ValueError):
        ds_default.vector("+k")  # k not in default vocabulary


@pytest.mark.parametrize(
    "vocab_input,ctx,expected_keys",
    [
        (None, does_not_raise(), {"x", "y", "z"}),  # default created by constructor
        ({"a": (1, 2, 3)}, does_not_raise(), {"a"}),
        ({"K": (0, 0.6, 0.8)}, does_not_raise(), {"k"}),  # normalized to lowercase
        ({"x": (1, 0, 0)}, does_not_raise(), {"x"}),
        ({"x": np.array((0, 1, 0))}, does_not_raise(), {"x"}),
        ({"ab": (1, 0, 0)}, pytest.raises(ValueError), set()),  # bad key length
        ({1: (1, 0, 0)}, pytest.raises(ValueError), set()),  # non-str key
        ({"x": (1, 0)}, pytest.raises(ValueError), set()),  # value length !=3
        ({"y": (1, 2, "a")}, pytest.raises((ValueError, TypeError)), set()),  # non-numeric entry
        ({"z": np.array([1, 2])}, pytest.raises(ValueError), set()),  # wrong-shape array
    ],
)
def test_vocabulary_constructor_and_setter(vocab_input, ctx, expected_keys):
    with ctx:
        if vocab_input is None:
            ds = DirectionShorthand()
        else:
            ds = DirectionShorthand(vocabulary=vocab_input)
        if ctx is does_not_raise():
            assert set(ds.vocabulary.keys()) == expected_keys


@pytest.mark.parametrize(
    "initial_vocab,set_value,ctx,expected_keys",
    [
        ({"x": (1, 0, 0)}, {"k": (0, 1, 0)}, does_not_raise(), {"k"}),
        ({"x": (1, 0, 0)}, {"K": (0, 1, 0)}, does_not_raise(), {"k"}),
        ({"x": (1, 0, 0)}, {"ab": (1, 0, 0)}, pytest.raises(ValueError), set()),
        ({"x": (1, 0, 0)}, {1: (1, 0, 0)}, pytest.raises(ValueError), set()),
        ({"x": (1, 0, 0)}, {"y": (1, 0)}, pytest.raises(ValueError), set()),
    ],
)
def test_vocabulary_setter_parametrized(initial_vocab, set_value, ctx, expected_keys):
    ds = DirectionShorthand(vocabulary=initial_vocab)
    with ctx:
        ds.vocabulary = set_value
        if ctx is does_not_raise():
            assert set(ds.vocabulary.keys()) == expected_keys


def test_vector_sign_letter_positions():
    ds = DirectionShorthand(vocabulary={"a": (1, 2, 3)})
    assert np.allclose(ds.vector("+a"), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(ds.vector("a+"), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        ds.vector("a")
    with pytest.raises(ValueError):
        ds.vector("+a+")


def test_vector_returns_copy_and_sign_behavior():
    ds = DirectionShorthand(vocabulary={"x": (1, 0, 0)})
    v1 = ds.vector("x+")
    v2 = ds.vector("x+")
    assert v1 is not v2
    v1[0] = 999.0
    assert np.allclose(v2, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(ds.vector("x-"), np.array([-1.0, 0.0, 0.0]))


def test_custom_vocab_with_non_unit_vectors_not_normalized():
    ds = DirectionShorthand(vocabulary={"r": (2, 0, 0)})
    assert np.allclose(ds.vector("r+"), np.array([2.0, 0.0, 0.0]))


def test_repr_contains_entries():
    ds = DirectionShorthand(vocabulary={"x": (1, 0, 0), "y": (0, 1, 0)})
    r = repr(ds)
    assert "x:" in r and "y:" in r

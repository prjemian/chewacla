"""
Test the adhoc module.

prompt> Write parametrized pytests
        using a context manager
        with parameter for pytest.raises(exception) or does_not_raise() for no exception
        (from contextlib import nullcontext as does_not_raise)
        when using pytest.raises(match=text), enclose with re.escape(text)
        label all tests with the class name
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from chewacla.adhoc import AHReflection
from chewacla.adhoc import Chewacla
from chewacla.adhoc import _AHLattice
from chewacla.adhoc import expand_direction_map
from chewacla.shorthand import DirectionShorthand
from chewacla.utils import R_axis

# ------------ expand_direction_map --------------------


@pytest.mark.parametrize(
    "ds, stage_map, expected_output, expected_exception",
    [
        # Valid input case
        (
            DirectionShorthand(),
            {"a": "x+", "b": "y-", "c": "z+"},
            {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, -1.0, 0.0]), "c": np.array([0.0, 0.0, 1.0])},
            does_not_raise(),
        ),
        # Invalid key type in stage_map
        (
            DirectionShorthand(),
            {1: "x+", "b": "y-"},  # Key is an integer
            None,
            pytest.raises(TypeError, match="stage_map keys must be str"),
        ),
        # Invalid value type in stage_map
        (
            DirectionShorthand(),
            {"a": "x+", "b": 2},  # Value is an integer
            None,
            pytest.raises(TypeError, match="stage_map values must be str"),
        ),
        # Invalid shorthand code
        (
            DirectionShorthand(),
            {"a": "invalid_code"},  # Invalid shorthand
            None,
            pytest.raises(
                ValueError,
                match=re.escape("Expected 2-character string like 'x+' or '+x'. Got: 'invalid_code'"),
            ),
        ),
        # Mixed valid and invalid inputs
        (
            DirectionShorthand(),
            {"a": "x+", "b": "invalid_code"},  # One valid and one invalid shorthand
            None,
            pytest.raises(
                ValueError,
                match=re.escape("Expected 2-character string like 'x+' or '+x'. Got: 'invalid_code'"),
            ),
        ),
    ],
)
def test_expand_direction_map(ds, stage_map, expected_output, expected_exception):
    with expected_exception:
        result = expand_direction_map(ds, stage_map)
        if expected_output is not None:
            # Use np.array_equal for comparing NumPy arrays
            for key in expected_output:
                assert np.array_equal(result[key], expected_output[key])


# ------------ _AHLattice --------------------


@pytest.mark.parametrize(
    "a, b, c, alpha, beta, gamma, expected_B, expected_exception",
    [
        # Valid initialization
        (5.0, 5.0, 5.0, 90.0, 90.0, 90.0, 2 * np.pi / 5.0 * np.eye(3), does_not_raise()),
        # Invalid lengths
        (
            -1.0,
            5.0,
            5.0,
            90.0,
            90.0,
            90.0,
            None,
            pytest.raises(ValueError, match=re.escape("lattice lengths a, b, c must be positive")),
        ),
        # Invalid angles
        (
            5.0,
            5.0,
            5.0,
            0.0,
            90.0,
            90.0,
            None,
            pytest.raises(ValueError, match=re.escape("alpha must be in (0, 180) degrees")),
        ),
        (
            5.0,
            5.0,
            5.0,
            90.0,
            180.0,
            90.0,
            None,
            pytest.raises(ValueError, match=re.escape("beta must be in (0, 180) degrees")),
        ),
        (
            5.0,
            5.0,
            5.0,
            90.0,
            90.0,
            180.0,
            None,
            pytest.raises(ValueError, match=re.escape("gamma must be in (0, 180) degrees")),
        ),
        # Degenerate cell (invalid angles)
        (
            5.0,
            5.0,
            5.0,
            0.0,
            90.0,
            90.0,
            None,
            pytest.raises(ValueError, match=re.escape("alpha must be in (0, 180) degrees")),
        ),
        (
            5.0,
            5.0,
            5.0,
            90.0,
            90.0,
            0.0,
            None,
            pytest.raises(ValueError, match=re.escape("gamma must be in (0, 180) degrees")),
        ),
    ],
)
def test_ahlattice(a, b, c, alpha, beta, gamma, expected_B, expected_exception):
    with expected_exception:
        lattice = _AHLattice(a, b, c, alpha, beta, gamma)
        if expected_B is not None:
            B = lattice.B
            assert B.shape == (3, 3)  # Check shape
            assert np.allclose(B, expected_B)  # Check values


@pytest.mark.parametrize(
    "a, b, c, alpha, beta, gamma, expected_dict",
    [
        (
            5.0,
            5.0,
            5.0,
            90.0,
            90.0,
            90.0,
            {
                "a": 5.0,
                "b": 5.0,
                "c": 5.0,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
        ),
        (
            5.123456,
            5.654321,
            5.987654,
            90.0,
            90.0,
            90.0,
            {
                "a": 5.123456,
                "b": 5.654321,
                "c": 5.987654,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
            },
        ),
    ],
)
def test_ahlattice_to_dict(a, b, c, alpha, beta, gamma, expected_dict):
    lattice = _AHLattice(a, b, c, alpha, beta, gamma)
    assert lattice.to_dict() == expected_dict


@pytest.mark.parametrize(
    "a, b, c, alpha, beta, gamma, expected_repr",
    [
        (5.0, 5.0, 5.0, 90.0, 90.0, 90.0, "_AHLattice(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0)"),
    ],
)
def test_ahlattice_repr(a, b, c, alpha, beta, gamma, expected_repr):
    lattice = _AHLattice(a, b, c, alpha, beta, gamma)
    assert repr(lattice) == expected_repr


# ------------ _AHReflection --------------------


@pytest.mark.parametrize(
    "input_value, expected_dict, expected_exception",
    [
        ({"h": 1, "k": 2.0}, {"h": 1.0, "k": 2.0}, does_not_raise()),
        ({}, {}, does_not_raise()),
        ({"h": np.float64(2.5)}, {"h": 2.5}, does_not_raise()),
        ([("h", 1)], None, pytest.raises(TypeError, match=re.escape("pseudos must be a Mapping[str, numeric]"))),
        ({1: 2}, None, pytest.raises(TypeError, match=re.escape("pseudos keys must be strings"))),
        ({"h": "a"}, None, pytest.raises(TypeError, match=re.escape("pseudos values must be numeric"))),
    ],
    ids=[
        "valid_ints_floats",
        "empty",
        "numpy_float",
        "non_mapping",
        "non_str_key",
        "non_numeric_value",
    ],
)
def test_AHReflection_pseudos_setter(input_value, expected_dict, expected_exception):
    """Test _AHReflection.pseudos setter with various valid and invalid inputs."""
    refl = AHReflection("refl", {}, {})
    with expected_exception:
        refl.pseudos = input_value
        if expected_dict is not None:
            assert refl.pseudos == expected_dict
            # ensure values are converted to plain Python floats
            for v in refl.pseudos.values():
                assert isinstance(v, float)


@pytest.mark.parametrize(
    "initial, name, value",
    [
        ({"h": 1}, "k", 2),
        ({}, "h", 3.5),
    ],
    ids=["add_to_existing", "add_to_empty"],
)
def test_AHReflection_set_get_remove_pseudo(initial, name, value):
    refl = AHReflection("refl", initial, {})
    # set and get
    refl.set_pseudo(name, value)
    assert refl.get_pseudo(name) == float(value)
    assert refl.pseudos[name] == float(value)

    # remove and ensure missing afterwards
    refl.remove_pseudo(name)
    assert refl.get_pseudo(name, None) is None


def test_validate_against_and_chewacla_factory():
    c = Chewacla({"s": "y+"}, {"d": "y+"})

    # valid reflection via factory
    r = c.make_reflection("one", {"h": 1, "k": 0, "l": 0}, {"s": 14.4, "d": 28.8})
    assert isinstance(r, AHReflection)

    # addReflection should accept already-validated reflection
    c.addReflection(r)
    assert "one" in c.reflections

    # factory should raise when keys mismatch
    with pytest.raises(ValueError):
        c.make_reflection("bad", {"h": 1}, {"x": 1})
    # Note: removal/key-errors are tested elsewhere; no extra assertions here.


def test_addReflection_duplicates_and_force():
    c = Chewacla({"s": "y+"}, {"d": "y+"})
    r1 = c.make_reflection("dup", {"h": 1, "k": 0, "l": 0}, {"s": 1, "d": 2})
    c.addReflection(r1)
    assert "dup" in c.reflections

    r2 = c.make_reflection("dup", {"h": 1, "k": 1, "l": 0}, {"s": 3, "d": 4})
    with pytest.raises(ValueError, match=re.escape("Reflection 'dup' already exists; pass force=True to replace")):
        c.addReflection(r2)

    # allow replacement when forced
    c.addReflection(r2, force=True)
    assert c.reflections["dup"].pseudos["k"] == 1.0


@pytest.mark.parametrize(
    "input_value, expected_dict, expected_exception",
    [
        ({"phi": 1, "chi": 2.0}, {"phi": 1.0, "chi": 2.0}, does_not_raise()),
        ({}, {}, does_not_raise()),
        ({"omega": np.float64(2.5)}, {"omega": 2.5}, does_not_raise()),
        ([("phi", 1)], None, pytest.raises(TypeError, match=re.escape("reals must be a Mapping[str, numeric]"))),
        ({1: 2}, None, pytest.raises(TypeError, match=re.escape("reals keys must be strings"))),
        ({"phi": "a"}, None, pytest.raises(TypeError, match=re.escape("reals values must be numeric"))),
    ],
    ids=["valid_ints_floats", "empty", "numpy_float", "non_mapping", "non_str_key", "non_numeric_value"],
)
def test_AHReflection_reals_setter(input_value, expected_dict, expected_exception):
    refl = AHReflection("refl", {}, {})
    with expected_exception:
        refl.reals = input_value
        if expected_dict is not None:
            assert refl.reals == expected_dict
            for v in refl.reals.values():
                assert isinstance(v, float)


@pytest.mark.parametrize(
    "initial, name, value",
    [
        ({"phi": 10.0}, "chi", 20.5),
        ({}, "omega", 5),
    ],
)
def test_AHReflection_set_get_remove_real(initial, name, value):
    refl = AHReflection("refl", {}, initial)
    refl.set_real(name, value)
    assert refl.get_real(name) == float(value)
    assert refl.reals[name] == float(value)

    refl.remove_real(name)
    assert refl.get_real(name, None) is None
    with pytest.raises(KeyError):
        refl.remove_real(name)


def test_AHReflection_repr():
    a = AHReflection("a", {"h": 1}, {"phi": 10.0}, wavelength=1.0)
    r = repr(a)
    assert "AHReflection" in r
    assert "pseudos=" in r
    assert "reals=" in r


@pytest.mark.parametrize(
    "other, expected",
    [
        (AHReflection("o1", {"h": 1.0}, {"phi": 10.0}, wavelength=1.0), True),
        (AHReflection("o2", {"h": 1.0 + 1e-9}, {"phi": 10.0}, wavelength=1.0), True),
        (AHReflection("o3", {"k": 1.0}, {"phi": 10.0}, wavelength=1.0), False),
        (AHReflection("o4", {"h": 1.0}, {"chi": 10.0}, wavelength=1.0), False),
        (AHReflection("o5", {"h": 1.0}, {"phi": 10.0}, wavelength=2.0), False),
        (object(), False),
    ],
    ids=[
        "identical",
        "close_values",
        "diff_pseudo_keys",
        "diff_real_keys",
        "diff_wavelength",
        "other_type",
    ],
)
def test_AHReflection_eq_param(other, expected):
    a = AHReflection("a", {"h": 1}, {"phi": 10.0}, wavelength=1.0)
    assert (a == other) is expected


# ------------ Chewacla --------------------


def test_Chewacla_init_and_properties():
    sample_stage = {"a": "x+"}
    detector_stage = {"d": "y+"}
    c = Chewacla(sample_stage, detector_stage)
    assert isinstance(c, Chewacla)

    # sample/detector stage expansion and raw storage
    assert c.raw_sample_stage == sample_stage
    assert c.raw_detector_stage == detector_stage
    assert np.array_equal(c.sample_stage["a"], np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(c.detector_stage["d"], np.array([0.0, 1.0, 0.0]))

    # defaults
    assert c.wavelength == 1.54
    assert len(c.reflections) == 0

    # incident beam default and assignment
    assert np.allclose(c.incident_beam, np.array([1.0, 0.0, 0.0]))
    arr = np.array([0.0, 1.0, 0.0])
    c.incident_beam = arr
    assert np.array_equal(c.raw_incident_beam, arr)
    assert np.array_equal(c.incident_beam, arr)

    # lattice setter
    c.lattice = (2, 3, 4, 90, 90, 90)
    assert isinstance(c.lattice, _AHLattice)
    assert c.lattice.a == 2.0


@pytest.mark.parametrize(
    "value, expected_len, check, expected_exception",
    [
        # set with dict
        ({"one": AHReflection("one", {"h": 1}, {"phi": 1.0})}, 1, None, does_not_raise()),
        # set to None should clear existing entries
        (None, 0, None, does_not_raise()),
        # set with mapping and verify stored item
        ({"x": AHReflection("x", {"h": 2}, {"phi": 2.0})}, 1, ("get_pseudo", "x", "h", 2.0), does_not_raise()),
        # set with iterable of pairs
        (
            [("a", AHReflection("a", {"h": 3}, {"phi": 3.0})), ("b", AHReflection("b", {"h": 4}, {"phi": 4.0}))],
            2,
            ("keys", ("a", "b")),
            does_not_raise(),
        ),
        # invalid type
        (123, None, None, pytest.raises(TypeError)),
    ],
    ids=["dict", "none_clears", "mapping", "pairs", "invalid_type"],
)
def test_Chewacla_reflections_setter_and_types(value, expected_len, check, expected_exception):
    c = Chewacla({"a": "x+"}, {"d": "y+"})

    # For the None case ensure there's something to clear
    if value is None:
        c.reflections = {"pre": AHReflection("pre", {"h": 0}, {"phi": 0.0})}

    with expected_exception:
        c.reflections = value

        if isinstance(expected_len, int):
            assert len(c.reflections) == expected_len

        if check is not None:
            if check[0] == "get_pseudo":
                _, key, pseudo, expected_val = check
                assert c.reflections[key].get_pseudo(pseudo) == expected_val
            elif check[0] == "keys":
                _, keys = check
                assert all(k in c.reflections for k in keys)


def test_Chewacla_calc_UB_BL67_requires_two_reflections():
    c = Chewacla({"a": "x+"}, {"d": "y+"})
    # with zero reflections
    with pytest.raises(ValueError, match="requires exactly two reflections"):
        c.calc_UB_BL67()


def test_Chewacla_addReflection_success_and_errors():
    c = Chewacla({"a": "x+"}, {"d": "y+"})
    # reflection with incorrect real-axis name should raise
    bad = AHReflection("bad", {"h": 1, "k": 0, "l": 0}, {"phi": 1.0})
    with pytest.raises(ValueError):
        c.addReflection(bad)

    # valid reflection: pseudos must include h,k,l and reals must match instrument axes ('a','d')
    good = AHReflection("one", {"h": 1, "k": 0, "l": 0}, {"a": 14.4, "d": 28.8})
    c.addReflection(good)
    assert "one" in c.reflections

    # add by reflection (object holds name) — another valid one
    c.addReflection(AHReflection("two", {"h": 2, "k": 0, "l": 0}, {"a": 14.4, "d": 28.8}))
    assert "two" in c.reflections

    # invalid calls
    with pytest.raises(TypeError):
        c.addReflection(123)  # not an AHReflection
    with pytest.raises(TypeError):
        c.addReflection(object())  # not an AHReflection

    # with one reflection
    c.reflections = {"one": AHReflection("one", {"h": 1}, {"phi": 1.0})}
    with pytest.raises(ValueError, match="requires exactly two reflections"):
        c.calc_UB_BL67()


@pytest.mark.parametrize(
    "modes_expected",
    [
        (['default']),
    ],
    ids=["default_list"],
)
def test_Chewacla_modes_property(modes_expected):
    c = Chewacla({"a": "x+"}, {"d": "y+"})
    assert isinstance(c.modes, list)
    assert c.modes == modes_expected


@pytest.mark.parametrize(
    "initial, set_value, expected, expected_exception",
    [
        # valid set to supported mode
        (None, 'default', 'default', does_not_raise()),
        # invalid mode value
        (None, 'invalid_mode', None, pytest.raises(ValueError, match=re.escape("Invalid mode: invalid_mode"))),
    ],
)
def test_Chewacla_mode_setter(initial, set_value, expected, expected_exception):
    c = Chewacla({"a": "x+"}, {"d": "y+"})
    if initial is not None:
        c.mode = initial
    with expected_exception:
        c.mode = set_value
        if expected is not None:
            assert c.mode == expected


def test_sample_rotation_matrix_single_axis():
    c = Chewacla({"s": "z+"}, {"d": "x+"})
    R = c._sample_rotation_matrix({"s": 90})
    x = np.array([1.0, 0.0, 0.0])
    y = R @ x
    assert np.allclose(y, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_sample_rotation_matrix_multi_axis_composition():
    # order should follow insertion order of sample_stage
    c = Chewacla({"a": "z+", "b": "x+"}, {"d": "y+"})
    angles = {"a": 90, "b": 90}
    R = c._sample_rotation_matrix(angles)
    # expected = I @ R_axis(z,90deg) @ R_axis(x,90deg)
    rad = np.pi / 180.0
    R_expected = np.eye(3) @ R_axis(np.array([0.0, 0.0, 1.0]), 90 * rad) @ R_axis(np.array([1.0, 0.0, 0.0]), 90 * rad)
    assert np.allclose(R, R_expected)


@pytest.mark.parametrize(
    "axes, match",
    [
        ({}, "missing sample axes"),
        ({"x": 10, "y": 20}, "unexpected sample axes"),
    ],
)
def test_sample_rotation_matrix_missing_or_extra_keys(axes, match):
    c = Chewacla({"a": "x+"}, {"d": "y+"})
    with pytest.raises(ValueError, match=match):
        c._sample_rotation_matrix(axes)


def test_sample_rotation_matrix_non_numeric_angle():
    c = Chewacla({"s": "z+"}, {"d": "x+"})
    with pytest.raises(TypeError, match="must be numeric"):
        c._sample_rotation_matrix({"s": "not-a-number"})

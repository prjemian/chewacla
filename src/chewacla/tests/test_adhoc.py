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

from chewacla.adhoc import Chewacla
from chewacla.adhoc import _AHLattice
from chewacla.adhoc import _AHReflection
from chewacla.adhoc import _AHReflectionList
from chewacla.adhoc import expand_direction_map
from chewacla.shorthand import DirectionShorthand

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
    refl = _AHReflection({}, {})
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
    refl = _AHReflection(initial, {})
    # set and get
    refl.set_pseudo(name, value)
    assert refl.get_pseudo(name) == float(value)
    assert refl.pseudos[name] == float(value)

    # remove and ensure missing afterwards
    refl.remove_pseudo(name)
    assert refl.get_pseudo(name, None) is None
    with pytest.raises(KeyError):
        refl.remove_pseudo(name)


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
    refl = _AHReflection({}, {})
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
    refl = _AHReflection({}, initial)
    refl.set_real(name, value)
    assert refl.get_real(name) == float(value)
    assert refl.reals[name] == float(value)

    refl.remove_real(name)
    assert refl.get_real(name, None) is None
    with pytest.raises(KeyError):
        refl.remove_real(name)


def test_AHReflection_repr():
    a = _AHReflection({"h": 1}, {"phi": 10.0}, wavelength=1.0)
    r = repr(a)
    assert "_AHReflection" in r
    assert "pseudos=" in r
    assert "reals=" in r


@pytest.mark.parametrize(
    "other, expected",
    [
        (_AHReflection({"h": 1.0}, {"phi": 10.0}, wavelength=1.0), True),
        (_AHReflection({"h": 1.0 + 1e-9}, {"phi": 10.0}, wavelength=1.0), True),
        (_AHReflection({"k": 1.0}, {"phi": 10.0}, wavelength=1.0), False),
        (_AHReflection({"h": 1.0}, {"chi": 10.0}, wavelength=1.0), False),
        (_AHReflection({"h": 1.0}, {"phi": 10.0}, wavelength=2.0), False),
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
    a = _AHReflection({"h": 1}, {"phi": 10.0}, wavelength=1.0)
    assert (a == other) is expected


# ------------ _AHReflectionList --------------------


def test_AHReflectionList_init_and_basic_ops():
    # empty init
    lst = _AHReflectionList()
    assert len(lst) == 0

    # init with mapping
    a = _AHReflection({"h": 1}, {"phi": 1.0})
    b = _AHReflection({"h": 2}, {"phi": 2.0})
    m = {"first": a, "second": b}
    lst2 = _AHReflectionList(m)
    assert len(lst2) == 2
    assert "first" in lst2
    assert lst2.get("first") is a


def test_AHReflectionList_set_pop_contains_len_iter():
    lst = _AHReflectionList()
    a = _AHReflection({"h": 1}, {"phi": 1.0})

    # set with valid inputs
    lst.set("one", a)
    assert len(lst) == 1
    assert lst.get("one") is a

    # type errors for set
    with pytest.raises(TypeError, match=re.escape("name must be a str")):
        lst.set(1, a)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=re.escape("reflection must be an _AHReflection instance")):
        lst.set("bad", object())  # type: ignore[arg-type]

    # pop behaviour
    popped = lst.pop("one")
    assert popped is a
    assert len(lst) == 0
    with pytest.raises(KeyError):
        lst.pop("one")


def test_AHReflectionList_items_names_values_clear_and_repr():
    # create >10 items to exercise repr truncation
    items = {f"r{i}": _AHReflection({"h": i}, {"phi": float(i)}) for i in range(11)}
    lst = _AHReflectionList(items)
    assert len(lst) == 11

    # items/names/values
    names = list(lst.names())
    assert set(names) == set(items.keys())
    items_list = list(lst.items())
    assert all(isinstance(k, str) and isinstance(v, _AHReflection) for k, v in items_list)
    vals = list(lst.values())
    assert all(isinstance(v, _AHReflection) for v in vals)

    # clear
    lst.clear()
    assert len(lst) == 0

    # repr truncation: create 11 again to check '+1 more' text
    lst = _AHReflectionList(items)
    r = repr(lst)
    assert lst.__class__.__name__ in r
    assert "+1 more" in r


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


def test_Chewacla_reflections_setter_and_types():
    c = Chewacla({"a": "x+"}, {"d": "y+"})

    # set with _AHReflectionList
    r = _AHReflection({"h": 1}, {"phi": 1.0})
    rl = _AHReflectionList({"one": r})
    c.reflections = rl
    assert len(c.reflections) == 1

    # set to None clears
    c.reflections = None
    assert len(c.reflections) == 0

    # set with mapping
    mapping = {"x": _AHReflection({"h": 2}, {"phi": 2.0})}
    c.reflections = mapping
    assert len(c.reflections) == 1
    assert c.reflections.get("x").get_pseudo("h") == 2.0

    # set with iterable of pairs
    pairs = [("a", _AHReflection({"h": 3}, {"phi": 3.0})), ("b", _AHReflection({"h": 4}, {"phi": 4.0}))]
    c.reflections = pairs
    assert len(c.reflections) == 2
    assert "a" in c.reflections and "b" in c.reflections

    # invalid type should raise TypeError
    with pytest.raises(TypeError, match=re.escape("reflections must be an _AHReflectionList, mapping,")):
        c.reflections = 123  # type: ignore[assignment]


def test_Chewacla_calc_UB_BL67_requires_two_reflections():
    c = Chewacla({"a": "x+"}, {"d": "y+"})
    # with zero reflections
    with pytest.raises(ValueError, match="requires exactly two reflections"):
        c.calc_UB_BL67()


def test_Chewacla_addReflection_success_and_errors():
    c = Chewacla({"a": "x+"}, {"d": "y+"})
    r = _AHReflection({"h": 1}, {"phi": 1.0})

    # add by (name, reflection)
    c.addReflection("one", r)
    assert "one" in c.reflections

    # add by pair (new API: name + reflection)
    c.addReflection("two", _AHReflection({"h": 2}, {"phi": 2.0}))
    assert "two" in c.reflections

    # invalid calls
    with pytest.raises(TypeError):
        c.addReflection(123)  # not a pair
    with pytest.raises(TypeError):
        c.addReflection("bad", object())  # not an _AHReflection

    # with one reflection
    c.reflections = {"one": _AHReflection({"h": 1}, {"phi": 1.0})}
    with pytest.raises(ValueError, match="requires exactly two reflections"):
        c.calc_UB_BL67()

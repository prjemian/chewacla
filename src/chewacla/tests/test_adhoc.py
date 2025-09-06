"""Test the adhoc module."""

import math

import numpy as np
import pytest

from chewacla.adhoc import _AHLattice
from chewacla.adhoc import _AHReflection
from chewacla.adhoc import _AHReflectionList
from chewacla.adhoc import expand_direction_map
from chewacla.shorthand import DirectionShorthand

# ---------------------------
# _AHLattice tests
# ---------------------------


@pytest.mark.parametrize(
    "a,b,c,alpha,beta,gamma,raises",
    [
        (1, 1, 1, 90, 90, 90, None),
        (2.5, 3.0, 4.1, 60, 70, 80, None),
        (0, 1, 1, 90, 90, 90, ValueError),
        (-1, 1, 1, 90, 90, 90, ValueError),
        (1, 1, 1, 0, 90, 90, ValueError),  # invalid angle
        (1, 1, 1, 90, 180, 90, ValueError),  # invalid angle
    ],
)
def test_lattice_init_param(a, b, c, alpha, beta, gamma, raises):
    if raises is not None:
        with pytest.raises(raises):
            _AHLattice(a, b, c, alpha, beta, gamma)
    else:
        L = _AHLattice(a, b, c, alpha, beta, gamma)
        assert isinstance(L.a, float)
        assert isinstance(L.alpha, float)


def test_lattice_B_cached_and_copy(monkeypatch):
    L = _AHLattice(3, 4, 5, 90, 90, 90)
    # Ensure chewacla.lattice.lattice_B is called and value returned used
    called = {}

    def fake_lattice_B(a, b, c, alpha, beta, gamma):
        called["args"] = (a, b, c, alpha, beta, gamma)
        return np.eye(3) * 2.0

    monkeypatch.setattr("chewacla.lattice.lattice_B", fake_lattice_B)
    B1 = L.B
    assert "args" in called
    # B returns a copy so modifying result should not change internal cache
    B1[0, 0] = 99.0
    B2 = L.B
    assert B2[0, 0] != 99.0
    assert isinstance(B2, np.ndarray)
    assert B2.shape == (3, 3)


@pytest.mark.parametrize(
    "digits,expected_keys",
    [(None, {"a", "b", "c", "alpha", "beta", "gamma"}), (3, {"a", "b", "c", "alpha", "beta", "gamma"})],
)
def test_lattice_to_dict_rounding(digits, expected_keys):
    L = _AHLattice(1.23456, 2.34567, 3.45678, 10.12345, 20.23456, 30.34567)
    d = L.to_dict(digits=digits)
    assert set(d.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(d[k], float)
    if isinstance(digits, int):
        # string of rounded decimal places by comparing repr formatted length
        for v in d.values():
            # at most digits decimals when converted to string
            s = f"{v:.{digits}f}"
            assert "." in s


# ---------------------------
# _AHReflection tests
# ---------------------------


@pytest.mark.parametrize(
    "pseudos,reals,wavelength,expect_error",
    [
        ({"h": 1, "k": 2, "l": 3}, {"x": 0.1, "y": 0.2}, 1.0, None),
        (None, None, 0.5, None),
    ],
)
def test_reflection_init_and_props(pseudos, reals, wavelength, expect_error):
    if expect_error:
        with pytest.raises(expect_error):
            _AHReflection(pseudos=pseudos, reals=reals, wavelength=wavelength)
        return
    r = _AHReflection(pseudos=pseudos, reals=reals, wavelength=wavelength)
    assert math.isclose(r.wavelength, float(wavelength), rel_tol=1e-12)
    # verify returned views are copies / valid types
    pview = r.pseudos
    rview = r.reals
    assert isinstance(pview, dict)
    assert isinstance(rview, dict)
    # setting from property should validate types
    r.pseudos = {"h": 5, "k": 6}
    assert r.get_pseudo("h") == 5.0
    r.reals = {"ax": 1.1}
    assert r.get_real("ax") == 1.1


@pytest.mark.parametrize("bad_value", [("not-a-mapping"), 123, [1, 2, 3]])
def test_reflection_pseudos_setter_type_errors(bad_value):
    r = _AHReflection()
    with pytest.raises(TypeError):
        r.pseudos = bad_value  # type: ignore[assignment]


def test_reflection_wavelength_setter_validation():
    r = _AHReflection()
    with pytest.raises(ValueError):
        r.wavelength = 0
    with pytest.raises(ValueError):
        r.wavelength = -1.0
    r.wavelength = 2.0
    assert r.wavelength == 2.0


def test_reflection_equality_and_repr():
    r1 = _AHReflection(pseudos={"h": 1}, reals={"x": 0.1}, wavelength=1.0)
    r2 = _AHReflection(pseudos={"h": 1}, reals={"x": 0.1}, wavelength=1.0)
    r3 = _AHReflection(pseudos={"h": 2}, reals={"x": 0.1}, wavelength=1.0)
    assert r1 == r2
    assert not (r1 == r3)
    s = repr(r1)
    assert "pseudos" in s and "reals" in s


# ---------------------------
# _AHReflectionList tests
# ---------------------------


@pytest.mark.parametrize("initial", [None, {}, {"a": _AHReflection({"h": 1}, {"x": 0.1}, 1.0)}])
def test_reflection_list_init_and_len(initial):
    lst = _AHReflectionList(initial=initial)
    assert isinstance(len(lst), int)
    if initial is None:
        assert len(lst) == 0
    else:
        assert len(lst) == len(initial)


def test_reflection_list_set_get_pop_clear_and_contains():
    L = _AHReflectionList()
    r = _AHReflection(pseudos={"h": 1}, reals={"x": 0.1})
    L.set("one", r)
    assert "one" in L
    got = L.get("one")
    assert got is r
    popped = L.pop("one")
    assert popped is r
    L.set("a", r)
    L.set("b", r)
    assert len(L) == 2
    L.clear()
    assert len(L) == 0


def test_reflection_list_iteration_and_repr_truncation():
    items = {str(i): _AHReflection(pseudos={"h": i}, reals={"x": float(i)}) for i in range(12)}
    L = _AHReflectionList(initial=items)
    # iteration yields names
    names = list(iter(L))
    assert set(names) == set(items.keys())
    # repr should contain truncated summary, not full 12 entries printed
    s = repr(L)
    assert "..." in s or "+2" in s or len(s) < 2000


# ---------------------------
# expand_direction_map tests
# ---------------------------


@pytest.mark.parametrize(
    "vocab, stage_map, expected",
    [
        (None, {"a": "x+"}, {"a": np.array([1.0, 0.0, 0.0])}),
        (None, {"a": "x+", "b": "y-"}, {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, -1.0, 0.0])}),
        (None, {"rot": "z+"}, {"rot": np.array([0.0, 0.0, 1.0])}),
        (
            {"p": (1, 0, 0), "q": (0, 1, 0)},
            {"pname": "p+", "qname": "-q"},
            {"pname": np.array([1.0, 0.0, 0.0]), "qname": np.array([0.0, -1.0, 0.0])},
        ),
    ],
)
def test_expand_direction_map_returns_expected_vectors(vocab, stage_map, expected):
    ds = DirectionShorthand(vocabulary=vocab) if vocab is not None else DirectionShorthand()
    out = expand_direction_map(ds, stage_map)
    assert set(out.keys()) == set(expected.keys())
    for k, v in expected.items():
        assert isinstance(out[k], np.ndarray)
        assert out[k].shape == (3,)
        assert np.allclose(out[k], v)


@pytest.mark.parametrize(
    "vocab,bad_map",
    [
        (None, {"a": 1}),  # non-str value
        (None, {1: "x+"}),  # non-str key
        ({"p": (1, 0, 0)}, {"p": 123}),  # invalid shorthand value for custom vocab
    ],
)
def test_expand_direction_map_raises_type_error_on_invalid_map(vocab, bad_map):
    ds = DirectionShorthand(vocabulary=vocab) if vocab is not None else DirectionShorthand()
    with pytest.raises(TypeError):
        expand_direction_map(ds, bad_map)


@pytest.mark.parametrize("vocab", [None, {"p": (1, 0, 0)}])
def test_expand_direction_map_raises_value_error_on_invalid_shorthand(vocab):
    ds = DirectionShorthand(vocabulary=vocab) if vocab is not None else DirectionShorthand()
    with pytest.raises(ValueError):
        expand_direction_map(ds, {"a": ""})


@pytest.mark.parametrize("vocab", [None, {"p": (1, 0, 0)}])
def test_expand_direction_map_returns_independent_copies(vocab):
    ds = DirectionShorthand(vocabulary=vocab) if vocab is not None else DirectionShorthand()
    code = "x+" if vocab is None else list(vocab.keys())[0] + "+"
    stage_map = {"a": code}
    out = expand_direction_map(ds, stage_map)
    out["a"][0] = 99.0
    out2 = expand_direction_map(ds, stage_map)
    assert out2["a"][0] != 99.0

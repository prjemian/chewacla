"""
Chewacla: *ad hoc* diffractometer with rotation stages described by the caller.

This module provides a lightweight, dictionary-driven representation of a
diffractometer and supporting types used for small, interactive workflows.

Key classes
-----------
- :class:`~chewacla.adhoc.Chewacla`: *ad hoc* diffractometer.
- :class:`~chewacla.adhoc.AHReflection`: orienting reflection: (h,k,l) & angles (& optional wavelength).

Example
-------
Construct a simple *Chewacla* diffractometer and add a reflection:

.. code-block:: python
    :linenos:

    from chewacla.adhoc import Chewacla, AHReflection

    c = Chewacla({"s": "y+"}, {"d": "y+"})
    c.lattice = 1, 1, 1, 90, 90, 90
    r = AHReflection("one", {"h": 1, "k": 0, "l": 0}, {"s": 14.4, "d": 28.8})
    c.addReflection(r)

Utilities
---------
- ``expand_direction_map``: expand shorthand direction codes (``'x+'``, ``'y-'``)
    into unit vectors using the `DirectionShorthand` vocabulary.
"""

import numbers
import reprlib
from collections.abc import Mapping
from collections.abc import Sequence
from functools import wraps
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
from hklpy2.misc import IDENTITY_MATRIX_3X3

from chewacla.shorthand import DirectionMap
from chewacla.shorthand import DirectionMapInput
from chewacla.shorthand import DirectionShorthand
from chewacla.shorthand import DirectionVector
from chewacla.shorthand import DirectionVectorInput
from chewacla.shorthand import unit_vector
from chewacla.utils import stage_rotation_matrix

TAU = 2 * np.pi
"""Prefactor, either 1 or 2 pi"""

DEFAULT_WAVELENGTH = 1.54
"""Approximately copper K-alpha in angstroms."""
DEFAULT_LATTICE_PARAMS = (1, 1, 1, 90, 90, 90)
"""Trivial cubic crystal."""


# --- validators and decorator -------------------------------------------------
def _validate_length(name: str, val: float) -> None:
    if val <= 0:
        # keep the original message used in tests
        raise ValueError("lattice lengths a, b, c must be positive")


def _validate_angle(name: str, val: float) -> None:
    if not (0.0 < val < 180.0):
        raise ValueError(f"{name} must be in (0, 180) degrees")


def _validated_setter(attr_name: str, validator):
    """Decorator for property setters: convert to float, validate, set internal
    _<attr_name>, and invalidate cached self._B.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, value):
            val = float(value)
            validator(attr_name, val)
            setattr(self, f"_{attr_name}", val)
            # invalidate cached B
            self._B = None

        return wrapper

    return decorator


def _compare_axis_sets(kind: str, reflection_name: str, defined: set, given: set) -> None:
    """Compare two sets of axis/key names and raise ValueError on mismatch.

    Parameters
    ----------
    kind:
        'real' or 'pseudo' — selects wording for the error message.
    reflection_name:
        Name of the reflection for error context.
    defined:
        The canonical set of names expected by the instrument.
    given:
        The set of names provided by the reflection.

    Raises
    ------
    ValueError
        If the given set does not exactly match the defined set.
    """
    missing = sorted(list(defined - given))
    extra = sorted(list(given - defined))
    if not missing and not extra:
        return
    parts: list[str] = []
    if missing:
        parts.append(f"missing {('real axes' if kind == 'real' else 'pseudo keys')}: {missing}")
    if extra:
        parts.append(f"unexpected {('real axes' if kind == 'real' else 'pseudo keys')}: {extra}")
    if kind == "real":
        raise ValueError(
            f"Reflection {reflection_name!r} real-axis names do not match instrument axes: " + ", ".join(parts)
        )
    else:
        raise ValueError(f"Reflection {reflection_name!r} pseudos do not match expected keys: " + ", ".join(parts))


def expand_direction_map(
    ds: DirectionShorthand,
    stage_map: DirectionMapInput,
) -> Dict[str, np.ndarray]:
    """Return dict: {axis name: direction} where shorthand is direction, a unit vector.

    Example::

        ds = DirectionShorthand()
        rotation_axes = {"a": "x+", "b": "y-", "c": "z+"}
        expanded = expand_direction_map(ds, rotation_axes)
        # expanded -> {"a": array([1.,0.,0.]), "b": array([0.,-1.,0.]), ...}
    """
    out: Dict[str, np.ndarray] = {}
    for axis_name, code in stage_map.items():
        if not isinstance(axis_name, str):
            raise TypeError("stage_map keys must be str")
        if not isinstance(code, str):
            raise TypeError("stage_map values must be str")
        vec = ds.vector(code)
        out[axis_name] = np.asarray(vec, dtype=float)
    return out


class _AHLattice:
    """Internal container for lattice parameters and B-matrix computation."""

    a: float  # angstrom
    r"""Crystal unit cell length along $\hat x$ axis."""
    b: float  # angstrom
    r"""Crystal unit cell length at angle gamma from $\hat x$ in $\hat x$-$\hat y$ plane."""
    c: float  # angstrom
    """Crystal unit cell length"""
    alpha: float  # degrees
    r"""Angle between c axis and $\hat x$-$\hat y$ plane."""
    beta: float  # degrees
    """Angle between a and c axes in a-c plane."""
    gamma: float  # degrees
    r"""Angle between a and b axes in $\hat x$-$\hat y$ plane."""

    _B: Optional[np.ndarray]
    """Crystalline sample orientation matrix."""
    digits: int | None = None
    """Display precision (number of digits), default is full precision."""

    def __init__(
        self,
        a: numbers.Real,
        b: numbers.Real,
        c: numbers.Real,
        alpha: numbers.Real,
        beta: numbers.Real,
        gamma: numbers.Real,
    ) -> None:
        self._B = None  # initial default is unset

        # convert and validate lengths
        # use property setters (which validate and invalidate _B)
        self.a = a
        self.b = b
        self.c = c

        # convert and validate angles (degrees)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # --- lattice parameter properties ---------------------------------
    @property
    def a(self) -> float:
        return self._a

    @a.setter
    @_validated_setter("a", _validate_length)
    def a(self, value: numbers.Real) -> None:
        # validation and setting handled by decorator
        pass

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    @_validated_setter("b", _validate_length)
    def b(self, value: numbers.Real) -> None:
        # validation and setting handled by decorator
        pass

    @property
    def c(self) -> float:
        return self._c

    @c.setter
    @_validated_setter("c", _validate_length)
    def c(self, value: numbers.Real) -> None:
        # validation and setting handled by decorator
        pass

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    @_validated_setter("alpha", _validate_angle)
    def alpha(self, value: numbers.Real) -> None:
        # validation and setting handled by decorator
        pass

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    @_validated_setter("beta", _validate_angle)
    def beta(self, value: numbers.Real) -> None:
        # validation and setting handled by decorator
        pass

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    @_validated_setter("gamma", _validate_angle)
    def gamma(self, value: numbers.Real) -> None:
        # validation and setting handled by decorator
        pass

    @property
    def B(self) -> np.ndarray:
        """Return the lattice B matrix (3x3 ndarray) [per BL67].

        * BL67: Busing&Levy Acta Cyst. 22, 457 (1967)
        * https://repo.or.cz/hkl.git/blob/HEAD:/hkl/hkl-lattice.c

        Compute the reciprocal-lattice B matrix from unit-cell parameters.
        This matrix depends only on the crystal cell parameters.

        Input Units:

        * angles alpha, beta, gamma are given in degrees
        * a, b, c are in the same units as the wavelength

        Returns:
            B: (3,3) numpy.ndarray (float64)

        Raises:
            ValueError: if the cell is degenerate (invalid angles/parameters).

        Note:
            This implementation uses the Busing & Levy convention and includes the
            factor 2π (TAU) so that G = B @ h has units of Å⁻¹.
        """
        if self._B is None:
            # compute B from lattice parameters
            if self.a <= 0 or self.b <= 0 or self.c <= 0:
                raise ValueError("a,b,c must be > 0")

            # convert degrees -> radians once
            rad = np.pi / 180.0
            alpha_r = self.alpha * rad
            beta_r = self.beta * rad
            gamma_r = self.gamma * rad

            ca = np.cos(alpha_r)
            cb = np.cos(beta_r)
            cg = np.cos(gamma_r)
            sa = np.sin(alpha_r)
            sb = np.sin(beta_r)
            sg = np.sin(gamma_r)

            # underflow
            tol = 10 ** (-(self.digits if self.digits is not None else 12))

            # metric determinant factor
            D_val = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
            if D_val <= tol:
                raise ValueError(f"Invalid unit cell (D <= {tol}): D={D_val}")
            inv_sqrtD = 1.0 / np.sqrt(D_val)

            # guard against near-zero sines (angles near 0 or 180 deg)
            if (abs(sg) < tol) or (abs(sb) < tol) or (abs(sa) < tol):
                raise ValueError("Unit cell angles produce near-zero sine terms; degenerate cell")

            # precompute scaled denominators
            inv_a = inv_sqrtD / self.a
            inv_b = inv_sqrtD / self.b
            inv_c = inv_sqrtD / self.c

            # Canonical upper-triangular B matrix giving reciprocal vectors in Cartesian (Å^-1)
            # following Busing & Levy conventions with 2π factor (TAU)
            B = TAU * np.array(
                [
                    [inv_a, -cg * inv_a / sg, (cb * cg - ca) * inv_a / (sg * sb)],
                    [0.0, inv_b / sg, (ca * cg - cb) * inv_b / (sg * sb)],
                    [0.0, 0.0, sa * inv_c / (sb * sg)],
                ],
                dtype=np.float64,
            )

            self._B = B
        return self._B.copy()

    def to_dict(self, digits: Optional[int] = None) -> Dict[str, float]:
        """Return lattice constants as a dict.

        If `digits` is an int, numeric values are rounded to that many decimal places.
        If `digits` is None (default), full precision floats are returned.
        """

        def fmt(x: float) -> float:
            return float(round(x, digits)) if isinstance(digits, int) else float(x)

        return {
            "a": fmt(self.a),
            "b": fmt(self.b),
            "c": fmt(self.c),
            "alpha": fmt(self.alpha),
            "beta": fmt(self.beta),
            "gamma": fmt(self.gamma),
        }

    def __repr__(self) -> str:
        """Nice text representation."""
        vals = self.to_dict(self.digits)
        body = ", ".join(f"{k}={v}" for k, v in vals.items())
        return f"{self.__class__.__name__}({body})"


class AHReflection:
    """Orienting reflection used by the AdHocDiffractometer."""

    _name: str
    """(internal) name given to this reflection by the caller."""
    _pseudos: Dict[str, float]
    """(internal) dict of pseudo-axis names to float values (e.g., h,k,l)."""
    _reals: Dict[str, float]
    """(internal) dict of real-axis names to float values (diffractometer angles)."""
    _wavelength: float
    """(internal) Wavelength associated with this reflection (in length units, typically Å)."""

    def __init__(
        self,
        name: str,
        pseudos: Mapping[str, float],
        reals: Mapping[str, float],
        wavelength: Optional[float] = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        self._name = name

        # use the public setters to validate and store the mappings
        self.pseudos = pseudos
        self.reals = reals
        self.wavelength = DEFAULT_WAVELENGTH if wavelength is None else float(wavelength)

    @property
    def name(self) -> str:
        """Name given to this reflection (read-only)."""
        return self._name

    # --- wavelength property -------------------------------------------------
    @property
    def wavelength(self) -> float:
        """Wavelength (float)."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        val = float(value)
        if val <= 0:
            raise ValueError("wavelength must be positive")
        self._wavelength = val

    # --- pseudos property ---------------------------------------------------

    @property
    def pseudos(self) -> Mapping[str, float]:
        """Read-only view of pseudos (hkl) mapping."""
        return dict(self._pseudos)

    @pseudos.setter
    def pseudos(self, value: Mapping[str, float]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("pseudos must be a Mapping[str, numeric]")
        new: Dict[str, float] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("pseudos keys must be strings")
            if not isinstance(v, numbers.Real):
                raise TypeError("pseudos values must be numeric")
            new[k] = float(v)
        self._pseudos = new

    def set_pseudo(self, name: str, value: float) -> None:
        """Set a single pseudo value."""
        self._pseudos[name] = float(value)

    def get_pseudo(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Get a single pseudo value or default if missing."""
        return self._pseudos.get(name, default)

    def remove_pseudo(self, name: str) -> None:
        """Remove a pseudo entry; KeyError if absent."""
        del self._pseudos[name]

    # --- reals property -----------------------------------------------------
    @property
    def reals(self) -> Mapping[str, float]:
        """Read-only view of reals mapping."""
        return dict(self._reals)

    @reals.setter
    def reals(self, value: Mapping[str, float]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("reals must be a Mapping[str, numeric]")
        new: Dict[str, float] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("reals keys must be strings")
            if not isinstance(v, numbers.Real):
                raise TypeError("reals values must be numeric")
            new[k] = float(v)
        self._reals = new

    def set_real(self, name: str, value: float) -> None:
        """Set a single real value."""
        self._reals[name] = float(value)

    def get_real(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Get a single real value or default if missing."""
        return self._reals.get(name, default)

    def remove_real(self, name: str) -> None:
        """Remove a real entry; KeyError if absent."""
        del self._reals[name]

    # --- representation and equality ----------------------------------------
    def __repr__(self) -> str:
        """Nice, truncated text representation for long dicts."""
        r = reprlib.Repr()
        r.maxlist = 10
        body = [
            f"name={self._name!r}",
            f"pseudos={r.repr(self.pseudos)}",
            f"reals={r.repr(self.reals)}",
            f"wavelength={self._wavelength!r}",
        ]
        return f"{self.__class__.__name__}({body})"

    def __eq__(self, other: object) -> bool:
        """Compare with 'other' reflection for equality with numeric tolerance."""
        if not isinstance(other, AHReflection):
            return NotImplemented

        # compare keys using public properties
        if set(self.pseudos) != set(other.pseudos):
            return False
        if set(self.reals) != set(other.reals):
            return False

        # compare values with tolerance
        for k in self.pseudos:
            if not np.isclose(self.pseudos[k], other.pseudos[k]):
                return False
        for k in self.reals:
            if not np.isclose(self.reals[k], other.reals[k]):
                return False

        return bool(np.isclose(self._wavelength, other._wavelength))

    def validate_against(self, expected_reals: Optional[Sequence[str]] = None, expected_pseudos: Optional[Sequence[str]] = None) -> None:
        """Validate this reflection's keys against expected instrument axes.

        Parameters
        ----------
        expected_reals:
            Sequence of real-axis names expected by the instrument (e.g. sample/detector axes).
        expected_pseudos:
            Sequence of pseudo-axis keys expected (usually 'h','k','l').

        Raises
        ------
        ValueError
            If the reflection's keys do not exactly match the expected sets.
        """
        # Use the public properties rather than internal attributes to preserve
        # encapsulation. `self.pseudos` and `self.reals` return dict copies,
        # so converting them to sets yields the key sets expected by
        # `_compare_axis_sets`.
        if expected_pseudos is not None:
            _compare_axis_sets("pseudo", self.name, set(expected_pseudos), set(self.pseudos))
        if expected_reals is not None:
            _compare_axis_sets("real", self.name, set(expected_reals), set(self.reals))


class Chewacla:
    """
    The *ad hoc* diffractometer with stages as described by a dictionary.

    Notes
    -----
    - Reflections are managed by name. Use :meth:`addReflection` to add an
        :class:`AHReflection` to the instrument. By default adding a reflection
        with a name that already exists raises ``ValueError`` to prevent
        accidental overwrites; pass ``force=True`` to replace an existing
        reflection deliberately.
    """

    _incident_beam: DirectionVector
    """Unit vector describing the direction of the incident beam."""
    _sample_stage: DirectionMap
    """names and unit vectors for sample stage rotations"""
    _detector_stage: DirectionMap
    """names and unit vectors for detector stage rotations"""
    _lattice: _AHLattice
    """crystal lattice parameters (angstroms and degrees)"""
    _wavelength: float
    """Wavelength of incident beam (angstroms)"""
    _reflections: Dict[str, AHReflection]
    """Orienting reflections (name -> AHReflection) to be used in computation of UB matrix."""
    U: np.ndarray
    """Goniometer orientation matrix"""
    UB: np.ndarray
    """Crystal orientation matrix"""

    _ds: DirectionShorthand
    """(internal) DirectionShorthand object"""

    def __init__(
        self,
        sample_stage: DirectionMapInput,
        detector_stage: DirectionMapInput,
        wavelength: Optional[float] = None,
        incident_beam: Optional[DirectionVectorInput] = None,
        direction_map: Optional[DirectionMapInput] = None,
    ) -> None:
        # Initialize internal direction shorthand
        self._ds = DirectionShorthand(direction_map)

        # raw inputs preserved for repr/debug
        self.raw_incident_beam = None
        self.raw_sample_stage = None
        self.raw_detector_stage = None

        # Set basic instrument descriptors. Use provided values or sensible defaults.
        self.incident_beam = "+x" if incident_beam is None else incident_beam
        self.sample_stage = sample_stage
        self.detector_stage = detector_stage
        self.wavelength = DEFAULT_WAVELENGTH if wavelength is None else float(wavelength)

        self.mode = self.modes[0]  # first one is the default diffractometer mode
        # Lattice defaults and reflection storage
        self.lattice = DEFAULT_LATTICE_PARAMS
        self._reflections = {}  # no reflections defined yet

        # Orientation matrices default to identity
        self.U = np.asarray(IDENTITY_MATRIX_3X3)
        self.UB = np.asarray(IDENTITY_MATRIX_3X3)

    def __repr__(self) -> str:
        """Nice text representation."""
        body = [
            f"incident_beam={self.raw_incident_beam!r}",
            f"sample_stage={self.raw_sample_stage!r}",
            f"detector_stage={self.raw_detector_stage!r}",
            f"mode={self.mode!r}",
        ]
        return f"{self.__class__.__name__}({', '.join(body)})"

    def addReflection(self, reflection: AHReflection, force: bool = False) -> None:
        """Add a single reflection using its own `name` attribute as the key.

        Parameters:

        - reflection: an already-constructed `AHReflection` instance with a valid
          `name` attribute (str). The reflection's `name` will be used as the key
          when storing it in the manager.
        """
        if not isinstance(reflection, AHReflection):
            raise TypeError("reflection must be an AHReflection instance")
        name = reflection.name
        if not isinstance(name, str):
            raise TypeError("reflection.name must be a str")

        # validate against instrument axes (raises ValueError on mismatch)
        reflection.validate_against(self.real_axis_names, self.pseudo_axis_names)

        # prevent accidental overwrites unless explicitly forced
        if (name in self._reflections) and not force:
            raise ValueError(f"Reflection {name!r} already exists; pass force=True to replace")

        # finally store/replace the reflection in the dict
        self._reflections[name] = reflection

    def make_reflection(self, name: str, pseudos: Mapping[str, float], reals: Mapping[str, float], wavelength: Optional[float] = None) -> AHReflection:
        """Factory: construct and validate an AHReflection using this instrument's axes.

        The returned AHReflection has already been validated against the Chewacla
        `real_axis_names` and `pseudo_axis_names`.
        """
        wl = self.wavelength if wavelength is None else float(wavelength)
        r = AHReflection(name, pseudos, reals, wl)
        r.validate_against(self.real_axis_names, self.pseudo_axis_names)
        return r

    def calc_UB_BL67(self) -> np.ndarray:
        """Calculate the U & UB matrices from the given reflections."""
        if len(self.reflections) != 2:
            raise ValueError(
                "Busing & Levy method requires exactly two reflections to compute"
                f" UB matrix. {len(self.reflections)} reflection(s) are defined."
            )
        # TODO: proceed
        return self.UB

    def forward(self, pseudos: Dict[str, float]) -> Sequence[Dict[str, float]]:
        return [{}]  # TODO:

    def inverse(self, reals: Dict[str, float]) -> Dict[str, float]:
        return {}  # TODO:

    # -------------- internal methods

    def _sample_rotation_matrix(self, axes: Mapping[str, float]) -> np.ndarray:
        """Rotation of the sample motors into the lab coordinates.
        
        Parameters
        ----------

        axes:
            Dictionary of sample stage axis names to angles in degrees.
        """
        # Validate keys (tests expect a specific message wording)
        if not isinstance(axes, Mapping):
            raise TypeError("axes must be a mapping of axis-name -> angle_degrees")
        defined = set(self.sample_stage.keys())
        given = set(axes.keys())
        missing = sorted(list(defined - given))
        extra = sorted(list(given - defined))
        if missing or extra:
            parts: list[str] = []
            if missing:
                parts.append(f"missing sample axes: {missing}")
            if extra:
                parts.append(f"unexpected sample axes: {extra}")
            raise ValueError("; ".join(parts))

        # Delegate to the shared utility implementation
        return stage_rotation_matrix(self.sample_stage, axes)

    # -------------- getter/setter property methods

    @property
    def detector_stage(self) -> DirectionMap:
        """Describes the detector stage stack of rotations."""
        return self._detector_stage

    @detector_stage.setter
    def detector_stage(self, value: DirectionMapInput) -> None:
        """Accept mapping-like; minimal conversion here."""
        self.raw_detector_stage = value
        self._detector_stage = expand_direction_map(self._ds, value)

    @property
    def incident_beam(self) -> DirectionVector:
        """Unit vector describing the direction of the incident beam."""
        return self._incident_beam

    @incident_beam.setter
    def incident_beam(self, raw: DirectionVectorInput) -> None:
        """Define the direction of the incident beam."""
        value = raw
        self.raw_incident_beam = raw
        if isinstance(value, str):
            value = self._ds.vector(value)
        if isinstance(value, (Sequence | np.ndarray)):
            self._incident_beam = value
            return
        self._incident_beam = unit_vector(value)

    @property
    def lattice(self) -> _AHLattice:
        """Return the Lattice object."""
        return self._lattice

    @lattice.setter
    def lattice(self, lattice_constants) -> None:
        """Set the crystal lattice parameters."""
        self._lattice = _AHLattice(*lattice_constants)

    @property
    def modes(self) -> list[str]:
        """List of the supported diffractometer forward calculation modes."""
        return ["default"]  # TODO: expand

    @property
    def mode(self) -> list[str]:
        """List of the supported diffractometer forward calculation modes."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        """Set the current diffractometer forward calculation mode."""
        if value not in self.modes:
            raise ValueError(f"Invalid mode: {value}. Supported modes are: {self.modes}")
        self._mode = value

    @property
    def pseudo_axis_names(self) -> list[str]:
        """List of the pseudo axis names."""
        return "h k l".split()

    @property
    def real_axis_names(self) -> list[str]:
        """List of the real axis names."""
        keys = list(self.sample_stage)
        for k in self.detector_stage:
            if k not in keys:
                keys.append(k)
        return keys

    @property
    def reflections(self) -> dict:
        """Return the reflection mapping (name -> AHReflection)."""
        return self._reflections

    @reflections.setter
    def reflections(self, value: Optional[Mapping[str, AHReflection]]) -> None:
        """Define the orienting reflections.

        Behaviour:
        - None (or {}) clears the current reflections.
        - Any Mapping or iterable of pairs of reflections (name -> AHReflection) is converted is accepted (copied).
        """
        if value is None:
            self._reflections.clear()
            return

        if isinstance(value, Mapping):
            self._reflections = dict(value)
            return

        # accept iterable of (name, reflection) pairs
        try:
            items = dict(value)  # will raise if not iterable of pairs
        except Exception as exc:
            raise TypeError("reflections must be a dict or iterable of (name, reflection) pairs") from exc
        self._reflections = items

    @property
    def sample_stage(self) -> DirectionMap:
        """Describes the sample stage stack of rotations."""
        return self._sample_stage

    @sample_stage.setter
    def sample_stage(self, value: DirectionMapInput) -> None:
        """Accept mapping-like; minimal conversion here."""
        self.raw_sample_stage = value
        self._sample_stage = expand_direction_map(self._ds, value)

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Any) -> None:
        v = float(value)
        if v <= 0:
            raise ValueError("Wavelength must be positive")
        self._wavelength = v

"""
Chewacla: *ad hoc* diffractometer with stages as described by a dictionary.

..
    .. autosummary::

        ~Chewacla
        ~expand_direction_map

    .. rubric:: Internal use only
    .. autosummary::

        ~_AHLattice
        ~_AHReflection
        ~_AHReflectionList
"""

import numbers
import reprlib
from collections.abc import Mapping
from collections.abc import Sequence
from functools import wraps
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple

import numpy as np
from hklpy2.misc import IDENTITY_MATRIX_3X3

from chewacla.shorthand import DirectionMap
from chewacla.shorthand import DirectionMapInput
from chewacla.shorthand import DirectionShorthand
from chewacla.shorthand import DirectionVector
from chewacla.shorthand import DirectionVectorInput
from chewacla.shorthand import unit_vector

TAU = 2 * np.pi
"""Prefactor, either 1 or 2 pi"""

DEFAULT_WAVELENGTH = 1.54


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
    """Crystal lattice of sample on the AdHocDiffractometer."""

    a: float  # angstrom
    b: float  # angstrom
    c: float  # angstrom
    alpha: float  # degrees
    beta: float  # degrees
    gamma: float  # degrees

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


# TODO: refactor AHReflection classes into Chewacla?


class AHReflection:  # TODO: needs to know diffractometer axes
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
        wavelength: float = DEFAULT_WAVELENGTH,
    ) -> None:
        # store name and internal copies (mutable dict) to avoid external modification surprises
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        self._name = name
        self.pseudos = dict(pseudos)
        self.reals = dict(reals)
        self.wavelength = float(wavelength)

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
            f"pseudos={r.repr(self._pseudos)}",
            f"reals={r.repr(self._reals)}",
            f"wavelength={self._wavelength!r}",
        ]
        return f"{self.__class__.__name__}({body})"

    def __eq__(self, other: object) -> bool:
        """Compare with 'other' reflection for equality with numeric tolerance."""
        if not isinstance(other, AHReflection):
            return NotImplemented

        # compare keys
        if set(self._pseudos.keys()) != set(other._pseudos.keys()):
            return False
        if set(self._reals.keys()) != set(other._reals.keys()):
            return False

        # compare values with tolerance
        for k in self._pseudos:
            if not np.isclose(self._pseudos[k], other._pseudos[k]):
                return False
        for k in self._reals:
            if not np.isclose(self._reals[k], other._reals[k]):
                return False

        return bool(np.isclose(self._wavelength, other._wavelength))


class _AHReflectionList:
    """Manage orienting reflections as named _AHReflection objects.

    Behaviors:
    - Stores reflections in an internal dict.
    - Supports get/set by key, deletion, iteration, length, clear.
    - Provides a concise, readable repr that truncates long contents.
    """

    def __init__(self, initial: Optional[Mapping[str, AHReflection]] = None) -> None:
        if initial is None:
            self._items: Dict[str, AHReflection] = {}
        else:
            # copy to prevent external mutation
            self._items = dict(initial)

    def get(self, name: str) -> AHReflection:
        """Return the reflection for name or raise KeyError if missing."""
        return self._items[name]

    def set(self, name: str, reflection: AHReflection) -> None:
        """Set a reflection by name."""
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        if not isinstance(reflection, AHReflection):
            raise TypeError("reflection must be an _AHReflection instance")
        self._items[name] = reflection

    def pop(self, name: str) -> AHReflection:
        """Remove and return the reflection for name; raises KeyError if absent."""
        return self._items.pop(name)

    def clear(self) -> None:
        """Remove all stored reflections."""
        self._items.clear()

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def items(self) -> Iterator[Tuple[str, AHReflection]]:
        return iter(self._items.items())

    def names(self) -> Iterator[str]:
        """Just the names of the reflections."""
        return iter(self._items.keys())

    def values(self) -> Iterator[AHReflection]:
        return iter(self._items.values())

    def __repr__(self) -> str:
        """Nice text representation."""
        r = reprlib.Repr()
        r.maxlist = 10
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in list(self._items.items())[:10])
        more = "" if len(self._items) <= 10 else f", ... (+{len(self._items) - 10} more)"
        return f"{self.__class__.__name__}({{{inner}{more}}})"


class Chewacla:
    """
    The *ad hoc* diffractometer with stages as described by a dictionary.
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
    _reflections: _AHReflectionList
    """Orienting reflections to be used in computation of UB matrix."""
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
        self._ds = DirectionShorthand(direction_map)

        self.raw_incident_beam = None
        self.raw_sample_stage = None
        self.raw_detector_stage = None

        self.incident_beam = "+x" if incident_beam is None else incident_beam
        self.sample_stage = sample_stage
        self.detector_stage = detector_stage
        self.wavelength = 1.54 if wavelength is None else float(wavelength)

        self.lattice = (1, 1, 1, 90, 90, 90)
        self.reflections = _AHReflectionList()
        self.U = np.asarray(IDENTITY_MATRIX_3X3)
        self.UB = np.asarray(IDENTITY_MATRIX_3X3)

    def __repr__(self) -> str:
        """Nice text representation."""
        body = [
            f"incident_beam={self.raw_incident_beam!r}",
            f"sample_stage={self.raw_sample_stage!r}",
            f"detector_stage={self.raw_detector_stage!r}",
        ]
        return f"{self.__class__.__name__}({', '.join(body)})"

    def addReflection(self, reflection: AHReflection) -> None:
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

        self.reflections.set(name, reflection)

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

    # TODO: mode property getter & setter

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
    def reflections(self) -> _AHReflectionList:
        """Return the reflection manager for name->_AHReflection access."""
        return self._reflections

    @reflections.setter
    def reflections(self, value: Optional[Mapping[str, AHReflection]]) -> None:
        """Replace reflection storage. Accepts _AHReflectionList, mapping, or None."""
        if value is None:
            self._reflections.clear()
            return

        if isinstance(value, _AHReflectionList):
            # copy internal dict to avoid aliasing
            self._reflections = _AHReflectionList(dict(value._items))  # type: ignore[attr-defined]
            return

        if isinstance(value, Mapping):
            self._reflections = _AHReflectionList(value)
            return

        # accept iterable of (_AHReflection) by requiring (name, reflection) pairs
        try:
            items = dict(value)  # will raise if not iterable of pairs
        except Exception as exc:
            raise TypeError(
                "reflections must be an _AHReflectionList, mapping,"
                #
                " or iterable of (name, reflection) pairs"
            ) from exc
        self._reflections = _AHReflectionList(items)

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

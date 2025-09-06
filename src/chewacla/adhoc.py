"""
Chewacla: *ad hoc* diffractometer with stages as described by a dictionary.

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
from collections.abc import Mapping as _ABCMapping
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Tuple

import numpy as np
from hklpy2.misc import IDENTITY_MATRIX_3X3
from numpy.typing import NDArray

import chewacla.lattice
from chewacla.shorthand import DirectionMap
from chewacla.shorthand import DirectionShorthand
from chewacla.shorthand import DirectionVector
from chewacla.shorthand import x_hat


class _AHLattice:
    """Crystal lattice of sample on the AdHocDiffractometer."""

    a: float  # angstrom
    b: float  # angstrom
    c: float  # angstrom
    alpha: float  # degrees
    beta: float  # degrees
    gamma: float  # degrees
    # TODO: B should update if any of these change
    
    _B: Optional[NDArray]
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
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            raise ValueError("lattice lengths a, b, c must be positive")

        # convert and validate angles (degrees)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        for name, val in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
        ):
            if not (0.0 < val < 180.0):
                raise ValueError(f"{name} must be in (0, 180) degrees")

    @property
    def B(self) -> np.ndarray:
        """Return a copy of the lattice B matrix (3x3 ndarray)."""
        if self._B is None:
            self._B = np.asarray(
                chewacla.lattice.lattice_B(
                    self.a,
                    self.b,
                    self.c,
                    self.alpha,
                    self.beta,
                    self.gamma,
                ),
                dtype=float,
            )
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


class _AHReflection:
    """Orienting reflection used only by the AdHocDiffractometer.

    ..
        Attributes
        ----------
        _pseudos
            Mapping of pseudo-axis names to their values (units depend on usage).
        _reals
            Mapping of real-axis names to their values (units depend on usage).
        _wavelength
            Wavelength associated with this reflection (in same length units as the
            rest of the system, typically Angstroms).
    """

    _pseudos: MutableMapping[str, float]
    """Ordered dictionary of $hkl$ values."""
    _reals: MutableMapping[str, float]
    """Ordered dictionary of diffractometer rotation axis values."""
    _wavelength: float
    """Wavelength (angstrom) of the incident radiation."""

    def __init__(
        self,
        pseudos: Optional[Mapping[str, float]] = None,
        reals: Optional[Mapping[str, float]] = None,
        wavelength: float = 1.0,
    ) -> None:
        # store internal copies (mutable) to avoid external modification surprises
        self._pseudos: MutableMapping[str, float] = dict(pseudos or {})
        self._reals: MutableMapping[str, float] = dict(reals or {})
        self._wavelength: float = float(wavelength)

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
        # use collections.abc.Mapping for runtime isinstance checks
        if not isinstance(value, _ABCMapping):
            raise TypeError("pseudos must be a Mapping[str, numeric]")
        new: MutableMapping[str, float] = {}
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
        # use collections.abc.Mapping for runtime isinstance checks
        if not isinstance(value, _ABCMapping):
            raise TypeError("reals must be a Mapping[str, numeric]")
        new: MutableMapping[str, float] = {}
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
        """Nice text representation."""
        return (
            f"{self.__class__.__name__}("
            f"pseudos={self._pseudos!r}, reals={self._reals!r}, wavelength={self._wavelength!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare with 'other' reflection for equality."""
        if not isinstance(other, _AHReflection):
            return NotImplemented
        return (
            self._pseudos == other._pseudos
            and self._reals == other._reals
            and np.isclose(self._wavelength, other._wavelength)
        )


class _AHReflectionList:
    """Manage a mapping of name -> _AHReflection.

    Behaviors:
    - Stores reflections in an internal dict.
    - Supports get/set by key, deletion, iteration, length, clear.
    - Provides a concise, readable repr that truncates long contents.
    """

    def __init__(self, initial: Optional[_ABCMapping[str, _AHReflection]] = None) -> None:
        if initial is None:
            self._items: MutableMapping[str, _AHReflection] = {}
        else:
            # copy to prevent external mutation
            self._items = dict(initial)

    def get(self, name: str) -> _AHReflection:
        """Return the reflection for name or raise KeyError if missing."""
        return self._items[name]

    def set(self, name: str, reflection: _AHReflection) -> None:
        """Set a reflection by name."""
        if not isinstance(name, str):
            raise TypeError("name must be a str")
        if not isinstance(reflection, _AHReflection):
            raise TypeError("reflection must be an _AHReflection instance")
        self._items[name] = reflection

    def pop(self, name: str) -> _AHReflection:
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

    def items(self) -> Iterator[Tuple[str, _AHReflection]]:
        return iter(self._items.items())

    def names(self) -> Iterator[str]:
        """Just the names of the reflections."""
        return iter(self._items.keys())

    def values(self) -> Iterator[_AHReflection]:
        return iter(self._items.values())

    def __repr__(self) -> str:
        """Nice text representation."""
        # use reprlib to avoid extremely long output
        r = reprlib.Repr()
        r.maxlist = 10
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in list(self._items.items())[:10])
        more = "" if len(self._items) <= 10 else f", ... (+{len(self._items) - 10} more)"
        return f"{self.__class__.__name__}({{{inner}{more}}})"


def expand_direction_map(
    ds: DirectionShorthand,
    stage_map: Mapping[str, str],
) -> Dict[str, np.ndarray]:
    """Return dict: {axis name: direction} where shorthand is direction is unit vector.

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
    U: NDArray
    """Goniometer orientation matrix"""
    UB: NDArray
    """Crystal orientation matrix"""

    _ds: DirectionShorthand
    """(internal) DirectionShorthand object"""

    # TODO: Apply consistent use of "unit vector" term throughout

    def __init__(
        self,
        sample_stage: Any,
        detector_stage: Any,
        wavelength: Optional[float] = None,
        incident_beam: Optional[DirectionVector] = None,
        direction_map: DirectionMap = None,
    ) -> None:
        # +x unit vector  # TODO: use "+x" literal
        self.incident_beam = x_hat if incident_beam is None else incident_beam
        self.sample_stage = sample_stage
        self.detector_stage = detector_stage
        self.wavelength = 1.54 if wavelength is None else float(wavelength)

        self._ds = DirectionShorthand(direction_map)
        self.lattice = (1, 1, 1, 90, 90, 90)
        self.reflections = []
        self.U = IDENTITY_MATRIX_3X3
        self.UB = IDENTITY_MATRIX_3X3

    def calc_UB_BL67(self) -> NDArray:
        """Calculate the U & UB matrices from the given reflections."""
        if len(self.reflections) != 2:
            raise ValueError(
                "Busing & Levy method requires exactly two reflections to compute"
                f" UB matrix. {len(self.reflections)} reflection(s) are defined."
            )
        # TODO: proceed

    # -------------- getter/setter property methods

    @property
    def detector_stage(self) -> DirectionMap:
        """Describes the detector stage stack of rotations."""
        return self._detector_stage

    @detector_stage.setter
    def detector_stage(self, value: Any) -> None:
        """Accept mapping-like; minimal conversion here."""
        self._detector_stage = self.expand_direction_map(self._ds, value)

    @property
    def incident_beam(self) -> DirectionVector:
        """Unit vector describing the direction of the incident beam."""
        return self._incident_beam

    @incident_beam.setter
    def incident_beam(self, value: Any) -> None:
        """Accept DirectionVector or array-like of shape (3,)."""
        if isinstance(value, DirectionVector):
            self._incident_beam = value
            return
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError("incident_beam must be of shape (3,)")
        self._incident_beam = DirectionVector(arr)  # type: ignore[arg-type]

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
    def reflections(self, value: Optional[Mapping[str, _AHReflection]]) -> None:
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
    def sample_stage(self, value: Any) -> None:
        """Accept mapping-like; minimal conversion here."""
        self._sample_stage = self.expand_direction_map(self._ds, value)

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Any) -> None:
        # TODO: rule: must be a positive number
        self._wavelength = float(value)

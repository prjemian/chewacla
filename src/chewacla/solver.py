"""
Solver for hklpy2.

.. substitutions file is at the root of the docs
.. include:: /substitutions.txt

.. autosummary::

    ~ChewaclaSolver
"""

from typing import Mapping
from typing import Sequence

from hklpy2 import SolverBase
from hklpy2.backends.base import Lattice
from hklpy2.backends.base import Reflection
from hklpy2.backends.base import Sample
from hklpy2.misc import IDENTITY_MATRIX_3X3, istype
from pyRestTable import Table

from chewacla.bl1967 import calcUB_BusingLevy
from chewacla.shorthand import DirectionMap
from chewacla.shorthand import DirectionVector
from chewacla.lattice import lattice_B


class ChewaclaSolver(SolverBase):
    """Solver for hklpy2."""

    # https://github.com/bluesky/hklpy2/blob/main/hklpy2/backends/base.py

    from chewacla import __version__

    name = "chewacla"
    """Name of this Solver."""

    version = __version__
    """Version of this Solver."""

    _geometries: Mapping = {}

    @classmethod
    def addGeometry(
        cls,
        key: str,
        incident: DirectionVector | str,
        sample: DirectionMap,
        detector: DirectionMap,
    ) -> None:
        """
        Add/update a geometry named 'key'.

        Add geometries before creating a diffractometer object.

        PARAMETERS

        key *str* :
            Name of the geometry.  Should be a single word.
        incident : DirectionVector | str
            Direction of the incident beam.  Either an array of float with shape (3,)
            or a shorthand (interpreted by :class:`chewacla.shorthand.DirectionShorthand`).
        sample : dict[name, DirectionVector | str]
            Describes the stack of rotational axes for the sample stage.
            Keys are the names of the axes, values are the the rotation axis.
            The order is significant, starting from the outermost rotation.
        detector : dict[name, DirectionVector | str]
            Describes the stack of rotational axes for the detector stage.
            Keys are the names of the axes, values are the the rotation axis.
            The order is significant, starting from the outermost rotation.

        Axis names can be duplicated in the sample and detector, indicating they
        share the same axis.

        EXAMPLES::

            >>> import hklpy2
            >>> SolverClass = hklpy2.get_solver("chewacla")
            >>> SolverClass.addGeometry(
                "fourc",
                incident="x+",
                sample=dict(omega="z+", chi="x+", phi="z+"),
                detector=dict(ttheta="z+"),
            )
            >>> SolverClass.addGeometry(
                "mys2d2",
                incident=(0, 0, 1),
                sample=dict(omega="x-", chi="z+"),
                detector=dict(delta="x-", gamma="z+"),
            )
            >>> SolverClass.geometries()
            ["fourc", "mys2d2"]
            >>> fourc = SolverClass("fourc")
            >>> mys2d2 = SolverClass("mys2d2")

        """
        cls._geometries[key] = dict(
            incident=incident,
            sample=sample,
            detector=detector,
        )

    def addReflection(self, reflection: Reflection) -> None:
        """Add coordinates of a diffraction condition (a reflection)."""
        # TODO

    def calculate_UB(
        self,
        r1: Reflection,
        r2: Reflection,
    ) -> list[list[float]]:
        geom = self.geometry_structure
        if "sample" in geom:
            sample_axes = geom["sample"]
            self.UB = calcUB_BusingLevy(r1, r2, self.B, sample_axes)
        return self.UB

    @property
    def extra_axis_names(self) -> list[str]:
        """Ordered list of any extra axis names (such as x, y, z)."""
        # Do NOT sort.
        return []  # no extra axes

    def forward(self, pseudos: dict) -> Sequence[Mapping[str, float]]:
        """Compute list of solutions(reals) from pseudos (hkl -> [angles])."""
        # based on geometry and mode
        return [{}]  # TODO

    @classmethod
    def geometries(cls) -> Sequence[str]:
        """
        Ordered list of the geometry names.

        EXAMPLES::

            >>> import hklpy2
            >>> SolverClass = hklpy2.get_solver("chewacla")
            >>> SolverClass.addGeometry("example", ...)
            >>> solver = Solver("chewacla")
            >>> solver.geometries()
            ["example"]
        """
        return list(cls._geometries.keys())

    @property
    def geometry_structure(self) -> Mapping:
        """Return the current geometry dict."""
        return self._geometries.get(self.geometry, {})

    def inverse(self, reals: dict) -> Mapping[str, float]:
        """Compute dict of pseudos from reals (angles -> hkl)."""
        ...  # TODO
        return {}

    @property
    def lattice(self) -> Lattice:
        """
        Crystal lattice parameters.
        """
        return self._lattice

    @lattice.setter
    def lattice(self, value: Lattice):
        if not istype(value, Lattice):
            raise TypeError(f"Must supply {Lattice} object, received {value!r}")
        self._lattice = value
        self.B = lattice_B(**value)

    @property
    def modes(self) -> Sequence[str]:
        """List of the geometry operating modes."""
        return ["default"]

    @property
    def pseudo_axis_names(self) -> Sequence[str]:
        """Ordered list of the pseudo axis names (such as h, k, l)."""
        # Do NOT sort.
        return "h k l".split()

    @property
    def real_axis_names(self) -> Sequence[str]:
        """Ordered list of the real axis names (such as th, tth)."""
        geom = self.geometry_structure
        if len(geom) == 0:
            return []

        names = list(geom["sample"])
        for key in geom["detector"]:
            if key not in names:
                names.append(key)
        return names

    def refineLattice(self, reflections: Sequence[Reflection]) -> Lattice:
        """Refine the lattice parameters from a list of reflections."""
        raise NotImplementedError("Lattice parameter refinement not implemented.")

    def removeAllReflections(self) -> None:
        """Remove all reflections."""
        ...  # TODO

    @property
    def sample(self) -> object:
        """
        Crystalline sample.
        """
        return self._sample

    @sample.setter
    def sample(self, value: Sample):
        if not istype(value, Sample):
            raise TypeError(f"Must supply {Sample} object, received {value!r}")
        self._sample = value
        # TODO: structure of the Sample object is not well-defined
        lattice = value.get("lattice")
        if isinstance(lattice, Mapping):
            self.lattice = lattice

    @property
    def summary(self) -> Table:
        """
        Table of this geometry (modes, axes).

        .. seealso:: https://blueskyproject.io/hklpy2/diffractometers.html#available-solver-geometry-tables
        """
        table = Table()
        table.labels = "mode pseudo(s) real(s) writable(s) extra(s)".split()
        # TODO:
        # sdict = self._summary_dict
        # for mode_name, mode in sdict["modes"].items():
        #     self.mode = mode_name
        #     row = [
        #         mode_name,
        #         ", ".join(sdict["pseudos"]),
        #         ", ".join(sdict["reals"]),
        #         ", ".join(mode["reals"]),
        #         ", ".join(mode["extras"]),
        #     ]
        #     table.addRow(row)
        return table

    @property
    def UB(self):
        """Orientation matrix (3x3)."""
        if "_UB" not in dir(self):
            self._UB = IDENTITY_MATRIX_3X3
        return self._UB

    @UB.setter
    def UB(self, matrix):
        """Orientation matrix (3x3)."""
        self._UB = matrix

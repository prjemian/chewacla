"""
Solver for hklpy2.

.. substitutions file is at the root of the docs
.. include:: /substitutions.txt

.. autosummary::

    ~ChewaclaSolver
"""

from hklpy2 import SolverBase
from hklpy2.blocks.lattice import Lattice
from hklpy2.blocks.reflection import Reflection
from hklpy2.misc import IDENTITY_MATRIX_3X3
from pyRestTable import Table


class ChewaclaSolver(SolverBase):
    """Solver for hklpy2."""

    # https://github.com/bluesky/hklpy2/blob/main/hklpy2/backends/base.py

    from . import __version__

    name = "chewacla"
    """Name of this Solver."""

    version = __version__
    """Version of this Solver."""

    def addReflection(self, reflection: Reflection) -> None:
        """Add coordinates of a diffraction condition (a reflection)."""
        # TODO

    def calculate_UB(
        self,
        r1: Reflection,
        r2: Reflection,
    ) -> list[list[float]]:
        # self.UB = calcUB_BusingLevy(r1, r2, sample axes dict...)  # TODO
        return self.UB

    @property
    def extra_axis_names(self) -> list[str]:
        """Ordered list of any extra axis names (such as x, y, z)."""
        # Do NOT sort.
        return []  # TODO

    def forward(self, pseudos: dict) -> list[dict[str, float]]:
        """Compute list of solutions(reals) from pseudos (hkl -> [angles])."""
        # based on geometry and mode
        return [{}]  # TODO

    @classmethod
    def geometries(cls) -> list[str]:
        """
        Ordered list of the geometry names.

        EXAMPLES::

            >>> from hklpy2 import get_solver
            >>> solver = Solver("chewacla")
            >>> solver.geometries()
            []
        """
        return []  # TODO

    def inverse(self, reals: dict) -> dict[str, float]:
        """Compute dict of pseudos from reals (angles -> hkl)."""
        ...  # TODO
        return {}

    @property
    def modes(self) -> list[str]:
        """List of the geometry operating modes."""
        return []  # TODO

    @property
    def pseudo_axis_names(self) -> list[str]:
        """Ordered list of the pseudo axis names (such as h, k, l)."""
        # Do NOT sort.
        return []  # TODO

    @property
    def real_axis_names(self) -> list[str]:
        """Ordered list of the real axis names (such as th, tth)."""
        # Do NOT sort.
        return []  # TODO

    def refineLattice(self, reflections: list[Reflection]) -> Lattice:
        """Refine the lattice parameters from a list of reflections."""
        ...  # TODO

    def removeAllReflections(self) -> None:
        """Remove all reflections."""
        ...  # TODO

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

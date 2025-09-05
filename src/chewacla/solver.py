"""
Solver for hklpy2.

.. autosummary::

    ~ChewaclaSolver
"""

from hklpy2 import SolverBase
from hklpy2.blocks.reflection import Reflection
from hklpy2.blocks.lattice import Lattice


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
        ...  # TODO
        # return self.UB

    def forward(self, pseudos: dict) -> list[dict[str, float]]:
        """Compute list of solutions(reals) from pseudos (hkl -> [angles])."""
        # based on geometry and mode
        # return [{}]
        ...  # TODO

    @classmethod
    def geometries(cls) -> list[str]:
        """
        Ordered list of the geometry names.

        EXAMPLES::

            >>> from hklpy2 import get_solver
            >>> Solver = get_solver("no_op")
            >>> Solver.geometries()
            []
            >>> solver = Solver("TH TTH Q")
            >>> solver.geometries()
            []
        """
        ...  # TODO
        # return []

    def inverse(self, reals: dict) -> dict[str, float]:
        """Compute dict of pseudos from reals (angles -> hkl)."""
        ...  # TODO
        # return {}

    @property
    def modes(self) -> list[str]:
        """List of the geometry operating modes."""
        ...  # TODO
        # return []

    @property
    def pseudo_axis_names(self) -> list[str]:
        """Ordered list of the pseudo axis names (such as h, k, l)."""
        # Do NOT sort.
        # return []
        ...  # TODO

    @property
    def real_axis_names(self) -> list[str]:
        """Ordered list of the real axis names (such as th, tth)."""
        # Do NOT sort.
        # return []
        ...  # TODO

    def refineLattice(self, reflections: list[Reflection]) -> Lattice:
        """Refine the lattice parameters from a list of reflections."""
        ...  # TODO

    def removeAllReflections(self) -> None:
        """Remove all reflections."""
        ...  # TODO

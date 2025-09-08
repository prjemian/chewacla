"""
Describe diffraction vectors with a shorthand vocabulary.

.. autosummary::

    ~DirectionShorthand
    ~unit_vector
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray

DirectionVector = NDArray[np.float64]  # type-only alias
"""Unit vector description of a direction."""

DirectionVectorInput = Union[DirectionVector, Sequence[float], str]
"""Allowed variations for user input."""

DirectionMap = Dict[str, DirectionVector]  # type-only alias
"""Ordered dictionary of DirectionVectors, keyed by rotational axis names."""

DirectionMapInput = Dict[str, DirectionVectorInput]  # type-only alias
"""Allowed variations for user input."""

x_hat: DirectionVector = np.array((1, 0, 0), dtype=float)
r"""Basis vector, $\hat{x}$"""

y_hat: DirectionVector = np.array((0, 1, 0), dtype=float)
r"""Basis vector, $\hat{y}$"""

z_hat: DirectionVector = np.array((0, 0, 1), dtype=float)
r"""Basis vector, $\hat{z}$"""


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return a unit vector from a length-3 ndarray.

    Parameters
    ----------
    v : np.ndarray
        Input array of shape (3,).

    Returns
    -------
    np.ndarray
        New array of shape (3,) with unit length.

    Raises
    ------
    TypeError
        If `v` is not an array-like of numeric type.
    ValueError
        If `v` does not have shape (3,) or has zero length.
    """
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError("input must be a 1-D array of length 3")
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return arr / norm


class DirectionShorthand:
    """
    Describe vectors with shorthand from xrayutilities.

    .. autosummary::

        ~vector
        ~vocabulary

    Example::

        >>> ds = DirectionShorthand()
        >>> ds.vector('x+')
        [1 0 0]
    """

    def __init__(
        self,
        vocabulary: Optional[DirectionMap | Mapping[str, Sequence[float]]] | None = None,
    ):
        if vocabulary is None:
            vocabulary = {
                "x": (1, 0, 0),
                "y": (0, 1, 0),
                "z": (0, 0, 1),
                # kappa axis, here's a definition with alpha=50 dgrees, in yz plane
                # "k": (0, np.cos(50), np.sin(50)),  # kappa axis, 50 degrees in yz plane
                # The caller provides this, the angle and/or plane could be different.
            }
        self.vocabulary = vocabulary

    def vector(self, symbol: str) -> np.ndarray:
        """
        Convert a symbol like 'x+' or '+x' to a numpy array (copy).

        Behavior:
        - Accepts 'x+' (letter then sign) or '+x' (sign then letter).
        - Does NOT permute characters; validates positions instead.
        - Whitespace allowed; case-insensitive.
        """
        if not isinstance(symbol, str):
            raise TypeError("symbol must be a string")
        s = symbol.strip().lower()
        if len(s) != 2:
            raise ValueError(f"Expected 2-character string like 'x+' or '+x'. Got: {symbol!r}")

        # now s is exactly two chars; identify sign and letter deterministically
        if s[0] in "+-" and s[1].isalpha():
            sign, name = s[0], s[1]
        elif s[1] in "+-" and s[0].isalpha():
            sign, name = s[1], s[0]
        else:
            raise ValueError(
                f"Invalid symbol {symbol!r}; must be"
                #
                " sign and single letter in positions (±letter or letter±)"
            )

        if not name.isalpha() or len(name) != 1:
            raise ValueError(f"Invalid axis character {name!r}; must be a single letter")

        if name not in self.vocabulary:
            raise ValueError(
                f"Unknown axis {name!r}."
                #
                f" Allowed: {sorted(self.vocabulary.keys())}"
            )

        vec = np.asarray(self.vocabulary[name], dtype=float).copy()
        return vec if sign == "+" else -vec

    @property
    def vocabulary(self) -> DirectionMap:
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, value: DirectionMap) -> None:
        norm = {}
        for k, v in value.items():
            if not isinstance(k, str) or len(k) != 1:
                raise ValueError("vocabulary keys must be single-letter axis strings")
            arr = np.asarray(v, dtype=float)
            if arr.shape != (3,):
                raise ValueError("vocabulary values must be 3-element sequences")
            norm[k.lower()] = arr
        self._vocabulary = norm

    def __repr__(self) -> str:
        vocab_repr = ", ".join(f"{k}: {v}" for k, v in self.vocabulary.items())
        return f"{self.__class__.__name__}(vocabulary={{ {vocab_repr} }})"

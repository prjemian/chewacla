"""
Describe diffraction vectors with a shorthand vocabulary.

.. autosummary::

    ~DirectionShorthand
"""

from typing import Mapping
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

DirectionVector = Sequence[float]
"""Unit vector description of a direction."""

DirectionMap = Mapping[str, DirectionVector | str]
"""Ordered dictionary of DirectionVectors, keyed by rotational axis names."""

x_hat: DirectionVector = np.array((1, 0, 0))
y_hat: DirectionVector = np.array((0, 1, 0))
z_hat: DirectionVector = np.array((0, 0, 1))


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

    def __init__(self, vocabulary: DirectionMap | None = None):
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

    def vector(self, symbol: str) -> NDArray:
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

        # Determine which position is sign and which is letter without permuting input
        if s[0] in "+-":
            sign = s[0]
            name = s[1]
        elif s[1] in "+-":
            name = s[0]
            sign = s[1]
        else:
            raise ValueError(f"Missing sign in symbol {symbol!r}; must include '+' or '-'")

        if not name.isalpha() or len(name) != 1:
            raise ValueError(f"Invalid axis character {name!r}; must be a single letter")

        if name not in self.vocabulary:
            raise ValueError(f"Unknown axis {name!r}. Allowed: {sorted(self.vocabulary.keys())}")

        vec = self.vocabulary[name].astype(float).copy()
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
        return f"DirectionShorthand(vocabulary={{ {vocab_repr} }})"

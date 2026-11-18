"""Protect prepared inputs from partial notebook edits.

A prepared transform is a calculation for one set of coordinates. If an
array changes while its quadrature weights do not, both arrays can look
valid but describe different observables. These small preparation helpers
make an independent, read-only snapshot. Construct a new geometry object
to change its coordinates; doing so needs no interpreter restart.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def immutable_copy(value):
    """Copy numerical arrays and containers into independently owned read-only data.

    A NumPy writeable=False flag on an owned allocation can be reversed.
    An immutable bytes buffer prevents that reversal as well. This copying
    occurs only during preparation, not in interpolation or quadrature loops.

    Arguments:
      value = array, mapping, sequence or scalar defining prepared inputs.
    Returns:
      an immutable array/mapping/tuple, or the unchanged scalar/object.
    Raises:
      ValueError for object arrays, whose byte representation contains pointers.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Prepared inputs require numerical or fixed-width strings.")
        result = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype)
        return result.reshape(value.shape)
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            result[name] = immutable_copy(value=item)
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.append(immutable_copy(value=item))
        return tuple(result)
    return value


class PreparedGeometry:
    """Keep a completed geometry consistent while allowing internal result caches.

    Public attributes hold the prepared definition. Names beginning with
    ``_`` hold internal bookkeeping, such as a retained mean.
    The constructor calls _seal_geometry after all validation/preparation;
    later assignment to a public attribute stops with a named instruction.
    """

    def _seal_geometry(self):
        """Take read-only snapshots and prevent replacement of public settings."""
        for name, value in list(self.__dict__.items()):
            if not name.startswith("_"):
                object.__setattr__(self, name, immutable_copy(value=value))
        object.__setattr__(self, "_geometry_sealed", True)

    def __setattr__(self, name, value):
        """Reject a partial geometry change after its dependent tables exist."""
        if getattr(self, "_geometry_sealed", False) and not name.startswith("_"):
            raise AttributeError(
                f"Prepared {type(self).__name__}.{name} cannot be changed; "
                "construct a new engine with the new settings in this kernel."
            )
        object.__setattr__(self, name, value)

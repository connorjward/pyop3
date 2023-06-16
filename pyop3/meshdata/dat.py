from __future__ import annotations

from typing import FrozenSet, Hashable, Sequence

import pytools

from pyop3.axis import Axis, AxisTree
from pyop3.distarray import MultiArray

# from pyop3.tree import Node, Tree, previsit
from pyop3.utils import UniqueNameGenerator, checked_zip, just_one

__all__ = ["Dat"]


class Dat:
    def __init__(self, space, dtype=None, *, data=None, name=None):
        data = MultiArray(space.axes, name=name, data=data, dtype=dtype)

        self.space = space
        self._data = data

    def __getitem__(self, indices: IndexTree) -> IndexedMultiArray:
        # TODO fail if we don't fully index the dat, this is because spaces can have
        # variable orderings so the resulting temporary would have undefined shape
        # if not is_fully_indexed(self.array.axes, indices):
        #     raise ValueError("Dats must be fully indexed")
        return self._data[indices]

    # TODO: Use darray as the name of the property for the underlying data structure.
    @property
    def data(self):
        return self._data.data

from __future__ import annotations

from typing import FrozenSet, Hashable, Sequence

import pytools

from pyop3.axis import Axis, AxisTree
from pyop3.distarray import MultiArray
from pyop3.meshdata.base import MeshDataCarrier

# from pyop3.tree import Node, Tree, previsit
from pyop3.utils import checked_zip, just_one

__all__ = ["Dat"]


class Dat(MeshDataCarrier):
    def __init__(self, space, data=None, *, dtype=None, name=None):
        array = MultiArray(space.axes, data=data, dtype=dtype, name=name)

        self.space = space
        self._array = array

    @property
    def array(self):
        return self._array

    @property
    def name(self):
        return self.array.name

    @property
    def data(self):
        return self.array.data

    @property
    def comm(self):
        return self.space.comm

    @property
    def internal_comm(self):
        return self.space.internal_comm

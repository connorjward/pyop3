import collections
import dataclasses

import numpy as np

import pyop3.utils


class Map:

    dtype = np.int32

    _name_generator = pyop3.utils._NameGenerator(prefix="map")

    def __init__(self, arity):
        self.arity = arity
        self.name = next(self._name_generator)


# This can also be thought of as a 'domain' or 'set', needs bikeshedding
class Index:
    def __init__(self, extent, mesh, parent=None):
        self.extent = extent
        self.mesh = mesh
        self.parent = parent

        if parent:
            self.map = Map(extent)
        else:
            self.map = None


def closure(indices):
    # if index is a sequence (e.g. extruded) then return a tuple of closures,
    # one per input index
    # TODO The closure of a dense array is not straightforward as it depends
    # on things like orientation and offsets
    if isinstance(indices, collections.abc.Sequence):
        return tuple(index.mesh.closure(index) for index in indices)
    else:
        index = indices
        return index.mesh.closure(index)


def star(index):
    # if index is a sequence (e.g. extruded) then return a tuple of stars,
    # one per input index
    if isinstance(index, collections.abc.Sequence):
        return tuple(idx.mesh.star(idx) for idx in index)
    else:
        return index.mesh.star(index)

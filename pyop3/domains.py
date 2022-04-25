import abc
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


class Domain(abc.ABC):

    _name_generator = pyop3.utils._NameGenerator(prefix="dom")

    @property
    def mesh(self):
        if not self._mesh:
            raise ValueError("Domain is not associated with a mesh")
        return self._mesh


class SparseDomain(Domain):
    def __init__(self, extent, *, mesh=None, name=None, parent=None):
        self.extent = extent
        self._mesh = mesh
        self.parent = parent

        if name:
            self.name = name
        else:
            self.name = next(self._name_generator)

        # TODO Eliminate this
        self.start = 0
        self.stop = extent
        self.step = 1


class DenseDomain(Domain):
    def __init__(self, *args, mesh=None, name=None, parent=None):
        if len(args) == 1:
            (stop,) = args
            start = 0
            step = 1
        elif len(args) == 2:
            start, stop = args
            step = 1
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise ValueError

        self.start = start
        self.stop = stop
        self.step = step
        self._mesh = mesh
        self.parent = parent

        if name:
            self.name = name
        else:
            self.name = next(self._name_generator)

    @property
    def extent(self):
        return (self.stop - self.start) // self.step


def closure(index):
    return index.mesh.closure(index)


def star(index):
    return index.mesh.star(index)

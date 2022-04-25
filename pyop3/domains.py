import abc
import collections
import dataclasses
from typing import Optional

import numpy as np

import pyop3.utils


# class Map:
#
#     dtype = np.int32
#
#     _name_generator = pyop3.utils._NameGenerator(prefix="map")
#
#     def __init__(self, arity):
#         self.arity = arity
#         self.name = next(self._name_generator)


# class Parameter:
#     ...
#
#
# class Literal(Parameter):
#     def __init__(self, value):
#         self.value = value
#
#
# class Variable(Parameter):
#     ...


class Domain:
    ...

    # def __init__(self, start, stop, step=1, *, mesh=None):
    #     self.start = start
    #     self.stop = stop
    #     self.step = step
    #     self.mesh = mesh
    #
    # @property
    # def count(self):
    #     return (self.stop - self.start) // self.step
    #
    # @property
    # def index(self):
    #     return Index(self)


class SparseDomain(Domain):

    """for i in range(start, stop, step):
          j = map[i]"""

    _name_generator = pyop3.utils._NameGenerator(prefix="map")

    def __init__(self, parent, arity, *, mesh=None, name=None):
        self.parent = parent
        self.arity = arity
        self.mesh = mesh
        self.name = name or next(self._name_generator)

        self.start = 0
        self.stop = arity
        self.step = 1

        self.extent = arity

    @property
    def index(self):
        return Index(self)


class DenseDomain(Domain):
    def __init__(self, start, stop, step=1, *, mesh=None):
        self.start = start
        self.stop = stop
        self.step = step
        self.mesh = mesh

    @property
    def index(self):
        return Index(self)

    @property
    def extent(self):
        try:
            return (self.stop- self.start) // self.step
        except:
            return None


class CompositeDomain(Domain):
    def __init__(self, *domains):
        self.domains = domains

    @property
    def index(self):
        return Index(self)

    def __iter__(self):
        return iter(self.domains)



@dataclasses.dataclass(frozen=True)
class Index:
    domain: Domain


# class Map:
#     """A map from one domain to another."""
#
#     def __init__(self, from_index: Index, to_index: Index):
#         self.from_index = from_index
#         self.to_index = to_index
#
#     @property
#     def extent(self):
#         return self.to_index.domain.extent
#
#     @property
#     def index(self):
#         return self.to_index


def closure(index):
    return index.domain.mesh.closure(index)


def star(index):
    return index.domain.mesh.star(index)

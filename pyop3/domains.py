import numbers

import dtlutils
import numpy as np


class Map:

    dtype = np.int32

    _name_generator = dtlutils.NameGenerator(prefix="map")

    def __init__(self, arity):
        self.arity = arity
        self.name = next(self._name_generator)


class Index:
    """Class representing a point in a plex."""

    _name_generator = dtlutils.NameGenerator(prefix="i")

    def __init__(self, domain, name=None):
        self.domain = domain
        self.name = name or next(self._name_generator)


class Domain:
    def __init__(self, extent, mesh, parent_index=None):
        self.extent = extent
        self.mesh = mesh

        self.parent_index = parent_index

        self.index = Index(self)

        if parent_index:
            self.map = Map(extent)
        else:
            self.map = None

    @property
    def loopy_domain(self):
        iname = self.index.name
        return f"{{ [{iname}]: 0 <= {iname} < {self.extent} }}"


def closure(index):
    return index.domain.mesh.closure(index)


def star(index):
    return index.domain.mesh.star(index)

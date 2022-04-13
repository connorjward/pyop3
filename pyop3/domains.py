import abc
import enum

import dtl
import dtlutils

from dtlpp import RealVectorSpace


class Restriction(enum.Enum):

    CLOSURE = enum.auto()
    STAR = enum.auto()

    def __str__(self):
        return self.name


class PointIndex(dtl.Index):
    """Class representing a point in a plex."""

    _name_generator = dtlutils.NameGenerator(prefix="p")

    def __init__(self, point_set, name=None):
        name = name or next(self._name_generator)
        extent = point_set.extent

        super().__init__(name, extent)
        self.point_set = point_set



class PointSet(dtl.Node, abc.ABC):
    """A set of plex points."""

    def __init__(self):
        self.index = PointIndex(self)

    @property
    def operands(self):
        return (self.index,)


class FreePointSet(PointSet):
    """An unrestricted domain (must be the outermost one)."""

    def __init__(self, name, extent):
        self.name = name
        self.extent = extent  # FIXME this should not be defined as it is runtime
        super().__init__()

    def __str__(self):
        return self.name


class RestrictedPointSet(PointSet):
    """E.g. closure(i)."""

    _count = 0

    def __init__(self, parent_index: PointIndex, restriction: Restriction, extent):
        self.parent_index = parent_index
        self.restriction = restriction


        tspace = dtl.TensorSpace([
            RealVectorSpace(parent_index.point_set.extent),
            RealVectorSpace(parent_index.point_set.extent),
            RealVectorSpace(extent)
            ])
        self.restriction_tensor = dtl.TensorVariable(tspace, f"{restriction}{self._count}")
        self._count += 1

        self.extent = extent
        super().__init__()

    def __str__(self):
        if self.restriction == Restriction.CLOSURE:
            return f"closure({self.parent_index.name})"
        if self.restriction == Restriction.STAR:
            return f"star({self.parent_index.name})"
        else:
            raise AssertionError


# FIXME we should determine the arity from point_index.point_set.mesh.get_arity(p, CLOSURE)
# or similar
def closure(point_index, arity):
    return RestrictedPointSet(point_index, Restriction.CLOSURE, arity)


# FIXME we should determine the arity from point_index.point_set.mesh.get_arity(p, CLOSURE)
# or similar
def star(point_index, arity):
    return RestrictedPointSet(point_index, Restriction.STAR, arity)

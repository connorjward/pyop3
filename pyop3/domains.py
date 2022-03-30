import abc
import enum

import pyop3.utils


class Restriction(enum.Enum):

    CLOSURE = enum.auto()
    STAR = enum.auto()

    def __str__(self):
        return self.name


class Index(abc.ABC):
    def __str__(self):
        return self.name


class PointIndex(Index):
    """Class representing a point in a plex."""

    _name_generator = pyop3.utils.NameGenerator(prefix="p")

    def __init__(self, point_set, name=None):
        self.point_set = point_set
        self.name = name or next(self._name_generator)


class CountIndex(Index):

    _name_generator = pyop3.utils.NameGenerator(prefix="i")

    def __init__(self, point_set):
        self.point_set = point_set
        self.name = next(self._name_generator)


class PointSet(abc.ABC):
    """A set of plex points."""

    def __init__(self):
        self.count_index = CountIndex(self)
        self.point_index = PointIndex(self)

    @property
    def index(self):
        """Alias for point_index."""
        return self.point_index


class FreePointSet(PointSet):
    """An unrestricted domain (must be the outermost one)."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    def __str__(self):
        return self.name


class RestrictedPointSet(PointSet):
    """E.g. closure(i)."""

    def __init__(self, parent_index: PointIndex, restriction: Restriction):
        self.parent_index = parent_index
        self.restriction = restriction
        super().__init__()

    def __str__(self):
        if self.restriction == Restriction.CLOSURE:
            return f"closure({self.parent_index.name})"
        else:
            raise AssertionError


def closure(point_index):
    return RestrictedPointSet(point_index, Restriction.CLOSURE)


def star(point_index):
    return RestrictedPointSet(point_index, Restriction.STAR)

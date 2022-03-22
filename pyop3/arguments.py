import abc
from pyop3.loops import Assign


class Argument(abc.ABC):
    """Data that gets passed about."""


class Dat(Argument):
    def __init__(self, name):
        self.name = name

    def __getitem__(self, domain):
        """You can index a dat with a domain to get an argument to a loop.

        **collective**
        """
        return DatSlice(self, domain)

    def assign(self, other):
        return Assign(self, other)


class DatSlice(Dat):

    def __init__(self, parent, point_set):
        self.parent = parent
        self.point_set = point_set

    @property
    def name(self):
        return self.parent.name

    def __str__(self):
        return f"{self.name}[{self.point_set}]"

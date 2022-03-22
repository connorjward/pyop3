import abc
from pyop3.loops import Assign
from pyop3 import domains, utils


class Tensor(abc.ABC):
    """Data that gets passed about."""

    shape = None
    free_indices = None
    """See the TSFC paper for a good explanation of these attributes"""

    def restrict(self, restriction: domains.RestrictedPointSet):
        return RestrictedTensor(self, restriction)

    def apply_multiindex(self, indices):
        indices = utils.as_tuple(indices)
        raise NotImplementedError


class Dat(Tensor):
    def __init__(self, name):
        self.name = name

    def __getitem__(self, key):
        """You can index a dat with a domain to get an argument to a loop.

        **collective**
        """
        if isinstance(key, domains.Index):
            return self.apply_multiindex(key)
        elif isinstance(key, domains.RestrictedPointSet):
            return self.restrict(key)
        else:
            raise TypeError

    def assign(self, other):
        return Assign(self, other)


class RestrictedTensor(Tensor):

    def __init__(self, parent, restriction: domains.RestrictedPointSet):
        self.parent = parent
        self.restriction = restriction


class DatSlice(Dat):

    def __init__(self, parent, point_set):
        self.parent = parent
        self.point_set = point_set

    @property
    def name(self):
        return self.parent.name

    def __str__(self):
        return f"{self.name}[{self.point_set}]"

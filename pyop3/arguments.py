import abc

import pyop3.domains
import pyop3.exprs
import pyop3.utils


class Tensor(abc.ABC):
    """Data that gets passed about."""

    shape = None
    free_indices = None
    """See the TSFC paper for a good explanation of these attributes"""

    def restrict(self, restriction: pyop3.domains.RestrictedPointSet):
        return RestrictedTensor(self, restriction)

    def apply_multiindex(self, indices):
        indices = pyop3.utils.as_tuple(indices)
        raise NotImplementedError


class Dat(Tensor):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __getitem__(self, key):
        """You can index a dat with a domain to get an argument to a loop.

        **collective**
        """
        if isinstance(key, pyop3.domains.Index):
            return self.apply_multiindex(key)
        elif isinstance(key, pyop3.domains.RestrictedPointSet):
            return self.restrict(key)
        else:
            raise TypeError

    def assign(self, other):
        return pyop3.exprs.Assign(self, other)


class RestrictedTensor(Tensor):
    def __init__(self, parent, restriction: pyop3.domains.RestrictedPointSet):
        self.parent = parent
        self.restriction = restriction

    def __str__(self):
        return f"{self.parent}[{self.restriction}]"

import collections
from typing import Tuple

import pyop3.domains
import pyop3.exprs
import pyop3.utils
from pyop3.domains import Index


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "tensor"

    def __init__(self, indices=(), *, name=None):
        if not isinstance(indices, collections.abc.Sequence):
            indices = (indices,)
        if not name:
            name = self.name_generator.generate(self.prefix)

        self.indices = indices
        self.name = name

    def __getitem__(self, indices):
        """You can index a dat with a domain to get an argument to a loop.

        **collective**
        """
        if not isinstance(indices, collections.abc.Sequence):
            indices = (indices,)

        if len(indices) > len(self.indices):
            raise ValueError

        new_indices = indices + self.indices[len(indices):]
        return self.copy(indices=new_indices)

    def copy(self, **kwargs):
        indices = kwargs.get("indices", self.indices)
        name = kwargs.get("name", self.name)
        return type(self)(indices=indices, name=name)


def Global(*, name=None):
    if not name:
        name = Tensor.name_generator.generate(prefix="global")
    return Tensor(name=name)


def Dat(index: Index, *, name=None):
    if not name:
        name = Tensor.name_generator.generate(prefix="dat")
    return Tensor(index, name=name)


def Mat(indices: Tuple[Index, Index], *, name=None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(indices, name=name)

import collections
from typing import Tuple, Union

import pyop3.domains
import pyop3.exprs
import pyop3.utils
from pyop3.domains import Index


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "tensor"

    def __init__(self, domains=(), indices=(), *, name: str=None):
        if not isinstance(domains, collections.abc.Sequence):
            domains = (domains,)
        if not isinstance(indices, collections.abc.Sequence):
            indices = (indices,)
        if not name:
            name = self.name_generator.generate(self.prefix)

        self.domains = domains
        self.indices = indices
        self.name = name

    def __getitem__(self, indices: Union[Index, Tuple[Index, ...]]):
        if not isinstance(indices, collections.abc.Sequence):
            indices = (indices,)

        if len(indices) > len(self.domains):
            raise ValueError

        return self.copy(indices=self.indices+indices)

    @property
    def shape_indices(self):
        """Indices corresponding to unindexed 'shape' in the tensor."""
        # TODO This doesn't seem quite right. I need a nice way to resolve
        # free indices and shape whilst still retaining some notion of the
        # original shape information (needed for codegen).
        return self.domains[len(self.indices):]

    def copy(self, **kwargs):
        domains = kwargs.get("domains", self.domains)
        indices = kwargs.get("indices", self.indices)
        name = kwargs.get("name", self.name)
        return type(self)(domains=domains, indices=indices, name=name)


def Global(*, name: str=None):
    if not name:
        name = Tensor.name_generator.generate(prefix="global")
    return Tensor(name=name)


def Dat(domain: Index, *, name: str=None):
    if not name:
        name = Tensor.name_generator.generate(prefix="dat")
    return Tensor(domain, name=name)


def Mat(domains: Tuple[Index, Index], *, name: str=None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(domains, name=name)

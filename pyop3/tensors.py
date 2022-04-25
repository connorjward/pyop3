import collections
from typing import Tuple, Union
import numbers

import pyop3.domains
import pyop3.exprs
import pyop3.utils
from pyop3.domains import Index, Domain, CompositeDomain


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "tensor"

    def __init__(self, domains=(), indices=(), *, name: str = None, parent=None):
        if not isinstance(domains, collections.abc.Sequence):
            domains = (domains,)
        if not isinstance(indices, collections.abc.Sequence):
            indices = (indices,)

        new_domains = []

        for domain in domains:
            if isinstance(domain, CompositeDomain):
                new_domains.extend(domain.domains)
            else:
                new_domains.append(domain)

        self.domains = tuple(new_domains)

        new_indices = []

        for index in indices:
            if isinstance(index.domain, CompositeDomain):
                for subdomain in index.domain.domains:
                    new_indices.append(subdomain.index)
            else:
                new_indices.append(index)

        self.indices = tuple(new_indices)
        self.name = name or self.name_generator.generate(self.prefix)

        # this is not very satisfying but I need to be able to get the original
        # shape of the tensor back for passing in to loopy
        self.parent = parent

    def __getitem__(self, indices: Union[Index, Domain, Tuple[Domain, ...]]):
        if not isinstance(indices, collections.abc.Sequence):
            indices = (indices,)

        if len(indices) > len(self.domains):
            raise ValueError

        new_indices = []
        new_domains = []
        for index in indices:
            if isinstance(index, collections.abc.Sequence):
                new_domains.extend(index)
            elif isinstance(index, Index):
                new_indices.append(index)
            elif isinstance(index, Domain):
                new_domains.append(index)
            elif isinstance(index, slice):
                raise NotImplementedError
            elif isinstance(index, numbers.Integral):
                new_indices.append(index)
            else:
                raise ValueError

        return self.copy(domains=tuple(new_domains) + self.domains[len(indices):],
                indices=self.indices + tuple(new_indices), parent=self)

    @property
    def broadcast_indices(self):
        return tuple(d.index for d in self.domains)

    @property
    def orig_shape(self):
        tensor = self
        while tensor.parent:
            tensor = self.parent
        assert not tensor.indices
        if not tensor.domains:
            return ()
        return (None,) + tuple(d.extent for d in tensor.domains[1:])

    def copy(self, **kwargs):
        domains = kwargs.get("domains", self.domains)
        indices = kwargs.get("indices", self.indices)
        name = kwargs.get("name", self.name)
        parent = kwargs.get("parent", self.parent)
        return type(self)(domains=domains, indices=indices, name=name, parent=parent)


def Global(*, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="global")
    return Tensor(name=name)


def Dat(domain: Domain, *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="dat")
    return Tensor(domain, name=name)


def Mat(domains: Tuple[Domain, Domain], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(domains, name=name)

import abc
import collections
import dataclasses
from typing import Tuple, Union
import numbers

import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "x"

    def __init__(self, indices=(), *, mesh=None, name: str = None):
        # indices can be either indices or ranges/slices
        # TODO should not need as_tuple
        self.indices = as_tuple(indices)
        self.mesh = mesh
        self.name = name or self.name_generator.generate(self.prefix)

    def __getitem__(self, indices):
        indices = as_tuple(indices)

        # if len(indices) > len(self.unbound_indices):
        #     raise ValueError

        new_indices = list(self.indices)
        index_iterator = list(indices)
        for i, index in enumerate(self.indices):
            # we can only index into domains that have not yet been fully indexed
            # if index.is_vector and index_iterator:
            if isinstance(index, Range) and index_iterator:
                new_indices[i] = index_iterator.pop(0)

        return self.copy(indices=tuple(new_indices))

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def broadcast_indices(self):
        return self.unbound_indices

    @property
    def unindexed_domains(self):
        return tuple(dom for dom, idx in zip(self.domains, self.indices) if idx == slice(None))

    @property
    def unbound_indices(self):
        return tuple(idx for idx in self.indices if isinstance(idx, Range))

    # Range also has this property
    @property
    def index(self):
        # FIXME This will fail if more than one unbound index
        (range_,) = self.unbound_indices
        return Index(self[range_.index], range_)

    @property
    def unindexed_shape(self):
        return tuple(len(dom.range) for dom in self.domains)

    @property
    def is_vector(self):
        return self.order == 1

    def copy(self, **kwargs):
        indices = kwargs.get("indices", self.indices)
        mesh = kwargs.get("indices", self.mesh)
        name = kwargs.get("name", self.name)
        return type(self)(indices=indices, mesh=mesh, name=name)


class Range:
    def __init__(self, *args, mesh=None):
        try:
            self.start, self.stop = args
        except ValueError:
            self.start = 0
            (self.stop,) = args

        self.mesh = mesh

    @property
    def size(self):
        return self.stop - self.start

    @property
    def index(self):
        return Index(Tensor(), self)


@dataclasses.dataclass(frozen=True)
class Index:
    scalar: Tensor
    range: Range

    # just for hacks
    @property
    def domain(self):
        return self.range


def Global(*, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="global")
    return Tensor(name=name)


def Dat(shape: Tuple[int, ...], *, name: str = None) -> Tensor:
    shape = as_tuple(shape)
    indices = tuple(Range(extent) for extent in shape)
    if not name:
        name = Tensor.name_generator.generate(prefix="dat")
    return Tensor(indices, name=name)


def Mat(domains: Tuple["Domain", "Domain"], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(domains, name=name)

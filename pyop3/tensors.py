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

    # @property
    # def shape(self):
    #     return 

    @property
    def order(self):
        return len(self.broadcast_indices)

    @property
    def broadcast_indices(self):
        return self.unbound_indices

    @property
    def unindexed_domains(self):
        return tuple(dom for dom, idx in zip(self.domains, self.indices) if idx == slice(None))

    @property
    def unbound_indices(self):
        # TODO replace with slices
        return tuple(idx for idx in self.indices if isinstance(idx, Range))

    @property
    def index(self):
        (range_,) = self.unbound_indices
        return Index(self, range_.start, range_.stop)

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


# class Range(Tensor):
class Range:
    def __init__(self, *args, mesh=None):
        try:
            self.start, self.stop = args
        except ValueError:
            self.start = 0
            (self.stop,) = args

        self.mesh = mesh
        self.name = "range"
        self.indices = ()

    @property
    def size(self):
        return self.stop - self.start

    @property
    def order(self):
        return 1

    @property
    def index(self):
        return Index(self, self.start, self.stop)


@dataclasses.dataclass(frozen=True)
class Index:
    tensor: Tensor
    start: Union[Tensor, int]
    stop: Union[Tensor, int]

    def __post_init__(self):
        # must be a vector for now (no multi-indexing)
        assert self.tensor.order == 1

    @property
    def domain(self):
        return self.start, self.stop


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

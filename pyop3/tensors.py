import abc
import itertools
import collections
import dataclasses
from typing import Tuple, Union, Any
import numbers

import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple


def as_domain(domain):
    if isinstance(domain, Domain):
        return domain
    elif isinstance(domain, numbers.Integral):
        return Domain(0, domain)
    else:
        raise ValueError


# TODO inherit from slice? how can we accomodate actual slices and integers?
class Domain:
    @property
    def index(self):
        return Index(self)

    @property
    @abc.abstractmethod
    def range(self):
        pass


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "ten"

    def __init__(self, dims=(), *, mesh=None, name: str = None, prefix: str=None):
        # dims is a tuple of domains (indexing tensors) and indices
        assert all(isinstance(dim, (Index, Domain)) for dim in dims)

        self.dims = dims
        self.mesh = mesh
        self.name = name or self.name_generator.generate(prefix or self.prefix)

    def __getitem__(self, indices):
        indices = as_tuple(indices)

        if len(indices) > self.order:
            raise ValueError

        # we can only index dims that are not already indexed
        new_dims = list(self.dims)
        indices = list(indices)
        for i, dim in enumerate(self.dims):
            if not isinstance(dim, Index) and indices:
                new_dims[i] = indices.pop(0)

        assert not indices
        return self.copy(dims=tuple(new_dims))

    def __str__(self):
        return f"{self.name}[{','.join(str(idx) for idx in self.indices)}]"

    @property
    def indices(self):
        return tuple(dim for dim in self.dims if isinstance(dim, Index))

    @property
    def domains(self):
        return tuple(dim for dim in self.dims if isinstance(dim, Domain))

    @property
    def broadcast_domains(self):
        domains = set()
        for dim in self.dims:
            domains |= self._get_broadcast_domains(dim)
        return frozenset(domains)

    @classmethod
    def _get_broadcast_domains(cls, dim):
        if isinstance(dim, Index):
            return frozenset()
        elif isinstance(dim, Slice):
            return frozenset({dim.domain})
        elif isinstance(dim, Map):
            return frozenset({dim.range}) | cls._get_broadcast_domains(dim.from_index)
        else:
            raise AssertionError

        domains = set()
        # the magic here is we broadcast everything that is not an index
        # e.g. cone(star(p)) has two broadcast domains
        # TODO Think about if we have a Range here - is that possible?
        for dim in filter(lambda d: isinstance(d, Tensor), dims):
            domains |= cls._get_broadcast_domains(dim.dims)
        return frozenset(domains)

    @property
    def orig_shape(self):
        shape = []
        for domain in self.dims:
            if isinstance(domain, Index):
                domain = domain.domain
            while not isinstance(domain, Slice):
                domain = domain.from_index.domain
            shape.append(domain.range)
        return tuple(shape)

    # @property
    # def domain(self):
    #     try:
    #         (dom,) = self.shape
    #         return dom
    #     except ValueError:
    #         raise TypeError

    @property
    def order(self):
        return len(self.domains)

    @property
    def is_scalar(self):
        return self.order == 0

    @property
    def is_vector(self):
        return self.order == 1

    def copy(self, **kwargs):
        dims = kwargs.get("dims", self.dims)
        mesh = kwargs.get("mesh", self.mesh)
        name = kwargs.get("name", self.name)
        return type(self)(dims=dims, mesh=mesh, name=name)


@dataclasses.dataclass(frozen=True)
class Range:
    start: int
    stop: int


class Map(Tensor, Domain):

    def __init__(self, from_index, slice_, **kwargs):
        if isinstance(slice_, int):
            slice_ = Slice(slice_)

        dims = from_index, slice_

        super().__init__(dims, prefix="map", **kwargs)

    @property
    def from_index(self):
        return self.dims[0]

    @property
    def slice(self):
        return self.dims[1]

    @property
    def range(self):
        return self.slice.range


@dataclasses.dataclass(frozen=True)
class Index:
    domain: Domain


# This CANNOT be a subclass of Tensor. This is because if it does then it will need
# to define dims, which would be itself!
# We do need the indexer parent class so that we can have maps acting as dims
class Slice(Domain):
    def __init__(self, *args, mesh=None):
        try:
            start, stop = args
        except ValueError:
            start = 0
            (stop,) = args
        self.start = start
        self.stop = stop
        self.mesh = mesh

    @property
    def range(self):
        return Range(self.start, self.stop)


def Global(*, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="global")
    return Tensor(name=name)


def Dat(shape: Tuple[int, ...], *, prefix="dat", **kwargs) -> Tensor:
    shape = as_tuple(shape)
    dims = tuple(Slice(size) for size in shape)
    return Tensor(dims, prefix=prefix, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(shape, name=name)

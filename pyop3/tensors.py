import abc
import itertools
import collections
import dataclasses
from typing import Tuple, Union, Any
import numbers

import pytools
import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple


class ValidDimension(abc.ABC):
    """Does it make sense to index a tensor with this object?"""


class IndexedDimension(abc.ABC):
    """Is this dimension 'free' or has it already been bound to an index?"""


class FancyIndex(ValidDimension, abc.ABC):
    """Name inspired from numpy. This allows you to slice something with a
    list of indices."""

    @property
    @abc.abstractmethod
    def index(self):
        pass

@dataclasses.dataclass(frozen=True)
class Index(ValidDimension, IndexedDimension):
    space: FancyIndex


class Multiindex:
    def __init__(self, *indices: Index):
        self.indices = indices


# This CANNOT be a subclass of Tensor. This is because if it does then it will need
# to define dims, which would be itself!
class Range(FancyIndex):
    def __init__(self, *args, mesh=None):
        if len(args) == 1:
            start, stop, step = 0, *args, 1
        elif len(args) == 2:
            start, stop, step = (*args, 1)
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise ValueError

        self.start = start
        self.stop = stop
        self.step = step

    @property
    def index(self):
        return Index(self)

    @property
    def size(self):
        return (self.stop - self.start) // self.step


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "ten"

    def __new__(cls, dims=(), **kwargs):
        try:
            dim1, dim2 = dims
            assert isinstance(dim1, IndexedDimension)
            assert not isinstance(dim2, IndexedDimension)
            return NonAffineMap(dims, **kwargs)
        except:
            return super().__new__(cls, dims, **kwargs)

    def __init__(self, dims, *, name: str = None, prefix: str=None):
        # dims is a tuple of domains (indexing tensors) and indices
        assert all(isinstance(dim, ValidDimension) for dim in dims)

        self.dims = dims
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
    def spaces(self):
        return tuple(dim for dim in self.dims if isinstance(dim, Space))

    @property
    def shape(self):
        return tuple(space.size for space in self.spaces)

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
            return frozenset({dim})
        elif isinstance(dim, Map):
            return frozenset({dim.to_space}) | cls._get_broadcast_domains(dim.from_space)
        else:
            raise AssertionError

    @property
    def orig_shape(self):
        shape = []
        for dim in self.dims:
            space = dim.space if isinstance(dim, Index) else dim
            while isinstance(space, Map):
                space = space.from_space

            # this is because a maps space can either be an index or slice
            space = space.space if isinstance(space, Index) else space

            # FIXME This requires some thought about bin-ops
            # shape.append(space.size)
            shape.append(space.stop)
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
        return len(self.spaces)

    @property
    def is_scalar(self):
        return self.order == 0

    @property
    def is_vector(self):
        return self.order == 1

    def copy(self, **kwargs):
        dims = kwargs.get("dims", self.dims)
        name = kwargs.get("name", self.name)
        return type(self)(dims=dims, name=name)


class Map(FancyIndex, IndexedDimension, abc.ABC):
    @property
    @abc.abstractmethod
    def from_dim(self):
        pass

    @property
    @abc.abstractmethod
    def to_dim(self):
        pass

    @property
    def index(self):
        return self.to_dim.index


class NonAffineMap(Map, Tensor):

    prefix = "map"

    @property
    def from_dim(self):
        dim, _ = self.dims
        assert isinstance(dim, IndexedDimension)
        return dim

    @property
    def to_dim(self):
        _, dim = self.dims
        assert not isinstance(dim, IndexedDimension)
        return dim


class AffineMap(Map):
    def __init__(self, from_dim, offsets=1, strides=1):
        self.from_dim = from_dim
        self.offsets = as_tuple(offsets)
        self.strides = as_tuple(strides)

    @property
    def arity(self):
        return pytools.single_valued([len(self.offsets), len(self.strides)])

    # FIXME This will break but do I want to set this in super().__init__ or here?
    @property
    def from_dim(self):
        return self.from_dim

    @property
    def to_dim(self):
        return Range(self.arity)


def Global(*, name: str = None):
    return Tensor(name=name, prefix="glob")


def Dat(shape: Tuple[int, ...], *, prefix="dat", **kwargs) -> Tensor:
    shape = as_tuple(shape)
    dims = tuple(Range(size) for size in shape)
    return Tensor(dims, prefix=prefix, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(shape, name=name)

import abc
import itertools
import collections
import dataclasses
from typing import Tuple, Union, Any
import numbers

import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple


class ValidDimension(abc.ABC):
    """Does it make sense to index a tensor with this object?"""


class Space(ValidDimension, abc.ABC):
    @property
    @abc.abstractmethod
    def mesh(self):
        pass

    @property
    @abc.abstractmethod
    def size(self):
        pass

    @property
    def index(self):
        return Index(self)


# This CANNOT be a subclass of Tensor. This is because if it does then it will need
# to define dims, which would be itself!
# We do need the indexer parent class so that we can have maps acting as dims
class Slice(Space):
    def __init__(self, *args, mesh=None):
        if not args or len(args) > 3:
            raise ValueError

        if len(args) == 1:
            self.start, self.stop, self.step = 0, *args, 1
        elif len(args) == 2:
            self.start, self.stop, self.step = (*args, 1)
        else:
            self.start, self.stop, self.step = args

        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    @property
    def size(self):
        return (self.stop - self.start) // self.step


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "ten"

    def __init__(self, dims=(), *, name: str = None, prefix: str=None):
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


class Map(Tensor, Space):

    def __init__(self, from_space, arity, *, mesh=None):
        self._arity = arity
        self._mesh = mesh

        dims = from_space, Slice(arity)
        super().__init__(dims, prefix="map")

    # TODO rename to from_dim and to_dim (latter always a slice)
    @property
    def from_space(self):
        return self.dims[0]

    @property
    def to_space(self):
        return self.dims[1]

    @property
    def mesh(self):
        return self._mesh

    @property
    def size(self):
        return self._arity


@dataclasses.dataclass(frozen=True)
class Index(ValidDimension):
    space: Space


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

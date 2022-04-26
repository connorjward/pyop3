import abc
import collections
import dataclasses
from typing import Tuple, Union
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


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "ten"

    def __init__(self, shape=(), indices=(), *, mesh=None, name: str = None, prefix: str=None):
        self.shape = tuple(as_domain(dom) for dom in as_tuple(shape))
        self.indices = as_tuple(indices)
        self.mesh = mesh
        self.name = name or self.name_generator.generate(prefix or self.prefix)

    def __getitem__(self, indices):
        indices = as_tuple(indices)

        # if len(indices) > len(self.broadcast_indices):
        #     raise ValueError

        # FIXME You cannot index over the top of slices since these are considered indices
        # index_iterator = list(indices)
        # for i, dom in enumerate(self.indices):
        #     if index not in self.within_indices and index_iterator:
        #         new_index = index_iterator.pop(0)
        #         if self.is_slice(new_index):
        #             # add another broadcast index
        #             if isinstance(new_index, Tensor):
        #                 new_indices[i] = new_index.index
        #             else:
        #                 assert isinstance(new_index, slice)
        #                 assert new_index.step == 1
        #                 new_indices[i] = Range(new_index.start, new_index.stop).index
        #         else:
        #             new_indices[i] = new_index
        #             new_within_indices.add(index)

        return self.copy(
            shape=self.shape[len(indices):],
            indices=self.indices + indices
        )

    @property
    def domain(self):
        try:
            (dom,) = self.shape
            return dom
        except ValueError:
            raise TypeError

    @property
    def order(self):
        return len(self.shape)

    @property
    def is_scalar(self):
        return self.order == 0

    @property
    def is_vector(self):
        return self.order == 1

    # @property
    # def broadcast_indices(self):
    #     return tuple(idx for idx in self.indices if idx.is_vector)
    #
    # @property
    # def within_indices(self):
    #     return tuple(idx for idx in self.indices if idx.is_scalar)

    def copy(self, **kwargs):
        shape = kwargs.get("shape", self.shape)
        indices = kwargs.get("indices", self.indices)
        mesh = kwargs.get("indices", self.mesh)
        name = kwargs.get("name", self.name)
        return type(self)(shape=shape, indices=indices, mesh=mesh, name=name)


@dataclasses.dataclass
class Domain:
    start: Tensor
    stop: Tensor

    def __len__(self):
        return self.stop - self.start

    def to_range(self):
        return Range(self.start, self.stop)


def Range(*args, **kwargs) -> Tensor:
    try:
        start, stop = args
    except ValueError:
        start = 0
        (stop,) = args
    shape = Domain(start, stop)
    return Tensor(shape, prefix="count", **kwargs)


def Global(*, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="global")
    return Tensor(name=name)


def Dat(shape: Tuple[int, ...], *, name: str = None) -> Tensor:
    return Tensor(shape, name=name, prefix="dat")


def Mat(shape: Tuple[int, ...], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(shape, name=name)

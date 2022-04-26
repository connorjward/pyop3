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

    # TODO open question about whether it is better to have broadcast_indices as input
    def __init__(self, indices=(), within_indices=frozenset(), *, mesh=None, name: str = None, prefix: str=None):
        # within_indices are the ones that are not slices
        assert len(within_indices) <= len(indices)

        self.indices = indices
        self.within_indices = within_indices
        self.mesh = mesh
        self.name = name or self.name_generator.generate(prefix or self.prefix)

    def __getitem__(self, indices):
        indices = as_tuple(indices)

        if len(indices) > len(self.broadcast_indices):
            raise ValueError

        new_indices = list(self.indices)
        new_within_indices = set(self.within_indices)
        index_iterator = list(indices)
        for i, index in enumerate(self.indices):
            if index not in self.within_indices and index_iterator:
                new_index = index_iterator.pop(0)
                if self.is_slice(new_index):
                    # add another broadcast index
                    if isinstance(new_index, Tensor):
                        new_indices[i] = new_index.index
                    else:
                        assert isinstance(new_index, slice)
                        assert new_index.step == 1
                        new_indices[i] = Range(new_index.start, new_index.stop).index
                else:
                    new_indices[i] = new_index
                    new_within_indices.add(index)

        return self.copy(indices=tuple(new_indices), within_indices=frozenset(new_within_indices))

    @property
    def order(self):
        return len(self.broadcast_indices)

    @property
    def broadcast_indices(self):
        return tuple(idx for idx in self.indices if idx not in self.within_indices)

    # TODO which name is better?
    @property
    def unbound_indices(self):
        return self.broadcast_indices

    @property
    def index(self):
        assert self.order == 1
        bindex, = self.broadcast_indices
        return Index(self, bindex.start, bindex.stop)

    def copy(self, **kwargs):
        indices = kwargs.get("indices", self.indices)
        within_indices = kwargs.get("within_indices", self.within_indices)
        mesh = kwargs.get("indices", self.mesh)
        name = kwargs.get("name", self.name)
        return type(self)(indices=indices, within_indices=within_indices, mesh=mesh, name=name)

    @staticmethod
    def is_slice(index):
        return isinstance(index, (slice, Tensor)) or index is None


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

        self.index = Index(self, self.start, self.stop)

    @property
    def order(self):
        return 1


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
    indices = tuple(Range(extent).index for extent in shape)
    if not name:
        name = Tensor.name_generator.generate(prefix="dat")
    return Tensor(indices, name=name)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(shape, name=name)

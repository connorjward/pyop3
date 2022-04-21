import collections.abc

import pyop3.domains
import pyop3.exprs


def flatten_inames(index):
    if index.domain.parent_index:
        return frozenset({index.name}) + flatten_inames(index.domain.parent_index)
    else:
        return frozenset({index.name})


class IndexedTensor:
    def __init__(self, tensor, indices, broadcast=False):
        if type(tensor) != Dat:
            raise NotImplementedError("Only dealing with dats atm")

        self.tensor = tensor

        # if we are indexing a vector indices must have length 1
        self.indices = indices

        (self.index,) = indices

        # this is needed for taking slices
        self.broadcast = broadcast

    @property
    def within_inames(self):
        (index,) = self.indices

        # if broadcasting then the innermost index is not included here
        if self.broadcast:
            index = index.domain.parent_index

        return flatten_inames(index)

    @property
    def name(self):
        return self.tensor.name


class Tensor:
    ...


class Global(Tensor):

    shape = ()

    def __init__(self, name):
        self.name = name


class Dat(Tensor):
    def __init__(self, shape, name):
        if not isinstance(shape, collections.abc.Sequence):
            shape = (shape,)

        self.shape = tuple(shape)
        self.name = name

    def __str__(self):
        return self.name

    def __getitem__(self, index):
        """You can index a dat with a domain to get an argument to a loop.

        **collective**
        """
        # index must be a PointIndex (which is a multiindex plus maps)
        # because you can index either with dat[star(p)] or dat[star(p).index]
        if isinstance(index, pyop3.domains.Domain):
            index = index.index
            broadcast_indices = frozenset({index})
        else:
            broadcast_indices = frozenset()

        return IndexedTensor(self, (index,), broadcast_indices)


class Mat(Tensor):
    def __init__(self, shape, name):
        ...

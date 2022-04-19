import collections.abc
from typing import Tuple


class PointIndex:

    def __init__(self, set):
        self.set = set


class PointSet:

    def __init__(self, dtl_expr=None):
        self.index = PointIndex(self)
        if dtl_expr:
            self.dtl_expr = dtl_expr
        else:
            self.dtl_expr = "I_ijk" # some identity tensor thing indexed with 3 indices (the middle one is unity)


def closure(index: PointIndex):
    te = "closure expression_i,j,k"
    return PointSet(index.set.dtl_expr*te)


class Tensor:

    def __init__(self, shape: Tuple[int]):
        self.shape = shape

    def __getitem__(self, indices):
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)

        if len(indices) != len(self.shape):
            raise ValueError  # currently can only index tensors into scalar expressions

        return IndexedTensor(indices)

    @property
    def order(self):
        return len(self.space)


class IndexedTensor(Tensor):

    shape = ()

    def __init__(self, indices):
        self.indices = indices

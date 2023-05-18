import abc
from typing import Sequence

import pytools

from pyop3.distarray.base import DistributedArray
from pyop3.distarray.multiarray import MultiArray
from pyop3.distarray.petsc import PetscMat
from pyop3.index import IndexTree
from pyop3.multiaxis import fill_shape
from pyop3.utils import just_one


class IndexedArray(pytools.ImmutableRecord, abc.ABC):
    fields = {"data", "indices"}

    def __init__(self, data: DistributedArray, indices: Sequence[IndexTree]) -> None:
        super().__init__()
        self.data = data
        self.indices = tuple(indices)


class IndexedMultiArray(IndexedArray):
    def __init__(self, data: MultiArray, index: IndexTree) -> None:
        # lets you do myarray[...]
        if index is Ellipsis:
            index = fill_shape(data.axes)

        super().__init__(data, [index])

    @property
    def index(self):
        return just_one(self.indices)


class IndexedPetscMat(IndexedArray):
    def __init__(
        self, data: PetscMat, row_index: IndexTree, col_index: IndexTree
    ) -> None:
        super().__init__(data, [row_index, col_index])

    @property
    def row_index(self):
        return self.indices[0]

    @property
    def col_index(self):
        return self.indices[1]

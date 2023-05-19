from typing import FrozenSet, Hashable, Sequence

import pytools

from pyop3.multiaxis import MultiAxis, MultiAxisComponent, MultiAxisTree
from pyop3.tree import Node, Tree, previsit
from pyop3.utils import UniqueNameGenerator, checked_zip, just_one

__all__ = ["Dat"]


class Dat:
    def __init__(self, mesh, layout):
        # this does not work nicely with inserting the mesh axis since only parts
        # of the mesh axis are desired
        # oh crud, what does that do for the global numbering/permutation that we have?
        # I suppose that we only need to use certain bits of the global numbering.
        # In other words certain entries have zero DoFs? Maybe add zero-DoF axes
        # to the list prior to doing the function below?
        layout += ConstrainedMultiAxis(mesh.axis, priority=10)

    def __getitem__(self, indices: IndexTree) -> IndexedMultiArray:
        if not is_fully_indexed(self.array.axes, indices):
            raise ValueError("Dats must be fully indexed")
        return self.array[indices]

    # TODO: Use darray as the name of the property for the underlying data structure.
    @property
    def data(self):
        return self.darray.data


def create_dat(mesh, axes) -> Dat:
    # axes is a list of multi-axes with additional constraint information
    pass

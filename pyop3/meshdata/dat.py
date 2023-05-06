from typing import FrozenSet, Hashable, Sequence

import pytools

from pyop3.multiaxis import MultiAxis
from pyop3.tree import Node, Tree
from pyop3.utils import Node, checked_zip, UniqueNameGenerator


__all__ = ["Dat"]


DEFAULT_PRIORITY = 100


class ConstrainedMultiAxis(Node):
    fields = {"axes", "within_axes", "priority", "id"}
    # TODO We could use 'id' to set the priority
    # via commandline options

    _id_generator = UniqueNameGenerator("_ConstrainedMultiAxis_")

    def __init__(
        self,
        axes: Sequence[MultiAxis],
        *,
        within_axes: FrozenSet[Hashable] = frozenset(),
        priority: int = DEFAULT_PRIORITY,
        id: Hashable | None = None
    ):
        self.axes = tuple(axes)
        self.within_axes = within_axes
        self.priority = priority
        super().__init__(id or next(self._id_generator))


class Dat:
    def __init__(self, mesh, layout):
        # this does not work nicely with inserting the mesh axis since only parts
        # of the mesh axis are desired
        # oh crud, what does that do for the global numbering/permutation that we have?
        # I suppose that we only need to use certain bits of the global numbering.
        # In other words certain entries have zero DoFs? Maybe add zero-DoF axes
        # to the list prior to doing the function below?
        layout += ConstrainedMultiAxis(mesh.axis, priority=10)


def create_dat(mesh, axes) -> Dat:
    # axes is a list of multi-axes with additional constraint information
    pass


def order_axes(layout):
    # TODO add missing bits to the mesh axes (zero-sized) here or,
    # more likely, in the dat constructor
    tree = Tree()
    for axes in layout:
        inserted = _insert_axes(tree, axes)
        if not inserted:
            raise ValueError("Axes do not obey the provided constraints")

    # now turn the tree into a proper multi-axis
    return _create_multiaxis(tree)


def _can_insert_before(new_axes, current_axes, within_labels):
    return (
        new_axes.priority < current_axes.priority
        and new_axes.within_labels <= within_labels)


def _can_insert_after(new_axes, current_axes, within_labels):
    return (
        new_axes.priority >= current_axes.priority
        and new_axes.within_labels <= within_labels | current_axes.label)


def _insert_axes(
    tree: Tree,
    new_axes: ConstrainedMultiAxis,
    current_axes: ConstrainedMultiAxis | None = None,
    within_labels: FrozenSet[Hashable] = frozenset(),
):
    inserted = False
    if (not current_axes or _can_insert_before(new_axes, current_axes, within_labels):
        parent_axes = tree.parent(current_axes)
        subtree = tree.pop_subtree(current_axes)
        tree.add_node(new_axes, parent_axes)
        for _ in new_axes:
            tree.add_subtree(subtree, new_axes, uniquify=True)
        inserted = True

    elif (tree.is_leaf(current_axes)
            and _can_insert_after(new_axes, current_axes, within_labels)):
        tree.add_nodes([new_axes]*tree.nchildren(current_axes), current_axes)
        inserted = True

    else:
        for child in tree.children(current_axes):
            inserted = inserted or _insert_axes(
                tree, new_axes, child, within_labels|current_axes.label)
    return inserted


def _create_multiaxis(tree):
    if not tree:
        raise ValueError

    axis = tree.value.axis
    if tree.children:
        parts = [
            part.copy(subaxis=_create_multiaxis(child))
            for part, child in checked_zip(axis.parts, tree.children)]
    else:
        parts = axis.parts
    return axis.copy(parts=parts)

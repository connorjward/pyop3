import pytools

from pyop3.multiaxis import MultiAxis
from pyop3.utils import Node, checked_zip


__all__ = ["Dat"]


class ConstrainedMultiAxis(pytools.ImmutableRecord):
    fields = {"axis", "within_parts", "priority"}

    def __init__(self, axis: MultiAxis, within_parts=frozenset(), priority: int = 100):
        # not sure if this is valid - what about extruded?
        if any(part.subaxis for part in axis.parts):
            raise ValueError("Provided axes must not have subaxes")

        self.axis = axis
        self.within_parts = within_parts
        self.priority = priority

    @property
    def id(self):
        return self.axis.id


class Dat:
    def __init__(self, mesh, layout):
        layout += ConstrainedMultiAxis(mesh.axis, priority=10)


def create_dat(mesh, axes) -> Dat:
    # axes is a list of multi-axes with additional constraint information
    pass


def order_axes(axes):
    # TODO add missing bits to the mesh axes (zero-sized) here or,
    # more likely, in the dat constructor
    tree = None
    for axis in axes:
        tree = _insert_axis(tree, axis)

    # now turn the tree into a proper multi-axis
    return _create_multiaxis(tree)


# this does not work nicely with inserting the mesh axis since only parts
# of the mesh axis are desired
# oh crud, what does that do for the global numbering/permutation that we have?
# I suppose that we only need to use certain bits of the global numbering.
# In other words certain entries have zero DoFs? Maybe add zero-DoF axes
# to the list prior to doing the function below?
def _insert_axis(tree, new_axis, within_parts=frozenset()):
    if not tree:
        return Node(new_axis)

    # it bothers me that a lower priority is actually a higher priority (semantically)
    if new_axis.priority < tree.value.priority:
        return Node(new_axis, children=[tree for _ in new_axis.axis.parts])

    if not tree.children:
        return tree.copy(children=[Node(new_axis) for _ in tree.value.axis.parts])

    # TODO raise an error if an axis is not placed anywhere
    new_children = [
        _insert_axis(child, new_axis, within_parts|part.label)
        for child, part in checked_zip(tree.children, tree.value.axis.parts)]
    return tree.copy(children=new_children)


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

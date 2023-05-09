from typing import FrozenSet, Hashable, Sequence

import pytools

from pyop3.multiaxis import MultiAxisTree, MultiAxisComponent
from pyop3.tree import Node, Tree, previsit
from pyop3.utils import checked_zip, UniqueNameGenerator, just_one


__all__ = ["Dat"]


DEFAULT_PRIORITY = 100


class InvalidConstraintsException(Exception):
    pass


class ConstrainedMultiAxis(pytools.ImmutableRecord):
    fields = {"axis", "priority", "within_labels"}
    # TODO We could use 'label' to set the priority
    # via commandline options

    def __init__(
        self,
        axis: Sequence[MultiAxisComponent],
        *,
        priority: int = DEFAULT_PRIORITY,
        within_labels: FrozenSet[Hashable] = frozenset(),
    ):
        self.axis = tuple(axis)
        self.priority = priority
        self.within_labels = frozenset(within_labels)
        super().__init__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(axis=({', '.join(str(axis_cpt) for axis_cpt in self.axis)}), priority={self.priority}, within_labels={self.within_labels})"


class ConstrainedMultiAxisNode(Node):
    fields = Node.fields | {"axis", "parent_label"}

    _id_generator = UniqueNameGenerator("_id_ConstrainedMultiAxisNode")

    def __init__(
        self, axis, parent_label: Hashable | None = None, *, id: Hashable | None = None
    ):
        self.axis = axis
        self.parent_label = parent_label
        super().__init__(id or next(self._id_generator))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis}, parent_label={self.parent_label})"


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
    axes = Tree()
    layout = list(layout)
    history = set()
    while layout:
        if tuple(layout) in history:
            raise ValueError("Seen this before, cyclic")
        history.add(tuple(layout))

        constrained_axis = layout.pop(0)
        inserted = _insert_axis(axes, constrained_axis, None)
        if not inserted:
            layout.append(constrained_axis)

    # should be able to delete
    # _check_constraints(tree)

    # now turn the tree into a proper multi-axis
    return _create_multiaxis(axes)
    return axes


def _insert_axis(
    tree: Tree,
    new_axes: ConstrainedMultiAxis,
    current_axes: ConstrainedMultiAxisNode,
    within_labels: FrozenSet[Hashable] = frozenset(),
):
    if tree.root and current_axes is None:
        current_axes = tree.root
    # cleanup
    # don't count this one
    within_labels -= {None}

    inserted = False

    if not tree.root:
        if not new_axes.within_labels:
            tree.add_node(ConstrainedMultiAxisNode(new_axes))
            inserted = True
        else:
            return False

    # elif _can_insert_before(new_axes, current_axes, within_labels):
    elif new_axes.priority < current_axes.axis.priority:
        # breakpoint()
        if new_axes.within_labels <= within_labels | {current_axes.parent_label}:
            # diagram or something?
            parent_axes = tree.parent(current_axes)

            subtree = tree.pop_subtree(current_axes)

            new_node = ConstrainedMultiAxisNode(new_axes, current_axes.parent_label)
            tree.add_node(new_node, parent_axes)

            # must already obey the constraints - so stick all back in
            for axis_cpt in new_axes.axis:
                stree = subtree.copy()
                stree.replace_node(stree.root.copy(parent_label=axis_cpt.label))
                tree.add_subtree(stree, new_node, uniquify=True)
            inserted = True
        else:
            # The priority is less so the axes should definitely
            # not be inserted below here - do not recurse
            pass

    elif tree.is_leaf(current_axes):
        assert new_axes.priority >= current_axes.axis.priority
        for axis_cpt in current_axes.axis.axis:
            if new_axes.within_labels <= within_labels | {axis_cpt.label}:
                new_node = ConstrainedMultiAxisNode(
                    new_axes, parent_label=axis_cpt.label
                )
                tree.add_node(new_node, parent=current_axes)  # make unique?
            inserted = True

    else:
        for child in tree.children(current_axes):
            inserted = inserted or _insert_axis(
                tree, new_axes, child, within_labels | {current_axes.parent_label}
            )
    return inserted


def _check_constraints(ctree):
    def check(node, within_labels):
        if (parent := ctree.parent(node)) and parent.priority > node.priority:
            raise ConstraintsNotMetException
        if node.within_labels > within_labels:
            raise ConstraintsNotMetException

        return within_labels | {node.label}

    previsit(ctree, check, ctree.root, frozenset())


def _create_multiaxis(tree: Tree) -> MultiAxisTree:
    axtree = MultiAxisTree()

    # note: constrained things contain multiple nodes
    def build(constrained_axes, *_):
        if parent := tree.parent(constrained_axes):
            target_parent = just_one(
                axis_cpt
                for axis_cpt in parent.axis.axis
                if axis_cpt.label == constrained_axes.parent_label
            )
        else:
            target_parent = axtree.root

        for i, part in enumerate(constrained_axes.axis.axis):
            axtree.add_node(part, target_parent, uniquify=True)

    previsit(tree, build)
    return axtree

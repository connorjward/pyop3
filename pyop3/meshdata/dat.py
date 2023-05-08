from typing import FrozenSet, Hashable, Sequence

import pytools

from pyop3.multiaxis import MultiAxisTree, MultiAxisComponent
from pyop3.tree import Node, Tree, previsit
from pyop3.utils import checked_zip, UniqueNameGenerator


__all__ = ["Dat"]


DEFAULT_PRIORITY = 100


class ConstraintsNotMetException(Exception):
    pass


class ConstrainedMultiAxis(Node):
    fields = {"axes", "within_labels", "priority", "label"} | Node.fields
    # TODO We could use 'label' to set the priority
    # via commandline options

    _label_generator = UniqueNameGenerator("_label_ConstrainedMultiAxis_")
    _id_generator = UniqueNameGenerator("_id_ConstrainedMultiAxis_")

    def __init__(
        self,
        axes: Sequence[MultiAxisComponent],
        *,
        within_labels: FrozenSet[Hashable] = frozenset(),
        priority: int = DEFAULT_PRIORITY,
        label: Hashable | None = None,
        id: Hashable | None = None,
    ):
        self.axes = tuple(axes)
        self.within_labels = frozenset(within_labels)
        self.priority = priority
        self.label = label or next(self._label_generator)
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
    layout = list(layout)
    history = set()
    while layout:
        if layout in history:
            raise ValueError("Seen this before, cyclic")

        history.add(tuple(layout))

        axes = layout.pop(0)
        inserted = _insert_axes(tree, axes, tree.root)
        if not inserted:
            layout.append(axes)

    _check_constraints(tree)

    # now turn the tree into a proper multi-axis
    return _create_multiaxis(tree)


def _can_insert_before(new_axes, current_axes, within_labels):
    breakpoint()
    # 1. compare priorities
    if new_axes.priority < current_axes.priority:
        return True
    # 2. equal priorities, look at dependencies
    elif (
        new_axes.priority == current_axes.priority
        # and new_axes.label in current_axes.within_labels
    ):
        return True
    # 3. cannot insert before
    else:
        return False


def _can_insert_after(new_axes, current_axes, within_labels):
    # breakpoint()
    return new_axes.priority >= current_axes.priority


def _insert_axes(
    tree: Tree,
    new_axes: ConstrainedMultiAxis,
    current_axes: ConstrainedMultiAxis,
    within_labels: FrozenSet[Hashable] = frozenset(),
):
    inserted = False

    if not tree.root:
        tree.root = new_axes
        inserted = True

    elif _can_insert_before(new_axes, current_axes, within_labels):
        parent_axes = tree.parent(current_axes)
        subtree = tree.pop_subtree(current_axes)
        tree.add_node(new_axes, parent_axes)
        for _ in range(len(new_axes.axes)):
            tree.add_subtree(subtree, new_axes, uniquify=True)
        inserted = True

    elif (tree.is_leaf(current_axes)):
            # and _can_insert_after(new_axes, current_axes, within_labels)):

        # FIXME: this won't work with constraints!

        tree.add_nodes([new_axes]*len(current_axes.axes), current_axes, uniquify=True)
        inserted = True

    else:
        for child in tree.children(current_axes):
            inserted = inserted or _insert_axes(
                tree, new_axes, child, within_labels|current_axes.label)
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
        if (parent := tree.parent(constrained_axes)):
            child_index = tree.children(parent).index(constrained_axes)
            target_parent = parent.axes[child_index]
        else:
            target_parent = axtree.root

        for i, part in enumerate(constrained_axes.axes):
            axtree.add_node(part, target_parent, uniquify=True)

    previsit(tree, build)
    return axtree

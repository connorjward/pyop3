import dataclasses
from typing import Any

from collections.abc import Hashable, Sequence

import pytools
from pyop3.utils import strictly_all, just_one


__all__ = ["Node", "Tree"]


class NodeNotFoundException(Exception):
    pass


class Node(pytools.ImmutableRecord):
    fields = {"id"}

    def __init__(self, id: Hashable):
        self.id = id


# FIXME I run into trouble when I modify multiaxistrees after
# they have been set up. I think I should spike the modifiers, or set
# frozen or something to prevent this - needs some thought
class Tree(pytools.RecordWithoutPickling):
    fields = set()

    def __init__(self):
        super().__init__()
        self.reset()

    def __str__(self):
        return self._stringify()

    def add_node(self, node: Node, parent: Node | str | None = None) -> None:
        if not parent:
            self._add_root(node)
        else:
            parent = self._as_node(parent)
            self._check_exists(parent)
            self._ids_to_nodes[node.id] = node
            self._parent_to_children[parent.id] += (node.id,)
            self._parent_to_children[node.id] = ()
            self._child_to_parent[node.id] = parent.id

    def add_nodes(self, nodes: Sequence[Node], parent: Node | str) -> None:
        for node in nodes:
            self.add_node(node, parent)

    def replace_node(self, node: Node) -> None:
        """Replace a node in the tree with another.

        The new node must have the same ``id`` as the old one.

        This function is useful when one wants nodes to be immutable.

        Parameters
        ----------
        node
            The new node to be inserted.

        """
        if node.id not in self._node_ids:
            raise NodeNotFoundException(
                f"{node.id} is not in the tree, replacement is not possible")
        self._ids_to_nodes[node.id] = node

    def parent(self, node: Node | str) -> Node | None:
        node = self._as_node(node)

        if node == self.root:
            return None

        try:
            parent_id = self._child_to_parent[node.id]
        except KeyError:
            raise NodeNotFoundException(f"{node.id} is not present in the tree")
        return self._ids_to_nodes[parent_id]

    def children(self, node: Node | str) -> tuple[Node]:
        node = self._as_node(node)

        try:
            child_ids = self._parent_to_children[node.id]
        except KeyError:
            raise NodeNotFoundException(f"{node.id} is not present in the tree")
        return tuple(self._ids_to_nodes[child_id] for child_id in child_ids)

    def node(self, node_id: str) -> Node:
        """Return the node from the tree matching the provided ``id``.

        Raises an exception if ``id`` is not found in the tree.
        """
        try:
            return self._ids_to_nodes[node_id]
        except KeyError:
            raise NodeNotFoundException(f"{node_id} is not present in the tree")

    @property
    def root(self) -> Node | None:
        return self._root

    @root.setter
    def root(self, node) -> None:
        self.reset()
        self._add_root(node)

    @property
    def leaves(self) -> tuple[Node]:
        return tuple(
            node for nid, node in self._ids_to_nodes.items()
            if not self._parent_to_children[nid])

    @property
    def leaf(self) -> Node:
        return just_one(self.leaves)

    def is_leaf(self, node: Node | str) -> bool:
        node = self._as_node(node)
        self._check_exists(node)
        return len(self._parent_to_children[node.id]) == 0

    @property
    def is_empty(self) -> bool:
        return not self.root

    @property
    def depth(self) -> int:
        if self.is_empty:
            return 0
        count = lambda _, *o: max(o or [0]) + 1
        return postvisit(self, count)

    def reset(self):
        self._root = None
        self._ids_to_nodes = {}
        self._parent_to_children = {}
        self._child_to_parent = {}

    def copy(self, **kwargs):
        dup = super().copy(**kwargs)
        dup._root = self._root.copy()
        dup._ids_to_nodes = self._ids_to_nodes.copy()
        dup._parent_to_children = self._parent_to_children.copy()
        dup._child_to_parent = self._child_to_parent.copy()
        return dup

    def _check_exists(self, node: Node | str) -> None:
        if (node_id := self._as_node(node).id) not in self._node_ids:
            raise NodeNotFoundException(f"{node_id} is not present in the tree")

    def _as_node(self, node: Node | str) -> Node:
        return node if isinstance(node, Node) else self.node(node)

    @property
    def _node_ids(self):
        return self._ids_to_nodes.keys()

    def _add_root(self, node: Node):
        if self.root:
            raise ValueError("The tree already has a root")
        self._root = node
        self._ids_to_nodes[node.id] = node
        self._parent_to_children[node.id] = ()
        self._child_to_parent[node.id] = None

    def _stringify(
        self,
        node: Node | str | None = None,
        begin_prefix: str = "",
        cont_prefix: str = "",
    ) -> list[str] | str:
        if not node:
            node = self.root
        node = self._as_node(node)

        nodestr = [f"{begin_prefix}{node}"]
        for i, child in enumerate(children := self.children(node)):
            last_child = i == len(children) - 1
            next_begin_prefix = f"{cont_prefix}{'└' if last_child else '├'}──➤ "
            next_cont_prefix = f"{cont_prefix}{' ' if last_child else '│'}    "
            nodestr += self._stringify(child, next_begin_prefix, next_cont_prefix)

        if not strictly_all([begin_prefix, cont_prefix]):
            return "\n".join(nodestr)
        else:
            return nodestr


def postvisit(tree, fn, current_node: Node | None = None) -> Any:
    """Traverse the tree in postorder.

    # TODO rewrite
    Parameters
    ----------
    tree: Tree
        The tree to be visited.
    fn: function(node, *fn_children)
        A function to be applied at each node. The function should take
        the node to be visited as its first argument, and the results of
        visiting its children as any further arguments.
    """
    if tree.is_empty:
        raise RuntimeError("Cannot traverse an empty tree")

    current_node = current_node or tree.root
    return fn(
        current_node,
        *(postvisit(tree, fn, child) for child in tree.children(current_node)))

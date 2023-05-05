import dataclasses

from collections.abc import Hashable, Sequence

import pytools


__all__ = ["Node", "Tree"]


class NodeNotFoundException(Exception):
    pass


class Node(pytools.ImmutableRecord):
    fields = {"id"}

    def __init__(self, id: Hashable):
        self.id = id


class Tree:
    def __init__(self):
        self.reset()

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

    @property
    def root(self) -> Node | None:
        return self._root

    @root.setter
    def root(self, node) -> None:
        self.reset()
        self._add_root(node)

    @property
    def is_empty(self) -> bool:
        return not self.root

    def reset(self):
        self._root = None
        self._ids_to_nodes = {}
        self._parent_to_children = {}
        self._child_to_parent = {}

    def _check_exists(self, node: Node | str) -> None:
        if (node_id := self._as_node(node).id) not in self._node_ids:
            raise NodeNotFoundException(f"{node_id} is not present in the tree")

    def _as_node(self, node: Node | str) -> Node:
        return node if isinstance(node, Node) else self._node_from_id(node)

    def _node_from_id(self, node_id: str) -> Node:
        try:
            return self._ids_to_nodes[node_id]
        except KeyError:
            raise NodeNotFoundException(f"{node_id} is not present in the tree")

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

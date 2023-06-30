from __future__ import annotations

import collections
import functools
from collections.abc import Hashable, Sequence
from typing import Any, Mapping

import pyrsistent
import pytools

from pyop3.utils import (
    UniqueNameGenerator,
    apply_at,
    as_tuple,
    flatten,
    has_unique_entries,
    just_one,
    some_but_not_all,
    strictly_all,
    unique,
)

# can I declare this as more obviously a type?
Id = Hashable
Label = Hashable


class NodeNotFoundException(Exception):
    pass


class FrozenTreeException(Exception):
    pass


class Node(pytools.ImmutableRecord):
    fields = {"degree", "id"}

    _lazy_id_generator = None

    def __init__(self, degree: int, id: Id | None = None):
        self.degree = degree
        self.id = id or next(self._id_generator)

    @classmethod
    @property
    def _id_generator(cls):
        if not cls._lazy_id_generator:
            cls._lazy_id_generator = UniqueNameGenerator(f"_{cls.__name__}_id")
        return cls._lazy_id_generator


class NodeComponent(pytools.ImmutableRecord):
    fields = {"label", "id"}

    _lazy_label_generator = None
    _lazy_id_generator = None

    def __init__(self, label: Label | None = None, id: Id | None = None):
        self.label = label if label is not None else next(self._label_generator)
        self.id = id if id is not None else next(self._id_generator)

    @classmethod
    @property
    def _label_generator(cls):
        if not cls._lazy_label_generator:
            cls._lazy_label_generator = UniqueNameGenerator(f"_label_{cls.__name__}")
        return cls._lazy_label_generator

    @classmethod
    @property
    def _id_generator(cls):
        if not cls._lazy_id_generator:
            cls._lazy_id_generator = UniqueNameGenerator(f"_id_{cls.__name__}")
        return cls._lazy_id_generator


class FixedAryTree(pytools.ImmutableRecord):
    fields = {"root", "parent_to_children"}

    def __init__(
        self,
        root: Node | None = None,
        parent_to_children: Mapping[Id, Node] | None = None,
    ) -> None:
        if root:
            if parent_to_children:
                parent_to_children = {
                    parent_id: as_tuple(children)
                    for parent_id, children in parent_to_children.items()
                }

                parent_to_children |= {
                    node.id: (None,) * node.degree
                    for node in filter(None, flatten(list(parent_to_children.values())))
                    if node.id not in parent_to_children
                }

                node_ids = [
                    node.id
                    for node in filter(None, flatten(list(parent_to_children.values())))
                ] + [root.id]
                if not has_unique_entries(node_ids):
                    raise ValueError("Nodes with duplicate IDs found")
                if any(parent_id not in node_ids for parent_id in parent_to_children):
                    raise ValueError("Parent ID not found")
                if any(
                    len(parent_to_children[node.id]) != node.degree
                    for node in filter(None, flatten(list(parent_to_children.values())))
                ):
                    raise ValueError("Node found with the wrong number of children")
            else:
                parent_to_children = {root.id: (None,) * root.degree}
        else:
            if parent_to_children:
                raise ValueError("Tree cannot have children without a root")
            else:
                parent_to_children = {}

        self.root = root
        self.parent_to_children = pyrsistent.freeze(parent_to_children)

    def __str__(self):
        return self._stringify()

    def __contains__(self, node: Node | str) -> bool:
        return self._as_node(node) in self.nodes

    def children(self, node: Node | str) -> tuple[Node]:
        node_id = self._as_existing_node_id(node)
        return self.parent_to_children[node_id]

    def put_node(
        self,
        node: Node,
        parent: Node | Id | None = None,
        component_index: int | None = None,
        uniquify: bool = False,
    ) -> None:
        if parent is None:
            if component_index is not None:
                raise ValueError("Cannot specify a component index when adding a root")
            if self.root:
                raise ValueError("Cannot add multiple roots")
            assert not self.parent_to_children
            return self.copy(root=node)
        else:
            if node in self:
                if uniquify:
                    node = node.copy(id=self._first_unique_id(node.id))
                else:
                    raise ValueError("Cannot insert a node with the same ID")

            if component_index is None:
                if parent.degree == 1:
                    component_index = 0
                else:
                    raise ValueError(
                        "Must specify a component index for axes with multiple components"
                    )

            parent_id = self._as_existing_node_id(parent)
            new_parent_to_children = dict(self.parent_to_children)
            new_children = list(new_parent_to_children[parent_id])
            new_children[component_index] = node
            new_parent_to_children[parent_id] = new_children
            return self.copy(parent_to_children=new_parent_to_children)

    # def replace_node(self, node: Node, loc=None):
    #     loc = loc or node.id
    #
    #     if loc not in self.node_ids:
    #         raise ValueError
    #
    #     parent = self.parent(loc)
    #
    #     new_parent_to_children = dict(self.parent_to_children)
    #     if parent:  # ie not root
    #         node_index = [
    #             child.id for child in self.parent_to_children[parent.id]
    #         ].index(node.id)
    #         new_children = list(self.parent_to_children[parent.id])
    #         new_children[node_index] = node
    #         new_parent_to_children[parent.id] = new_children
    #         return type(self)(self.root, new_parent_to_children)
    #     else:
    #         # this won't work if we tinker with IDs
    #         return type(self)(node, self.parent_to_children.copy())

    def child(self, parent, cpt) -> Node:
        parent = self._as_node(parent)
        cpt_label = self._as_component_label(cpt)
        cpt_index = [c.label for c in parent.components].index(cpt_label)
        return self.parent_to_children[parent.id][cpt_index]

    # def as_node
    # """Return the node from the tree matching the provided ``id``.
    #
    # Raises an exception if ``id`` is not found in the tree.
    # """
    #     try:
    #         return just_one(node for node in self.nodes if node.id == node_id)
    #     except:
    #         raise NodeNotFoundException(f"{node_id} is not present in the tree")

    @functools.cached_property
    def node_ids(self) -> frozenset[Id]:
        return frozenset(node.id for node in self.nodes)

    @functools.cached_property
    def child_to_parent(self):
        return {
            child: self.id_to_node[parent_id]
            for parent_id, children in self.parent_to_children.items()
            for child in children
        }

    @functools.cached_property
    def id_to_node(self):
        return {node.id: node for node in self.nodes}

    @functools.cached_property
    def nodes(self) -> frozenset[Node]:
        return frozenset({self.root}) | {
            node
            for node in filter(None, flatten(list(self.parent_to_children.values())))
        }

    def _as_existing_node(self, node: Node | str) -> Node:
        node = node if isinstance(node, Node) else self.find_node(node)
        if node.id not in self.node_ids:
            raise NodeNotFoundException(f"{node.id} is not present in the tree")
        return node

    def _as_existing_node_id(self, node: Node | Id) -> Id:
        node_id = node.id if isinstance(node, Node) else node
        if node_id not in self.node_ids:
            raise NodeNotFoundException(f"{node_id} is not present in the tree")
        return node_id

    def _as_component_label(self, cpt: NodeComponent | Label) -> Label:
        if isinstance(cpt, NodeComponent):
            return cpt.label
        else:
            return cpt

    def parent(self, node: Node | Id) -> Node | None:
        node = self._as_existing_node(node)

        if node == self.root:
            return None
        else:
            return self.child_to_parent[node]

    def pop_subtree(self, subroot: Node | str) -> "Tree":
        subroot = self._as_node(subroot)
        self._check_exists(subroot)

        if self._is_root(subroot):
            subtree = self.copy()
            self.reset()
            return subtree

        subtree = Tree(subroot)

        nodes_and_parents = []

        def collect_node_and_parent(node, _):
            nodes_and_parents.append((node, self.parent(node)))

        previsit(self, collect_node_and_parent, subroot)

        for node, parent in nodes_and_parents:
            if node != subroot:
                subtree.add_node(node, parent)
            del self._ids_to_nodes[node.id]
            del self._child_to_parent[node.id]
            self.parent_to_children[parent.id] = tuple(
                child
                for child in self.parent_to_children[parent.id]
                if child != node.id
            )

        return subtree

    def add_subtree(
        self,
        subtree: LabelledTree,
        parent: Node | Id | None = None,
        component: NodeComponent | Label | None = None,
        uniquify: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        etc
            ...
        uniquify
            If ``False``, duplicate ``ids`` between the tree and subtree
            will raise an exception. If ``True``, the ``ids`` will be changed
            to avoid the clash.
        """
        if uniquify:
            raise NotImplementedError("TODO")

        if some_but_not_all([parent, component]):
            raise ValueError("Both parent and component must be defined")

        if not parent:
            return subtree
        else:
            parent = self._as_node(parent)
            cpt_label = self._as_component_label(component)
            cpt_index = [c.label for c in parent.components].index(cpt_label)
            new_parent_to_children = {
                p: list(ch) for p, ch in self.parent_to_children.items()
            }
            new_parent_to_children[parent.id][cpt_index] = subtree.root
            new_parent_to_children |= subtree.parent_to_children
            return self.copy(parent_to_children=new_parent_to_children)

    # alias, better?
    def _to_node_id(self, arg):
        return self._as_node_id(arg)

    # @property
    # def leaves(self) -> tuple[Node]:
    #     return tuple(
    #         (node, cidx)
    #         for node in self.nodes
    #         for cidx in range(node.degree)
    #         if self.parent_to_children[node.id][cidx] is None
    #     )

    @property
    def leaf(self) -> Node:
        return just_one(self.leaves)

    def is_leaf(self, node: Node | str) -> bool:
        node = self._as_node(node)
        self._check_exists(node)
        return all(child is None for child in self.parent_to_children[node.id])

    @property
    def is_empty(self) -> bool:
        return not self.root

    @property
    def depth(self) -> int:
        if self.is_empty:
            return 0
        count = lambda _, *o: max(o or [0]) + 1
        return postvisit(self, count)

    def _check_exists(self, node: Node | str) -> None:
        if (node_id := self._as_node(node).id) not in self.node_ids:
            raise NodeNotFoundException(f"{node_id} is not present in the tree")

    def _first_unique_id(self, node: Node | Hashable, sep: str = "_") -> str:
        orig_node_id = self._as_node_id(node)
        if orig_node_id not in self:
            return orig_node_id

        counter = 0
        node_id = f"{orig_node_id}{sep}{counter}"
        while node_id in self:
            counter += 1
            node_id = f"{orig_node_id}{sep}{counter}"
        return node_id

    def _as_node(self, node: Node | Hashable) -> Node:
        return node if isinstance(node, Node) else self.find_node(node)

    def _as_node_id(self, node: Node | Hashable) -> Hashable:
        return node.id if isinstance(node, Node) else node

    def _stringify(
        self,
        node: Node | Hashable | None = None,
        begin_prefix: str = "",
        cont_prefix: str = "",
    ) -> list[str] | str:
        node = self._as_node(node) if node else self.root

        nodestr = [f"{begin_prefix}{node}"]
        for i, child in enumerate(children := self.children(node)):
            last_child = i == len(children) - 1
            next_begin_prefix = f"{cont_prefix}{'└' if last_child else '├'}──➤ "
            next_cont_prefix = f"{cont_prefix}{' ' if last_child else '│'}    "
            if child is not None:
                nodestr += self._stringify(child, next_begin_prefix, next_cont_prefix)
            else:
                nodestr += [f"{next_begin_prefix}None"]

        if not strictly_all([begin_prefix, cont_prefix]):
            return "\n".join(nodestr)
        else:
            return nodestr


# FIXME Components should live here I think
class LabelledNode(Node):
    fields = Node.fields | {"label"}

    _lazy_label_generator = None

    def __init__(self, label: Hashable | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.label = label or next(self._label_generator)

    def with_modified_component(self, component_index: int | None = None, **kwargs):
        if component_index is None:
            if self.degree == 1:
                component_index = 0
            else:
                raise ValueError(
                    "Must specify a component index for multi-component nodes"
                )

        new_components = apply_at(
            lambda c: c.copy(**kwargs), self.components, component_index
        )
        return self.copy(components=new_components)

    @classmethod
    @property
    def _label_generator(cls):
        if not cls._lazy_label_generator:
            cls._lazy_label_generator = UniqueNameGenerator(f"_{cls.__name__}_label")
        return cls._lazy_label_generator


NodePath = dict[Hashable, Hashable]
"""Mapping from axis labels to component labels."""
# wrong now


class LabelledTree(FixedAryTree):
    def with_modified_node(self, node, **kwargs):
        return self.replace_node(node.copy(**kwargs))

    def with_modified_component(self, node, component_index=None, **kwargs):
        return self.replace_node(
            node.with_modified_component(component_index, **kwargs)
        )

    def put_node(
        self,
        node: Node,
        parent: Mapping[Node | Hashable, int] | Node | Hashable | None = None,
        component_index=None,
        uniquify: bool = False,
    ) -> None:
        if isinstance(parent, Mapping):
            parent = self._node_from_path(parent)
        return super().put_node(node, parent, component_index, uniquify)

    def _node_from_path(self, path: Mapping[Node | Hashable, int]) -> Node:
        if not path:
            return self.root

        path_ = path.copy()
        node = self.root
        while path_:
            component_index = path_.pop(node.label)
            node = self.parent_to_children[node.id][component_index]
        return node

    # def find_node(
    #     self, loc: Mapping[Hashable, int] | tuple[Node | Hashable, int] | None = None
    # ) -> LabelledNode:
    #     if isinstance(loc, Mapping):
    #         return self._node_from_path(loc)
    #     else:
    #         return super().find_node(loc)

    # def children(self, node: LabelledNode | Hashable | NodePath) -> tuple[Node]:
    #     if isinstance(node, Mapping):
    #         child = self.from_path(node)
    #         return (child,) if child else ()
    #     else:
    #         return super().children(node)

    # def add_subtree(
    #     self,
    #     subtree: "Tree",
    #     parent: NodePath = None,
    #     uniquify: bool = False,
    # ) -> None:
    #     """
    #     Parameters
    #     ----------
    #     etc
    #         ...
    #     uniquify
    #         If ``False``, duplicate ``ids`` between the tree and subtree
    #         will raise an exception. If ``True``, the ``ids`` will be changed
    #         to avoid the clash.
    #     """
    #     if parent:
    #         myparent, parentlabel = self._node_from_path(parent)
    #         self._check_exists(myparent)
    #         myouterpath = self._node_to_path[myparent.id] or {}
    #         myouterpath |= {myparent.label: parentlabel}
    #     else:
    #         myouterpath = {}
    #
    #     def add_subtree_node(node, parent_id):
    #         if parent_id == myparent.id:
    #             path = myouterpath
    #         else:
    #             path = myouterpath | subtree._node_to_path[parent_id]
    #         if uniquify:
    #             node = node.copy(id=self._first_unique_id(node))
    #         self.add_node(node, path)
    #         return node.id
    #
    #     previsit(subtree, add_subtree_node, None, myparent.id)

    def pop_subtree(self, subroot: Node | str) -> "Tree":
        subroot = self._as_node(subroot)
        self._check_exists(subroot)

        if self._is_root(subroot):
            subtree = self.copy()
            self.reset()
            return subtree

        subtree = type(self)(subroot)

        nodes_and_parents = []

        def collect_node_and_parent(node, _):
            nodes_and_parents.append((node, self._child_to_parent_and_label[node.id]))

        previsit(self, collect_node_and_parent, subroot)

        for node, parent in nodes_and_parents:
            parent_id, component_label = parent
            if node != subroot:
                subtree.add_node(node, parent)
            del self._ids_to_nodes[node.id]
            del self._child_to_parent[node.id]
            self.parent_to_children[parent_id] = tuple(
                child
                for child in self.parent_to_children[parent_id]
                if child != node.id
            )

            del self._parent_and_label_to_child[(parent_id, component_label)]
            del self._child_to_parent_and_label[node.id]
            del self._node_to_path[node.id]
            for component in node.components:
                del self._parent_and_label_to_child[(node.id, component.label)]

        return subtree

    def child_by_label(self, node: LabelledNode | Hashable, label: Hashable):
        node_id = self._as_existing_node_id(node)
        child = self._parent_and_label_to_child[node_id, label]
        if child is not None:
            return self._as_node(child)
        else:
            return None

    @classmethod
    def from_dict(
        cls,
        node_dict: dict[Node, Hashable],
        set_up: bool = False,
    ) -> "LabelledTree":  # -> subclass?
        tree = cls()
        node_queue = list(node_dict.keys())
        history = set()
        while node_queue:
            if tuple(node_queue) in history:
                raise ValueError("cycle!")

            history.add(tuple(node_queue))

            node = node_queue.pop(0)
            parent_info = node_dict[node]
            if parent_info is None:
                tree.add_node(node)
            else:
                parent_id, parent_label = parent_info
                if parent_id in tree._node_ids:
                    tree.add_node(node, (parent_id, parent_label))
                else:
                    node_queue.append(node)

        if set_up:
            tree.set_up()

        return tree

    @functools.cached_property
    def _paths(self):
        def paths_fn(node, component_index, prev):
            prev = prev or []
            new_path = prev + [(node, component_index)]
            paths_[node, component_index] = new_path
            return new_path

        paths_ = {}
        previsit(self, paths_fn)
        return pyrsistent.freeze(paths_)

    @functools.cached_property
    def leaves(self) -> tuple[tuple[Node, NodeComponent]]:
        """Return the leaves of the tree."""

        def leaves_fn(node, cpt, prev):
            if not self.child(node, cpt):
                leaves_.append((node, cpt))

        leaves_ = []
        previsit(self, leaves_fn)
        return tuple(leaves_)

    def ancestors(self, node, component_index):
        """Return the ancestors of a ``(node_id, component_label)`` 2-tuple."""
        return self.path(node, component_index)[:-1]

    def path(self, node, component_index):
        return self._paths[node, component_index]


# better alias?
MultiTree = LabelledTree


# def previsit(
#     tree, fn, current_node: Node | None = None, prev_result: Any | None = None
# ) -> Any:
#     if tree.is_empty:
#         raise RuntimeError("Cannot traverse an empty tree")
#
#     current_node = current_node or tree.root
#
#     result = fn(current_node, prev_result)
#     for child in tree.children(current_node):
#         previsit(tree, fn, child, result)
#
#
def previsit(
    tree,
    fn,
    current_node: Node | None = None,
    prev=None,
) -> Any:
    if tree.is_empty:
        raise RuntimeError("Cannot traverse an empty tree")

    current_node = current_node or tree.root
    for cpt in current_node.components:
        next = fn(current_node, cpt, prev)
        if subnode := tree.child(current_node, cpt):
            previsit(tree, fn, subnode, next)


def postvisit(tree, fn, current_node: Node | None = None, **kwargs) -> Any:
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
        *(
            postvisit(tree, fn, child, **kwargs)
            for child in filter(None, tree.children(current_node))
        ),
        **kwargs,
    )

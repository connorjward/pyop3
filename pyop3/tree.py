from __future__ import annotations

import collections
import functools
from collections.abc import Hashable, Sequence
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

import pyrsistent
import pytools
from pyrsistent import freeze, pmap

from pyop3.utils import (
    Id,
    Label,
    LabelledImmutableRecord,
    UniquelyIdentifiedImmutableRecord,
    apply_at,
    as_tuple,
    checked_zip,
    flatten,
    has_unique_entries,
    just_one,
    map_when,
    some_but_not_all,
    strictly_all,
    unique,
)


class NodeNotFoundException(Exception):
    pass


class NodeData(pytools.ImmutableRecord):
    pass


# type aliases
NodeId = Id
ComponentLabel = Label


class LabelledNode(LabelledImmutableRecord):
    pass


class StrictLabelledNode(LabelledNode):
    fields = LabelledNode.fields | {"component_labels"}

    def __init__(self, component_labels, **kwargs):
        super().__init__(**kwargs)
        self.component_labels = as_tuple(component_labels)

    @property
    def degree(self) -> int:
        return len(self.component_labels)


# old alias
Node = LabelledNode


# TODO I don't think that this should be considered an immutable record. The fields
# relate to one another and it encourages mutability (via copy) rather than using a
# specific interface
class LabelledTree(pytools.ImmutableRecord):
    fields = {"root", "parent_to_children"}

    def __init__(self, root, parent_to_children):
        self.root = root
        self.parent_to_children = freeze(parent_to_children)

    def __str__(self):
        return self._stringify()

    def __contains__(self, node: Union[Node, str]) -> bool:
        return self._as_node(node) in self.nodes

    @property
    def is_empty(self) -> bool:
        return not self.root

    def _stringify(
        self,
        node: Optional[Union[Node, Hashable]] = None,
        begin_prefix: str = "",
        cont_prefix: str = "",
    ) -> Union[List[str], str]:
        if self.is_empty:
            return "empty"
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


# TODO is this a good inheritance choice? Messes with MutableLabelledTree/FrozenLabelledTree
class StrictLabelledTree(LabelledTree):
    def __init__(
        self,
        root: Optional[Node] = None,
        parent_to_children: Optional[Mapping[Id, Node]] = None,
    ) -> None:
        if root:
            if parent_to_children:
                parent_to_children = {
                    parent_id: as_tuple(children)
                    for parent_id, children in parent_to_children.items()
                }

                parent_to_children.update(
                    {
                        node.id: (None,) * node.degree
                        for node in filter(
                            None, flatten(list(parent_to_children.values()))
                        )
                        if node.id not in parent_to_children
                    }
                )

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

        super().__init__(root, parent_to_children)

    @property
    def depth(self) -> int:
        if self.is_empty:
            return 0
        count = lambda _, *o: max(o or [0]) + 1
        return postvisit(self, count)

    def children(self, node: Union[Node, str]) -> Tuple[Node]:
        node_id = self._as_node_id(node)
        return self.parent_to_children[node_id]

    def child(
        self, parent: Union[LabelledNode, NodeId], component_label: ComponentLabel
    ) -> LabelledNode:
        parent = self._as_node(parent)
        cpt_index = parent.component_labels.index(component_label)
        return self.parent_to_children[parent.id][cpt_index]

    def add_node(
        self,
        node: Node,
        parent: Optional[Union[Node, Id]] = None,
        parent_component: Optional[Label] = None,
        uniquify: bool = False,
    ) -> None:
        if parent is None:
            if self.root:
                raise ValueError("Cannot add multiple roots")
            if parent_component is not None:
                raise ValueError("Cannot specify a component when adding a root")
            assert not self.parent_to_children
            return self.copy(root=node)
        else:
            parent = self._as_node(parent)
            if parent_component is None:
                if len(parent.components) == 1:
                    parent_cpt_label = parent.components[0].label
                else:
                    raise ValueError(
                        "Must specify a component for parents with multiple components"
                    )
            else:
                parent_cpt_label = parent_component

            cpt_index = parent.component_labels.index(parent_cpt_label)

            if self.parent_to_children[parent.id][cpt_index] is not None:
                raise ValueError("Node already exists at this location")

            if node in self:
                if uniquify:
                    node = node.copy(id=self._first_unique_id(node.id))
                else:
                    raise ValueError("Cannot insert a node with the same ID")

            new_parent_to_children = {
                k: list(v) for k, v in self.parent_to_children.items()
            }
            new_parent_to_children[parent.id][cpt_index] = node
            return self.copy(parent_to_children=new_parent_to_children)

    # old alias
    put_node = add_node

    def replace_node(self, old: Union[Node, Id], new: Node) -> LabelledTree:
        old = self._as_node(old)

        new_root = self.root
        new_parent_to_children = {
            k: list(v) for k, v in self.parent_to_children.items()
        }
        new_parent_to_children[new.id] = new_parent_to_children.pop(old.id)

        if old == self.root:
            new_root = new
        else:
            parent_node, parent_cpt = self.parent(old)
            parent_cpt_index = parent_node.component_labels.index(parent_cpt.label)
            new_parent_to_children[parent_node.id][parent_cpt_index] = new

        return self.copy(root=new_root, parent_to_children=new_parent_to_children)

    # old alias
    with_node = replace_node

    @functools.cached_property
    def node_ids(self) -> frozenset[Id]:
        return frozenset(node.id for node in self.nodes)

    @functools.cached_property
    def child_to_parent(self) -> Dict[Node, Tuple[Node, NodeComponent]]:
        child_to_parent_ = {}
        for parent_id, children in self.parent_to_children.items():
            parent = self._as_node(parent_id)
            for cpt, child in checked_zip(parent.components, children):
                if child is None:
                    continue
                child_to_parent_[child] = (parent, cpt)
        return child_to_parent_

    @functools.cached_property
    def id_to_node(self):
        return {node.id: node for node in self.nodes}

    @functools.cached_property
    def nodes(self) -> Frozenset[Node]:
        if self.is_empty:
            return frozenset()
        return frozenset({self.root}) | {
            node
            for node in filter(None, flatten(list(self.parent_to_children.values())))
        }

    def parent(self, node: Union[Node, Id]) -> Optional[Tuple[Node, NodeComponent]]:
        node = self._as_node(node)
        if node == self.root:
            return None
        else:
            return self.child_to_parent[node]

    def pop_subtree(self, subroot: Union[Node, str]) -> Tree:
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
        parent: Optional[Union[Node, Id]] = None,
        component: Optional[Union[NodeComponent, Label]] = None,
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
            cpt_index = parent.component_labels.index(component)
            new_parent_to_children = {
                p: list(ch) for p, ch in self.parent_to_children.items()
            }
            new_parent_to_children[parent.id][cpt_index] = subtree.root
            new_parent_to_children.update(subtree.parent_to_children)
            return self.copy(parent_to_children=new_parent_to_children)

    # alias, better?
    def _to_node_id(self, arg):
        return self._as_node_id(arg)

    def _check_exists(self, node: Union[Node, str]) -> None:
        if (node_id := self._as_node(node).id) not in self.node_ids:
            raise NodeNotFoundException(f"{node_id} is not present in the tree")

    def _first_unique_id(self, node: Union[Node, Hashable], sep: str = "_") -> str:
        orig_node_id = self._as_node_id(node)
        if orig_node_id not in self:
            return orig_node_id

        counter = 0
        node_id = f"{orig_node_id}{sep}{counter}"
        while node_id in self:
            counter += 1
            node_id = f"{orig_node_id}{sep}{counter}"
        return node_id

    def _as_node(self, node: Union[LabelledNode, Id]) -> Node:
        return node if isinstance(node, Node) else self.id_to_node[node]

    def _as_node_id(self, node: Union[Node, Id]) -> Id:
        return node.id if isinstance(node, Node) else node

    def with_modified_node(self, node: Union[Node, Id], **kwargs):
        return self.replace_node(node, node.copy(**kwargs))

    def with_modified_component(
        self,
        node: Node,
        component: Optional[Union[NodeComponent, Label]] = None,
        **kwargs,
    ):
        return self.replace_node(
            node, node.with_modified_component(component, **kwargs)
        )

    def pop_subtree(self, subroot: Union[Node, str]) -> "Tree":
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

    def child_by_label(self, node: Union[LabelledNode, Hashable], label: Hashable):
        node_id = self._as_node_id(node)
        child = self._parent_and_label_to_child[node_id, label]
        if child is not None:
            return self._as_node(child)
        else:
            return None

    @classmethod
    def from_dict(
        cls,
        node_dict: Dict[Node, Hashable],
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
        def paths_fn(node, component_label, current_path):
            if current_path is None:
                current_path = ()
            new_path = current_path + ((node.label, component_label),)
            paths_[node.id, component_label] = new_path
            return new_path

        paths_ = {}
        previsit(self, paths_fn)
        return pmap(paths_)

    # TODO interface choice about whether we want whole nodes, ids or labels in paths
    # maybe need to distinguish between paths, ancestors and label-only?
    @functools.cached_property
    def _paths_with_nodes(self):
        def paths_fn(node, component_label, current_path):
            if current_path is None:
                current_path = ()
            new_path = current_path + ((node, component_label),)
            paths_[node.id, component_label] = new_path
            return new_path

        paths_ = {}
        previsit(self, paths_fn)
        return pmap(paths_)

    @functools.cached_property
    def leaves(self) -> Tuple[Tuple[Node, ComponentLabel]]:
        """Return the leaves of the tree."""
        leaves_ = []

        def leaves_fn(node, cpt, prev):
            if not self.child(node, cpt):
                leaves_.append((node, cpt))

        previsit(self, leaves_fn)
        return tuple(leaves_)

    @property
    def leaf(self) -> Node:
        return just_one(self.leaves)

    def is_leaf(self, node: Union[Node, str]) -> bool:
        node = self._as_node(node)
        self._check_exists(node)
        return all(child is None for child in self.parent_to_children[node.id])

    def ancestors(self, node, component_label):
        """Return the ancestors of a ``(node_id, component_label)`` 2-tuple."""
        return pmap(
            {
                nd: cpt
                for nd, cpt in self.path(node, component_label).items()
                if nd != node.label
            }
        )

    def path(self, node, component_label, ordered=False):
        node_id = self._as_node_id(node)
        path_ = self._paths[node_id, component_label]
        if ordered:
            return path_
        else:
            return pmap(path_)

    def path_with_nodes(
        self, node, component_label, ordered=False, and_components=False
    ):
        node_id = self._as_node_id(node)
        path_ = self._paths_with_nodes[node_id, component_label]
        if and_components:
            path_ = tuple(
                (ax, just_one(cpt for cpt in ax.components if cpt.label == clabel))
                for ax, clabel in path_
            )
        if ordered:
            return path_
        else:
            return pmap(path_)

    def _node_from_path(self, path: Mapping[Union[Node, Hashable], int]) -> Node:
        if not path:
            return None

        path_ = dict(path)
        node = self.root
        while True:
            cpt_label = path_.pop(node.label)
            cpt_index = node.component_labels.index(cpt_label)
            new_node = self.parent_to_children[node.id][cpt_index]

            # if we are a leaf then return the final bit
            if path_:
                node = new_node
            else:
                return node, node.components[cpt_index]
        assert False, "shouldn't get this far"


NodePath = Dict[Hashable, Hashable]
"""Mapping from axis labels to component labels."""
# wrong now


def previsit(
    tree,
    fn,
    current_node: Optional[Node] = None,
    prev=None,
) -> Any:
    if tree.is_empty:
        raise RuntimeError("Cannot traverse an empty tree")

    current_node = current_node or tree.root
    for cpt_label in current_node.component_labels:
        next = fn(current_node, cpt_label, prev)
        if subnode := tree.child(current_node, cpt_label):
            previsit(tree, fn, subnode, next)


def postvisit(tree, fn, current_node: Optional[Node] = None, **kwargs) -> Any:
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

from __future__ import annotations

import abc
import collections
import functools
from collections.abc import Hashable, Sequence
from functools import cached_property
from itertools import chain
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

import pyrsistent
import pytools
from pyrsistent import freeze, pmap

from pyop3.utils import (
    Id,
    Identified,
    Label,
    Labelled,
    apply_at,
    as_tuple,
    checked_zip,
    deprecated,
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


class EmptyTreeException(Exception):
    pass


class Node(pytools.ImmutableRecord, Identified):
    fields = {"id"}

    def __init__(self, id=None):
        pytools.ImmutableRecord.__init__(self)
        Identified.__init__(self, id)


class AbstractTree(abc.ABC):
    def __init__(self, parent_to_children=None):
        self.parent_to_children = self._parse_parent_to_children(parent_to_children)

    def __str__(self):
        return self._stringify()

    def __contains__(self, node) -> bool:
        return self._as_node(node) in self.nodes

    def __bool__(self) -> bool:
        """Return `True` if the tree is non-empty."""
        return not self.is_empty

    @classmethod
    def from_nest(cls, nest, **kwargs) -> AxisTree:
        root, parent_to_children = cls._from_nest(nest)
        parent_to_children.update({None: [root]})
        return cls(parent_to_children, **kwargs)

    @property
    def root(self):
        if not self.is_empty:
            return just_one(self.parent_to_children[None])
        else:
            return None

    @property
    def is_empty(self) -> bool:
        return not self.parent_to_children

    @property
    def depth(self) -> int:
        if self.is_empty:
            return 0
        count = lambda _, *o: max(o or [0]) + 1
        return postvisit(self, count)

    @cached_property
    def node_ids(self):
        return frozenset(node.id for node in self.nodes)

    @cached_property
    def child_to_parent(self):
        child_to_parent_ = {}
        for parent_id, children in self.parent_to_children.items():
            parent = self._as_node(parent_id)
            for i, child in enumerate(children):
                child_to_parent_[child] = (parent, i)
        return child_to_parent_

    @cached_property
    def id_to_node(self):
        return freeze({node.id: node for node in self.nodes})

    @cached_property
    def nodes(self):
        if self.is_empty:
            return frozenset()
        return frozenset(
            {
                node
                for node in chain.from_iterable(self.parent_to_children.values())
                if node is not None
            }
        )

    @property
    @abc.abstractmethod
    def leaves(self):
        """Return the leaves of the tree."""
        pass

    @property
    def leaf(self):
        return just_one(self.leaves)

    def is_leaf(self, node):
        return self._as_node(node) in self.leaves

    def parent(self, node):
        node = self._as_node(node)
        return self.child_to_parent[node]

    def children(self, node):
        node_id = self._as_node_id(node)
        return self.parent_to_children.get(node_id, ())

    # TODO, could be improved
    @staticmethod
    def _parse_parent_to_children(parent_to_children):
        if not parent_to_children:
            return pmap()
        elif isinstance(parent_to_children, Node):
            # just passing root
            return freeze({None: (parent_to_children,)})
        else:
            parent_to_children = dict(parent_to_children)
            if None not in parent_to_children:
                raise ValueError("Root missing from tree")
            elif len(parent_to_children[None]) != 1:
                raise ValueError("Multiple roots provided, this is not allowed")
            else:
                node_ids = [
                    node.id
                    for node in chain.from_iterable(parent_to_children.values())
                    if node is not None
                ]
                if not has_unique_entries(node_ids):
                    raise ValueError("Nodes with duplicate IDs found")
                if any(
                    parent_id not in node_ids
                    for parent_id in parent_to_children.keys() - {None}
                ):
                    raise ValueError("Tree is disconnected")
            return freeze(parent_to_children)

    @staticmethod
    def _parse_node(node):
        if isinstance(node, Node):
            return node
        else:
            raise TypeError(f"No handler defined for {type(node).__name__}")

    def _stringify(
        self,
        node=None,
        begin_prefix="",
        cont_prefix="",
    ):
        if self.is_empty:
            return "<empty>"

        node = node or self.root

        nodestr = [f"{begin_prefix}{node}"]
        children = self.children(node)
        for i, child in enumerate(children):
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

    def _as_node(self, node):
        if node is None:
            return None
        else:
            return node if isinstance(node, Node) else self.id_to_node[node]

    @staticmethod
    def _as_node_id(node):
        return node.id if isinstance(node, Node) else node


class Tree(AbstractTree):
    @cached_property
    def leaves(self):
        return tuple(
            node
            for node in self.nodes
            if all(c is None for c in self.parent_to_children.get(node.id, ()))
        )

    def add_node(
        self,
        node,
        parent=None,
        uniquify=False,
    ):
        if parent is None:
            if not self.is_empty:
                raise ValueError("Cannot add multiple roots")
            return type(self)(node)
        else:
            parent = self._as_node(parent)
            if node in self:
                if uniquify:
                    node = node.copy(id=node.unique_id())
                else:
                    raise ValueError("Cannot insert a node with the same ID")

            parent_to_children = {
                k: list(v) for k, v in self.parent_to_children.items()
            }
            parent_to_children[parent.id].append(node)
            return type(self)(self.root, parent_to_children)

    @classmethod
    def _from_nest(cls, nest):
        # TODO add appropriate exception classes
        if isinstance(nest, collections.abc.Mapping):
            assert len(nest) == 1
            node, subnodes = just_one(nest.items())
            node = cls._parse_node(node)

            if isinstance(subnodes, collections.abc.Mapping):
                if len(subnodes) == 1 and isinstance(just_one(subnodes.keys()), Node):
                    # just one subnode
                    subnodes = [subnodes]
                else:
                    raise ValueError
            elif not isinstance(subnodes, collections.abc.Sequence):
                subnodes = [subnodes]

            children = []
            parent_to_children = {}
            for subnode in subnodes:
                subnode_, sub_p2c = cls._from_nest(subnode)
                children.append(subnode_)
                parent_to_children.update(sub_p2c)
            parent_to_children[node.id] = children
            return node, parent_to_children
        else:
            node = cls._parse_node(nest)
            return node, {}


class LabelledNodeComponent(pytools.ImmutableRecord, Labelled):
    fields = {"label"}

    def __init__(self, label=None):
        pytools.ImmutableRecord.__init__(self)
        Labelled.__init__(self, label)


class MultiComponentLabelledNode(Node, Labelled):
    fields = Node.fields | {"components", "label"}

    def __init__(self, components, label=None, *, id=None):
        Node.__init__(self, id)
        Labelled.__init__(self, label)
        self.components = as_tuple(components)

    @property
    def degree(self) -> int:
        return len(self.components)

    @property
    def component_labels(self):
        return tuple(c.label for c in self.components)

    @property
    def component(self):
        return just_one(self.components)


class LabelledTree(AbstractTree):
    @deprecated("child")
    def component_child(self, parent, component):
        return self.child(parent, component)

    def child(self, parent, component):
        clabel = self._as_component_label(component)
        cidx = parent.component_labels.index(clabel)
        try:
            return self.parent_to_children[parent.id][cidx]
        except (KeyError, IndexError):
            return None

    @cached_property
    def leaves(self):
        return tuple(
            (node, cpt)
            for node in self.nodes
            for cidx, cpt in enumerate(node.components)
            if self.parent_to_children.get(node.id, [None] * node.degree)[cidx] is None
        )

    def add_node(
        self,
        node,
        parent=None,
        parent_component=None,
        uniquify=False,
    ):
        if parent is None:
            if not self.is_empty:
                raise ValueError("Cannot add multiple roots")
            return type(self)(root)
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
            return type(self)(new_parent_to_children)

    def replace_node(self, old_node, new_node):
        parent_to_children = {k: list(v) for k, v in self.parent_to_children.items()}
        parent_to_children[new_node.id] = parent_to_children.pop(old_node.id)
        parent, pidx = self.parent(old_node)
        parent_id = parent.id if parent is not None else None
        parent_to_children[parent_id][pidx] = new_node
        return type(self)(parent_to_children)

    def with_modified_node(self, node, **kwargs):
        node = self._as_node(node)
        return self.replace_node(node, node.copy(**kwargs))

    def with_modified_component(self, node, component, **kwargs):
        return self.replace_node(
            node, node.with_modified_component(component, **kwargs)
        )

    # invalid for frozen trees
    def add_subtree(
        self,
        subtree,
        parent=None,
        component=None,
        uniquify: bool = False,
    ):
        """
        Parameters
        ----------
        etc
            ...
        uniquify
            If ``False``, duplicate ``ids`` between the tree and subtree
            will raise an exception. If ``True``, the ``ids`` will be changed
            to avoid the clash.

        Notes
        -----
        This function returns a parent-to-children mapping instead of a new tree
        because it is non-trivial to unpick the impact of adding new nodes to the
        tree. For example a new star forest may need to be computed. It, for now,
        is preferable to make trees as "immutable as possible".
        """
        if uniquify:
            raise NotImplementedError("TODO")

        if some_but_not_all([parent, component]):
            raise ValueError(
                "Either both or neither of parent and component must be defined"
            )

        if not parent:
            raise NotImplementedError("TODO")

        assert isinstance(parent, MultiComponentLabelledNode)
        cidx = parent.component_labels.index(component.label)
        parent_to_children = {p: list(ch) for p, ch in self.parent_to_children.items()}

        sub_p2c = dict(subtree.parent_to_children)
        subroot = just_one(sub_p2c.pop(None))
        parent_to_children[parent.id][cidx] = subroot
        parent_to_children.update(sub_p2c)
        return type(self)(parent_to_children)

    @cached_property
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
    @cached_property
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

    def ancestors(self, node, component_label):
        """Return the ancestors of a ``(node_id, component_label)`` 2-tuple."""
        return pmap(
            {
                nd: cpt
                for nd, cpt in self.path(node, component_label).items()
                if nd != node.label
            }
        )

    def path(self, node, component, ordered=False):
        clabel = self._as_component_label(component)
        node_id = self._as_node_id(node)
        path_ = self._paths[node_id, clabel]
        if ordered:
            return path_
        else:
            return pmap(path_)

    def path_with_nodes(
        self, node, component_label, ordered=False, and_components=False
    ):
        component_label = self._as_component_label(component_label)
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

    def _node_from_path(self, path):
        if not path:
            return None

        path_ = dict(path)
        node = self.root
        while True:
            cpt_label = path_.pop(node.label)
            cpt_index = node.component_labels.index(cpt_label)
            new_node = self.parent_to_children.get(node.id, [None] * node.degree)[
                cpt_index
            ]

            # if we are a leaf then return the final bit
            if path_:
                node = new_node
            else:
                return node, node.components[cpt_index]
        assert False, "shouldn't get this far"

    # bad name
    def detailed_path(self, path):
        node = self._node_from_path(path)
        if node is None:
            return pmap()
        else:
            return self.path_with_nodes(*node, and_components=True)

    # this method is crap, if it fails I don't get any useful feedback!
    def is_valid_path(self, path):
        try:
            self._node_from_path(path)
            return True
        except:
            return False

    def find_component(self, node_label, cpt_label, also_node=False):
        """Return the first component in the tree matching the given labels.

        Notes
        -----
        This will return the first component matching the labels. Multiple may exist
        but we assume that they are identical.

        """
        for node in self.nodes:
            if node.label == node_label:
                for cpt in node.components:
                    if cpt.label == cpt_label:
                        if also_node:
                            return node, cpt
                        else:
                            return cpt
        raise ValueError("Matching component not found")

    @classmethod
    def _from_nest(cls, nest):
        # TODO add appropriate exception classes
        if isinstance(nest, collections.abc.Mapping):
            assert len(nest) == 1
            node, subnodes = just_one(nest.items())
            node = cls._parse_node(node)

            if isinstance(subnodes, collections.abc.Mapping):
                if len(subnodes) == 1 and isinstance(
                    just_one(subnodes.keys()), MultiComponentLabelledNode
                ):
                    # just one subnode
                    cidxs = [0]
                    subnodes = [subnodes]
                else:
                    # mapping of component labels to subnodes
                    cidxs = [
                        node.component_labels.index(clabel)
                        for clabel in subnodes.keys()
                    ]
                    subnodes = subnodes.values()
            elif isinstance(subnodes, collections.abc.Sequence):
                cidxs = range(node.degree)
            else:
                if node.degree != 1:
                    raise ValueError
                cidxs = [0]
                subnodes = [subnodes]

            children = [None] * node.degree
            parent_to_children = {}
            for cidx, subnode in checked_zip(cidxs, subnodes):
                subnode_, sub_p2c = cls._from_nest(subnode)
                children[cidx] = subnode_
                parent_to_children.update(sub_p2c)
            parent_to_children[node.id] = children
            return node, parent_to_children
        else:
            node = cls._parse_node(nest)
            return node, {node.id: [None] * node.degree}

    # TODO, could be improved, same as other Tree apart from [None, None, ...] bit
    @staticmethod
    def _parse_parent_to_children(parent_to_children):
        if not parent_to_children:
            return pmap()

        if isinstance(parent_to_children, Node):
            # just passing root
            parent_to_children = {None: (parent_to_children,)}
        else:
            parent_to_children = dict(parent_to_children)

        if None not in parent_to_children:
            raise ValueError("Root missing from tree")
        if len(parent_to_children[None]) != 1:
            raise ValueError("Multiple roots provided, this is not allowed")

        nodes = [
            node
            for node in chain.from_iterable(parent_to_children.values())
            if node is not None
        ]
        node_ids = [n.id for n in nodes]
        if not has_unique_entries(node_ids):
            raise ValueError("Nodes with duplicate IDs found")
        if any(
            parent_id not in node_ids
            for parent_id in parent_to_children.keys() - {None}
        ):
            raise ValueError("Tree is disconnected")
        for node in nodes:
            if node.id not in parent_to_children.keys():
                parent_to_children[node.id] = [None] * node.degree
        return freeze(parent_to_children)

    @staticmethod
    def _parse_node(node):
        if isinstance(node, MultiComponentLabelledNode):
            return node
        else:
            raise TypeError(f"No handler defined for {type(node).__name__}")

    @staticmethod
    def _as_component_label(component):
        if isinstance(component, LabelledNodeComponent):
            return component.label
        else:
            return component


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

from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import itertools
import math
import numbers
import sys
from functools import cached_property
from typing import Any, Collection, Hashable, Mapping, Sequence

import numpy as np
import pymbolic as pym
import pyrsistent
import pytools
from mpi4py import MPI
from pyrsistent import freeze, pmap

from pyop3.axtree import (
    Axis,
    AxisComponent,
    AxisTree,
    AxisVariable,
    ContextAware,
    ContextFree,
    ContextSensitive,
    LoopIterable,
)
from pyop3.axtree.layout import _as_int
from pyop3.axtree.tree import (
    ContextSensitiveAxisTree,
    ContextSensitiveLoopIterable,
    ExpressionEvaluator,
    PartialAxisTree,
)
from pyop3.dtypes import IntType, get_mpi_dtype
from pyop3.lang import KernelArgument
from pyop3.tree import LabelledTree, MultiComponentLabelledNode, Node, Tree, postvisit
from pyop3.utils import (
    Identified,
    Labelled,
    as_tuple,
    checked_zip,
    is_single_valued,
    just_one,
    merge_dicts,
)

bsearch = pym.var("mybsearch")


# FIXME this is copied from loopexpr2loopy VariableReplacer
class IndexExpressionReplacer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        self._replace_map = replace_map

    def map_axis_variable(self, expr):
        return self._replace_map.get(expr.axis_label, expr)

    def map_multi_array(self, expr):
        from pyop3.array.harray import MultiArrayVariable

        index_exprs = {ax: self.rec(iexpr) for ax, iexpr in expr.index_exprs.items()}
        return MultiArrayVariable(expr.array, expr.target_path, index_exprs)

    def map_loop_index(self, expr):
        # this is hacky, if I make this raise a KeyError then we fail in indexing
        return self._replace_map.get((expr.name, expr.axis), expr)


class IndexTree(LabelledTree):
    pass


def parse_index_tree(parent_to_children, loop_context):
    new_parent_to_children = parse_parent_to_children(parent_to_children, loop_context)

    return pmap(new_parent_to_children), loop_context


def parse_parent_to_children(parent_to_children, loop_context, parent=None):
    if parent in parent_to_children:
        new_children = []
        subparents_to_children = []
        for child in parent_to_children[parent]:
            if child is None:
                continue
            child = apply_loop_context(child, loop_context)
            new_children.append(child)
            subparents_to_children.append(
                parse_parent_to_children(parent_to_children, loop_context, child.id)
            )

        return pmap({parent: tuple(new_children)}) | merge_dicts(subparents_to_children)
    else:
        return pmap()


class DatamapCollector(pym.mapper.CombineMapper):
    def combine(self, values):
        return merge_dicts(values)

    def map_constant(self, expr):
        return {}

    map_variable = map_constant

    def map_called_map(self, called_map):
        dmap = self.rec(called_map.parameters)
        return dmap | called_map.function.map_component.datamap


_datamap_collector = DatamapCollector()


def collect_datamap_from_expression(expr: pym.primitives.Expr) -> dict:
    return _datamap_collector(expr)


class SliceComponent(pytools.ImmutableRecord, abc.ABC):
    fields = {"component"}

    def __init__(self, component):
        super().__init__()
        self.component = component


class AffineSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"start", "stop", "step"}

    def __init__(self, component, start=None, stop=None, step=None):
        super().__init__(component)
        # use None for the default args here since that agrees with Python slices
        self.start = start if start is not None else 0
        self.stop = stop
        self.step = step if step is not None else 1

    @property
    def datamap(self):
        return pmap()


class Subset(SliceComponent):
    fields = SliceComponent.fields | {"array"}

    def __init__(self, component, array: MultiArray):
        super().__init__(component)
        self.array = array

    @property
    def datamap(self):
        return self.array.datamap


class MapComponent(pytools.ImmutableRecord, Labelled, abc.ABC):
    fields = {"target_axis", "target_component", "label"}

    def __init__(self, target_axis, target_component, *, label=None):
        pytools.ImmutableRecord.__init__(self)
        Labelled.__init__(self, label)
        self.target_axis = target_axis
        self.target_component = target_component

    @property
    @abc.abstractmethod
    def arity(self):
        pass


# TODO: Implement AffineMapComponent
class TabulatedMapComponent(MapComponent):
    fields = MapComponent.fields | {"array"}

    def __init__(self, target_axis, target_component, array, *, label=None):
        super().__init__(target_axis, target_component, label=label)
        self.array = array

    @property
    def arity(self):
        # TODO clean this up in AxisTree
        axes = self.array.axes
        leaf_axis, leaf_clabel = axes.leaf
        leaf_cidx = leaf_axis.component_index(leaf_clabel)
        return leaf_axis.components[leaf_cidx].count

    # old alias
    @property
    def data(self):
        return self.array

    @functools.cached_property
    def datamap(self):
        return self.array.datamap


class Index(MultiComponentLabelledNode):
    fields = MultiComponentLabelledNode.fields | {"component_labels"}

    def __init__(self, label=None, *, component_labels=None, id=None):
        super().__init__(label, id=id)
        self._component_labels = component_labels

    @property
    @abc.abstractmethod
    def leaf_target_paths(self):
        # rename to just target paths?
        pass

    @property
    def component_labels(self):
        if self._component_labels is None:
            # do this for now (since leaf_target_paths currently requires an
            # instantiated object to determine)
            self._component_labels = tuple(
                self.unique_label() for _ in self.leaf_target_paths
            )
        return self._component_labels


class ContextFreeIndex(Index, ContextFree, abc.ABC):
    @property
    def axes(self):
        return self._tree.axes

    @property
    def target_paths(self):
        return self._tree.target_paths

    @cached_property
    def _tree(self):
        """

        Notes
        -----
        This method will deliberately not work for slices since slices
        require additional existing axis information in order to be valid.

        """
        return as_index_tree(self)


class ContextSensitiveIndex(Index, ContextSensitive, abc.ABC):
    def __init__(self, context_map, *, id=None):
        Index.__init__(self, id)
        ContextSensitive.__init__(self, context_map)


class AbstractLoopIndex(
    pytools.ImmutableRecord, KernelArgument, Identified, ContextAware, abc.ABC
):
    dtype = IntType
    fields = {"id"}

    def __init__(self, id=None):
        pytools.ImmutableRecord.__init__(self)
        Identified.__init__(self, id)


# Is this really an index? I dont think it's valid in an index tree
class LoopIndex(AbstractLoopIndex):
    """
    Parameters
    ----------
    iterset: AxisTree or ContextSensitiveAxisTree (!!!)
        Only add context later on

    """

    def __init__(self, iterset, *, id=None):
        super().__init__(id=id)
        self.iterset = iterset

    @cached_property
    def local_index(self):
        return LocalLoopIndex(self)

    @property
    def i(self):
        return self.local_index

    def with_context(self, context):
        iterset = self.iterset.with_context(context)
        path = context[self.id]
        return ContextFreeLoopIndex(iterset, path, id=self.id)

    # unsure if this is required
    @property
    def datamap(self):
        return self.iterset.datamap


# FIXME class hierarchy is very confusing
class ContextFreeLoopIndex(ContextFreeIndex):
    def __init__(self, iterset: AxisTree, path, *, id=None):
        super().__init__(id=id)
        self.iterset = iterset
        self.path = freeze(path)

    @property
    def leaf_target_paths(self):
        return (self.path,)

    @property
    def axes(self):
        return AxisTree()

    @property
    def target_paths(self):
        return freeze({None: self.path})

    @property
    def index_exprs(self):
        return freeze(
            {None: {axis: LoopIndexVariable(self, axis) for axis in self.path.keys()}}
        )

    @property
    def layout_exprs(self):
        # FIXME, no clue if this is right or not
        return freeze({None: 0})

    @property
    def domain_index_exprs(self):
        return pmap()

    @property
    def datamap(self):
        return self.iterset.datamap

    def iter(self, stuff=pmap()):
        if not isinstance(self.iterset, AxisTree):
            raise NotImplementedError
        return iter_axis_tree(
            self.iterset, self.iterset.target_paths, self.iterset.index_exprs, stuff
        )


class LocalLoopIndex(AbstractLoopIndex):
    """Class representing a 'local' index."""

    def __init__(self, loop_index: LoopIndex, *, id=None):
        super().__init__(id)
        self.loop_index = loop_index

    @property
    def iterset(self):
        return self.loop_index.iterset

    def with_context(self, context):
        # not sure about this
        iterset = self.loop_index.iterset.with_context(context)
        path = context[self.id]
        return ContextFreeLoopIndex(iterset, path, id=self.id)

    @property
    def datamap(self):
        return self.loop_index.datamap


# TODO I want a Slice to have "bits" like a Map/CalledMap does
class Slice(ContextFreeIndex):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    fields = Index.fields | {"axis", "slices"} - {"label"}

    def __init__(self, axis, slices, *, id=None):
        super().__init__(label=axis, id=id)
        self.axis = axis
        self.slices = as_tuple(slices)

    @cached_property
    def leaf_target_paths(self):
        return tuple(
            freeze({self.axis: subslice.component}) for subslice in self.slices
        )

    @property
    def datamap(self):
        return merge_dicts([s.datamap for s in self.slices])


class Map(pytools.ImmutableRecord):
    """

    Notes
    -----
    This class *cannot* be used as an index. Instead, one must use a
    `CalledMap` which can be formed from a `Map` using call syntax.
    """

    fields = {"connectivity", "name"}

    def __init__(self, connectivity, name, **kwargs) -> None:
        super().__init__(**kwargs)
        self.connectivity = connectivity
        self.name = name

    def __call__(self, index):
        return CalledMap(self, index)

    @cached_property
    def datamap(self):
        data = {}
        for bit in self.connectivity.values():
            for map_cpt in bit:
                data.update(map_cpt.datamap)
        return pmap(data)


class CalledMap(LoopIterable):
    def __init__(self, map, from_index):
        self.map = map
        self.from_index = from_index

    def __getitem__(self, indices):
        raise NotImplementedError("TODO")

    def index(self) -> LoopIndex:
        context_map = {
            ctx: _index_axes(itree, ctx) for ctx, itree in as_index_forest(self).items()
        }
        context_sensitive_axes = ContextSensitiveAxisTree(context_map)
        return LoopIndex(context_sensitive_axes)

    def with_context(self, context):
        cf_index = self.from_index.with_context(context)
        return ContextFreeCalledMap(self.map, cf_index)

    @property
    def name(self):
        return self.map.name

    @property
    def connectivity(self):
        return self.map.connectivity


class ContextFreeCalledMap(Index, ContextFree):
    def __init__(self, map, index, *, id=None):
        super().__init__(id=id)
        self.map = map
        # better to call it "input_index"?
        self.index = index

    @property
    def name(self) -> str:
        return self.map.name

    @cached_property
    def leaf_target_paths(self):
        return tuple(
            freeze({mcpt.target_axis: mcpt.target_component})
            for path in self.index.leaf_target_paths
            for mcpt in self.map.connectivity[path]
        )

    @cached_property
    def axes(self):
        return self._axes_info[0]

    @cached_property
    def target_paths(self):
        return self._axes_info[1]

    @cached_property
    def index_exprs(self):
        return self._axes_info[2]

    @cached_property
    def layout_exprs(self):
        return self._axes_info[3]

    @cached_property
    def domain_index_exprs(self):
        return self._axes_info[4]

    # TODO This is bad design, unroll the traversal and store as properties
    @cached_property
    def _axes_info(self):
        return collect_shape_index_callback(self)


class LoopIndexVariable(pym.primitives.Variable):
    init_arg_names = ("index", "axis")

    mapper_method = sys.intern("map_loop_index")

    def __init__(self, index, axis):
        super().__init__(index.id)
        self.index = index
        self.axis = axis

    def __getinitargs__(self):
        # FIXME The following is wrong, but it gives us the repr we want
        # return (self.index, self.axis)
        return (self.index.id, self.axis)

    @property
    def id(self):
        return self.name

    @property
    def datamap(self):
        return self.index.datamap


class ContextSensitiveCalledMap(ContextSensitiveLoopIterable):
    pass


@functools.singledispatch
def apply_loop_context(arg, loop_context, *, axes, path):
    from pyop3.array import HierarchicalArray

    if isinstance(arg, HierarchicalArray):
        parent = axes._node_from_path(path)
        if parent is not None:
            parent_axis, parent_cpt = parent
            target_axis = axes.child(parent_axis, parent_cpt)
        else:
            target_axis = axes.root
        slice_cpts = []
        # potentially a bad idea to apply the subset to all components. Might want to match
        # labels. In fact I enforce that here and so multiple components would break things.
        # Not sure what the right approach is. This is also potentially tricky for multi-level
        # subsets
        array_axis, array_component = arg.axes.leaf
        for cpt in target_axis.components:
            slice_cpt = Subset(cpt.label, arg)
            slice_cpts.append(slice_cpt)
        return Slice(target_axis.label, slice_cpts)
    elif isinstance(arg, str):
        # component label
        # FIXME this is not right, only works at top level
        return Slice(axes.root.label, AffineSliceComponent(arg))
    elif isinstance(arg, numbers.Integral):
        return apply_loop_context(
            slice(arg, arg + 1), loop_context, axes=axes, path=path
        )
    else:
        raise TypeError


@apply_loop_context.register
def _(index: Index, loop_context, **kwargs):
    return index.with_context(loop_context)


@apply_loop_context.register
def _(index: Axis, *args, **kwargs):
    return Slice(index.label, [AffineSliceComponent(c.label) for c in index.components])


@apply_loop_context.register
def _(slice_: slice, loop_context, axes, path):
    parent = axes._node_from_path(path)
    if parent is not None:
        parent_axis, parent_cpt = parent
        target_axis = axes.child(parent_axis, parent_cpt)
    else:
        target_axis = axes.root
    slice_cpts = []
    for cpt in target_axis.components:
        slice_cpt = AffineSliceComponent(
            cpt.label, slice_.start, slice_.stop, slice_.step
        )
        slice_cpts.append(slice_cpt)
    return Slice(target_axis.label, slice_cpts)


def combine_contexts(contexts):
    new_contexts = []
    for mycontexts in itertools.product(*contexts):
        new_contexts.append(pmap(merge_dicts(mycontexts)))
    return new_contexts


def is_fully_indexed(axes: AxisTree, indices: IndexTree) -> bool:
    """Check that the provided indices are compatible with the axis tree."""
    # To check for correctness we ensure that all of the paths through the
    # index tree generate valid paths through the axis tree.
    for leaf_index, component_label in indices.leaves:
        # this maps indices to the specific component being accessed
        # use this to find the right target_path
        index_path = indices.path_with_nodes(leaf_index, component_label)

        full_target_path = {}
        for index, cpt_label in index_path.items():
            # select the target_path corresponding to this component label
            cidx = index.component_labels.index(cpt_label)
            full_target_path |= index.target_paths[cidx]

        # the axis addressed by the full path should be a leaf, else we are
        # not fully indexing the array
        final_axis, final_cpt = axes._node_from_path(full_target_path)
        if axes.child(final_axis, final_cpt) is not None:
            return False

    return True


def _collect_datamap(index, *subdatamaps, itree):
    return index.datamap | merge_dicts(subdatamaps)


def index_tree_from_ellipsis(axes, current_axis=None, first_call=True):
    current_axis = current_axis or axes.root
    slice_components = []
    subroots = []
    subtrees = []
    for component in current_axis.components:
        slice_components.append(AffineSliceComponent(component.label))

        if subaxis := axes.child(current_axis, component):
            subroot, subtree = index_tree_from_ellipsis(axes, subaxis, first_call=False)
            subroots.append(subroot)
            subtrees.append(subtree)
        else:
            subroots.append(None)
            subtrees.append({})

    fullslice = Slice(current_axis.label, slice_components)
    myslice = fullslice

    if first_call:
        return IndexTree(myslice, pmap({myslice.id: subroots}) | merge_dicts(subtrees))
    else:
        return myslice, pmap({myslice.id: subroots}) | merge_dicts(subtrees)


def index_tree_from_iterable(
    indices, loop_context, axes=None, path=pmap(), first_call=False
):
    index, *subindices = indices

    index = apply_loop_context(index, loop_context, axes=axes, path=path)
    assert isinstance(index, ContextFree)

    if subindices:
        children = []
        subtrees = []

        # if index.axes.is_empty:
        #     index_keyss = [[None]]
        # else:
        #     index_keyss = []
        #     for leaf_axis, leaf_cpt in index.axes.leaves:
        #         source_path = index.axes.path(leaf_axis, leaf_cpt)
        #         index_keys = [None] + [
        #             (axis.id, cpt.label)
        #             for axis, cpt in index.axes.detailed_path(source_path).items()
        #         ]
        #         index_keyss.append(index_keys)

        # for index_keys in index_keyss:
        for target_path in index.leaf_target_paths:
            path_ = path | target_path

            child, subtree = index_tree_from_iterable(
                subindices, loop_context, axes, path_
            )
            children.append(child)
            subtrees.append(subtree)

        parent_to_children = pmap({index.id: children}) | merge_dicts(subtrees)
    else:
        parent_to_children = {}

    if first_call:
        assert None not in parent_to_children
        parent_to_children |= {None: [index]}
        return IndexTree(parent_to_children, loop_context=loop_context)
    else:
        return index, parent_to_children


# not sure that this is a useful method, want to have context instead?
@functools.singledispatch
def as_index_tree(arg, loop_context, **kwargs):
    if isinstance(arg, collections.abc.Iterable):
        return index_tree_from_iterable(arg, loop_context, first_call=True, **kwargs)
    else:
        raise TypeError


@as_index_tree.register
def _(index: Index, ctx, **kwargs):
    return IndexTree(index, loop_context=ctx)


@as_index_tree.register
def _(called_map: CalledMap, ctx, **kwargs):
    # index_tree = as_index_tree(called_map.from_index)
    cf_called_map = called_map.with_context(ctx)
    return IndexTree(cf_called_map, loop_context=ctx)
    #
    # index_tree_ = index_tree.add_node(cf_called_map, index_tree.leaf)
    # # because loop contexts are an attribute!
    # index_tree_ = IndexTree(index_tree_.parent_to_children, loop_context=ctx)
    # return index_tree_


@as_index_tree.register
def _(index: AbstractLoopIndex, context, **kwargs):
    index = index.with_context(context)
    return IndexTree(index, loop_context=context)


@functools.singledispatch
def as_index_forest(arg: Any, *, axes=None, path=pmap(), **kwargs):
    from pyop3.array import HierarchicalArray

    if isinstance(arg, HierarchicalArray):
        # NOTE: This is the same behaviour as for slices
        parent = axes._node_from_path(path)
        if parent is not None:
            parent_axis, parent_cpt = parent
            target_axis = axes.child(parent_axis, parent_cpt)
        else:
            target_axis = axes.root

        if target_axis.degree > 1:
            raise ValueError(
                "Passing arrays as indices is only allowed when there is no ambiguity"
            )

        slice_cpt = Subset(target_axis.component.label, arg)
        slice_ = Slice(target_axis.label, [slice_cpt])
        return freeze({pmap(): IndexTree(slice_)})
    else:
        raise TypeError(f"No handler provided for {type(arg).__name__}")


@as_index_forest.register
def _(indices: collections.abc.Sequence, *, path=pmap(), loop_context=pmap(), **kwargs):
    index, *subindices = indices

    # FIXME This fails because strings are considered sequences, perhaps we should
    # cast component labels into their own type?
    # if isinstance(index, collections.abc.Sequence):
    #     # what's the right exception? Some sort of BadIndexException?
    #     raise ValueError("Nested iterables are not supported")

    forest = {}
    # TODO, it is a bad pattern to build a forest here when I really just want to convert
    # a single index
    for context, tree in as_index_forest(
        index, path=path, loop_context=loop_context, **kwargs
    ).items():
        # converting a single index should only produce index trees with depth 1
        assert tree.depth == 1
        cf_index = tree.root

        if subindices:
            for clabel, target_path in checked_zip(
                cf_index.component_labels, cf_index.leaf_target_paths
            ):
                path_ = path | target_path

                subforest = as_index_forest(
                    subindices,
                    path=path | target_path,
                    loop_context=loop_context | context,
                    **kwargs,
                )
                for subctx, subtree in subforest.items():
                    forest[subctx] = tree.add_subtree(subtree, cf_index, clabel)
        else:
            forest[context] = tree
    return freeze(forest)


@as_index_forest.register
def _(index_tree: IndexTree, **kwargs):
    return freeze({pmap(): index_tree})


@as_index_forest.register
def _(index: ContextFreeIndex, **kwargs):
    return freeze({pmap(): IndexTree(index)})


# TODO This function can definitely be refactored
@as_index_forest.register
def _(index: AbstractLoopIndex, *, loop_context=pmap(), **kwargs):
    local = isinstance(index, LocalLoopIndex)

    forest = {}
    if isinstance(index.iterset, ContextSensitive):
        for context, axes in index.iterset.context_map.items():
            if axes.is_empty:
                source_path = pmap()
                target_path = axes.target_paths.get(None, pmap())

                if local:
                    context_ = (
                        loop_context | context | {index.local_index.id: source_path}
                    )
                else:
                    context_ = loop_context | context | {index.id: target_path}

                cf_index = index.with_context(context_)
                forest[context_] = IndexTree(cf_index)
            else:
                for leaf in axes.leaves:
                    source_path = axes.path(*leaf)
                    target_path = axes.target_paths.get(None, pmap())
                    for axis, cpt in axes.path_with_nodes(
                        *leaf, and_components=True
                    ).items():
                        target_path |= axes.target_paths.get((axis.id, cpt.label), {})

                    if local:
                        context_ = loop_context | context | {index.id: source_path}
                    else:
                        context_ = loop_context | context | {index.id: target_path}

                    cf_index = index.with_context(context_)
                    forest[context_] = IndexTree(cf_index)
    else:
        assert isinstance(index.iterset, ContextFree)
        for leaf_axis, leaf_cpt in index.iterset.leaves:
            source_path = index.iterset.path(leaf_axis, leaf_cpt)
            target_path = index.iterset.target_paths.get(None, pmap())
            for axis, cpt in index.iterset.path_with_nodes(
                leaf_axis, leaf_cpt, and_components=True
            ).items():
                target_path |= index.iterset.target_paths[axis.id, cpt.label]
            if local:
                context = loop_context | {index.id: source_path}
            else:
                context = loop_context | {index.id: target_path}

            cf_index = index.with_context(context)
            forest[context] = IndexTree(cf_index)
    return freeze(forest)


@as_index_forest.register
def _(called_map: CalledMap, **kwargs):
    forest = {}
    input_forest = as_index_forest(called_map.from_index, **kwargs)
    for context in input_forest.keys():
        cf_called_map = called_map.with_context(context)
        forest[context] = IndexTree(cf_called_map)
    return freeze(forest)


@as_index_forest.register
def _(index: numbers.Integral, **kwargs):
    return as_index_forest(slice(index, index + 1), **kwargs)


@as_index_forest.register
def _(slice_: slice, *, axes=None, path=pmap(), loop_context=pmap(), **kwargs):
    if axes is None:
        raise RuntimeError("invalid slice usage")

    parent = axes._node_from_path(path)
    if parent is not None:
        parent_axis, parent_cpt = parent
        target_axis = axes.child(parent_axis, parent_cpt)
    else:
        target_axis = axes.root

    if target_axis.degree > 1:
        # badindexexception?
        raise ValueError(
            "Cannot slice multi-component things using generic slices, ambiguous"
        )

    slice_cpt = AffineSliceComponent(
        target_axis.component.label, slice_.start, slice_.stop, slice_.step
    )
    slice_ = Slice(target_axis.label, [slice_cpt])
    return freeze({loop_context: IndexTree(slice_)})


@as_index_forest.register
def _(label: str, *, axes, **kwargs):
    # if we use a string then we assume we are taking a full slice of the
    # top level axis
    axis = axes.root
    component = just_one(c for c in axis.components if c.label == label)
    slice_ = Slice(axis.label, [AffineSliceComponent(component.label)])
    return as_index_forest(slice_, axes=axes, **kwargs)


@functools.singledispatch
def collect_shape_index_callback(index, *args, **kwargs):
    raise TypeError(f"No handler provided for {type(index)}")


# @collect_shape_index_callback.register
# def _(loop_index: LoopIndex, *, loop_indices, **kwargs):
#     return collect_shape_index_callback(
#         loop_index.with_context(loop_indices), loop_indices=loop_indices, **kwargs
#     )


@collect_shape_index_callback.register
def _(loop_index: ContextFreeLoopIndex, *, loop_indices, **kwargs):
    return (
        loop_index.axes,
        loop_index.target_paths,
        loop_index.index_exprs,
        loop_index.layout_exprs,
        loop_index.domain_index_exprs,
    )


# @collect_shape_index_callback.register
# def _(local_index: LocalLoopIndex, *args, loop_indices, **kwargs):
#     path = loop_indices[local_index.id]
#
#     loop_index = local_index.loop_index
#     iterset = loop_index.iterset
#
#     target_path_per_cpt = pmap({None: path})
#     index_exprs_per_cpt = pmap(
#         {
#             None: pmap(
#                 {axis: LoopIndexVariable(local_index, axis) for axis in path.keys()}
#             )
#         }
#     )
#
#     layout_exprs_per_cpt = pmap({None: 0})
#     return (
#         PartialAxisTree(),
#         target_path_per_cpt,
#         index_exprs_per_cpt,
#         layout_exprs_per_cpt,
#         pmap(),
# )


@collect_shape_index_callback.register
def _(slice_: Slice, *, prev_axes, **kwargs):
    from pyop3.array.harray import MultiArrayVariable

    components = []
    target_path_per_subslice = []
    index_exprs_per_subslice = []
    layout_exprs_per_subslice = []

    axis_label = slice_.label

    for subslice in slice_.slices:
        # we are assuming that axes with the same label *must* be identical. They are
        # only allowed to differ in that they have different IDs.
        target_axis, target_cpt = prev_axes.find_component(
            slice_.axis, subslice.component, also_node=True
        )

        if isinstance(subslice, AffineSliceComponent):
            if subslice.stop is None:
                stop = target_cpt.count
            else:
                stop = subslice.stop
            size = math.ceil((stop - subslice.start) / subslice.step)
        else:
            assert isinstance(subslice, Subset)
            size = subslice.array.axes.leaf_component.count
        cpt = AxisComponent(size, label=subslice.component)
        components.append(cpt)

        target_path_per_subslice.append(pmap({slice_.axis: subslice.component}))

        newvar = AxisVariable(axis_label)
        layout_var = AxisVariable(slice_.axis)
        if isinstance(subslice, AffineSliceComponent):
            index_exprs_per_subslice.append(
                pmap({slice_.axis: newvar * subslice.step + subslice.start})
            )
            layout_exprs_per_subslice.append(
                pmap({slice_.label: (layout_var - subslice.start) // subslice.step})
            )
        else:
            assert isinstance(subslice, Subset)

            # below is also used for maps - cleanup
            subset_array = subslice.array
            subset_axes = subset_array.axes

            # must be single component
            source_path = subset_axes.path(*subset_axes.leaf)
            index_keys = [None] + [
                (axis.id, cpt.label)
                for axis, cpt in subset_axes.detailed_path(source_path).items()
            ]
            my_target_path = merge_dicts(
                subset_array.target_paths.get(key, {}) for key in index_keys
            )
            old_index_exprs = merge_dicts(
                subset_array.index_exprs.get(key, {}) for key in index_keys
            )

            my_index_exprs = {}
            index_expr_replace_map = {subset_axes.leaf_axis.label: newvar}
            replacer = IndexExpressionReplacer(index_expr_replace_map)
            for axlabel, index_expr in old_index_exprs.items():
                my_index_exprs[axlabel] = replacer(index_expr)
            subset_var = MultiArrayVariable(
                subslice.array, my_target_path, my_index_exprs
            )

            index_exprs_per_subslice.append(pmap({slice_.axis: subset_var}))
            layout_exprs_per_subslice.append(
                pmap({slice_.label: bsearch(subset_var, layout_var)})
            )

    axis = Axis(components, label=axis_label)
    axes = PartialAxisTree(axis)
    target_path_per_component = {}
    index_exprs_per_component = {}
    layout_exprs_per_component = {}
    for cpt, target_path, index_exprs, layout_exprs in checked_zip(
        components,
        target_path_per_subslice,
        index_exprs_per_subslice,
        layout_exprs_per_subslice,
    ):
        target_path_per_component[axis.id, cpt.label] = target_path
        index_exprs_per_component[axis.id, cpt.label] = index_exprs
        layout_exprs_per_component[axis.id, cpt.label] = layout_exprs
    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
        pmap(),
    )


@collect_shape_index_callback.register
def _(called_map: ContextFreeCalledMap, **kwargs):
    (
        prior_axes,
        prior_target_path_per_cpt,
        prior_index_exprs_per_cpt,
        _,
        prior_domain_index_exprs_per_cpt,
    ) = collect_shape_index_callback(called_map.index, **kwargs)

    if not prior_axes:
        prior_target_path = prior_target_path_per_cpt[None]
        prior_index_exprs = prior_index_exprs_per_cpt[None]
        (
            axis,
            target_path_per_cpt,
            index_exprs_per_cpt,
            layout_exprs_per_cpt,
            domain_index_exprs_per_cpt,
        ) = _make_leaf_axis_from_called_map(
            called_map, prior_target_path, prior_index_exprs
        )
        axes = PartialAxisTree(axis)

    else:
        axes = prior_axes
        target_path_per_cpt = {}
        index_exprs_per_cpt = {}
        layout_exprs_per_cpt = {}
        domain_index_exprs_per_cpt = {}
        for prior_leaf_axis, prior_leaf_cpt in prior_axes.leaves:
            prior_target_path = prior_target_path_per_cpt.get(None, pmap())
            prior_index_exprs = prior_index_exprs_per_cpt.get(None, pmap())

            for myaxis, mycomponent_label in prior_axes.path_with_nodes(
                prior_leaf_axis.id, prior_leaf_cpt
            ).items():
                prior_target_path |= prior_target_path_per_cpt[
                    myaxis.id, mycomponent_label
                ]
                prior_index_exprs |= prior_index_exprs_per_cpt[
                    myaxis.id, mycomponent_label
                ]

            (
                subaxis,
                subtarget_paths,
                subindex_exprs,
                sublayout_exprs,
                subdomain_index_exprs,
            ) = _make_leaf_axis_from_called_map(
                called_map, prior_target_path, prior_index_exprs
            )
            axes = axes.add_subtree(
                PartialAxisTree(subaxis), prior_leaf_axis, prior_leaf_cpt
            )
            target_path_per_cpt.update(subtarget_paths)
            index_exprs_per_cpt.update(subindex_exprs)
            layout_exprs_per_cpt.update(sublayout_exprs)
            domain_index_exprs_per_cpt.update(subdomain_index_exprs)

            # does this work?
            # need to track these for nested ragged things
            # breakpoint()
            # index_exprs_per_cpt.update({(prior_leaf_axis.id, prior_leaf_cpt.label): prior_index_exprs})
            domain_index_exprs_per_cpt.update(prior_domain_index_exprs_per_cpt)
            # layout_exprs_per_cpt.update(...)

    return (
        axes,
        freeze(target_path_per_cpt),
        freeze(index_exprs_per_cpt),
        freeze(layout_exprs_per_cpt),
        freeze(domain_index_exprs_per_cpt),
    )


def _make_leaf_axis_from_called_map(called_map, prior_target_path, prior_index_exprs):
    from pyop3.array.harray import CalledMapVariable

    axis_id = Axis.unique_id()
    components = []
    target_path_per_cpt = {}
    index_exprs_per_cpt = {}
    layout_exprs_per_cpt = {}
    domain_index_exprs_per_cpt = {}

    for map_cpt in called_map.map.connectivity[prior_target_path]:
        cpt = AxisComponent(map_cpt.arity, label=map_cpt.label)
        components.append(cpt)

        target_path_per_cpt[axis_id, cpt.label] = pmap(
            {map_cpt.target_axis: map_cpt.target_component}
        )

        axisvar = AxisVariable(called_map.name)

        if not isinstance(map_cpt, TabulatedMapComponent):
            raise NotImplementedError("Currently we assume only arrays here")

        map_array = map_cpt.array
        map_axes = map_array.axes

        assert map_axes.depth == 2

        source_path = map_axes.path(*map_axes.leaf)
        index_keys = [None] + [
            (axis.id, cpt.label)
            for axis, cpt in map_axes.detailed_path(source_path).items()
        ]
        my_target_path = merge_dicts(
            map_array.target_paths.get(key, {}) for key in index_keys
        )

        # the outer index is provided from "prior" whereas the inner one requires
        # a replacement
        map_leaf_axis, map_leaf_component = map_axes.leaf
        old_inner_index_expr = map_array.index_exprs[
            map_leaf_axis.id, map_leaf_component
        ]

        my_index_exprs = {}
        index_expr_replace_map = {map_axes.leaf_axis.label: axisvar}
        replacer = IndexExpressionReplacer(index_expr_replace_map)
        for axlabel, index_expr in old_inner_index_expr.items():
            my_index_exprs[axlabel] = replacer(index_expr)
        new_inner_index_expr = my_index_exprs

        # breakpoint()
        map_var = CalledMapVariable(
            map_cpt.array, my_target_path, prior_index_exprs, new_inner_index_expr
        )

        index_exprs_per_cpt[axis_id, cpt.label] = {
            # map_cpt.target_axis: map_var(prior_index_exprs | {called_map.name: axisvar})
            map_cpt.target_axis: map_var
        }

        # don't think that this is possible for maps
        layout_exprs_per_cpt[axis_id, cpt.label] = {
            called_map.name: pym.primitives.NaN(IntType)
        }

        domain_index_exprs_per_cpt[axis_id, cpt.label] = prior_index_exprs

    axis = Axis(components, label=called_map.name, id=axis_id)

    return (
        axis,
        target_path_per_cpt,
        index_exprs_per_cpt,
        layout_exprs_per_cpt,
        domain_index_exprs_per_cpt,
    )


def _index_axes(indices: IndexTree, loop_context, axes=None):
    (
        indexed_axes,
        tpaths,
        index_expr_per_target,
        layout_expr_per_target,
        domain_index_exprs,
    ) = _index_axes_rec(
        indices,
        current_index=indices.root,
        loop_indices=loop_context,
        prev_axes=axes,
    )

    # check that slices etc have not been missed
    if axes is not None:
        for leaf_iaxis, leaf_icpt in indexed_axes.leaves:
            target_path = dict(tpaths.get(None, {}))
            for iaxis, icpt in indexed_axes.path_with_nodes(
                leaf_iaxis, leaf_icpt
            ).items():
                target_path.update(tpaths.get((iaxis.id, icpt), {}))
            if not axes.is_valid_path(target_path, and_leaf=True):
                raise ValueError("incorrect/insufficient indices")

    return AxisTree(
        indexed_axes.parent_to_children,
        target_paths=tpaths,
        index_exprs=index_expr_per_target,
        layout_exprs=layout_expr_per_target,
        domain_index_exprs=domain_index_exprs,
    )


def _index_axes_rec(
    indices,
    *,
    current_index,
    **kwargs,
):
    index_data = collect_shape_index_callback(current_index, **kwargs)
    axes_per_index, *rest = index_data

    (
        target_path_per_cpt_per_index,
        index_exprs_per_cpt_per_index,
        layout_exprs_per_cpt_per_index,
        domain_index_exprs_per_cpt_per_index,
    ) = tuple(map(dict, rest))

    if axes_per_index:
        leafkeys = axes_per_index.leaves
    else:
        leafkeys = [None]

    subaxes = {}
    if current_index.id in indices.parent_to_children:
        for leafkey, subindex in checked_zip(
            leafkeys, indices.parent_to_children[current_index.id]
        ):
            if subindex is None:
                continue
            retval = _index_axes_rec(
                indices,
                current_index=subindex,
                **kwargs,
            )
            subaxes[leafkey] = retval[0]

            for key in retval[1].keys():
                if key in target_path_per_cpt_per_index:
                    target_path_per_cpt_per_index[key] = (
                        target_path_per_cpt_per_index[key] | retval[1][key]
                    )
                    index_exprs_per_cpt_per_index[key] = (
                        index_exprs_per_cpt_per_index[key] | retval[2][key]
                    )
                    layout_exprs_per_cpt_per_index[key] = (
                        layout_exprs_per_cpt_per_index[key] | retval[3][key]
                    )
                else:
                    target_path_per_cpt_per_index.update({key: retval[1][key]})
                    index_exprs_per_cpt_per_index.update({key: retval[2][key]})
                    layout_exprs_per_cpt_per_index.update({key: retval[3][key]})

                assert key not in domain_index_exprs_per_cpt_per_index
                domain_index_exprs_per_cpt_per_index[key] = retval[4].get(key, pmap())

    target_path_per_component = freeze(target_path_per_cpt_per_index)
    index_exprs_per_component = freeze(index_exprs_per_cpt_per_index)
    layout_exprs_per_component = freeze(layout_exprs_per_cpt_per_index)
    domain_index_exprs_per_cpt_per_index = freeze(domain_index_exprs_per_cpt_per_index)

    axes = axes_per_index
    for k, subax in subaxes.items():
        if subax is not None:
            if axes:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = subax

    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
        domain_index_exprs_per_cpt_per_index,
    )


def _compose_bits(
    axes,
    prev_target_paths,
    prev_index_exprs,
    prev_layout_exprs,
    indexed_axes,
    itarget_paths,
    iindex_exprs,
    ilayout_exprs,
    *,
    iaxis=None,
    target_path=pmap(),
    partial_index_exprs=pmap(),
    partial_layout_exprs=pmap(),
    visited_target_axes=frozenset(),
    target_path_acc=pmap(),
    index_exprs_acc=pmap(),
    layout_exprs_acc=pmap(),
):
    if not indexed_axes:
        return (
            freeze({None: itarget_paths.get(None, pmap())}),
            freeze({None: iindex_exprs.get(None, pmap())}),
            freeze({None: ilayout_exprs.get(None, pmap())}),
        )

    if iaxis is None:
        target_path |= itarget_paths.get(None, {})
        partial_index_exprs |= iindex_exprs.get(None, {})
        iaxis = indexed_axes.root

    target_path_per_cpt = collections.defaultdict(dict)
    index_exprs = collections.defaultdict(dict)
    layout_exprs = collections.defaultdict(dict)

    for cidx, icpt in enumerate(iaxis.components):
        new_target_path_acc = target_path_acc
        new_index_exprs_acc = index_exprs_acc

        itarget_path = itarget_paths.get((iaxis.id, icpt.label), {})
        # turn this into something else...
        new_target_path = target_path | itarget_path

        new_partial_index_exprs = partial_index_exprs | iindex_exprs.get(
            (iaxis.id, icpt.label), {}
        )
        new_partial_layout_exprs = dict(partial_layout_exprs)
        if (iaxis.id, icpt.label) in ilayout_exprs:
            new_partial_layout_exprs[iaxis.id, icpt.label] = ilayout_exprs[
                iaxis.id, icpt.label
            ]

        # if target_path is "complete" then do stuff, else pass responsibility to next func down
        new_visited_target_axes = visited_target_axes
        if axes.is_valid_path(new_target_path):
            detailed_path = axes.detailed_path(new_target_path)

            for target_axis, target_cpt in detailed_path.items():
                skip = target_axis.label in new_visited_target_axes
                new_visited_target_axes |= {target_axis.label}

                new_target_path_acc = new_target_path_acc | prev_target_paths.get(
                    (target_axis.id, target_cpt.label), {}
                )

                if not skip:
                    for myaxlabel, mycptlabel in prev_target_paths.get(
                        (target_axis.id, target_cpt.label), {}
                    ).items():
                        target_path_per_cpt[iaxis.id, icpt.label][
                            myaxlabel
                        ] = mycptlabel

                # do a replacement for index exprs
                # compose index expressions, this does an *inside* substitution
                # so the final replace map is target -> f(src)
                # loop over the original replace map and substitute each value
                # but drop some bits if indexed out... and final map is per component of the new axtree
                orig_index_exprs = prev_index_exprs[target_axis.id, target_cpt.label]
                for axis_label, index_expr in orig_index_exprs.items():
                    # new_index_expr = IndexExpressionReplacer(new_partial_index_exprs)(
                    new_index_expr = IndexExpressionReplacer(new_partial_index_exprs)(
                        index_expr
                    )
                    if not skip:
                        index_exprs[iaxis.id, icpt.label][
                            axis_label  # this axis label is the *final* target, unlike the intermediate target called target_axis here
                        ] = new_index_expr
                    new_index_exprs_acc = new_index_exprs_acc | {
                        axis_label: new_index_expr
                    }

            # now do the layout expressions, this is simpler since target path magic isnt needed
            # compose layout expressions, this does an *outside* substitution
            # so the final replace map is src -> h(final)
            # we start with src -> f(intermediate)
            # and intermediate -> g(final)

            # only do this if we are indexing an axis tree, not an array
            if prev_layout_exprs is not None:
                full_replace_map = merge_dicts(
                    [
                        prev_layout_exprs[tgt_ax.id, tgt_cpt.label]
                        for tgt_ax, tgt_cpt in detailed_path.items()
                    ]
                )
                for ikey, layout_expr in new_partial_layout_exprs.items():
                    # always 1:1 for layouts
                    mykey, myvalue = just_one(layout_expr.items())
                    mytargetpath = just_one(itarget_paths[ikey].keys())
                    layout_expr_replace_map = {
                        mytargetpath: full_replace_map[mytargetpath]
                    }
                    new_layout_expr = IndexExpressionReplacer(layout_expr_replace_map)(
                        myvalue
                    )
                    layout_exprs[ikey][mykey] = new_layout_expr

        isubaxis = indexed_axes.child(iaxis, icpt)
        if isubaxis:
            (
                subtarget_path,
                subindex_exprs,
                sublayout_exprs,
            ) = _compose_bits(
                axes,
                prev_target_paths,
                prev_index_exprs,
                prev_layout_exprs,
                indexed_axes,
                itarget_paths,
                iindex_exprs,
                ilayout_exprs,
                iaxis=isubaxis,
                target_path=new_target_path,
                partial_index_exprs=new_partial_index_exprs,
                partial_layout_exprs=new_partial_layout_exprs,
                visited_target_axes=new_visited_target_axes,
                target_path_acc=new_target_path_acc,
                index_exprs_acc=new_index_exprs_acc,
            )
            target_path_per_cpt.update(subtarget_path)
            index_exprs.update(subindex_exprs)
            layout_exprs.update(sublayout_exprs)

        else:
            pass

    return (
        freeze(dict(target_path_per_cpt)),
        freeze(dict(index_exprs)),
        freeze(dict(layout_exprs)),
    )


def iter_axis_tree(
    axes: AxisTree,
    target_paths,
    index_exprs,
    outer_loops=pmap(),
    axis=None,
    path=pmap(),
    indices=pmap(),
    target_path=None,
    index_exprs_acc=None,
):
    if target_path is None:
        assert index_exprs_acc is None
        target_path = target_paths.get(None, pmap())

        myindex_exprs = index_exprs.get(None, pmap())
        new_exprs = {}
        for axlabel, index_expr in myindex_exprs.items():
            new_index = ExpressionEvaluator(outer_loops)(index_expr)
            assert new_index != index_expr
            new_exprs[axlabel] = new_index
        index_exprs_acc = freeze(new_exprs)

    if axes.is_empty:
        yield pmap(), target_path, pmap(), index_exprs_acc
        return

    axis = axis or axes.root

    for component in axis.components:
        path_ = path | {axis.label: component.label}
        target_path_ = target_path | target_paths.get((axis.id, component.label), {})
        myindex_exprs = index_exprs[axis.id, component.label]
        subaxis = axes.child(axis, component)
        for pt in range(_as_int(component.count, path, indices)):
            new_exprs = {}
            for axlabel, index_expr in myindex_exprs.items():
                new_index = ExpressionEvaluator(
                    outer_loops | indices | {axis.label: pt}
                )(index_expr)
                assert new_index != index_expr
                new_exprs[axlabel] = new_index
            index_exprs_ = index_exprs_acc | new_exprs
            indices_ = indices | {axis.label: pt}
            if subaxis:
                yield from iter_axis_tree(
                    axes,
                    target_paths,
                    index_exprs,
                    outer_loops,
                    subaxis,
                    path_,
                    indices_,
                    target_path_,
                    index_exprs_,
                )
            else:
                yield path_, target_path_, indices_, index_exprs_


class ArrayPointLabel(enum.IntEnum):
    CORE = 0
    ROOT = 1
    LEAF = 2


class IterationPointType(enum.IntEnum):
    CORE = 0
    ROOT = 1
    LEAF = 2


# TODO This should work for multiple loop indices. One should really pass a loop expression.
def partition_iterset(index: LoopIndex, arrays):
    """Split an iteration set into core, root and leaf index sets.

    The distinction between these is as follows:

    * CORE: May be iterated over without any communication at all.
    * ROOT: Requires a leaf-to-root reduction (i.e. up-to-date SF roots).
    * LEAF: Requires a root-to-leaf broadcast (i.e. up-to-date SF leaves) and also up-to-date roots.

    The partitioning algorithm basically loops over the iteration set and marks entities
    in turn. Any entries whose stencils touch an SF leaf are marked LEAF and any that do
    not touch leaves but do roots are marked ROOT. Any remaining entities do not require
    the SF and are marked CORE.

    !!! NOTE !!!

    I am changing this behaviour. I think the distinction between ROOT and LEAF is not
    meaningful. These can be lumped together into NONCORE (i.e. 'requires communication
    to be complete before computation can happen').

    """
    from pyop3.array import HierarchicalArray

    # take first
    if index.iterset.depth > 1:
        raise NotImplementedError("Need a good way to sniff the parallel axis")
    paraxis = index.iterset.root

    # FIXME, need indices per component
    if len(paraxis.components) > 1:
        raise NotImplementedError

    # at a minimum this should be done per multi-axis instead of per array
    is_root_or_leaf_per_array = {}
    for array in arrays:
        # skip purely local arrays
        if not array.array.is_distributed:
            continue

        sf = array.array.sf  # the dof sf

        # mark leaves and roots
        is_root_or_leaf = np.full(sf.size, ArrayPointLabel.CORE, dtype=np.uint8)
        is_root_or_leaf[sf.iroot] = ArrayPointLabel.ROOT
        is_root_or_leaf[sf.ileaf] = ArrayPointLabel.LEAF

        is_root_or_leaf_per_array[array.name] = is_root_or_leaf

    labels = np.full(paraxis.size, IterationPointType.CORE, dtype=np.uint8)
    for path, target_path, indices, target_indices in index.iterset.iter():
        parindex = indices[paraxis.label]
        assert isinstance(parindex, numbers.Integral)

        replace_map = freeze(
            {(index.id, axis): i for axis, i in target_indices.items()}
        )

        for array in arrays:
            # skip purely local arrays
            if not array.array.is_distributed:
                continue
            if labels[parindex] == IterationPointType.LEAF:
                continue

            # loop over stencil
            array = array.with_context({index.id: target_path})

            for (
                array_path,
                array_target_path,
                array_indices,
                array_target_indices,
            ) in array.iter_indices(replace_map):
                offset = array.simple_offset(array_target_path, array_target_indices)

                point_label = is_root_or_leaf_per_array[array.name][offset]
                if point_label == ArrayPointLabel.LEAF:
                    labels[parindex] = IterationPointType.LEAF
                    break  # no point doing more analysis
                elif point_label == ArrayPointLabel.ROOT:
                    assert labels[parindex] != IterationPointType.LEAF
                    labels[parindex] = IterationPointType.ROOT
                else:
                    assert point_label == ArrayPointLabel.CORE
                    pass

    parcpt = just_one(paraxis.components)  # for now

    core = just_one(np.nonzero(labels == IterationPointType.CORE))
    root = just_one(np.nonzero(labels == IterationPointType.ROOT))
    leaf = just_one(np.nonzero(labels == IterationPointType.LEAF))

    subsets = []
    for data in [core, root, leaf]:
        # Constant?
        size = HierarchicalArray(
            AxisTree(), data=np.asarray([len(data)]), dtype=IntType
        )
        subset = HierarchicalArray(
            Axis([AxisComponent(size, parcpt.label)], paraxis.label), data=data
        )
        subsets.append(subset)
    subsets = tuple(subsets)

    # make a new iteration set over just these indices
    # index with just core (arbitrary)

    # need to use the existing labels here
    new_iterset = index.iterset[
        Slice(
            paraxis.label,
            [Subset(parcpt.label, subsets[0])],
        )
    ]

    return index.copy(iterset=new_iterset), subsets

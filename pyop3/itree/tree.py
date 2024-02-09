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
from pyrsistent import PMap, freeze, pmap

from pyop3.array import HierarchicalArray
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
from pyop3.tree import (
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    postvisit,
)
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
        # For test_map_composition to pass this needs to be able to have a fallback
        # TODO: Figure out a better, less silent, fix
        if expr.id in self._replace_map:
            return self._replace_map[expr.id][expr.axis]
        else:
            return expr


class IndexTree(LabelledTree):
    fields = LabelledTree.fields | {"outer_loops"}

    # TODO rename to node_map
    def __init__(self, parent_to_children, outer_loops=()):
        super().__init__(parent_to_children)
        assert isinstance(outer_loops, tuple)
        self.outer_loops = outer_loops

    @classmethod
    def from_nest(cls, nest):
        root, node_map = cls._from_nest(nest)
        node_map.update({None: [root]})
        return cls(node_map)


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


class SliceComponent(LabelledNodeComponent, abc.ABC):
    def __init__(self, component):
        super().__init__(component)

    @property
    def component(self):
        return self.label


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
        # TODO cleanup
        if self._component_labels is None:
            # do this for now (since leaf_target_paths currently requires an
            # instantiated object to determine)
            self._component_labels = tuple(
                self.unique_label() for _ in self.leaf_target_paths
            )
        return self._component_labels


class ContextFreeIndex(Index, ContextFree, abc.ABC):
    # The following is unimplemented but may prove useful
    # @property
    # def axes(self):
    #     return self._tree.axes
    #
    # @property
    # def target_paths(self):
    #     return self._tree.target_paths
    #
    # @cached_property
    # def _tree(self):
    #     """
    #
    #     Notes
    #     -----
    #     This method will deliberately not work for slices since slices
    #     require additional existing axis information in order to be valid.
    #
    #     """
    #     return as_index_tree(self)
    pass


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

    # @property
    # def paths(self):
    #     return tuple(self.iterset.path(*leaf) for leaf in self.iterset.leaves)
    #
    # NOTE: This is confusing terminology. A loop index can be context-sensitive
    # in two senses:
    # 1. axes.index() is context-sensitive if axes is multi-component
    # 2. axes[p].index() is context-sensitive if p is context-sensitive
    # I think this can be resolved by considering axes[p] and axes as "iterset"
    # and handling that separately.
    def with_context(self, context, *args):
        iterset = self.iterset.with_context(context)
        source_path, path = context[self.id]

        # think I want this sorted...
        slices = []
        axis = iterset.root
        while axis is not None:
            cpt = source_path[axis.label]
            slices.append(Slice(axis.label, AffineSliceComponent(cpt)))
            axis = iterset.child(axis, cpt)

        # the iterset is a single-component full slice of the overall iterset
        iterset_ = iterset[slices]
        return ContextFreeLoopIndex(iterset_, source_path, path, id=self.id)

    # unsure if this is required
    @property
    def datamap(self):
        return self.iterset.datamap


# FIXME class hierarchy is very confusing
class ContextFreeLoopIndex(ContextFreeIndex):
    def __init__(self, iterset: AxisTree, source_path, path, *, id=None):
        super().__init__(id=id)
        self.iterset = iterset
        self.source_path = freeze(source_path)
        self.path = freeze(path)

    def with_context(self, context, *args):
        return self

    @property
    def leaf_target_paths(self):
        return (self.path,)

    # TODO is this better as an alias for iterset?
    @property
    def axes(self):
        return AxisTree()

    @property
    def target_paths(self):
        return freeze({None: self.path})

    # should now be ignored
    @property
    def index_exprs(self):
        return freeze(
            {
                None: {
                    axis: LoopIndexVariable(self, axis) for axis in self.path.keys()
                },
            }
        )

    @property
    def loops(self):
        # return self.iterset.outer_loops | {
        #     LocalLoopIndexVariable(self, axis)
        #     for axis in self.iterset.path(*self.iterset.leaf).keys()
        # }
        return self.iterset.outer_loops + (self,)

    @property
    def layout_exprs(self):
        # FIXME, no clue if this is right or not
        return freeze({None: 0})

    @property
    def datamap(self):
        return self.iterset.datamap

    def iter(self, stuff=pmap()):
        if not isinstance(self.iterset, AxisTree):
            raise NotImplementedError
        return iter_axis_tree(
            self,
            self.iterset,
            self.iterset.target_paths,
            self.iterset.index_exprs,
            stuff,
        )


# TODO This is properly awful, needs a big cleanup
class ContextFreeLocalLoopIndex(ContextFreeLoopIndex):
    @property
    def index_exprs(self):
        return freeze(
            {
                None: {
                    axis: LocalLoopIndexVariable(self, axis)
                    for axis in self.path.keys()
                }
            }
        )


# class LocalLoopIndex(AbstractLoopIndex):
class LocalLoopIndex:
    """Class representing a 'local' index."""

    def __init__(self, loop_index: LoopIndex):
        # super().__init__(id)
        self.loop_index = loop_index

    # @property
    # def id(self):
    #     return self.loop_index.id

    @property
    def iterset(self):
        return self.loop_index.iterset

    def with_context(self, context, axes=None):
        # not sure about this
        iterset = self.loop_index.iterset.with_context(context)
        path, _ = context[self.loop_index.id]  # here different from LoopIndex
        return ContextFreeLocalLoopIndex(iterset, path, path, id=self.loop_index.id)

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
        # super().__init__(label=axis, id=id, component_labels=[s.label for s in slices])
        super().__init__(label=axis, id=id)
        self.axis = axis
        self.slices = as_tuple(slices)

    @property
    def components(self):
        return self.slices

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

    def __init__(self, connectivity, name=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.connectivity = connectivity

        # TODO delete entirely
        # self.name = name

    def __call__(self, index):
        return CalledMap(self, index)

    @cached_property
    def datamap(self):
        data = {}
        for bit in self.connectivity.values():
            for map_cpt in bit:
                data.update(map_cpt.datamap)
        return pmap(data)


class CalledMap(Identified, LoopIterable):
    def __init__(self, map, from_index, *, id=None):
        Identified.__init__(self, id=id)
        self.map = map
        self.from_index = from_index

    def __getitem__(self, indices):
        raise NotImplementedError("TODO")
        # figure out the current loop context, just a single loop index
        from_index = self.from_index
        while isinstance(from_index, CalledMap):
            from_index = from_index.from_index
        existing_loop_contexts = tuple(
            freeze({from_index.id: path}) for path in from_index.paths
        )

        index_forest = {}
        for existing_context in existing_loop_contexts:
            axes = self.with_context(existing_context)
            index_forest.update(
                as_index_forest(indices, axes=axes, loop_context=existing_context)
            )

        array_per_context = {}
        for loop_context, index_tree in index_forest.items():
            indexed_axes = _index_axes(index_tree, loop_context, self.axes)

            (
                target_paths,
                index_exprs,
                layout_exprs,
            ) = _compose_bits(
                self.axes,
                self.target_paths,
                self.index_exprs,
                None,
                indexed_axes,
                indexed_axes.target_paths,
                indexed_axes.index_exprs,
                indexed_axes.layout_exprs,
            )

            array_per_context[loop_context] = HierarchicalArray(
                indexed_axes,
                data=self.array,
                layouts=self.layouts,
                target_paths=target_paths,
                index_exprs=index_exprs,
                name=self.name,
                max_value=self.max_value,
            )
        return ContextSensitiveMultiArray(array_per_context)

    def index(self) -> LoopIndex:
        context_map = {
            ctx: _index_axes(itree, ctx) for ctx, itree in as_index_forest(self).items()
        }
        context_sensitive_axes = ContextSensitiveAxisTree(context_map)
        return LoopIndex(context_sensitive_axes)

    def iter(self, outer_loops=()):
        loop_context = merge_dicts(
            iter_entry.loop_context for iter_entry in outer_loops
        )
        cf_called_map = self.with_context(loop_context)
        # breakpoint()
        return iter_axis_tree(
            self.index(),
            cf_called_map.axes,
            cf_called_map.target_paths,
            cf_called_map.index_exprs,
            outer_loops,
        )

    def with_context(self, context, axes=None):
        # TODO stole this docstring from elsewhere, correct it
        """Remove map outputs that are not present in the axes.

        This is useful for the case where we have a general map acting on a
        restricted set of axes. An example would be a cell closure map (maps
        cells to cells, edges and vertices) acting on a data structure that
        only holds values on vertices. The cell-to-cell and cell-to-edge elements
        of the closure map would produce spurious entries in the index tree.

        If the map has no valid outputs then an exception will be raised.

        """
        cf_index = self.from_index.with_context(context, axes)
        leaf_target_paths = tuple(
            freeze({mcpt.target_axis: mcpt.target_component})
            for path in cf_index.leaf_target_paths
            for mcpt in self.connectivity[path]
            # if axes is None we are *building* the axes from this map
            if axes is None
            or axes.is_valid_path(
                {mcpt.target_axis: mcpt.target_component}, complete=False
            )
        )
        if len(leaf_target_paths) == 0:
            raise RuntimeError
        return ContextFreeCalledMap(self.map, cf_index, leaf_target_paths, id=self.id)

    @property
    def name(self):
        return self.map.name

    @property
    def connectivity(self):
        return self.map.connectivity


# class ContextFreeCalledMap(Index, ContextFree):
class ContextFreeCalledMap(Index):
    def __init__(self, map, index, leaf_target_paths, *, id=None):
        super().__init__(id=id)
        self.map = map
        # better to call it "input_index"?
        self.index = index
        self._leaf_target_paths = leaf_target_paths

        # alias for compat with ContextFreeCalledMap
        self.from_index = index

    # TODO cleanup
    def with_context(self, context, axes=None):
        # maybe this line isn't needed?
        # cf_index = self.from_index.with_context(context, axes)
        cf_index = self.index
        leaf_target_paths = tuple(
            freeze({mcpt.target_axis: mcpt.target_component})
            for path in cf_index.leaf_target_paths
            for mcpt in self.map.connectivity[path]
            # if axes is None we are *building* the axes from this map
            if axes is None
            or axes.is_valid_path(
                {mcpt.target_axis: mcpt.target_component}, complete=False
            )
        )
        if len(leaf_target_paths) == 0:
            raise RuntimeError
        return ContextFreeCalledMap(self.map, cf_index, leaf_target_paths, id=self.id)

    @property
    def name(self) -> str:
        return self.map.name

    # is this ever used?
    # @property
    # def components(self):
    #     return self.map.connectivity[self.index.target_paths]

    @property
    def leaf_target_paths(self):
        return self._leaf_target_paths

    #     return tuple(
    #         freeze({mcpt.target_axis: mcpt.target_component})
    #         for path in self.index.leaf_target_paths
    #         for mcpt in self.map.connectivity[path]
    #     )

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

    # TODO This is bad design, unroll the traversal and store as properties
    @cached_property
    def _axes_info(self):
        return collect_shape_index_callback(
            self, (), include_loop_index_shape=False, prev_axes=None
        )


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


class LocalLoopIndexVariable(LoopIndexVariable):
    pass


class ContextSensitiveCalledMap(ContextSensitiveLoopIterable):
    pass


# TODO make kwargs explicit
def as_index_forest(forest: Any, *, axes=None, **kwargs):
    if forest is Ellipsis:
        # full slice of all components
        assert axes is not None
        if axes.is_empty:
            raise NotImplementedError("TODO, think about this")
        forest = Slice(
            axes.root.label,
            [AffineSliceComponent(c.label) for c in axes.root.components],
        )

    forest = _as_index_forest(forest, axes=axes, **kwargs)
    assert isinstance(forest, dict), "must be ordered"
    # print(forest)
    if axes is not None:
        forest = _validated_index_forest(forest, axes=axes, **kwargs)
        forest_ = {}
        for ctx, index_tree in forest.items():
            forest_[ctx] = index_tree.copy(outer_loops=axes.outer_loops)
        forest = forest_
    return forest


@functools.singledispatch
def _as_index_forest(arg: Any, *, axes=None, path=pmap(), **kwargs):
    # FIXME no longer a cyclic import
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
        return {pmap(): IndexTree(slice_)}
    else:
        raise TypeError(f"No handler provided for {type(arg).__name__}")


@_as_index_forest.register
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
    for context, tree in _as_index_forest(
        index, path=path, loop_context=loop_context, **kwargs
    ).items():
        # converting a single index should only produce index trees with depth 1
        assert tree.depth == 1
        cf_index = tree.root

        if subindices:
            for clabel, target_path in checked_zip(
                cf_index.component_labels, cf_index.leaf_target_paths
            ):
                # if not kwargs["axes"].is_valid_path(path|target_path):
                #     continue
                subforest = _as_index_forest(
                    subindices,
                    path=path | target_path,
                    loop_context=loop_context | context,
                    **kwargs,
                )
                for subctx, subtree in subforest.items():
                    forest[subctx] = tree.add_subtree(subtree, cf_index, clabel)
        else:
            forest[context] = tree
    return forest


@_as_index_forest.register
def _(forest: collections.abc.Mapping, **kwargs):
    return forest


@_as_index_forest.register
def _(index_tree: IndexTree, **kwargs):
    return {pmap(): index_tree}


@_as_index_forest.register
def _(index: ContextFreeIndex, **kwargs):
    return {pmap(): IndexTree(index)}


# @_as_index_forest.register
# def _(index: ContextFreeCalledMap, **kwargs):
#     return {pmap(): IndexTree(index)}


# TODO This function can definitely be refactored
@_as_index_forest.register(AbstractLoopIndex)
@_as_index_forest.register(LocalLoopIndex)
def _(index, *, loop_context=pmap(), **kwargs):
    local = isinstance(index, LocalLoopIndex)

    forest = {}
    if isinstance(index.iterset, ContextSensitive):
        for context, axes in index.iterset.context_map.items():
            if axes.is_empty:
                source_path = pmap()
                target_path = axes.target_paths.get(None, pmap())

                context_ = (
                    loop_context | context | {index.id: (source_path, target_path)}
                )

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

                    context_ = (
                        loop_context | context | {index.id: (source_path, target_path)}
                    )

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
            # TODO cleanup
            my_id = index.id if not local else index.loop_index.id
            context = loop_context | {my_id: (source_path, target_path)}

            cf_index = index.with_context(context)
            forest[context] = IndexTree(cf_index)
    return forest


@_as_index_forest.register(CalledMap)
@_as_index_forest.register(ContextFreeCalledMap)
def _(called_map, *, axes, **kwargs):
    forest = {}
    input_forest = _as_index_forest(called_map.from_index, axes=axes, **kwargs)
    for context in input_forest.keys():
        cf_called_map = called_map.with_context(context, axes)
        # breakpoint()
        forest[context] = IndexTree(cf_called_map)
    return forest


@_as_index_forest.register
def _(index: numbers.Integral, **kwargs):
    return _as_index_forest(slice(index, index + 1), **kwargs)


@_as_index_forest.register
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
    return {loop_context: IndexTree(slice_)}


@_as_index_forest.register
def _(label: str, *, axes, **kwargs):
    # if we use a string then we assume we are taking a full slice of the
    # top level axis
    axis = axes.root
    component = just_one(c for c in axis.components if c.label == label)
    slice_ = Slice(axis.label, [AffineSliceComponent(component.label)])
    return _as_index_forest(slice_, axes=axes, **kwargs)


def _validated_index_forest(forest, *, axes):
    """
    Insert slices and check things work OK.
    """
    assert axes is not None, "Cannot validate if axes are unknown"

    return {ctx: _validated_index_tree(tree, axes=axes) for ctx, tree in forest.items()}


def _validated_index_tree(tree, index=None, *, axes, path=pmap()):
    if index is None:
        index = tree.root

    new_tree = IndexTree(index)

    all_leaves_skipped = True
    for clabel, path_ in checked_zip(index.component_labels, index.leaf_target_paths):
        # can I get rid of this check? The index tree should be correct
        if not axes.is_valid_path(path | path_, complete=False):
            continue

        all_leaves_skipped = False

        if subindex := tree.child(index, clabel):
            subtree = _validated_index_tree(
                tree,
                subindex,
                axes=axes,
                path=path | path_,
            )
        else:
            subtree = _collect_extra_slices(axes, path | path_)

        if subtree:
            new_tree = new_tree.add_subtree(
                subtree,
                index,
                clabel,
            )

    # TODO make this nicer
    assert not all_leaves_skipped, "this means leaf_target_paths missed everything"
    return new_tree


def _collect_extra_slices(axes, path, *, axis=None):
    if axis is None:
        axis = axes.root

    if axis.label in path:
        if subaxis := axes.child(axis, path[axis.label]):
            return _collect_extra_slices(axes, path, axis=subaxis)
        else:
            return None
    else:
        index_tree = IndexTree(
            Slice(axis.label, [AffineSliceComponent(c.label) for c in axis.components])
        )
        for cpt, clabel in checked_zip(
            axis.components, index_tree.root.component_labels
        ):
            if subaxis := axes.child(axis, cpt):
                subtree = _collect_extra_slices(axes, path, axis=subaxis)
                if subtree:
                    index_tree = index_tree.add_subtree(
                        subtree, index_tree.root, clabel
                    )
        return index_tree


@functools.singledispatch
def collect_shape_index_callback(index, *args, **kwargs):
    raise TypeError(f"No handler provided for {type(index)}")


@collect_shape_index_callback.register
def _(
    loop_index: ContextFreeLoopIndex,
    indices,
    *,
    include_loop_index_shape,
    debug=False,
    **kwargs,
):
    if include_loop_index_shape:
        assert False, "old code"
    else:
        # if debug:
        #     breakpoint()
        axes = loop_index.axes
        target_paths = loop_index.target_paths

        index_exprs = loop_index.index_exprs
        # breakpoint()
        # index_exprs = {axis: LocalLoopIndexVariable(loop_index, axis) for axis in loop_index.iterset.path(*loop_index.iterset.leaf)}
        #
        # index_exprs = {None: index_exprs}

    return (
        axes,
        target_paths,
        index_exprs,
        loop_index.layout_exprs,
        loop_index.loops,
    )


@collect_shape_index_callback.register
def _(slice_: Slice, indices, *, prev_axes, **kwargs):
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
            # TODO handle this is in a test, slices of ragged things
            if isinstance(target_cpt.count, HierarchicalArray):
                if (
                    subslice.stop is not None
                    or subslice.start != 0
                    or subslice.step != 1
                ):
                    raise NotImplementedError("TODO")
                if len(indices) == 0:
                    size = target_cpt.count
                else:
                    size = target_cpt.count[indices]
            else:
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
        (),  # no outer loops
    )


@collect_shape_index_callback.register
def _(
    called_map: ContextFreeCalledMap,
    indices,
    *,
    include_loop_index_shape,
    prev_axes,
    debug=False,
    **kwargs,
):
    if debug:
        breakpoint()
    (
        prior_axes,
        prior_target_path_per_cpt,
        prior_index_exprs_per_cpt,
        _,
        outer_loops,
    ) = collect_shape_index_callback(
        called_map.index,
        indices,
        include_loop_index_shape=include_loop_index_shape,
        prev_axes=prev_axes,
        **kwargs,
    )

    if not prior_axes:
        prior_target_path = prior_target_path_per_cpt[None]
        prior_index_exprs = prior_index_exprs_per_cpt[None]
        (
            axis,
            target_path_per_cpt,
            index_exprs_per_cpt,
            layout_exprs_per_cpt,
        ) = _make_leaf_axis_from_called_map(
            called_map,
            prior_target_path,
            prior_index_exprs,
            include_loop_index_shape,
            prev_axes,
        )
        axes = PartialAxisTree(axis)

    else:
        axes = PartialAxisTree(prior_axes.parent_to_children)
        target_path_per_cpt = {}
        index_exprs_per_cpt = {}
        layout_exprs_per_cpt = {}
        for prior_leaf_axis, prior_leaf_cpt in prior_axes.leaves:
            prior_target_path = prior_target_path_per_cpt.get(None, pmap())
            prior_index_exprs = prior_index_exprs_per_cpt.get(None, pmap())

            for myaxis, mycomponent_label in prior_axes.path_with_nodes(
                prior_leaf_axis.id, prior_leaf_cpt
            ).items():
                prior_target_path |= prior_target_path_per_cpt.get(
                    (myaxis.id, mycomponent_label), {}
                )
                prior_index_exprs |= prior_index_exprs_per_cpt.get(
                    (myaxis.id, mycomponent_label), {}
                )

            (
                subaxis,
                subtarget_paths,
                subindex_exprs,
                sublayout_exprs,
            ) = _make_leaf_axis_from_called_map(
                called_map,
                prior_target_path,
                prior_index_exprs,
                include_loop_index_shape,
                prev_axes,
            )

            axes = axes.add_subtree(
                PartialAxisTree(subaxis),
                prior_leaf_axis,
                prior_leaf_cpt,
            )
            target_path_per_cpt.update(subtarget_paths)
            index_exprs_per_cpt.update(subindex_exprs)
            layout_exprs_per_cpt.update(sublayout_exprs)

    return (
        axes,
        freeze(target_path_per_cpt),
        freeze(index_exprs_per_cpt),
        freeze(layout_exprs_per_cpt),
        outer_loops,
    )


def _make_leaf_axis_from_called_map(
    called_map,
    prior_target_path,
    prior_index_exprs,
    include_loop_index_shape,
    prev_axes,
):
    from pyop3.array.harray import CalledMapVariable

    axis_id = Axis.unique_id()
    components = []
    target_path_per_cpt = {}
    index_exprs_per_cpt = {}
    layout_exprs_per_cpt = {}

    all_skipped = True
    for map_cpt in called_map.map.connectivity[prior_target_path]:
        if prev_axes is not None and not prev_axes.is_valid_path(
            {map_cpt.target_axis: map_cpt.target_component}, complete=False
        ):
            continue

        all_skipped = False
        if (
            isinstance(map_cpt.arity, HierarchicalArray)
            and not include_loop_index_shape
        ):
            arity = map_cpt.arity[called_map.index]
        else:
            arity = map_cpt.arity
        cpt = AxisComponent(arity, label=map_cpt.label)
        components.append(cpt)

        target_path_per_cpt[axis_id, cpt.label] = pmap(
            {map_cpt.target_axis: map_cpt.target_component}
        )

        axisvar = AxisVariable(called_map.id)

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
            map_leaf_axis.id, map_leaf_component.label
        ]

        my_index_exprs = {}
        index_expr_replace_map = {map_axes.leaf_axis.label: axisvar}
        replacer = IndexExpressionReplacer(index_expr_replace_map)
        for axlabel, index_expr in old_inner_index_expr.items():
            my_index_exprs[axlabel] = replacer(index_expr)
        new_inner_index_expr = my_index_exprs

        map_var = CalledMapVariable(
            map_cpt.array, my_target_path, prior_index_exprs, new_inner_index_expr
        )

        index_exprs_per_cpt[axis_id, cpt.label] = {map_cpt.target_axis: map_var}

        # don't think that this is possible for maps
        layout_exprs_per_cpt[axis_id, cpt.label] = {
            called_map.id: pym.primitives.NaN(IntType)
        }

    if all_skipped:
        raise RuntimeError("map does not target any relevant axes")

    axis = Axis(components, label=called_map.id, id=axis_id)

    return (
        axis,
        target_path_per_cpt,
        index_exprs_per_cpt,
        layout_exprs_per_cpt,
    )


def _index_axes(
    indices: IndexTree,
    loop_context,
    axes=None,
    include_loop_index_shape=False,
    debug=False,
):
    # if debug:
    #     breakpoint()
    (
        indexed_axes,
        tpaths,
        index_expr_per_target,
        layout_expr_per_target,
        outer_loops,
    ) = _index_axes_rec(
        indices,
        (),
        current_index=indices.root,
        loop_indices=loop_context,
        prev_axes=axes,
        include_loop_index_shape=include_loop_index_shape,
        debug=debug,
    )

    # index trees should track outer loops, I think?
    outer_loops += indices.outer_loops

    # check that slices etc have not been missed
    assert not include_loop_index_shape, "old option"
    if axes is not None:
        for leaf_iaxis, leaf_icpt in indexed_axes.leaves:
            target_path = dict(tpaths.get(None, {}))
            for iaxis, icpt in indexed_axes.path_with_nodes(
                leaf_iaxis, leaf_icpt
            ).items():
                target_path.update(tpaths.get((iaxis.id, icpt), {}))
            if not axes.is_valid_path(target_path, leaf=True):
                raise ValueError("incorrect/insufficient indices")

    return AxisTree(
        indexed_axes.parent_to_children,
        target_paths=tpaths,
        index_exprs=index_expr_per_target,
        layout_exprs=layout_expr_per_target,
        outer_loops=outer_loops,
    )


def _index_axes_rec(
    indices,
    indices_acc,
    *,
    current_index,
    debug=False,
    **kwargs,
):
    index_data = collect_shape_index_callback(
        current_index, indices_acc, debug=debug, **kwargs
    )
    axes_per_index, *rest, outer_loops = index_data

    (
        target_path_per_cpt_per_index,
        index_exprs_per_cpt_per_index,
        layout_exprs_per_cpt_per_index,
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
            indices_acc_ = indices_acc + (current_index,)

            retval = _index_axes_rec(
                indices,
                indices_acc_,
                current_index=subindex,
                debug=debug,
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

            outer_loops += retval[4]

    target_path_per_component = freeze(target_path_per_cpt_per_index)
    index_exprs_per_component = freeze(index_exprs_per_cpt_per_index)
    layout_exprs_per_component = freeze(layout_exprs_per_cpt_per_index)

    axes = PartialAxisTree(axes_per_index.parent_to_children)
    for k, subax in subaxes.items():
        if subax is not None:
            if axes:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = PartialAxisTree(subax.parent_to_children)

    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
        outer_loops,
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
                orig_index_exprs = prev_index_exprs.get(
                    (target_axis.id, target_cpt.label), pmap()
                )
                for axis_label, index_expr in orig_index_exprs.items():
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
                        prev_layout_exprs.get((tgt_ax.id, tgt_cpt.label), pmap())
                        for tgt_ax, tgt_cpt in detailed_path.items()
                    ]
                )
                for ikey, layout_expr in new_partial_layout_exprs.items():
                    # always 1:1 for layouts
                    mykey, myvalue = just_one(layout_expr.items())
                    mytargetpath = just_one(itarget_paths[ikey].keys())
                    # layout_expr_replace_map = {
                    #     mytargetpath: full_replace_map[mytargetpath]
                    # }
                    layout_expr_replace_map = full_replace_map
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


@dataclasses.dataclass(frozen=True)
class IndexIteratorEntry:
    index: LoopIndex
    source_path: PMap
    target_path: PMap
    source_exprs: PMap
    target_exprs: PMap

    @property
    def loop_context(self):
        return freeze({self.index.id: (self.source_path, self.target_path)})

    @property
    def target_replace_map(self):
        return freeze(
            {
                self.index.id: (
                    {ax: expr for ax, expr in self.source_exprs.items()},
                    {ax: expr for ax, expr in self.target_exprs.items()},
                )
            }
        )


def iter_axis_tree(
    loop_index: LoopIndex,
    axes: AxisTree,
    target_paths,
    index_exprs,
    outer_loops=(),
    axis=None,
    path=pmap(),
    indices=pmap(),
    target_path=None,
    index_exprs_acc=None,
):
    outer_replace_map = merge_dicts(
        iter_entry.target_replace_map for iter_entry in outer_loops
    )
    if target_path is None:
        assert index_exprs_acc is None
        target_path = target_paths.get(None, pmap())

        myindex_exprs = index_exprs.get(None, pmap())
        evaluator = ExpressionEvaluator(outer_replace_map)
        new_exprs = {}
        for axlabel, index_expr in myindex_exprs.items():
            new_index = evaluator(index_expr)
            assert new_index != index_expr
            new_exprs[axlabel] = new_index
        index_exprs_acc = freeze(new_exprs)

    if axes.is_empty:
        yield IndexIteratorEntry(
            loop_index, pmap(), target_path, pmap(), index_exprs_acc
        )
        return

    axis = axis or axes.root

    for component in axis.components:
        # for efficiency do these outside the loop
        path_ = path | {axis.label: component.label}
        target_path_ = target_path | target_paths.get((axis.id, component.label), {})
        myindex_exprs = index_exprs.get((axis.id, component.label), pmap())
        subaxis = axes.child(axis, component)

        # bit of a hack
        if isinstance(component.count, HierarchicalArray):
            mypath = component.count.target_paths.get(None, {})
            myindices = component.count.index_exprs.get(None, {})
            if not component.count.axes.is_empty:
                for cax, ccpt in component.count.axes.path_with_nodes(
                    *component.count.axes.leaf
                ).items():
                    mypath.update(component.count.target_paths.get((cax.id, ccpt), {}))
                    myindices.update(
                        component.count.index_exprs.get((cax.id, ccpt), {})
                    )

            mypath = freeze(mypath)
            myindices = freeze(myindices)
            replace_map = outer_replace_map | indices
        else:
            mypath = pmap()
            myindices = pmap()
            replace_map = None

        for pt in range(_as_int(component.count, replace_map, mypath, myindices)):
            new_exprs = {}
            for axlabel, index_expr in myindex_exprs.items():
                new_index = ExpressionEvaluator(
                    outer_replace_map | indices | {axis.label: pt}
                )(index_expr)
                assert new_index != index_expr
                new_exprs[axlabel] = new_index
            # breakpoint()
            index_exprs_ = index_exprs_acc | new_exprs
            indices_ = indices | {axis.label: pt}
            if subaxis:
                yield from iter_axis_tree(
                    loop_index,
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
                yield IndexIteratorEntry(
                    loop_index, path_, target_path_, indices_, index_exprs_
                )


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
    # if index.iterset.depth > 1:
    #     raise NotImplementedError("Need a good way to sniff the parallel axis")
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
    for p in index.iterset.iter():
        # hack because I wrote bad code and mix up loop indices and itersets
        p = dataclasses.replace(p, index=index)

        parindex = p.source_exprs[paraxis.label]
        assert isinstance(parindex, numbers.Integral)

        for array in arrays:
            # skip purely local arrays
            if not array.array.is_distributed:
                continue
            if labels[parindex] == IterationPointType.LEAF:
                continue

            # loop over stencil
            array = array.with_context({index.id: (p.source_path, p.target_path)})

            for q in array.iter_indices({p}):
                offset = array.offset(q.target_exprs, q.target_path)

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

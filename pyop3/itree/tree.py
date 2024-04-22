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
import pytools
from pyrsistent import PMap, freeze, pmap, thaw

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
    AxisTree,
    ContextSensitiveAxisTree,
    ContextSensitiveLoopIterable,
    ExpressionEvaluator,
    IndexedAxisTree,
)
from pyop3.dtypes import IntType, get_mpi_dtype
from pyop3.lang import KernelArgument
from pyop3.tree import (
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    MutableLabelledTreeMixin,
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
    strictly_all,
)

bsearch = pym.var("mybsearch")


class IndexExpressionReplacer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map, loop_exprs=pmap()):
        self._replace_map = replace_map
        self._loop_exprs = loop_exprs

    def map_axis_variable(self, expr):
        return self._replace_map.get(expr.axis_label, expr)

    def map_array(self, array_var):
        indices = {ax: self.rec(expr) for ax, expr in array_var.indices.items()}
        return type(array_var)(array_var.array, indices, array_var.path)

    def map_loop_index(self, index):
        if index.id in self._loop_exprs:
            return self._loop_exprs[index.id][index.axis]
        else:
            return index


class IndexTree(MutableLabelledTreeMixin, LabelledTree):
    def __init__(self, node_map=pmap()):
        super().__init__(node_map)

    @classmethod
    def from_nest(cls, nest):
        root, node_map = cls._from_nest(nest)
        node_map.update({None: [root]})
        return cls(node_map)

    @classmethod
    def from_iterable(cls, iterable):
        # All iterable entries must be indices for now as we do no parsing
        root, *rest = iterable
        node_map = {None: (root,)}
        parent = root
        for index in rest:
            node_map.update({parent.id: (index,)})
            parent = index
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
    def __init__(self, component, *, label=None):
        super().__init__(label)
        self.component = component


class AffineSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"start", "stop", "step", "label_was_none"}

    # use None for the default args here since that agrees with Python slices
    def __init__(self, component, start=None, stop=None, step=None, *, label=None, **kwargs):
        label_was_none = label is None

        super().__init__(component, label=label, **kwargs)
        # could be None here
        self.start = start if start is not None else 0
        self.stop = stop
        # could be None here
        self.step = step if step is not None else 1

        # hack to force a relabelling
        self.label_was_none = label_was_none

    @property
    def datamap(self) -> PMap:
        return pmap()

    @property
    def is_full(self):
        return self.start == 0 and self.stop is None and self.step == 1


class SubsetSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"array"}

    def __init__(self, component, array, **kwargs):
        super().__init__(component, **kwargs)
        self.array = array

    @property
    def datamap(self) -> PMap:
        return self.array.datamap


# alternative name, better or worse?
Subset = SubsetSliceComponent


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
    fields = MapComponent.fields | {"array", "arity"}

    def __init__(self, target_axis, target_component, array, *, arity=None, label=None):
        # determine the arity from the provided array
        if arity is None:
            leaf_axis, leaf_clabel = array.axes.leaf
            leaf_cidx = leaf_axis.component_index(leaf_clabel)
            arity = leaf_axis.components[leaf_cidx].count

        super().__init__(target_axis, target_component, label=label)
        self.array = array
        self._arity = arity

    @property
    def arity(self):
        return self._arity

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

    @property
    def kernel_dtype(self):
        return self.dtype


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


class LoopIndexReplacer(pym.mapper.IdentityMapper):
    def __init__(self, index):
        super().__init__()
        self._index = index

    def map_axis_variable(self, axis_var):
        # this is unconditional, key error should not occur here
        return LocalLoopIndexVariable(self._index, axis_var.axis)

    def map_array(self, array_var):
        indices = {ax: self.rec(expr) for ax, expr in array_var.indices.items()}
        return type(array_var)(array_var.array, indices, array_var.path)


# FIXME class hierarchy is very confusing
class ContextFreeLoopIndex(ContextFreeIndex):
    fields = {"iterset", "source_path", "path", "id"}

    def __init__(self, iterset: AxisTree, source_path, path, *, id=None):
        super().__init__(id=id, label=id, component_labels=("XXX",))
        self.iterset = iterset
        self.source_path = freeze(source_path)
        self.path = freeze(path)

    def with_context(self, context, *args):
        return self

    @cached_property
    def local_index(self):
        return ContextFreeLocalLoopIndex(
            self.iterset, self.source_path, self.source_path, id=self.id
        )

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
        # if self.source_path != self.path and len(self.path) != 1:
        #     raise NotImplementedError("no idea what to do here")

        # Need to replace the index_exprs with LocalLoopIndexVariable equivs
        flat_index_exprs = {}
        replacer = LoopIndexReplacer(self)
        for axis in self.iterset.nodes:
            key = axis.id, axis.component.label
            for axis_label, orig_expr in self.iterset.index_exprs[key].items():
                new_expr = replacer(orig_expr)
                flat_index_exprs[axis_label] = new_expr

        return freeze({None: flat_index_exprs})

        # target = just_one(self.path.keys())
        # return freeze(
        #     {
        #         None: {
        #             target: LoopIndexVariable(self, axis)
        #             # for axis in self.source_path.keys()
        #             for axis in self.path.keys()
        #         },
        #     }
        # )

    @property
    def loops(self):
        # return self.iterset.outer_loops | {
        #     LocalLoopIndexVariable(self, axis)
        #     for axis in self.iterset.path(*self.iterset.leaf).keys()
        # }
        # return self.iterset.outer_loops + (self,)
        return (self,)

    @property
    def layout_exprs(self):
        # FIXME, no clue if this is right or not
        return freeze({None: 0})

    @property
    def datamap(self):
        return self.iterset.datamap

    def iter(self, stuff=pmap()):
        return iter_axis_tree(
            self,
            self.iterset,
            self.iterset.target_paths,
            self.iterset.index_exprs,
            stuff,
        )
        # return iter_loop(
        #     self,
        #     # stuff,
        # )


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


class ScalarIndex(ContextFreeIndex):
    fields = {"axis", "component", "value", "id"}

    def __init__(self, axis, component, value, *, id=None):
        super().__init__(axis, component_labels=["XXX"], id=id)
        self.axis = axis
        self.component = component
        self.value = value

    @property
    def leaf_target_paths(self):
        return (freeze({self.axis: self.component}),)


# TODO I want a Slice to have "bits" like a Map/CalledMap does
class Slice(ContextFreeIndex):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    fields = ContextFreeIndex.fields - {"component_labels"} | {"axis", "slices", "numbering", "label"}

    def __init__(self, axis, slices, *, numbering=None, id=None, label=None):
        slices = as_tuple(slices)
        component_labels = [s.label for s in slices]

        super().__init__(label=label, id=id, component_labels=component_labels)
        self.axis = axis
        self.slices = slices
        self.numbering = numbering

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

    fields = {"connectivity", "name", "numbering"}

    counter = 0

    def __init__(self, connectivity, name=None, *, numbering=None) -> None:
        # FIXME It is not appropriate to attach the numbering here because the
        # numbering may differ depending on the loop context.
        if numbering is not None and len(connectivity.keys()) != 1:
            raise NotImplementedError

        super().__init__()
        self.connectivity = freeze(connectivity)
        self.numbering = numbering

        # TODO delete entirely
        if name is None:
            # lazy unique name
            name = f"_Map_{self.counter}"
            self.counter += 1
        self.name = name

    def __call__(self, index):
        if isinstance(index, (ContextFreeIndex, ContextFreeCalledMap)):
            leaf_target_paths = tuple(
                freeze({mcpt.target_axis: mcpt.target_component})
                for path in index.leaf_target_paths
                for mcpt in self.connectivity[path]
            )
            return ContextFreeCalledMap(self, index, leaf_target_paths)
        else:
            return CalledMap(self, index)

    @cached_property
    def datamap(self):
        data = {}
        for bit in self.connectivity.values():
            for map_cpt in bit:
                data.update(map_cpt.datamap)
        return pmap(data)


class CalledMap(Identified, Labelled, LoopIterable):
    def __init__(self, map, from_index, *, id=None, label=None):
        Identified.__init__(self, id=id)
        Labelled.__init__(self, label=label)
        self.map = map
        self.from_index = from_index

    def __getitem__(self, indices):
        raise NotImplementedError("TODO")
        # figure out the current loop context, just a single loop index
        # from_index = self.from_index
        # while isinstance(from_index, CalledMap):
        #     from_index = from_index.from_index
        # existing_loop_contexts = tuple(
        #     freeze({from_index.id: path}) for path in from_index.paths
        # )
        #
        # index_forest = {}
        # for existing_context in existing_loop_contexts:
        #     axes = self.with_context(existing_context)
        #     index_forest.update(
        #         as_index_forest(indices, axes=axes, loop_context=existing_context)
        #     )
        #
        # array_per_context = {}
        # for loop_context, index_tree in index_forest.items():
        #     indexed_axes = index_axes(index_tree, loop_context, self.axes)
        #
        #     (
        #         target_paths,
        #         index_exprs,
        #         layout_exprs,
        #     ) = _compose_bits(
        #         self.axes,
        #         self.target_paths,
        #         self.index_exprs,
        #         None,
        #         indexed_axes,
        #         indexed_axes.target_paths,
        #         indexed_axes.index_exprs,
        #         indexed_axes.layout_exprs,
        #     )
        #
        #     array_per_context[loop_context] = HierarchicalArray(
        #         indexed_axes,
        #         data=self.array,
        #         layouts=self.layouts,
        #         target_paths=target_paths,
        #         index_exprs=index_exprs,
        #         name=self.name,
        #         max_value=self.max_value,
        #     )
        # return ContextSensitiveMultiArray(array_per_context)

    # def index(self) -> LoopIndex:
    #     context_map = {
    #         ctx: index_axes(itree, ctx) for ctx, itree in as_index_forest(self).items()
    #     }
    #     context_sensitive_axes = ContextSensitiveAxisTree(context_map)
    #     return LoopIndex(context_sensitive_axes)

    # def iter(self, outer_loops=()):
    #     loop_context = merge_dicts(
    #         iter_entry.loop_context for iter_entry in outer_loops
    #     )
    #     cf_called_map = self.with_context(loop_context)
    #     return iter_axis_tree(
    #         self.index(),
    #         cf_called_map.axes,
    #         cf_called_map.target_paths,
    #         cf_called_map.index_exprs,
    #         outer_loops,
    #     )

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
        )
        if len(leaf_target_paths) == 0:
            raise RuntimeError
        return ContextFreeCalledMap(
            self.map, cf_index, leaf_target_paths, id=self.id, label=self.label
        )

    @property
    def name(self):
        return self.map.name

    @property
    def connectivity(self):
        return self.map.connectivity


# class ContextFreeCalledMap(Index, ContextFree):
# TODO: ContextFreeIndex
class ContextFreeCalledMap(Index):
    # FIXME this is clumsy
    # fields = Index.fields | {"map", "index", "leaf_target_paths"} - {"label", "component_labels"}
    fields = {"map", "index", "leaf_target_paths", "label", "id"}

    def __init__(self, map, index, leaf_target_paths, *, id=None, label=None):
        super().__init__(id=id, label=label)
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
        return collect_shape_index_callback(self, (), prev_axes=None)


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


# don't use this anywhere
class LoopIndexEnumerateIndexVariable(pym.primitives.Leaf):
    """Variable representing the index of an enumerated index.

    The variable is equivalent to the index ``i`` in the expression

        for i, x in enumerate(X):
            ...

    Here, if ``X`` were composed of multiple axes, this class would
    be implemented like

        i = 0
        for x0 in X[0]:
            for x1 in X[1]:
                x = f(x0, x1)
                ...
                i += 1

    This class is very important because it allows us to express layouts
    when we materialise indexed things. An example is the maps that are
    required for indexing PETSc matrices.

    """

    init_arg_names = ("index",)

    mapper_method = sys.intern("map_enumerate")

    # This could perhaps support a target_axis argument in future were we
    # to have loop indices targeting multiple output axes.
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __getinitargs__(self) -> tuple:
        return (self.index,)

    @property
    def datamap(self) -> PMap:
        return self.index.datamap


class LocalLoopIndexVariable(LoopIndexVariable):
    pass


class ContextSensitiveCalledMap(ContextSensitiveLoopIterable):
    pass


# TODO make kwargs explicit
def as_index_forest(forest: Any, *, axes=None, strict=False, **kwargs):
    # TODO: I think that this is the wrong place for this to exist. Also
    # the implementation only seems to work for flat axes.
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

    # If axes are provided then check that the index tree is compatible
    # and add extra slices if required.
    if axes is not None:
        forest_ = {}
        for ctx, tree in forest.items():
            if not strict:
                # NOTE: This function doesn't always work. In particular if
                # the loop index is from an already indexed axis. This
                # requires more thought but for now just use the strict version
                # and provide additional information elsewhere.
                tree = _complete_index_tree(tree, axes)
            if not _index_tree_is_complete(tree, axes):
                raise ValueError("Index tree does not completely index axes")
            forest_[ctx] = tree
        forest = forest_

        # TODO: Clean this up, and explain why it's here.
        forest_ = {}
        for ctx, index_tree in forest.items():
            # forest_[ctx] = index_tree.copy(outer_loops=axes.outer_loops)
            forest_[ctx] = index_tree
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
        forest[context] = IndexTree(cf_called_map)
    return forest


@_as_index_forest.register
def _(index: numbers.Integral, *, axes, **kwargs):
    # If we are dealing with a multi-component axis (e.g. a mixed thing), then
    # indexing the axis with an integer will take a full slice of a particular
    # component, if no component exists with that label then an error is raised.
    # If instead we are dealing with a single-component axis, then the normal
    # numpy-style indexing will be used where a particular index from the
    # axis will be selected for.
    root = axes.root
    if len(root.components) > 1:
        component = just_one(c for c in root.components if c.label == index)
        if component.unit:
            index_ = ScalarIndex(root.label, component.label, 0)
        else:
            index_ = Slice(root.label, [AffineSliceComponent(component.label)])
    else:
        component = just_one(root.components)
        index_ = ScalarIndex(root.label, component.label, index)
    return _as_index_forest(index_, axes=axes, **kwargs)


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

    # If the component is marked as "unit" then indexing in this way will
    # fully consume the axis.
    if component.unit:
        slice_ = ScalarIndex(axis.label, component.label, 0)
    else:
        slice_ = Slice(axis.label, [AffineSliceComponent(component.label)])
    return _as_index_forest(slice_, axes=axes, **kwargs)


def _complete_index_tree(
    tree: IndexTree, axes: AxisTree, index=None, axis_path=pmap()
) -> IndexTree:
    """Add extra slices to the index tree to match the axes.

    Notes
    -----
    This function is currently only capable of adding additional slices if
    they are "innermost".

    """
    if index is None:
        index = tree.root

    tree_ = IndexTree(index)
    for component_label, path in checked_zip(
        index.component_labels, index.leaf_target_paths
    ):
        axis_path_ = axis_path | path
        if subindex := tree.child(index, component_label):
            subtree = _complete_index_tree(
                tree,
                axes,
                subindex,
                axis_path_,
            )
        else:
            # At the bottom of the index tree, add any extra slices if needed.
            subtree = _complete_index_tree_slices(axes, axis_path_)

        tree_ = tree_.add_subtree(subtree, index, component_label)
    return tree_


def _complete_index_tree_slices(axes: AxisTree, path: PMap, axis=None) -> IndexTree:
    if axis is None:
        axis = axes.root

    if axis.label in path:
        if subaxis := axes.child(axis, path[axis.label]):
            return _complete_index_tree_slices(axes, path, subaxis)
        else:
            return IndexTree()
    else:
        # Axis is missing from the index tree, use a full slice.
        slice_ = Slice(
            axis.label, [AffineSliceComponent(c.label) for c in axis.components]
        )
        tree = IndexTree(slice_)

        for axis_component, index_component in checked_zip(
            axis.components, slice_.component_labels
        ):
            if subaxis := axes.child(axis, axis_component):
                subtree = _complete_index_tree_slices(axes, path, subaxis)
                tree = tree.add_subtree(subtree, slice_, index_component)
        return tree


def _index_tree_is_complete(indices: IndexTree, axes: AxisTree):
    """Return whether the index tree completely indexes the axis tree."""
    # For each leaf in the index tree, collect the resulting axis path
    # and check that this is a leaf of the axis tree.
    for index_leaf_path in indices.ordered_leaf_paths_with_nodes:
        axis_path = {}
        for index, index_cpt_label in index_leaf_path:
            index_cpt_index = index.component_labels.index(index_cpt_label)
            for axis, axis_cpt in index.leaf_target_paths[index_cpt_index].items():
                assert axis not in axis_path, "Paths should not clash"
                axis_path[axis] = axis_cpt
        axis_path = freeze(axis_path)

        if axis_path not in axes.leaf_paths:
            return False

    # All leaves of the tree are complete
    return True


@functools.singledispatch
def collect_shape_index_callback(index, *args, **kwargs):
    raise TypeError(f"No handler provided for {type(index)}")


@collect_shape_index_callback.register
def _(
    loop_index: ContextFreeLoopIndex,
    indices,
    **kwargs,
):
    axes = loop_index.axes
    target_paths = loop_index.target_paths
    index_exprs = loop_index.index_exprs

    return (
        axes,
        target_paths,
        index_exprs,
        loop_index.layout_exprs,
        loop_index.loops,
        {},
    )


@collect_shape_index_callback.register
def _(index: ScalarIndex, indices, *, target_path_acc, prev_axes, **kwargs):
    target_path = freeze({None: just_one(index.leaf_target_paths)})
    index_exprs = freeze({None: {index.axis: index.value}})
    layout_exprs = freeze({None: 0})
    return (
        AxisTree(),
        target_path,
        index_exprs,
        layout_exprs,
        (),
        {},
    )


@collect_shape_index_callback.register
def _(slice_: Slice, indices, *, target_path_acc, prev_axes, **kwargs):
    from pyop3.array.harray import ArrayVar

    # If we are just taking a component from a multi-component array,
    # e.g. mesh.points["cells"], then relabelling the axes just leads to
    # needless confusion. For instance if we had
    #
    #     myslice0 = Slice("mesh", AffineSliceComponent("cells", step=2))
    #
    # then mesh.points[myslice0] would work but mesh.points["cells"][myslice0]
    # would fail.
    # As a counter example, if we have non-trivial subsets then this sort of
    # relabelling is essential for things to make sense. If we have two subsets:
    #
    #     subset0 = Slice("mesh", Subset("cells", [1, 2, 3]))
    #
    # and
    #
    #     subset1 = Slice("mesh", Subset("cells", [4, 5, 6]))
    #
    # then mesh.points[subset0][subset1] is confusing, should subset1 be
    # assumed to work on the already sliced axis? This can be a major source of
    # confusion for things like interior facets in Firedrake where the first slice
    # happens in one function and the other happens elsewhere. We hit situations like
    #
    #     mesh.interior_facets[interior_facets_I_want]
    #
    # conflicts with
    #
    #     mesh.interior_facets[facets_I_want]
    #
    # where one subset is given with facet numbering and the other with interior
    # facet numbering. The labels are the same so identifying this is really difficult.
    #
    # We fix this here by requiring that non-full slices perform a relabelling and
    # full slices do not.
    is_full_slice = all(
        isinstance(s, AffineSliceComponent) and s.is_full for s in slice_.slices
    )

    # It would be better here to relabel if a label is provided but default to keeping the same
    axis_label = slice_.axis if is_full_slice else slice_.label

    components = []
    target_path_per_subslice = []
    index_exprs_per_subslice = []
    layout_exprs_per_subslice = []

    if not prev_axes.is_valid_path(target_path_acc, complete=False):
        raise NotImplementedError(
            "If we swap axes around then we must check "
            "that we don't get clashes."
        )

        # previous code:
        # we are assuming that axes with the same label *must* be identical. They are
        # only allowed to differ in that they have different IDs.
        # target_axis, target_cpt = prev_axes.find_component(
        #     slice_.axis, subslice.component, also_node=True
        # )

    if not target_path_acc:
        target_axis = prev_axes.root
    else:
        parent = prev_axes._node_from_path(target_path_acc)
        target_axis = prev_axes.child(*parent)

    for subslice in slice_.slices:
        assert target_axis.label == slice_.axis
        target_cpt = just_one(
            c for c in target_axis.components if c.label == subslice.component
        )

        if isinstance(subslice, AffineSliceComponent):
            # TODO handle this is in a test, slices of ragged things
            if isinstance(target_cpt.count, HierarchicalArray):
                if target_cpt.distributed:
                    raise NotImplementedError

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

                if target_cpt.distributed:
                    if subslice.start != 0 or subslice.step != 1:
                        raise NotImplementedError

                    owned_count = min(target_cpt.owned_count, stop)
                    count = stop

                    if owned_count == count:
                        size = count
                    else:
                        size = (owned_count, count)
                else:
                    size = math.ceil((stop - subslice.start) / subslice.step)

        else:
            assert isinstance(subslice, Subset)
            size = subslice.array.axes.leaf_component.count

        if is_full_slice and subslice.label_was_none:
            mylabel = subslice.component
        else:
            mylabel = subslice.label
        cpt = AxisComponent(size, label=mylabel, unit=target_cpt.unit, rank_equal=target_cpt.rank_equal)
        components.append(cpt)

        target_path_per_subslice.append(pmap({slice_.axis: subslice.component}))

        newvar = AxisVariable(axis_label)
        layout_var = AxisVariable(slice_.axis)
        if isinstance(subslice, AffineSliceComponent):
            if is_full_slice:
                index_exprs_per_subslice.append(
                    freeze(
                        {
                            slice_.axis: newvar * subslice.step + subslice.start,
                        }
                    )
                )
            else:
                index_exprs_per_subslice.append(
                    freeze(
                        {
                            slice_.axis: newvar * subslice.step + subslice.start,
                            # slice_.label: AxisVariable(slice_.label),
                        }
                    )
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
                subset_array.axes.target_paths.get(key, {}) for key in index_keys
            )
            old_index_exprs = merge_dicts(
                subset_array.axes.index_exprs.get(key, {}) for key in index_keys
            )

            my_index_exprs = {}
            index_expr_replace_map = {subset_axes.leaf_axis.label: newvar}
            replacer = IndexExpressionReplacer(index_expr_replace_map)
            for axlabel, index_expr in old_index_exprs.items():
                my_index_exprs[axlabel] = replacer(index_expr)
            subset_var = ArrayVar(subslice.array, my_index_exprs, my_target_path)

            if is_full_slice:
                index_exprs_per_subslice.append(
                    freeze(
                        {
                            slice_.axis: subset_var,
                        }
                    )
                )
            else:
                index_exprs_per_subslice.append(
                    freeze(
                        {
                            slice_.axis: subset_var,
                            # slice_.label: AxisVariable(slice_.label),
                        }
                    )
                )
            layout_exprs_per_subslice.append(
                pmap({slice_.label: bsearch(subset_var, layout_var)})
            )

    axis = Axis(components, label=axis_label, numbering=slice_.numbering)
    axes = AxisTree(axis)
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
        freeze(target_path_per_component),
        freeze(index_exprs_per_component),
        freeze(layout_exprs_per_component),
        (),  # no outer loops
        {},
    )


@collect_shape_index_callback.register
def _(
    called_map: ContextFreeCalledMap,
    indices,
    *,
    prev_axes,
    **kwargs,
):
    (
        prior_axes,
        prior_target_path_per_cpt,
        prior_index_exprs_per_cpt,
        _,
        outer_loops,
        prior_extra_index_exprs,
    ) = collect_shape_index_callback(
        called_map.index,
        indices,
        prev_axes=prev_axes,
        **kwargs,
    )

    extra_index_exprs = dict(prior_extra_index_exprs)

    if not prior_axes:
        prior_target_path = prior_target_path_per_cpt[None]
        prior_index_exprs = prior_index_exprs_per_cpt[None]
        (
            axis,
            target_path_per_cpt,
            index_exprs_per_cpt,
            layout_exprs_per_cpt,
            more_extra_index_exprs,
        ) = _make_leaf_axis_from_called_map(
            called_map,
            prior_target_path,
            prior_index_exprs,
            prev_axes,
        )
        axes = AxisTree(axis)

        extra_index_exprs.update(more_extra_index_exprs)

    else:
        axes = AxisTree(prior_axes.node_map)
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
                subextra_index_exprs,
            ) = _make_leaf_axis_from_called_map(
                called_map,
                prior_target_path,
                prior_index_exprs,
                prev_axes,
            )

            axes = axes.add_subtree(
                AxisTree(subaxis),
                prior_leaf_axis,
                prior_leaf_cpt,
            )
            target_path_per_cpt.update(subtarget_paths)
            index_exprs_per_cpt.update(subindex_exprs)
            layout_exprs_per_cpt.update(sublayout_exprs)
            extra_index_exprs.update(subextra_index_exprs)

    return (
        axes,
        freeze(target_path_per_cpt),
        freeze(index_exprs_per_cpt),
        freeze(layout_exprs_per_cpt),
        outer_loops,
        freeze(extra_index_exprs),
    )


def _make_leaf_axis_from_called_map(
    called_map,
    prior_target_path,
    prior_index_exprs,
    prev_axes,
):
    from pyop3.array.harray import CalledMapVariable

    axis_id = Axis.unique_id()
    components = []
    target_path_per_cpt = {}
    index_exprs_per_cpt = {}
    layout_exprs_per_cpt = {}
    extra_index_exprs = {}

    all_skipped = True
    for map_cpt in called_map.map.connectivity[prior_target_path]:
        if prev_axes is not None and not prev_axes.is_valid_path(
            {map_cpt.target_axis: map_cpt.target_component}, complete=False
        ):
            continue

        all_skipped = False
        if isinstance(map_cpt.arity, HierarchicalArray):
            arity = map_cpt.arity[called_map.index]
        else:
            arity = map_cpt.arity
        cpt = AxisComponent(arity, label=map_cpt.label)
        components.append(cpt)

        target_path_per_cpt[axis_id, cpt.label] = pmap(
            {map_cpt.target_axis: map_cpt.target_component}
        )

        axisvar = AxisVariable(called_map.map.name)

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
            map_array.axes.target_paths.get(key, {}) for key in index_keys
        )

        # the outer index is provided from "prior" whereas the inner one requires
        # a replacement
        map_leaf_axis, map_leaf_component = map_axes.leaf
        old_inner_index_expr = map_array.axes.index_exprs[
            map_leaf_axis.id, map_leaf_component
        ]

        my_index_exprs = {}
        index_expr_replace_map = {map_axes.leaf_axis.label: axisvar}
        replacer = IndexExpressionReplacer(index_expr_replace_map)
        for axlabel, index_expr in old_inner_index_expr.items():
            my_index_exprs[axlabel] = replacer(index_expr)
        new_inner_index_expr = my_index_exprs

        map_var = CalledMapVariable(
            map_cpt.array,
            # order is important to avoid overwriting prior, cleanup
            # merge_dicts([prior_index_exprs, new_inner_index_expr]),
            merge_dicts([new_inner_index_expr, prior_index_exprs]),
            my_target_path,
        )

        index_exprs_per_cpt[axis_id, cpt.label] = {
            map_cpt.target_axis: map_var,
        }

        # also one for the new axis
        # Nooooo, bad idea
        extra_index_exprs[axis_id, cpt.label] = {
            # axisvar.axis: axisvar,
        }

        # don't think that this is possible for maps
        layout_exprs_per_cpt[axis_id, cpt.label] = {
            called_map.id: pym.primitives.NaN(IntType)
        }

    if all_skipped:
        raise RuntimeError("map does not target any relevant axes")

    axis = Axis(
        components,
        label=called_map.map.name,
        id=axis_id,
        numbering=called_map.map.numbering,
    )

    return (
        axis,
        target_path_per_cpt,
        index_exprs_per_cpt,
        layout_exprs_per_cpt,
        extra_index_exprs,
    )


def index_axes(
    indices: IndexTree,
    loop_context,
    axes=None,
):
    (
        indexed_axes,
        tpaths,
        index_expr_per_target,
        layout_expr_per_target,
        outer_loops,
    ) = _index_axes_rec(
        indices,
        (),
        pmap(),  # target_path
        current_index=indices.root,
        loop_indices=loop_context,
        prev_axes=axes,
    )

    outer_loops += axes.outer_loops

    # drop duplicates
    outer_loops_ = []
    allids = set()
    for ol in outer_loops:
        if ol.id in allids:
            continue
        outer_loops_.append(ol)
        allids.add(ol.id)
    outer_loops = tuple(outer_loops_)

    # check that slices etc have not been missed
    if axes is not None:
        for leaf_iaxis, leaf_icpt in indexed_axes.leaves:
            target_path = dict(tpaths.get(None, {}))
            for iaxis, icpt in indexed_axes.path_with_nodes(
                leaf_iaxis, leaf_icpt
            ).items():
                target_path.update(tpaths.get((iaxis.id, icpt), {}))
            if not axes.is_valid_path(target_path, leaf=True):
                raise ValueError("incorrect/insufficient indices")

    mytpaths = _acc_target_paths(indexed_axes, tpaths)
    myindex_expr_per_target = _acc_target_paths(indexed_axes, index_expr_per_target)

    return IndexedAxisTree(
        indexed_axes.node_map,
        axes.unindexed,
        target_paths=mytpaths,
        index_exprs=myindex_expr_per_target,
        # layout_exprs=mylayout_expr_per_target,
        layout_exprs={},  # disable for now
        outer_loops=outer_loops,
    )


def _acc_target_paths(axes, target_paths, axis=None, target_path_acc=None):
    if axes.is_empty:
        return target_paths

    target_paths_merged = {}

    if strictly_all(x is None for x in {axis, target_path_acc}):
        axis = axes.root
        target_path_acc = target_paths.get(None, pmap())
        target_paths_merged[None] = target_path_acc

    for component in axis.components:
        key = (axis.id, component.label)
        target_path_acc_ = target_path_acc | target_paths.get(key, {})
        target_paths_merged[key] = target_path_acc_

        if subaxis := axes.child(axis, component):
            target_paths_merged.update(
                _acc_target_paths(axes, target_paths, subaxis, target_path_acc_)
            )
    return freeze(target_paths_merged)


def _index_axes_rec(
    indices,
    indices_acc,
    target_path_acc,
    *,
    current_index,
    **kwargs,
):
    index_data = collect_shape_index_callback(
        current_index,
        indices_acc,
        target_path_acc=target_path_acc,
        **kwargs,
    )
    axes_per_index, *rest, outer_loops, extra_index_exprs = index_data

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
    if current_index.id in indices.node_map:
        for leafkey, subindex in checked_zip(
            leafkeys, indices.node_map[current_index.id]
        ):
            if subindex is None:
                continue
            indices_acc_ = indices_acc + (current_index,)

            target_path_acc_ = dict(target_path_acc)
            target_path_acc_.update(target_path_per_cpt_per_index.get(None, {}))
            if not axes_per_index.is_empty:
                for _ax, _cpt in axes_per_index.path_with_nodes(*leafkey).items():
                    target_path_acc_.update(
                        target_path_per_cpt_per_index.get((_ax.id, _cpt), {})
                    )
            target_path_acc_ = freeze(target_path_acc_)

            retval = _index_axes_rec(
                indices,
                indices_acc_,
                target_path_acc_,
                current_index=subindex,
                **kwargs,
            )
            subaxes[leafkey] = retval[0]

            for key in retval[1].keys():
                if key is None:
                    # if key is None then tie the things to the parent axis
                    if subaxes[leafkey].is_empty:
                        mykey = leafkey[0].id, leafkey[1]
                        # don't overwite (better with defaultdict)
                        if mykey in target_path_per_cpt_per_index:
                            # NOTE: .update will not currently work as retval has pmaps
                            target_path_per_cpt_per_index[mykey] = (
                                target_path_per_cpt_per_index[mykey] | retval[1][None]
                            )
                            index_exprs_per_cpt_per_index[mykey] = (
                                index_exprs_per_cpt_per_index[mykey] | retval[2][None]
                            )
                        else:
                            target_path_per_cpt_per_index[mykey] = retval[1][None]
                            index_exprs_per_cpt_per_index[mykey] = retval[2][None]
                elif key in target_path_per_cpt_per_index:
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
    index_exprs_per_component = thaw(index_exprs_per_cpt_per_index)
    for key, inner in extra_index_exprs.items():
        if key in index_exprs_per_component:
            for ax, expr in inner.items():
                assert ax not in index_exprs_per_component[key]
                index_exprs_per_component[key][ax] = expr
        else:
            index_exprs_per_component[key] = inner
    index_exprs_per_component = freeze(index_exprs_per_component)
    layout_exprs_per_component = freeze(layout_exprs_per_cpt_per_index)

    # This this is no longer necessary
    axes = AxisTree(axes_per_index.node_map)
    for k, subax in subaxes.items():
        if subax is not None:
            if axes:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = AxisTree(subax.node_map)

    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
        outer_loops,
    )


def compose_axes(indexed_axes, orig_axes):
    target_paths, index_exprs = _compose_axes_rec(indexed_axes, orig_axes)
    return IndexedAxisTree(
        indexed_axes.node_map,
        orig_axes.unindexed,
        target_paths=target_paths,
        index_exprs=index_exprs,
        layout_exprs={},
        outer_loops=indexed_axes.outer_loops,
    )


def _compose_axes_rec(indexed_axes, orig_axes, *, indexed_axis=None):
    composed_target_paths = collections.defaultdict(dict)
    composed_index_exprs = collections.defaultdict(dict)
    # composed_layout_exprs = defaultdict(dict)  # TODO

    if indexed_axis is None:
        ikey = None
        partial_target_path = indexed_axes.target_paths.get(ikey, {})
        partial_index_exprs = indexed_axes.index_exprs.get(ikey, {})

        if orig_axes.is_valid_path(partial_target_path):
            orig_axis, orig_component = orig_axes._node_from_path(partial_target_path)
            okey = (orig_axis.id, orig_component.label)

            # 1. Determine target_paths.
            composed_target_paths[ikey] = orig_axes.target_paths[okey]

            # 2. Determine index_exprs. This is done via an *inside* substitution
            # ... old below
            # so the final replace map is target -> f(src)
            # loop over the original replace map and substitute each value
            # but drop some bits if indexed out... and final map is per component of the new axtree
            replacer = IndexExpressionReplacer(partial_index_exprs)
            for oaxis_label, oindex_expr in orig_axes.index_exprs.get(
                okey, pmap()
            ).items():
                composed_index_exprs[ikey][oaxis_label] = replacer(oindex_expr)

        # Keep the bits that are already indexed out
        composed_target_paths[None].update(orig_axes.target_paths.get(None, {}))
        composed_index_exprs[None].update(orig_axes.index_exprs.get(None, {}))

        if indexed_axes.is_empty:
            # Can do nothing more, stop here
            return (freeze(composed_target_paths), freeze(composed_index_exprs))
        else:
            indexed_axis = indexed_axes.root

    for indexed_component in indexed_axis.components:
        ikey = (indexed_axis.id, indexed_component.label)
        partial_target_path = indexed_axes.target_paths.get(ikey, pmap())
        partial_index_exprs = indexed_axes.index_exprs.get(ikey, pmap())

        if orig_axes.is_valid_path(partial_target_path):
            orig_axis, orig_component = orig_axes._node_from_path(partial_target_path)
            okey = (orig_axis.id, orig_component.label)

            # 1. Determine target_paths.
            composed_target_paths[ikey] = orig_axes.target_paths.get(okey, pmap())

            # 2. Determine index_exprs. This is done via an *inside* substitution
            # ... old below
            # so the final replace map is target -> f(src)
            # loop over the original replace map and substitute each value
            # but drop some bits if indexed out... and final map is per component of the new axtree
            replacer = IndexExpressionReplacer(partial_index_exprs)
            for oaxis_label, oindex_expr in orig_axes.index_exprs.get(
                okey, pmap()
            ).items():
                composed_index_exprs[ikey][oaxis_label] = replacer(oindex_expr)

            # 3. Determine layout_exprs...
            # ...
            # now do the layout expressions, this is simpler since target path magic isnt needed
            # compose layout expressions, this does an *outside* substitution
            # so the final replace map is src -> h(final)
            # we start with src -> f(intermediate)
            # and intermediate -> g(final)

            # only do this if we are indexing an axis tree, not an array
            # if prev_layout_exprs is not None:
            #     full_replace_map = merge_dicts(
            #         [
            #             prev_layout_exprs.get((tgt_ax.id, tgt_cpt.label), pmap())
            #             for tgt_ax, tgt_cpt in detailed_path.items()
            #         ]
            #     )
            #     for ikey, layout_expr in new_partial_layout_exprs.items():
            #         # always 1:1 for layouts
            #         mykey, myvalue = just_one(layout_expr.items())
            #         mytargetpath = just_one(itarget_paths[ikey].keys())
            #         # layout_expr_replace_map = {
            #         #     mytargetpath: full_replace_map[mytargetpath]
            #         # }
            #         layout_expr_replace_map = full_replace_map
            #         new_layout_expr = IndexExpressionReplacer(layout_expr_replace_map)(
            #             myvalue
            #         )
            #
            #         # this is a trick to get things working in Firedrake, needs more
            #         # thought to understand what is going on
            #         if ikey in layout_exprs and mykey in layout_exprs[ikey]:
            #             assert layout_exprs[ikey][mykey] == new_layout_expr
            #         else:
            #             layout_exprs[ikey][mykey] = new_layout_expr

        if indexed_subaxis := indexed_axes.child(indexed_axis, indexed_component):
            (
                subtarget_paths,
                subindex_exprs,
                # sublayout_exprs,
            ) = _compose_axes_rec(
                indexed_axes,
                orig_axes,
                indexed_axis=indexed_subaxis,
            )
            composed_target_paths.update(subtarget_paths)
            composed_index_exprs.update(subindex_exprs)
            # composed_layout_exprs.update(sublayout_exprs)

    return (
        freeze(composed_target_paths),
        freeze(composed_index_exprs),
        # freeze(composed_layout_exprs),
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
    def replace_map(self):
        return freeze(
            {self.index.id: merge_dicts([self.source_exprs, self.target_exprs])}
        )

    @property
    def target_replace_map(self):
        return freeze(
            {
                self.index.id: {ax: expr for ax, expr in self.target_exprs.items()},
            }
        )

    @property
    def source_replace_map(self):
        return freeze(
            {
                self.index.id: {ax: expr for ax, expr in self.source_exprs.items()},
            }
        )


def iter_loop(loop):
    if len(loop.target_paths) != 1:
        raise NotImplementedError

    if loop.iterset.outer_loops:
        outer_loop = just_one(loop.iterset.outer_loops)
        for indices in outer_loop.iter():
            for i, index in enumerate(loop.iterset.iter(indices)):
                # hack needed because we mix up our source and target exprs
                axis_label = just_one(
                    just_one(loop.iterset.target_paths.values()).keys()
                )

                # source_path = {}
                source_expr = {loop.id: {axis_label: i}}

                target_expr_sym = merge_dicts(loop.iterset.index_exprs.values())[
                    axis_label
                ]
                replace_map = {axis_label: i}
                loop_exprs = merge_dicts(idx.target_replace_map for idx in indices)
                target_expr = ExpressionEvaluator(replace_map, loop_exprs)(
                    target_expr_sym
                )
                target_expr = {axis_label: target_expr}

                # new_exprs = {}
                # evaluator = ExpressionEvaluator(
                #     indices | {axis.label: pt}, outer_replace_map
                # )
                # for axlabel, index_expr in myindex_exprs.items():
                #     new_index = evaluator(index_expr)
                #     assert new_index != index_expr
                #     new_exprs[axlabel] = new_index

                index = IndexIteratorEntry(
                    loop, source_path, target_path, source_expr, target_expr
                )

                yield indices + (index,)
    else:
        for i, index in enumerate(loop.iterset.iter()):
            # hack needed because we mix up our source and target exprs
            axis_label = just_one(just_one(loop.iterset.target_paths.values()).keys())

            source_path = "NA"
            target_path = "NA"

            source_expr = {axis_label: i}

            target_expr_sym = merge_dicts(loop.iterset.index_exprs.values())[axis_label]
            replace_map = {axis_label: i}
            target_expr = ExpressionEvaluator(replace_map, {})(target_expr_sym)
            target_expr = {axis_label: target_expr}

            iter_entry = IndexIteratorEntry(
                loop,
                source_path,
                target_path,
                freeze(source_expr),
                freeze(target_expr),
            )
            yield (iter_entry,)


def iter_axis_tree(
    loop_index: LoopIndex,
    axes: AxisTree,
    target_paths,
    index_exprs,
    outer_loops=(),
    include_loops=False,
    axis=None,
    path=pmap(),
    indices=pmap(),
    target_path=None,
    index_exprs_acc=None,
):
    outer_replace_map = merge_dicts(
        # iter_entry.target_replace_map for iter_entry in outer_loops
        # iter_entry.source_replace_map
        iter_entry.replace_map
        for iter_entry in outer_loops
    )
    if target_path is None:
        assert index_exprs_acc is None
        target_path = target_paths.get(None, pmap())

        # Substitute the index exprs, which map target to source, into
        # indices, giving target index exprs
        myindex_exprs = index_exprs.get(None, pmap())
        evaluator = ExpressionEvaluator(indices, outer_replace_map)
        new_exprs = {}
        for axlabel, index_expr in myindex_exprs.items():
            # try:
            #     new_index = evaluator(index_expr)
            #     assert new_index != index_expr
            #     new_exprs[axlabel] = new_index
            # except UnrecognisedAxisException:
            #     pass
            new_index = evaluator(index_expr)
            # assert new_index != index_expr
            new_exprs[axlabel] = new_index
        index_exprs_acc = freeze(new_exprs)

    if axes.is_empty:
        if include_loops:
            # source_path =
            assert False, "old code"
        else:
            source_path = pmap()
            source_exprs = pmap()
        yield IndexIteratorEntry(
            loop_index, source_path, target_path, source_exprs, index_exprs_acc
        )
        return

    axis = axis or axes.root

    for component in axis.components:
        # for efficiency do these outside the loop
        path_ = path | {axis.label: component.label}
        target_path_ = target_path | target_paths.get((axis.id, component.label), {})
        myindex_exprs = index_exprs.get((axis.id, component.label), pmap())
        subaxis = axes.child(axis, component)

        # bit of a hack, I reckon this can go as we can just get it from component.count
        # inside as_int
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
            replace_map = indices
        else:
            mypath = pmap()
            myindices = pmap()
            replace_map = None

        for pt in range(
            _as_int(
                component.count,
                replace_map,
                loop_indices=outer_replace_map,
            )
        ):
            new_exprs = {}
            evaluator = ExpressionEvaluator(
                indices | {axis.label: pt}, outer_replace_map
            )
            for axlabel, index_expr in myindex_exprs.items():
                new_index = evaluator(index_expr)
                new_exprs[axlabel] = new_index
            index_exprs_ = index_exprs_acc | new_exprs
            indices_ = indices | {axis.label: pt}
            if subaxis:
                yield from iter_axis_tree(
                    loop_index,
                    axes,
                    target_paths,
                    index_exprs,
                    outer_loops,
                    include_loops,
                    subaxis,
                    path_,
                    indices_,
                    target_path_,
                    index_exprs_,
                )
            else:
                if include_loops:
                    assert False, "old code"
                    source_path = path_ | merge_dicts(
                        ol.source_path for ol in outer_loops
                    )
                    source_exprs = indices_ | merge_dicts(
                        ol.source_exprs for ol in outer_loops
                    )
                else:
                    source_path = path_
                    source_exprs = indices_
                yield IndexIteratorEntry(
                    loop_index, source_path, target_path_, source_exprs, index_exprs_
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

    """
    from pyop3.array import HierarchicalArray, Mat
    from pyop3.array.petsc import Sparsity

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
        # skip matrices
        # really nasty hack for now to handle indexed mats
        if isinstance(array, (Mat, Sparsity)) or not hasattr(array, "buffer"):
            continue

        # skip purely local arrays
        if not array.buffer.is_distributed:
            continue

        sf = array.buffer.sf  # the dof sf

        # mark leaves and roots
        is_root_or_leaf = np.full(sf.size, ArrayPointLabel.CORE, dtype=np.uint8)
        is_root_or_leaf[sf.iroot] = ArrayPointLabel.ROOT
        is_root_or_leaf[sf.ileaf] = ArrayPointLabel.LEAF

        is_root_or_leaf_per_array[array.name] = is_root_or_leaf

    labels = np.full(paraxis.size, IterationPointType.CORE, dtype=np.uint8)
    # for p in index.iterset.iter():
    #     # hack because I wrote bad code and mix up loop indices and itersets
    #     p = dataclasses.replace(p, index=index)
    for p in index.iter():
        parindex = p.source_exprs[paraxis.label]
        assert isinstance(parindex, numbers.Integral)

        for array in arrays:
            # same nasty hack
            if isinstance(array, (Mat, Sparsity)) or not hasattr(array, "buffer"):
                continue
            # skip purely local arrays
            if not array.buffer.is_distributed:
                continue
            if labels[parindex] == IterationPointType.LEAF:
                continue

            # loop over stencil
            array = array.with_context({index.id: (p.source_path, p.target_path)})

            for q in array.axes.iter({p}):
                # offset = array.axes.offset(q.target_exprs, q.target_path)
                offset = array.axes.offset(q.source_exprs, q.source_path, loop_exprs=p.replace_map)

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

    # I don't think this is working - instead everything touches a leaf
    # core = just_one(np.nonzero(labels == IterationPointType.CORE))
    # root = just_one(np.nonzero(labels == IterationPointType.ROOT))
    # leaf = just_one(np.nonzero(labels == IterationPointType.LEAF))
    core = np.asarray([], dtype=IntType)
    root = np.asarray([], dtype=IntType)
    leaf = np.arange(paraxis.size, dtype=IntType)

    subsets = []
    for data in [core, root, leaf]:
        # Constant? no, rank_equal=False
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
    mysubset = Slice(
        paraxis.label,
        [Subset(parcpt.label, subsets[0], label=parcpt.label)],
        label=paraxis.label,
    )
    new_iterset = index.iterset[mysubset]

    return index.copy(iterset=new_iterset), subsets

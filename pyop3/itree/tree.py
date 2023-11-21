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
    ContextFree,
    ContextSensitive,
    LoopIterable,
)
from pyop3.axtree.tree import (
    ContextSensitiveAxisTree,
    ContextSensitiveLoopIterable,
    ExpressionEvaluator,
    FrozenAxisTree,
    IndexedAxisTree,
    _as_int,
)
from pyop3.dtypes import IntType, get_mpi_dtype
from pyop3.extras.debug import print_if_rank, print_with_rank
from pyop3.tree import LabelledNode, LabelledTree, postvisit
from pyop3.utils import (
    LabelledImmutableRecord,
    UniquelyIdentifiedImmutableRecord,
    as_tuple,
    checked_zip,
    is_single_valued,
    just_one,
    merge_dicts,
)

bsearch = pym.var("mybsearch")

# just use a pmap for this
# class IndexForest:
#     def __init__(self, trees: Mapping[Mapping, IndexTree]):
#         # per loop context
#         self.trees = trees


# index trees are different to axis trees because we know less about
# the possible attaching components. In particular a CalledMap can
# have different "attaching components"/output components depending on
# the loop context. This is awful for a user to have to build since we
# need something like a SplitCalledMap. Instead we will just admit any
# parent_to_children map and do error checking when we convert it to shape.
class IndexTree(LabelledTree):
    def __init__(self, root, parent_to_children=pmap(), *, loop_context=pmap()):
        root, parent_to_children, loop_context = parse_index_tree(
            root, parent_to_children, loop_context
        )
        super().__init__(root, parent_to_children)
        self.loop_context = loop_context


def parse_index_tree(root, parent_to_children, loop_context):
    root = apply_loop_context(root, loop_context)
    new_parent_to_children = parse_parent_to_children(
        parent_to_children, root, loop_context
    )

    return root, pmap(new_parent_to_children), loop_context


def parse_parent_to_children(parent_to_children, parent, loop_context):
    if parent.id in parent_to_children:
        new_children = []
        subparents_to_children = []
        for child in parent_to_children[parent.id]:
            if child is None:
                continue
            child = apply_loop_context(child, loop_context)
            new_children.append(child)
            subparents_to_children.append(
                parse_parent_to_children(parent_to_children, child, loop_context)
            )

        return pmap({parent.id: tuple(new_children)}) | merge_dicts(
            subparents_to_children
        )
    else:
        return pmap()


IndexLabel = collections.namedtuple("IndexLabel", ["axis", "component"])


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


class SliceComponent(LabelledImmutableRecord, abc.ABC):
    fields = LabelledImmutableRecord.fields | {"component"}

    def __init__(self, component, **kwargs):
        super().__init__(**kwargs)
        self.component = component


class AffineSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"start", "stop", "step"}

    def __init__(self, component, start=None, stop=None, step=None, **kwargs):
        super().__init__(component, **kwargs)
        # use None for the default args here since that agrees with Python slices
        self.start = start if start is not None else 0
        self.stop = stop
        self.step = step if step is not None else 1

    @property
    def datamap(self):
        return pmap()


class Subset(SliceComponent):
    fields = SliceComponent.fields | {"array"}

    def __init__(self, component, array: MultiArray, **kwargs):
        super().__init__(component, **kwargs)
        self.array = array

    @property
    def datamap(self):
        return self.array.datamap


class MapComponent(LabelledImmutableRecord):
    fields = {
        "target_axis",
        "target_component",
        "arity",
    } | LabelledImmutableRecord.fields

    def __init__(self, target_axis, target_component, arity, **kwargs):
        super().__init__(**kwargs)
        self.target_axis = target_axis
        self.target_component = target_component
        self.arity = arity


class MapVariable(pym.primitives.Variable):
    """Pymbolic variable representing the action of a map."""

    mapper_method = sys.intern("map_map_variable")

    def __init__(self, full_map, map_component):
        super().__init__(map_component.array.name)
        self.full_map = full_map
        self.map_component = map_component

    def __call__(self, *args):
        return CalledMapVariable(self, *args)

    @functools.cached_property
    def datamap(self):
        return self.map_component.datamap


class CalledMapVariable(pym.primitives.Call):
    mapper_method = sys.intern("map_called_map")

    @functools.cached_property
    def datamap(self):
        return self.function.datamap | merge_dicts(
            idx.datamap for idx in self.parameters.values()
        )


class TabulatedMapComponent(MapComponent):
    fields = MapComponent.fields - {"arity"} | {"array"}

    def __init__(self, target_axis, target_component, array, **kwargs):
        arity = array.axes.leaf_component.count
        super().__init__(target_axis, target_component, arity, **kwargs)
        self.array = array

        # self.index_expr = MapVariable(self)

    # old alias
    @property
    def data(self):
        return self.array

    @functools.cached_property
    def datamap(self):
        return self.array.datamap


class AffineMapComponent(MapComponent):
    fields = MapComponent.fields | {"expr"}

    def __init__(self, from_labels, to_labels, arity, expr, **kwargs):
        """
        Parameters
        ----------
        expr:
            A 2-tuple of pymbolic variables and an expression. We need to split them
            like this because we need to know the order in which the variables
            correspond to the axis parts.
        """
        if len(expr[0]) != len(from_labels) + 1:
            raise ValueError("Wrong number of variables in expression")

        self.expr = expr
        super().__init__(from_labels, to_labels, arity, **kwargs)


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

    def __call__(self, index) -> Union[CalledMap, ContextSensitiveCalledMap]:
        return CalledMap(self, index)

    @functools.cached_property
    def datamap(self):
        data = {}
        for bit in self.connectivity.values():
            for map_cpt in bit:
                data.update(map_cpt.datamap)
        return pmap(data)


class Index(LabelledNode):
    @abc.abstractmethod
    def target_paths(self, context):
        pass


# ImmutableRecord?
class CalledMap(Index, LoopIterable):
    # This function cannot be part of an index tree because it has not specialised
    # to a particular loop index path.
    def __init__(self, map, from_index, **kwargs):
        self.map = map
        self.from_index = from_index
        Index.__init__(self, **kwargs)

    def __getitem__(self, indices):
        raise NotImplementedError("TODO")

    def index(self) -> LoopIndex:
        contexts = collect_loop_contexts(self)
        # FIXME this assumption is not always true
        context = just_one(contexts)
        axes, target_paths, index_exprs, layout_exprs = collect_shape_index_callback(
            self, loop_indices=context
        )

        axes = IndexedAxisTree(
            axes.root,
            axes.parent_to_children,
            target_paths,
            index_exprs,
            layout_exprs,
        )

        context_sensitive_axes = ContextSensitiveAxisTree({context: axes})
        return LoopIndex(context_sensitive_axes)

    @property
    def name(self):
        return self.map.name

    @property
    def connectivity(self):
        return self.map.connectivity

    def target_paths(self, context):
        targets = []
        for src_path in self.from_index.target_paths(context):
            for map_component in self.connectivity[src_path]:
                targets.append(
                    pmap({map_component.target_axis: map_component.target_component})
                )
        return tuple(targets)


class ContextSensitiveCalledMap(ContextSensitiveLoopIterable):
    pass


# no clue if this should be context free, only really makes sense for iterables
class AbstractLoopIndex(Index, abc.ABC):
    pass


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


class LoopIndex(AbstractLoopIndex):
    fields = AbstractLoopIndex.fields | {"iterset"}

    def __init__(self, iterset, **kwargs):
        super().__init__(**kwargs)
        self.iterset = iterset
        self.local_index = LocalLoopIndex(self)

    @property
    def i(self):
        return self.local_index

    @property
    def j(self):
        # is this evil?
        return self

    @property
    def datamap(self):
        return self.iterset.datamap

    def target_paths(self, context):
        return (context[self.id],)

    def iter(self, stuff=pmap()):
        if not isinstance(self.iterset, (IndexedAxisTree, FrozenAxisTree)):
            raise NotImplementedError
        return iter_axis_tree(self.iterset, stuff)


class LocalLoopIndex(AbstractLoopIndex):
    """Class representing a 'local' index."""

    def __init__(self, loop_index: LoopIndex, **kwargs):
        super().__init__(**kwargs)
        self.loop_index = loop_index

    def target_paths(self, context):
        return (context[self.id],)

    @property
    def datamap(self):
        return self.loop_index.datamap


# TODO I want a Slice to have "bits" like a Map/CalledMap does
class Slice(Index):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    fields = Index.fields | {"axis", "slices"}

    def __init__(self, axis, slices: Collection[SliceComponent], **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.slices = as_tuple(slices)

    def target_paths(self, context):
        return tuple(pmap({self.axis: subslice.component}) for subslice in self.slices)

    @property
    def datamap(self):
        return merge_dicts([s.datamap for s in self.slices])


@functools.singledispatch
def apply_loop_context(arg, loop_context, *, axes, path):
    from pyop3.distarray import MultiArray

    if isinstance(arg, MultiArray):
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
            slice_cpt = Subset(cpt.label, arg, label=array_component.label)
            slice_cpts.append(slice_cpt)
        return Slice(target_axis.label, slice_cpts, label=array_axis.label)
    elif isinstance(arg, numbers.Integral):
        return apply_loop_context(
            slice(arg, arg + 1), loop_context, axes=axes, path=path
        )
    else:
        raise TypeError


@apply_loop_context.register
def _(index: Index, loop_context, **kwargs):
    return index


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


@functools.singledispatch
def collect_loop_indices(arg):
    # cyclic import
    from pyop3.distarray import MultiArray

    if isinstance(arg, (MultiArray, Slice, slice, str)):
        return ()
    elif isinstance(arg, collections.abc.Iterable):
        return sum(map(collect_loop_indices, arg), ())
    else:
        raise NotImplementedError


@collect_loop_indices.register
def _(arg: LoopIndex):
    return (arg,)


@collect_loop_indices.register
def _(arg: LocalLoopIndex):
    return (arg,)


@collect_loop_indices.register
def _(arg: IndexTree):
    return collect_loop_indices(arg.root) + tuple(
        loop_index
        for child in arg.parent_to_children.values()
        for loop_index in collect_loop_indices(child)
    )


@collect_loop_indices.register
def _(arg: CalledMap):
    return collect_loop_indices(arg.from_index)


@collect_loop_indices.register
def _(arg: int):
    return ()


def loop_contexts_from_iterable(indices):
    all_loop_indices = tuple(
        loop_index for index in indices for loop_index in collect_loop_indices(index)
    )

    if len(all_loop_indices) == 0:
        return {}

    contexts = combine_contexts(
        [collect_loop_contexts(idx) for idx in all_loop_indices]
    )

    # add on context-free contexts, these cannot already be included
    for index in indices:
        if not isinstance(index, ContextSensitive):
            continue
        loop_index, paths = index.loop_context
        if loop_index in contexts[0].keys():
            raise AssertionError
        for ctx in contexts:
            ctx[loop_index.id] = paths
    return contexts


@functools.singledispatch
def collect_loop_contexts(arg, *args, **kwargs):
    # cyclic import
    from pyop3.distarray import MultiArray

    if isinstance(arg, (MultiArray, numbers.Integral)):
        return {}
    elif isinstance(arg, collections.abc.Iterable):
        return loop_contexts_from_iterable(arg)
    if arg is Ellipsis:
        return {}
    else:
        raise TypeError


@collect_loop_contexts.register
def _(index_tree: IndexTree):
    contexts = {}
    for loop_index, paths in index_tree.loop_context.items():
        contexts[loop_index] = [paths]
    return contexts


@collect_loop_contexts.register
def _(arg: LocalLoopIndex):
    return collect_loop_contexts(arg.loop_index, local=True)


@collect_loop_contexts.register
def _(arg: LoopIndex, local=False):
    if isinstance(arg.iterset, ContextSensitiveAxisTree):
        contexts = []
        for loop_context, axis_tree in arg.iterset.context_map.items():
            extra_source_context = {}
            extracontext = {}
            for leaf in axis_tree.leaves:
                source_path = axis_tree.path(*leaf)
                target_path = {}
                for axis, cpt in axis_tree.path_with_nodes(
                    *leaf, and_components=True
                ).items():
                    target_path.update(axis_tree.target_paths[axis.id, cpt.label])
                extra_source_context.update(source_path)
                extracontext.update(target_path)
            if local:
                contexts.append(
                    loop_context | {arg.local_index.id: pmap(extra_source_context)}
                )
            else:
                contexts.append(loop_context | {arg.id: pmap(extracontext)})
        return tuple(contexts)
    else:
        assert isinstance(arg.iterset, (FrozenAxisTree, IndexedAxisTree))
        iterset = arg.iterset
        contexts = []
        for leaf in iterset.leaves:
            source_path = iterset.path(*leaf)
            target_path = {}
            for axis, cpt in iterset.path_with_nodes(
                *leaf, and_components=True
            ).items():
                target_path.update(
                    iterset.target_paths[axis.id, cpt.label]
                    # iterset.paths[axis.id, cpt.label]
                )
            if local:
                contexts.append(pmap({arg.local_index.id: source_path}))
            else:
                contexts.append(pmap({arg.id: pmap(target_path)}))
        return tuple(contexts)


def _paths_from_called_map_loop_index(index, context):
    # terminal
    if isinstance(index, LoopIndex):
        return (context[index][1],)

    assert isinstance(index, CalledMap)
    paths = []
    for from_path in _paths_from_called_map_loop_index(index.from_index, context):
        for map_component in index.connectivity[from_path]:
            paths.append(
                (
                    pmap({index.label: map_component.label}),
                    pmap({map_component.target_axis: map_component.target_component}),
                )
            )
    return tuple(paths)


@collect_loop_contexts.register
def _(called_map: CalledMap):
    return collect_loop_contexts(called_map.from_index)


@collect_loop_contexts.register
def _(slice_: slice):
    return ()


@collect_loop_contexts.register
def _(slice_: Slice):
    return ()


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

    if subindices:
        children = []
        subtrees = []
        # used to be leaves...
        for target_path in index.target_paths(loop_context):
            assert target_path
            new_path = path | target_path
            child, subtree = index_tree_from_iterable(
                subindices, loop_context, axes, new_path
            )
            children.append(child)
            subtrees.append(subtree)

        root = index
        parent_to_children = pmap({index.id: children}) | merge_dicts(subtrees)
    else:
        root = index
        parent_to_children = pmap()

    if first_call:
        return IndexTree(root, parent_to_children, loop_context=loop_context)
    else:
        return root, parent_to_children


@functools.singledispatch
def as_index_tree(arg, loop_context, **kwargs):
    if isinstance(arg, collections.abc.Iterable):
        return index_tree_from_iterable(arg, loop_context, first_call=True, **kwargs)
    else:
        raise TypeError


@as_index_tree.register
def _(index: Index, ctx, **kwargs):
    return IndexTree(index, loop_context=ctx)


@functools.singledispatch
def as_index_forest(arg: Any, **kwargs):
    from pyop3.distarray import MultiArray

    if isinstance(arg, MultiArray):
        slice_ = apply_loop_context(arg, loop_context=pmap(), path=pmap(), **kwargs)
        return (IndexTree(slice_),)
    elif isinstance(arg, collections.abc.Iterable):
        loop_contexts = collect_loop_contexts(arg) or [pmap()]
        forest = []
        for context in loop_contexts:
            forest.append(as_index_tree(arg, context, **kwargs))
        return tuple(forest)
    else:
        raise TypeError


@as_index_forest.register
def _(index_tree: IndexTree, **kwargs):
    return (index_tree,)


@as_index_forest.register
def _(index: Index, **kwargs):
    loop_contexts = collect_loop_contexts(index) or [pmap()]
    forest = []
    for context in loop_contexts:
        forest.append(as_index_tree(index, context, **kwargs))
    return tuple(forest)


@as_index_forest.register
def _(slice_: slice, **kwargs):
    slice_ = apply_loop_context(slice_, loop_context=pmap(), path=pmap(), **kwargs)
    return (IndexTree(slice_),)


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


@collect_shape_index_callback.register
def _(loop_index: LoopIndex, *, loop_indices, **kwargs):
    # breakpoint()
    # path = loop_indices[loop_index]

    iterset = loop_index.iterset

    target_path_per_component = pmap({None: loop_indices[loop_index.id]})
    # fairly sure that here I want the *output* path of the loop indices
    index_exprs_per_component = pmap(
        {
            None: pmap(
                {
                    axis: LoopIndexVariable(loop_index, axis)
                    for axis in loop_indices[loop_index.id].keys()
                }
                # {axis: LoopIndexVariable(loop_index, axis) for axis in path.keys()}
            )
        }
    )
    layout_exprs_per_component = pmap({None: 0})
    return (
        AxisTree(),
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
    )


@collect_shape_index_callback.register
def _(local_index: LocalLoopIndex, *args, loop_indices, **kwargs):
    path = loop_indices[local_index.id]

    loop_index = local_index.loop_index
    iterset = loop_index.iterset

    target_path_per_cpt = pmap({None: path})
    index_exprs_per_cpt = pmap(
        {
            None: pmap(
                {axis: LoopIndexVariable(local_index, axis) for axis in path.keys()}
            )
        }
    )

    layout_exprs_per_cpt = pmap({None: 0})
    return (
        AxisTree(),
        target_path_per_cpt,
        index_exprs_per_cpt,
        layout_exprs_per_cpt,
    )


@collect_shape_index_callback.register
def _(slice_: Slice, *, prev_axes, **kwargs):
    components = []
    target_path_per_subslice = []
    index_exprs_per_subslice = []
    layout_exprs_per_subslice = []

    axis_label = slice_.label

    for subslice in slice_.slices:
        # we are assuming that axes with the same label *must* be identical. They are
        # only allowed to differ in that they have different IDs.
        cpt_label = subslice.label
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
        cpt = AxisComponent(size, label=cpt_label)
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
            index_exprs_per_subslice.append(
                pmap({slice_.axis: subslice.array.as_var()})
            )
            layout_exprs_per_subslice.append(
                pmap({slice_.label: bsearch(subslice.array.as_var(), layout_var)})
            )

    axes = FrozenAxisTree(Axis(components, label=axis_label))
    target_path_per_component = {}
    index_exprs_per_component = {}
    layout_exprs_per_component = {}
    for cpt, target_path, index_exprs, layout_exprs in checked_zip(
        components,
        target_path_per_subslice,
        index_exprs_per_subslice,
        layout_exprs_per_subslice,
    ):
        target_path_per_component[axes.root.id, cpt.label] = target_path
        index_exprs_per_component[axes.root.id, cpt.label] = index_exprs
        layout_exprs_per_component[axes.root.id, cpt.label] = layout_exprs
    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
    )


@collect_shape_index_callback.register
def _(called_map: CalledMap, **kwargs):
    (
        prior_axes,
        prior_target_path_per_cpt,
        prior_index_exprs_per_cpt,
        _,
    ) = collect_shape_index_callback(called_map.from_index, **kwargs)

    if prior_axes.is_empty:
        prior_target_path = prior_target_path_per_cpt[None]
        prior_index_exprs = prior_index_exprs_per_cpt[None]
        (
            axis,
            target_path_per_cpt,
            index_exprs_per_cpt,
            layout_exprs_per_cpt,
        ) = _make_leaf_axis_from_called_map(
            called_map, prior_target_path, prior_index_exprs
        )
        axes = AxisTree(axis)
    else:
        axes = prior_axes
        target_path_per_cpt = {}
        index_exprs_per_cpt = {}
        layout_exprs_per_cpt = {}
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
            ) = _make_leaf_axis_from_called_map(
                called_map, prior_target_path, prior_index_exprs
            )
            axes = axes.add_subaxis(subaxis, prior_leaf_axis, prior_leaf_cpt)
            target_path_per_cpt.update(subtarget_paths)
            index_exprs_per_cpt.update(subindex_exprs)
            layout_exprs_per_cpt.update(sublayout_exprs)

    return (
        axes,
        pmap(target_path_per_cpt),
        pmap(index_exprs_per_cpt),
        pmap(layout_exprs_per_cpt),
    )


def _make_leaf_axis_from_called_map(called_map, prior_target_path, prior_index_exprs):
    axis_id = Axis.unique_id()
    components = []
    target_path_per_cpt = {}
    index_exprs_per_cpt = {}
    layout_exprs_per_cpt = {}

    for map_cpt in called_map.map.connectivity[prior_target_path]:
        cpt = AxisComponent(map_cpt.arity, label=map_cpt.label)
        components.append(cpt)

        target_path_per_cpt[axis_id, cpt.label] = pmap(
            {map_cpt.target_axis: map_cpt.target_component}
        )

        map_var = MapVariable(called_map, map_cpt)
        axisvar = AxisVariable(called_map.name)

        index_exprs_per_cpt[axis_id, cpt.label] = {
            map_cpt.target_axis: map_var(prior_index_exprs | {called_map.name: axisvar})
        }

        # don't think that this is possible for maps
        layout_exprs_per_cpt[axis_id, cpt.label] = {
            called_map.name: pym.primitives.NaN(IntType)
        }

    axis = Axis(components, label=called_map.name, id=axis_id)

    return axis, target_path_per_cpt, index_exprs_per_cpt, layout_exprs_per_cpt


def _index_axes(axes, indices: IndexTree, loop_context):
    (
        indexed_axes,
        tpaths,
        index_expr_per_target,
        layout_expr_per_target,
    ) = _index_axes_rec(
        indices,
        current_index=indices.root,
        loop_indices=loop_context,
        prev_axes=axes,
    )

    if indexed_axes is None:
        indexed_axes = AxisTree()

    # return the new axes plus the new index expressions per leaf
    return indexed_axes.freeze(), tpaths, index_expr_per_target, layout_expr_per_target


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
    ) = tuple(map(dict, rest))

    if not axes_per_index.is_empty:
        leafkeys = [(ax.id, cpt) for ax, cpt in axes_per_index.leaves]
    else:
        leafkeys = [None]

    subaxes = {}
    for leafkey in leafkeys:
        if current_index.id in indices.parent_to_children:
            for subindex in indices.parent_to_children[current_index.id]:
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

    target_path_per_component = pmap(target_path_per_cpt_per_index)
    index_exprs_per_component = pmap(index_exprs_per_cpt_per_index)
    layout_exprs_per_component = pmap(layout_exprs_per_cpt_per_index)

    axes = axes_per_index
    for k, subax in subaxes.items():
        if subax is not None:
            if axes.root:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = subax

    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
    )


def index_axes(axes, index_tree):
    (
        indexed_axes,
        target_path_per_indexed_cpt,
        index_exprs_per_indexed_cpt,
        layout_exprs_per_indexed_cpt,
    ) = _index_axes(axes, index_tree, loop_context=index_tree.loop_context)

    target_paths, index_exprs, layout_exprs = _compose_bits(
        axes,
        indexed_axes,
        target_path_per_indexed_cpt,
        index_exprs_per_indexed_cpt,
        layout_exprs_per_indexed_cpt,
    )
    return IndexedAxisTree(
        indexed_axes.root,
        indexed_axes.parent_to_children,
        target_paths,
        index_exprs,
        layout_exprs,
    )


def _compose_bits(
    axes,
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
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if indexed_axes.is_empty:
        return (
            # pmap(),
            # pmap(),
            # pmap(),
            freeze({None: itarget_paths.get(None, pmap())}),
            freeze({None: iindex_exprs.get(None, pmap())}),
            freeze({None: ilayout_exprs.get(None, pmap())}),
        )

    if iaxis is None:
        target_path |= itarget_paths.get(None, {})
        partial_index_exprs |= iindex_exprs.get(None, {})
        # partial_layout_exprs |= ilayout_exprs.get(None, {})
        visited_target_axes = visited_target_axes.union(target_path.keys())
        iaxis = indexed_axes.root

    target_path_per_cpt = collections.defaultdict(dict)
    index_exprs = collections.defaultdict(dict)
    layout_exprs = collections.defaultdict(dict)

    for icpt in iaxis.components:
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
        # index_exprs[iaxis.id, icpt.label] = {}
        # layout_exprs[iaxis.id, icpt.label] = {}
        new_visited_target_axes = visited_target_axes
        if axes.is_valid_path(new_target_path):
            detailed_path = axes.detailed_path(new_target_path)

            for target_axis, target_cpt in detailed_path.items():
                skip = target_axis.label in new_visited_target_axes
                new_visited_target_axes |= {target_axis.label}

                new_target_path_acc = new_target_path_acc | axes.target_paths.get(
                    (target_axis.id, target_cpt.label), {}
                )

                if not skip:
                    for myaxlabel, mycptlabel in axes.target_paths.get(
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
                orig_index_exprs = axes.index_exprs[target_axis.id, target_cpt.label]
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
            # new_partial_index_exprs = pmap()

            # now do the layout expressions, this is simpler since target path magic isnt needed
            # compose layout expressions, this does an *outside* substitution
            # so the final replace map is src -> h(final)
            # we start with src -> f(intermediate)
            # and intermediate -> g(final)

            full_replace_map = merge_dicts(
                [
                    axes.layout_exprs[tgt_ax.id, tgt_cpt.label]
                    for tgt_ax, tgt_cpt in detailed_path.items()
                ]
            )
            for ikey, layout_expr in new_partial_layout_exprs.items():
                # always 1:1 for layouts
                mykey, myvalue = just_one(layout_expr.items())
                mytargetpath = just_one(itarget_paths[ikey].keys())
                layout_expr_replace_map = {mytargetpath: full_replace_map[mytargetpath]}
                new_layout_expr = IndexExpressionReplacer(layout_expr_replace_map)(
                    myvalue
                )
                layout_exprs[ikey][mykey] = new_layout_expr

        if isubaxis := indexed_axes.child(iaxis, icpt):
            (
                subtarget_path,
                subindex_exprs,
                sublayout_exprs,
            ) = _compose_bits(
                axes,
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
            # assert not skip  huh?
            # assert not new_partial_index_exprs
            # assert not new_partial_layout_exprs

    # breakpoint()
    return (
        freeze(dict(target_path_per_cpt)),
        freeze(dict(index_exprs)),
        freeze(dict(layout_exprs)),
    )


def iter_axis_tree(
    axes: AxisTree,
    outermap,
    axis=None,
    path=pmap(),
    indices=pmap(),
    target_path=pmap(),
    index_exprs=pmap(),
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if axes.is_empty:
        yield pmap(), pmap(), pmap(), pmap()
        return

    axis = axis or axes.root

    for component in axis.components:
        path_ = path | {axis.label: component.label}
        target_path_ = target_path | axes.target_paths.get(
            (axis.id, component.label), {}
        )
        myindex_exprs = axes.index_exprs[axis.id, component.label]
        subaxis = axes.child(axis, component)
        for pt in range(_as_int(component.count, path, indices)):
            new_exprs = {}
            for axlabel, index_expr in myindex_exprs.items():
                # need to replace *not* evaluate because the axis tree could be indexed
                # with a loop index whose value is not yet known (or pass in context info)
                # new_index = IndexExpressionReplacer({axis.label: pt})(index_expr)
                new_index = ExpressionEvaluator(outermap | indices | {axis.label: pt})(
                    index_expr
                )
                assert new_index != index_expr
                new_exprs[axlabel] = new_index
            index_exprs_ = index_exprs | new_exprs
            # index_exprs_ = index_exprs | myindex_exprs
            indices_ = indices | {axis.label: pt}
            if subaxis:
                yield from iter_axis_tree(
                    axes, outermap, subaxis, path_, indices_, target_path_, index_exprs_
                )
            else:
                # yield path_, index_exprs_, indices_
                yield path_, target_path_, indices_, index_exprs_
                # yield path_, indices_


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
    from pyop3.distarray.multiarray import Dat, IndexExpressionReplacer

    # take first
    # paraxis = [axis for axis in index.iterset.nodes if axis.sf is not None][0]
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
        if not array.orig_array.array.is_distributed:
            continue

        # take first
        # array_paraxes = [
        #     axis for axis in array.orig_array.axes.nodes if axis.sf is not None
        # ]
        #
        # array_paraxis = array_paraxes[0]
        # sf = array_paraxis.sf
        sf = array.orig_array.axes.sf  # the dof sf

        # mark leaves and roots
        is_root_or_leaf = np.full(sf.size, ArrayPointLabel.CORE, dtype=np.uint8)
        is_root_or_leaf[sf.iroot] = ArrayPointLabel.ROOT
        is_root_or_leaf[sf.ileaf] = ArrayPointLabel.LEAF

        # do this because we need to think of the indices here as a selector
        # rather than a map. We need to transform to the new numbering, hence we
        # need to apply the map default -> reordered, but the indexing semantics
        # are the opposite of this
        # is_root_or_leaf = is_root_or_leaf[array_paraxis.numbering]
        # this is equivalent to:
        # new_labels = np.empty_like(labels)
        # for i, l in enumerate(labels):
        #     j = array_paraxis._inverse_numbering[i]
        #     new_labels[j] = l
        # labels = new_labels

        is_root_or_leaf_per_array[array.name] = is_root_or_leaf

    labels = np.full(paraxis.size, IterationPointType.CORE, dtype=np.uint8)
    for path, target_path, indices, target_indices in index.iter():
        parindex = indices[paraxis.label]
        assert isinstance(parindex, numbers.Integral)

        replace_map = freeze(
            {(index.id, axis): i for axis, i in target_indices.items()}
        )

        for array in arrays:
            # skip purely local arrays
            if not array.orig_array.array.is_distributed:
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
            ) in array.axes.index().iter(replace_map):
                # allexprs = dict(array.axes.index_exprs.get(None, {}))
                # if not array.axes.is_empty:
                #     for myaxis, mycpt in array.axes.path_with_nodes(
                #         *array.axes._node_from_path(array_path)
                #     ).items():
                #         allexprs.update(array.axes.index_exprs[myaxis.id, mycpt])
                #
                offset = array.axes.offset(array_path, array_indices | replace_map)

                # allexprs is indexed with the "source" labels but we want a particular
                # "target" label, need to go backwards... or something
                # if len(target_path) != 1:
                #     raise NotImplementedError
                # target_parallel_axis_label = just_one(target_path.keys())
                # the_expr_i_want = allexprs[target_parallel_axis_label]
                #
                # # but this is for a particular component!! need to map component index to
                # # "full" one, how? or just do offset?
                # pt_index = pym.evaluate(
                #     the_expr_i_want,
                #     replace_map | array_indices,
                #     ExpressionEvaluator,
                # )
                # print_if_rank(1, "ptindex", pt_index)
                # assert isinstance(pt_index, numbers.Integral)

                # point_label = is_root_or_leaf_per_array[array.name][pt_index]
                point_label = is_root_or_leaf_per_array[array.name][offset]
                print_if_rank(1, "ptlabel", point_label)
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

    print_with_rank("arrayper", is_root_or_leaf_per_array)
    print_with_rank("labels", labels)

    core = just_one(np.nonzero(labels == IterationPointType.CORE))
    root = just_one(np.nonzero(labels == IterationPointType.ROOT))
    leaf = just_one(np.nonzero(labels == IterationPointType.LEAF))

    subsets = []
    for data in [core, root, leaf]:
        # Constant?
        size = Dat(AxisTree(), data=np.asarray([len(data)]), dtype=IntType)
        subset = Dat(
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
            [Subset(parcpt.label, subsets[0], label=parcpt.label)],
            label=paraxis.label,
        )
    ]

    return index.copy(iterset=new_iterset), subsets
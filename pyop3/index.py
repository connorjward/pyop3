from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import numbers
import sys
from typing import Any, Collection, Hashable, Mapping, Sequence

import pymbolic as pym
import pytools
from pyrsistent import pmap

from pyop3.axis import (
    Axis,
    AxisComponent,
    AxisTree,
    AxisVariable,
    ContextFree,
    ContextSensitive,
    LoopIterable,
)
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

        return pmap(
            {parent.id: tuple(new_children)} | merge_dicts(subparents_to_children)
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


class IndexedArray:
    """Container representing an object that has been indexed.

    For example ``dat[index]`` would produce ``Indexed(dat, index)``.

    """

    # note that axes here are specially modified to have the right layout functions
    # this needs to be done inside __getitem__
    def __init__(self, array: MultiArray, axis_trees, layout_exprs):
        assert False, "old code, used IndexedAxisTree instead"
        # from pyop3.codegen.loopexpr2loopy import _indexed_axes
        #
        # # The following tricksy bit of code builds a pretend AxisTree for the
        # # indexed object. It is complicated because the resultant AxisTree will
        # # have a different shape depending on the loop context (which is why we have
        # # SplitIndexTrees). We therefore store axes here split by loop context.
        # split_indices = {}
        # split_axes = {}
        # for loop_ctx, axes in obj.split_axes.items():
        #     indices = as_split_index_tree(indices, axes=axes, loop_context=loop_ctx)
        #     split_indices |= indices.index_trees
        #     for loop_ctx_, itree in indices.index_trees.items():
        #         # nasty hack because _indexed_axes currently expects a 3-tuple per loop index
        #         assert set(loop_ctx.keys()) <= set(loop_ctx_.keys())
        #         my_loop_context = {
        #             idx: (path, "not used", "not used")
        #             for idx, path in loop_ctx_.items()
        #         }
        #         split_axes[loop_ctx_] = _indexed_axes((axes, indices), my_loop_context)
        #
        # self.obj = obj
        # self.split_axes = pmap(split_axes)
        # self.indices = SplitIndexTree(split_indices)
        self.array = array
        self.axis_trees = axis_trees
        self.layout_exprs = layout_exprs

    def __getitem__(self, indices):
        return IndexedArray(self, axis_trees, layout_expr_per_axis_tree_per_leaf)

    @functools.cached_property
    def datamap(self):
        dmap = {}
        for expr_per_leaf in self.layout_exprs.values():
            for expr in expr_per_leaf.values():
                dmap |= collect_datamap_from_expression(expr)

        return (
            dmap
            | self.array.datamap
            | merge_dicts([axes.datamap for axes in self.axis_trees.values()])
        )

    @property
    def name(self):
        return self.array.name

    @property
    def dtype(self):
        return self.array.dtype


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
        return {}


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
        return CalledMapVariable(self, args)


class CalledMapVariable(pym.primitives.Call):
    mapper_method = sys.intern("map_called_map")


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

    # FIXME, naturally this is a placeholder
    fields = {"bits", "name"}

    def __init__(self, bits, name, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bits = bits
        self.name = name

    def __call__(self, index):
        return CalledMap(self, index)

    @functools.cached_property
    def datamap(self):
        data = {}
        for bit in self.bits.values():
            for map_cpt in bit:
                data |= map_cpt.datamap
        return data


class Index(LabelledNode):
    @abc.abstractmethod
    def target_paths(self, context):
        pass


# ImmutableRecord?
class CalledMap(Index, LoopIterable, UniquelyIdentifiedImmutableRecord):
    # This function cannot be part of an index tree because it has not specialised
    # to a particular loop index path.
    def __init__(self, map, from_index):
        self.map = map
        self.from_index = from_index
        UniquelyIdentifiedImmutableRecord.__init__(self)

    @property
    def name(self):
        return self.map.name

    @property
    def axes(self):
        raise NotImplementedError

    @property
    def index(self):
        raise NotImplementedError

    @property
    def enumerate(self):
        raise NotImplementedError

    # FIXME should be context-sensitive!
    @property
    def target_path_per_component(self):
        raise NotImplementedError("TODO")

    @property
    def index_exprs_per_component(self):
        raise NotImplementedError("TODO")

    @property
    def layout_exprs_per_component(self):
        raise NotImplementedError("TODO")

    @property
    def bits(self):
        return self.map.bits

    def target_paths(self, context):
        targets = []
        for src_path in self.from_index.target_paths(context):
            for map_component in self.bits[src_path]:
                targets.append(
                    pmap({map_component.target_axis: map_component.target_component})
                )
        return tuple(targets)


class AbstractLoopIndex(Index, abc.ABC):
    pass


# TODO just call this LoopIndex (inherit from AbstractLoopIndex)
class LoopIndex(AbstractLoopIndex):
    fields = AbstractLoopIndex.fields | {"iterset"}

    def __init__(self, iterset, **kwargs):
        # FIXME I think that an IndexTree should not know its component labels
        # we can do that in the dict. This is because it is context dependent.
        # for now just use one label (assume single component)
        super().__init__(**kwargs)
        self.iterset = iterset

    @property
    def datamap(self):
        return self.iterset.datamap

    def target_paths(self, context):
        iterset = self.iterset.with_context(context)
        paths = []
        for leaf in iterset.leaves:
            path = iterset.path(*leaf)
            paths.append(iterset.target_path_per_leaf[path])
        return tuple(paths)


class LocalLoopIndex(AbstractLoopIndex):
    """Class representing a 'local' index."""

    def __init__(self, loop_index: LoopIndex, **kwargs):
        super().__init__(**kwargs)
        self.loop_index = loop_index

    @property
    def target_paths(self):
        assert False, "dead code"
        return self.global_index.target_paths

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


# class ContextFreeCalledMap(Index, ContextFreeIndex):
#     def __init__(self, component_labels, map: CalledMap, from_index: ContextFreeIndex):
#         super().__init__(component_labels)
#         self.map = map
#         self.from_index = from_index
#
#     @property
#     def target_paths(self):
#         # hopefully I shouldn't have to do this introspection here, make a class attribute
#         all_map_components = {}
#         for map_components in self.map.bits.values():
#             for map_component in map_components:
#                 all_map_components[map_component.label] = map_component
#
#         targets = []
#         for component_label in self.component_labels:
#             map_component_label = component_label[-1]
#             selected_cpt = all_map_components[map_component_label]
#             target = pmap({selected_cpt.target_axis: selected_cpt.target_component})
#             targets.append(target)
#         return targets


class EnumeratedLoopIndex:
    def __init__(self, value: LoopIndex):
        self.index = LocalLoopIndex(value)
        self.value = value


# move with other axis trees
# A better name for this is "ContextSensitiveAxisTree"
class IndexedAxisTree(ContextSensitive):
    def __init__(self, axis_trees):
        ContextSensitive.__init__(self, axis_trees)
        self.axis_trees = pmap(axis_trees)

    def __getitem__(self, indices):
        new_axis_trees = {}
        for loop_context, axis_tree in self.axis_trees.items():
            context_sensitive_axes = axis_tree[indices]
            for (
                new_loop_context,
                ctx_free_axes,
            ) in context_sensitive_axes.values.items():
                new_axis_trees[loop_context | new_loop_context] = ctx_free_axes
        return IndexedAxisTree(new_axis_trees)

    def index(self):
        return LoopIndex(self)

    def enumerate(self):
        return EnumeratedLoopIndex(self.index())

    @functools.cached_property
    def datamap(self):
        return merge_dicts(axis_tree.datamap for axis_tree in self.axis_trees.values())


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


"""
FIXME This function is currently not really fit for purpose. It cannot cope
with multiple loop indices as they are not merged into the same context.

If we pass index trees here then we could get incompatible loop contexts
Probably should collect unrestricted loop indices as well as complete contexts?
"""


def loop_contexts_from_iterable(indices):
    # this is a bit tricky/unpleasant to write
    # FIXME this is currently very naive/limited, need to check for conflicts etc
    loop_contexts = []
    for index in indices:
        # TODO Should check the type of index here as it is more restrictive than at a top
        # level. One cannot have another iterable.
        ctx = collect_loop_context(index)
        if len(ctx) > 1:
            raise NotImplementedError("TODO")
        loop_contexts.extend(ctx)
    return tuple(loop_contexts)


@functools.singledispatch
def collect_loop_context(arg, *args, **kwargs):
    # cyclic import
    from pyop3.distarray import MultiArray

    if isinstance(arg, (MultiArray, numbers.Integral)):
        return ()
    elif isinstance(arg, collections.abc.Iterable):
        return loop_contexts_from_iterable(arg)
    if arg is Ellipsis:
        return ()
    else:
        raise TypeError


@collect_loop_context.register
def _(index_tree: IndexTree):
    return index_tree.loop_context


@collect_loop_context.register
def _(arg: LocalLoopIndex):
    return collect_loop_context(arg.loop_index)


@collect_loop_context.register
def _(arg: LoopIndex):
    from pyop3.axis import AxisTree

    if isinstance(arg.iterset, IndexedAxisTree):
        loop_contexts = []
        for loop_context, axis_tree in arg.iterset.axis_trees.items():
            extra_source_context = {}
            extracontext = {}
            for source_path, target_path in checked_zip(
                axis_tree.source_paths.values(), axis_tree.target_paths.values()
            ):
                extra_source_context |= source_path
                extracontext |= target_path
            loop_contexts.append(
                loop_context | {arg: (pmap(extra_source_context), pmap(extracontext))}
            )
        return loop_contexts
    else:
        assert isinstance(arg.iterset, AxisTree)

        iterset = arg.iterset
        contexts = []
        for leaf in iterset.leaves:
            source_path = iterset.path(*leaf)
            target_path = {}
            for axis, cpt in iterset.path_with_nodes(
                *leaf, and_components=True
            ).items():
                target_path |= iterset.target_path_per_component[axis.id, cpt.label]
            contexts.append((source_path, pmap(target_path)))
        return [
            pmap({arg: (source_path, target_path)})
            for source_path, target_path in contexts
        ]


@collect_loop_context.register
def _(called_map: CalledMap):
    return collect_loop_context(called_map.from_index)


@collect_loop_context.register
def _(slice_: slice):
    return ()


@collect_loop_context.register
def _(slice_: Slice):
    return ()


def _split_index_tree_from_ellipsis(
    axes: AxisTree,
    current_axis: Axis | None = None,
    loop_context=pmap(),
) -> IndexTree:
    assert False, "old code"
    current_axis = current_axis or axes.root

    subslices = []
    subtrees = []
    for cpt in current_axis.components:
        subslices.append(AffineSliceComponent(cpt.label))

        subaxis = axes.child(current_axis, cpt)
        if subaxis:
            subtrees.append(_index_tree_from_ellipsis(axes, subaxis, loop_context))
        else:
            subtrees.append(None)

    slice_ = Slice(current_axis.label, subslices)
    tree = IndexTree(slice_)
    for subslice, subtree in checked_zip(subslices, subtrees):
        if subtree is not None:
            tree = tree.add_subtree(subtree, slice_, subslice.component)
    return pmap({pmap(): tree})


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


def index_tree_from_ellipsis(axes, current_axis, first_call=True):
    assert False, "not needed"
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
        return IndexTree(myslice, {myslice.id: subroots} | merge_dicts(subtrees))
    else:
        return myslice, {myslice.id: subroots} | merge_dicts(subtrees)


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
        parent_to_children = pmap({index.id: children} | merge_dicts(subtrees))
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
def _(index: Index, ctx):
    return IndexTree(index, loop_context=ctx)


@functools.singledispatch
def as_index_forest(arg: Any, *, axes, **kwargs):
    from pyop3.distarray import MultiArray

    if isinstance(arg, MultiArray):
        slice_ = apply_loop_context(
            arg, loop_context=pmap(), path=pmap(), axes=axes, **kwargs
        )
        return (IndexTree(slice_),)
    elif isinstance(arg, collections.abc.Iterable):
        loop_contexts = collect_loop_context(arg) or [pmap()]
        return tuple(
            as_index_tree(arg, ctx, axes=axes, **kwargs) for ctx in loop_contexts
        )
    elif arg is Ellipsis:
        assert False, "dont go this way"
        return (index_tree_from_ellipsis(axes, axes.root),)
    else:
        raise TypeError


@as_index_forest.register
def _(index_tree: IndexTree, **kwargs):
    return (index_tree,)


@as_index_forest.register
def _(index: Index, **kwargs):
    loop_contexts = collect_loop_context(index) or [pmap()]
    return tuple(as_index_tree(index, ctx) for ctx in loop_contexts)


@as_index_forest.register
def _(slice_: slice, **kwargs):
    slice_ = apply_loop_context(slice_, loop_context=pmap(), path=pmap(), **kwargs)
    return (IndexTree(slice_),)


@functools.singledispatch
def collect_shape_index_callback(index, *args, **kwargs):
    raise TypeError(f"No handler provided for {type(index)}")


@collect_shape_index_callback.register
def _(loop_index: LoopIndex, *, loop_indices, **kwargs):
    _, path = loop_indices[loop_index]

    if isinstance(loop_index.iterset, IndexedAxisTree):
        iterset = just_one(loop_index.iterset.values.values())
    else:
        iterset = loop_index.iterset

    myleaf = iterset.orig_axes._node_from_path(path)
    visited_nodes = iterset.orig_axes.path_with_nodes(*myleaf, ordered=True)

    target_path_per_component = pmap(
        {None: pmap({node.label: cpt_label for node, cpt_label in visited_nodes})}
    )
    index_exprs_per_component = pmap(
        {
            None: pmap(
                {node.label: AxisVariable(node.label) for node, _ in visited_nodes}
            )
        }
    )
    layout_exprs_per_component = pmap()  # not implemented
    return (
        AxisTree(),
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
    )


@collect_shape_index_callback.register
def _(local_index: LocalLoopIndex, *args, loop_indices, **kwargs):
    loop_index = local_index.loop_index
    path, _ = loop_indices[loop_index]

    if isinstance(loop_index.iterset, IndexedAxisTree):
        iterset = just_one(loop_index.iterset.values.values())
    else:
        iterset = loop_index.iterset

    myleaf = iterset._node_from_path(path)
    visited_nodes = iterset.path_with_nodes(*myleaf, ordered=True)

    source_path = pmap()
    target_path = pmap({node.label: cpt_label for node, cpt_label in visited_nodes})

    # make LoopIndex property?
    index_expr_per_target_axis = {
        node.label: AxisVariable(node.label) for node, _ in visited_nodes
    }

    layout_exprs = {}  # not allowed I believe, or zero?
    return {
        pmap(): (source_path, target_path, index_expr_per_target_axis, layout_exprs)
    }, (AxisTree(),)


@collect_shape_index_callback.register
def _(slice_: Slice, *, prev_axes, **kwargs):
    components = []
    target_path_per_component = {}
    index_exprs_per_component = {}
    layout_exprs_per_component = {}

    for subslice in slice_.slices:
        # we are assuming that axes with the same label *must* be identical. They are
        # only allowed to differ in that they have different IDs.
        target_axis, target_cpt = prev_axes.find_component(
            slice_.axis, subslice.component, also_node=True
        )
        if isinstance(subslice, AffineSliceComponent):
            # FIXME should be ceiling
            if subslice.stop is None:
                stop = target_cpt.count
            else:
                stop = subslice.stop
            size = (stop - subslice.start) // subslice.step
        else:
            assert isinstance(subslice, Subset)
            size = subslice.array.axes.leaf_component.count
        cpt = AxisComponent(size, label=subslice.label)
        components.append(cpt)

        target_path_per_component[cpt] = pmap({target_axis.label: target_cpt.label})

        newvar = AxisVariable(slice_.label)
        if isinstance(subslice, AffineSliceComponent):
            index_exprs_per_component[cpt] = pmap(
                {slice_.axis: newvar * subslice.step + subslice.start}
            )
            layout_exprs_per_component[cpt] = pmap(
                {slice_.axis: (newvar - subslice.start) // subslice.step}
            )
        else:
            index_exprs_per_component[cpt] = pmap({slice_.axis: subslice.array})
            layout_exprs_per_component[cpt] = pmap({slice_.axis: "inverse search"})

    axes = AxisTree(Axis(components, label=slice_.label))
    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
    )


@collect_shape_index_callback.register
def _(called_map: CalledMap, preorder_ctx, **kwargs):
    leaves, index_data = collect_shape_index_callback(
        called_map.from_index, preorder_ctx, **kwargs
    )
    (axes,) = index_data

    leaf_keys = []
    target_path_per_leaf = []
    index_exprs_per_leaf = []
    layout_exprs_per_leaf = []

    for from_leaf_key, leaf in leaves.items():
        _, from_target_path, from_index_exprs, _ = leaf

        # clean this up, we know some of this at an earlier point (loop context)
        components = []
        index_exprs = []
        layout_exprs = []

        bits = called_map.map.bits[pmap(from_target_path)]
        for map_component in bits:  # each one of these is a new "leaf"
            cpt = AxisComponent(map_component.arity, label=map_component.label)
            components.append(cpt)

            map_var = MapVariable(called_map, map_component)
            axisvar = AxisVariable(called_map.name)

            # not super happy about this. The called variable doesn't now
            # necessarily know the right axis labels
            from_indices = tuple(
                index_expr for axis_label, index_expr in from_index_exprs.items()
            )

            index_exprs.append(
                {map_component.target_axis: map_var(*from_indices, axisvar)}
            )

            # don't think that this is possible for maps
            layout_exprs.append({map_component.target_axis: NotImplemented})

        axis = Axis(components, label=called_map.name)
        if axes.root:
            axes = axes.add_subaxis(axis, *from_leaf_key)
        else:
            axes = AxisTree(axis)

        for i, (cpt, mapcpt) in enumerate(checked_zip(components, bits)):
            leaf_keys.append((axis.id, cpt.label))

            target_path_per_leaf.append(
                pmap({mapcpt.target_axis: mapcpt.target_component})
            )
            index_exprs_per_leaf.append(index_exprs[i])
            layout_exprs_per_leaf.append(layout_exprs[i])

    leaves = {}
    for leaf_key, source_leaf, target_path, index_exprs, layout_exprs in checked_zip(
        leaf_keys,
        axes.leaves,
        target_path_per_leaf,
        index_exprs_per_leaf,
        layout_exprs_per_leaf,
    ):
        source_path = axes.path(*source_leaf)
        leaves[leaf_key] = (source_path, target_path, index_exprs, layout_exprs)
    return leaves, (axes,)


def index_axes(axes: AxisTree, indices: IndexTree, loop_context):
    # offsets are always scalar
    # if isinstance(indexed, CalledAxisTree):
    #     raise NotImplementedError
    # return AxisTree()

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

    # if axes is not None:
    #     return axes
    # else:
    #     return AxisTree()

    # return the new axes plus the new index expressions per leaf
    return indexed_axes, tpaths, index_expr_per_target, layout_expr_per_target


def _index_axes_rec(
    indices,
    *,
    current_index,
    # target_path_per_component=pmap(),
    # index_exprs_per_component=pmap(),
    # layout_exprs_per_component=pmap(),
    **kwargs,
):
    index_data = collect_shape_index_callback(current_index, **kwargs)
    axes_per_index, *rest = index_data

    (
        target_path_per_cpt_per_index,
        index_exprs_per_cpt_per_index,
        layout_exprs_per_cpt_per_index,
    ) = rest
    # new_target_path_per_cpt = target_path_per_component | target_path_per_cpt_per_index
    # new_index_exprs_per_cpt = index_exprs_per_component | index_exprs_per_cpt_per_index
    # new_layout_exprs_per_cpt = layout_exprs_per_component | layout_exprs_per_cpt_per_index

    if not axes_per_index.is_empty:
        leafkeys = [(ax.id, cpt.label) for ax, cpt in axes_per_index.leaves]
    else:
        leafkeys = [None]

    leafdata = {}
    for leafkey in leafkeys:
        if current_index.id in indices.parent_to_children:
            for subindex in indices.parent_to_children[current_index.id]:
                retval = _index_axes_rec(
                    indices,
                    current_index=subindex,
                    # target_path_per_component=new_target_path_per_cpt,
                    # index_exprs_per_component=new_index_exprs_per_cpt,
                    # layout_exprs_per_component=new_layout_exprs_per_cpt,
                    **kwargs,
                )
                leafdata[leafkey] = retval
        else:
            leafdata[leafkey] = (
                AxisTree(),
                target_path_per_cpt_per_index,
                index_exprs_per_cpt_per_index,
                layout_exprs_per_cpt_per_index,
            )

    target_path_per_component = {}
    index_exprs_per_component = {}
    layout_exprs_per_component = {}

    axes = axes_per_index
    # this is overly complicated, can be done above.
    for k, (subax, target_path, index_exprs, layout_exprs) in leafdata.items():
        if subax is not None:
            if axes.root:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = subax

        target_path_per_component |= target_path
        index_exprs_per_component |= index_exprs
        layout_exprs_per_component |= layout_exprs
    return (
        axes,
        target_path_per_component,
        index_exprs_per_component,
        layout_exprs_per_component,
    )

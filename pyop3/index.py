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

# cyclic import, avoid
# from pyop3.axis import Axis, AxisComponent, AxisTree
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


class LoopIterable(abc.ABC):
    """Abstract class representing something that can be looped over.

    In order for an object to be loop-able over it needs to have shape
    (``axes``) and an index expression per leaf of the shape. The simplest
    case is `AxisTree` since the index expression is just identity. This
    contrasts with something like an `IndexedLoopIterable` or `CalledMap`.
    For the former the index expression for ``axes[::2]`` would be ``2*i``
    and for the latter ``map(p)`` would be something like ``map[i, j]``.

    """

    @abc.abstractmethod
    def index(self) -> LoopIndex:
        pass

    @abc.abstractmethod
    def enumerate(self) -> EnumeratedLoopIndex:
        pass

    @property
    @abc.abstractmethod
    def axes(self):
        pass

    # this is a map from axes leaves to both target_paths and associated index expressions.
    @property
    @abc.abstractmethod
    def index_exprs(self):
        pass


class IndexedLoopIterable(LoopIterable):
    """Class representing an indexed object that can be looped over.

    Examples include ``axes[map(p)]`` or ``map(p)[::2]``.

    """


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


# We don't use this any more because index trees are eagerly consumed
# class SplitIndexTree:
#     """Container of `IndexTree`s distinguished by outer loop information.
#
#     This class is required because multi-component outer loops can lead to
#     ambiguity in the shape of the resulting `IndexTree`. Consider the loop:
#
#     .. code:: python
#
#         loop(p := mesh.points, kernel(dat0[closure(p)]))
#
#     In this case, assuming ``mesh`` to be at least 1-dimensional, ``p`` will
#     loop over multiple components (cells, edges, vertices, etc) and each
#     component will have a differently sized temporary. This is because
#     vertices map to themselves whereas, for example, edges map to themselves
#     *and* the incident vertices.
#
#     A `SplitIndexTree` is therefore useful as it allows the description of
#     an `IndexTree` *per possible configuration of relevant loop indices*.
#
#     """
#
#     def __init__(self, index_trees: pmap[pmap[LoopIndex, pmap[str, str]], IndexTree]):
#         # this is terribly unclear
#         if not is_single_valued([set(key.keys()) for key in index_trees.keys()]):
#             raise ValueError("Loop contexts must contain the same loop indices")
#
#         new_index_trees = {}
#         for key, itree in index_trees.items():
#             new_key = {}
#             for loop_index, path in key.items():
#                 if isinstance(loop_index, LocalLoopIndex):
#                     loop_index = loop_index.global_index
#                 new_key[loop_index] = path
#             new_index_trees[pmap(new_key)] = itree
#         self.index_trees = pmap(new_index_trees)
#
#     def __getitem__(self, loop_context):
#         key = {}
#         for loop_index, path in loop_context.items():
#             if isinstance(loop_index, LocalLoopIndex):
#                 loop_index = loop_index.global_index
#             if loop_index in self.loop_indices:
#                 key |= {loop_index: path}
#         key = pmap(key)
#         return self.index_trees[key]
#
#     @functools.cached_property
#     def loop_indices(self) -> frozenset[LoopIndex]:
#         # loop is used just for unpacking
#         for loop_context in self.index_trees.keys():
#             indices = set()
#             for loop_index in loop_context.keys():
#                 if isinstance(loop_index, LocalLoopIndex):
#                     loop_index = loop_index.global_index
#                 indices.add(loop_index)
#             return frozenset(indices)
#
#     @functools.cached_property
#     def datamap(self):
#         return merge_dicts([itree.datamap for itree in self.index_trees.values()])


def parse_index_tree(root, parent_to_children, loop_context):
    root = apply_loop_context(root, loop_context)
    new_parent_to_children = parse_parent_to_children(
        parent_to_children, root, loop_context
    )

    return root, pmap(new_parent_to_children), loop_context


def parse_parent_to_children(parent_to_children, parent, loop_context):
    assert isinstance(parent, ContextFreeIndex)
    new_children = []
    subparents_to_children = []

    if parent.id in parent_to_children:
        for child in parent_to_children[parent.id]:
            if child is None:
                continue
            child = apply_loop_context(child, loop_context)
            new_children.append(child)
            subparents_to_children.append(
                parse_parent_to_children(parent_to_children, child, loop_context)
            )
    else:
        for _ in parent.component_labels:
            new_children.append(None)

    return pmap({parent.id: new_children} | merge_dicts(subparents_to_children))


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


class ContextSensitiveIndex(abc.ABC):
    pass


class IndexedArray:
    """Container representing an object that has been indexed.

    For example ``dat[index]`` would produce ``Indexed(dat, index)``.

    """

    # note that axes here are specially modified to have the right layout functions
    # this needs to be done inside __getitem__
    def __init__(self, array: MultiArray, axis_trees, layout_exprs):
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

    def __init__(self, component, start=None, stop=None, step=None):
        super().__init__(component)
        # use None for the default args here since that agrees with Python slices
        self.start = start if start is not None else 0
        self.stop = stop
        self.step = step if step is not None else 1

    @property
    def datamap(self):
        return {}


class Subset(SliceComponent):
    fields = SliceComponent.fields | {"array"}

    def __init__(self, component, array: MultiArray):
        super().__init__(component)
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


# ImmutableRecord?
class CalledMap(LoopIterable, UniquelyIdentifiedImmutableRecord, ContextSensitiveIndex):
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

    @property
    def index_exprs(self):
        raise NotImplementedError

    @property
    def bits(self):
        return self.map.bits


class Index(LabelledNode):
    @property
    @abc.abstractmethod
    def target_paths(self):
        raise NotImplementedError


class LoopIndex(Index, ContextSensitiveIndex, abc.ABC):
    pass


# inherit from Index?
class ContextFreeIndex(abc.ABC):
    # make abstract etc
    # @property
    # def id(self):
    #     raise NotImplementedError
    pass


class ContextFreeLoopIndex(Index, ContextFreeIndex):
    def __init__(self, orig_index, labels, **kwargs):
        assert len(labels) == 1
        super().__init__(labels, id=orig_index.id, **kwargs)
        self.index = orig_index

    # TODO clean this up.
    @property
    def target_paths(self):
        return self.component_labels


# TODO just call this LoopIndex (inherit from AbstractLoopIndex)
class GlobalLoopIndex(LoopIndex):
    fields = LoopIndex.fields | {"iterset"}

    def __init__(self, iterset, **kwargs):
        # FIXME I think that an IndexTree should not know its component labels
        # we can do that in the dict. This is because it is context dependent.
        # for now just use one label (assume single component)
        cpt_labels = ["dont use"]
        super().__init__(cpt_labels, **kwargs)
        self.iterset = iterset

    @property
    def target_paths(self):
        return self.iterset.target_paths

    @property
    def datamap(self):
        return self.iterset.datamap


class LocalLoopIndex(LoopIndex):
    """Class representing a 'local' index."""

    def __init__(self, loop_index: LoopIndex):
        self.global_index = loop_index

    @property
    def target_paths(self):
        return self.global_index.target_paths

    @property
    def datamap(self):
        return self.global_index.datamap


# TODO I want a Slice to have "bits" like a Map/CalledMap does
class Slice(Index, ContextSensitiveIndex):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    fields = {"axis", "slices"} | Index.fields

    def __init__(self, axis, slices: Collection[SliceComponent], **kwargs):
        slices = as_tuple(slices)
        cpt_labels = [s.component for s in slices]
        super().__init__(cpt_labels, **kwargs)
        self.axis = axis
        self.slices = slices

    @property
    def target_paths(self):
        return tuple(pmap({self.axis: subslice.component}) for subslice in self.slices)

    @property
    def datamap(self):
        return merge_dicts([s.datamap for s in self.slices])


# TODO
class ContextFreeSlice(ContextFreeIndex):
    def __init__(self, orig_slice, labels):
        self.orig_slice = orig_slice
        self.component_labels = labels
        self.id = self.orig_slice.id

    @property
    def degree(self):
        return len(self.component_labels)


class ContextFreeCalledMap(Index, ContextFreeIndex):
    def __init__(self, component_labels, map: CalledMap, from_index: ContextFreeIndex):
        super().__init__(component_labels)
        self.map = map
        self.from_index = from_index

    @property
    def target_paths(self):
        # hopefully I shouldn't have to do this introspection here, make a class attribute
        all_map_components = {}
        for map_components in self.map.bits.values():
            for map_component in map_components:
                all_map_components[map_component.label] = map_component

        targets = []
        for component_label in self.component_labels:
            map_component_label = component_label[-1]
            selected_cpt = all_map_components[map_component_label]
            target = pmap({selected_cpt.target_axis: selected_cpt.target_component})
            targets.append(target)
        return targets

    # @property
    # def id(self):
    #     return self.map.id
    @property
    def name(self):
        return self.map.name

    @functools.cached_property
    def datamap(self):
        return self.map.map.datamap | self.from_index.datamap


#
#     # ick
#     @property
#     def bits(self):
#         return self.map.map.bits
#
#     @property
#     def name(self):
#         return self.map.map.name


class EnumeratedLoopIndex:
    def __init__(self, iterset: AxisTree):
        global_index = GlobalLoopIndex(iterset)
        local_index = LocalLoopIndex(global_index)

        self.global_index = global_index
        self.local_index = local_index


# it is probably a better pattern to give axis trees a "parent" option
class IndexedAxisTree:
    def __init__(self, axis_trees):
        self.axis_trees = pmap(axis_trees)

    def __getitem__(self, indices):
        new_axis_trees = {}
        for loop_context, axis_tree in self.axis_trees.items():
            new_axis_trees[loop_context] = axis_tree[indices]
        return IndexedAxisTree(new_axis_trees)

    @property
    def target_paths(self):
        # should move somewhere else
        from pyop3.codegen.loopexpr2loopy import collect_target_paths

        # TODO I don't think I need to pass loop_indices here (pmap() here)
        _, paths = collect_target_paths(self, pmap())
        return tuple(paths.values())

    def index(self):
        return GlobalLoopIndex(self)

    def enumerate(self):
        return EnumeratedLoopIndex(self)

    @functools.cached_property
    def datamap(self):
        return merge_dicts(axis_tree.datamap for axis_tree in self.axis_trees.values())


"""
25/08

* I want this function to return a nested dictionary of the form:

    {
        loop_context0: {
            {
                "root": CalledMap,
            }
        },
        loop_context1: {
            {
                "root": CalledMap,
            }
        },
    }

  Then things (e.g. slices) will attach below this:

    {
        loop_context0: {
            {
                "root": CalledMap,
                X: Slice,
                Y: Slice,
            }
        },
        loop_context1: {
            {
                "root": CalledMap,
                A: Slice,
                B: Slice,
                C: Slice,
            }
        },
    }

  where X, Y, A, B and C are some sort of path describing how they attach to the map.
  This will depend on the loop context and the advertised outputs of the map.
    
"""
# @_split_index.register
# def _(called_map: CalledMap) -> Mapping[Mapping[LoopIndex, Mapping], SplitCalledMap]:
#     index_forest = collections.defaultdict(dict)
#     for loop_context, inner_index_tree in _split_index(called_map.from_index).items():
#         index_forest[loop_context] = IndexTree(called_map)
#     return pmap(index_forest)


@functools.singledispatch
def apply_loop_context(arg, *args, **kwargs):
    raise TypeError


@apply_loop_context.register
def _(index: ContextFreeIndex, loop_context, **kwargs):
    return index


@apply_loop_context.register
def _(loop_index: LoopIndex, loop_context, **kwargs):
    # use the path as the component label
    component_label = loop_context[loop_index]
    return ContextFreeLoopIndex(loop_index, [component_label])


@apply_loop_context.register
def _(slice_: Slice, loop_context, **kwargs):
    return ContextFreeSlice(slice_, [cpt.label for cpt in slice_.slices])


@apply_loop_context.register
def _(called_map: CalledMap, loop_context, **kwargs):
    from_index = apply_loop_context(called_map.from_index, loop_context, **kwargs)

    new_labels = []
    for from_label, from_path in checked_zip(
        from_index.component_labels, from_index.target_paths
    ):
        map_components = called_map.bits[from_path]
        for map_component in map_components:
            # the new label should be the concatenation of previous ones
            new_label = as_tuple(from_label) + (map_component.label,)
            new_labels.append(new_label)

    return ContextFreeCalledMap(new_labels, called_map, from_index)


@apply_loop_context.register
def _(slice_: slice, loop_context, axes, path):
    parent_axis, parent_cpt = axes._node_from_path(path)
    target_axis = axes.child(parent_axis, parent_cpt)
    slice_cpts = []
    for cpt in target_axis.components:
        slice_cpt = AffineSliceComponent(
            cpt.label, slice_.start, slice_.stop, slice_.step
        )
        slice_cpts.append(slice_cpt)
    fullslice = Slice(target_axis.label, slice_cpts)
    return ContextFreeSlice(fullslice, [scpt.label for scpt in slice_cpts])


def loop_contexts_from_iterable(indices):
    # this is a bit tricky/unpleasant to write
    # FIXME this is currently very naive/limited, need to check for conflicts etc
    loop_contexts = []
    for index in indices:
        ctx = collect_loop_context(index)
        if len(ctx) > 1:
            raise NotImplementedError("TODO")
        loop_contexts.extend(ctx)
    return tuple(loop_contexts)


@functools.singledispatch
def collect_loop_context(arg, *args, **kwargs):
    if isinstance(arg, collections.abc.Iterable):
        return loop_contexts_from_iterable(arg)
    else:
        raise TypeError


@collect_loop_context.register
def _(arg: ContextFreeLoopIndex, *args, **kwargs):
    return [pmap({arg.index: just_one(arg.component_labels)})]


@collect_loop_context.register
def _(arg: LoopIndex):
    from pyop3.axis import AxisTree

    if isinstance(arg.iterset, IndexedAxisTree):
        loop_contexts = []
        for loop_context, axis_tree in arg.iterset.axis_trees.items():
            for leaf in axis_tree.leaves:
                loop_contexts.append(loop_context | pmap({arg: axis_tree.path(*leaf)}))
        return loop_contexts
    else:
        assert isinstance(arg.iterset, AxisTree)
        return [pmap({arg: arg.iterset.path(*leaf)}) for leaf in arg.iterset.leaves]


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
    myslice = ContextFreeSlice(fullslice, [c.label for c in slice_components])

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
        for target_path in index.target_paths:
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


# @as_index_tree.register
# def _(index_tree:IndexTree, loop_context):
#     assert index_tree.loop_context == loop_context
#     return index_tree


@as_index_tree.register
def _(index: ContextSensitiveIndex, ctx):
    return IndexTree(index, loop_context=ctx)


@functools.singledispatch
def as_index_forest(arg: Any, *, axes, **kwargs):
    if isinstance(arg, collections.abc.Iterable):
        loop_contexts = collect_loop_context(arg)
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
def _(index: ContextSensitiveIndex, **kwargs):
    loop_contexts = collect_loop_context(index) or [pmap()]
    return tuple(as_index_tree(index, ctx) for ctx in loop_contexts)


@as_index_forest.register
def _(slice_: slice, **kwargs):
    raise NotImplementedError

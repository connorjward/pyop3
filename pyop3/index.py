from __future__ import annotations

import abc
import collections
import dataclasses
import functools
from typing import Any, Collection, Hashable, Sequence

import pytools
from pyrsistent import pmap

from pyop3.axis import Axis, AxisComponent, AxisTree
from pyop3.tree import LabelledNode, LabelledTree, postvisit
from pyop3.utils import (
    LabelledImmutableRecord,
    as_tuple,
    checked_zip,
    is_single_valued,
    just_one,
    merge_dicts,
)


class IndexTree(LabelledTree):
    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
        return postvisit(self, _collect_datamap, itree=self)


class IndexTreeBag:
    def __init__(self, index_trees: pmap[pmap[LoopIndex, pmap[str, str]], IndexTree]):
        # this is terribly unclear
        if not is_single_valued([set(key.keys()) for key in index_trees.keys()]):
            raise ValueError("Loop contexts must contain the same loop indices")
        self.index_trees = index_trees

    def __getitem__(self, loop_context):
        key = {}
        for loop_index, path in loop_context.items():
            if loop_index in self.loop_indices:
                key |= {loop_index: path}
        key = pmap(key)
        return self.index_trees[key]

    @functools.cached_property
    def loop_indices(self) -> frozenset[LoopIndex]:
        for loop_context in self.index_trees.keys():
            return frozenset(loop_context.keys())

    @functools.cached_property
    def datamap(self):
        return merge_dicts([itree.datamap for itree in self.index_trees.values()])


IndexLabel = collections.namedtuple("IndexLabel", ["axis", "component"])


# is IndexedArray a better name? Probably
class Indexed:
    """Container representing an object that has been indexed.

    For example ``dat[index]`` would produce ``Indexed(dat, index)``.

    """

    def __init__(self, obj, indices):
        # Since loop indices can target different axes we need to have multiple index
        # trees. These are stored in a pmap mapping {loop_index: axis_path, ...} to
        # index tree. The idea is that each tree can be handled cleanly with explicitly
        # known axes. During code generation we can select the right tree.
        # Some thought is required about how we want to parse index trees/lists of indices
        # into these separable trees.
        if not isinstance(indices, IndexTreeBag):
            raise NotImplementedError("Some clever parsing is required here")

        self.obj = obj
        self.indices = indices

    # old alias, not right now we have a pmap of index trees rather than just a single one
    @property
    def itree(self):
        return self.indices

    def __getitem__(self, indices):
        from pyop3.distarray import MultiArray

        if not isinstance(self.obj, MultiArray) and not isinstance(
            indices, (IndexTree, Index)
        ):
            raise NotImplementedError(
                "Need to compute the temporary/intermediate axes for this to be allowed"
            )

        indices = as_index_tree(indices, "not currently an axis tree")
        return Indexed(self, indices)

    @functools.cached_property
    def datamap(self):
        return self.obj.datamap | self.itree.datamap

    @property
    def name(self):
        return self.obj.name

    @property
    def dtype(self):
        return self.obj.dtype


class SliceComponent(pytools.ImmutableRecord, abc.ABC):
    fields = {"component"}

    def __init__(self, component):
        super().__init__()
        self.component = component


class AffineSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"start", "stop", "step"}

    def __init__(self, component, start=0, stop=None, step=1):
        super().__init__(component)
        self.start = start
        self.stop = stop
        self.step = step


class Subset(SliceComponent):
    fields = SliceComponent.fields | {"array"}

    def __init__(self, component, array: MultiArray):
        super().__init__(component)
        self.array = array


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


class TabulatedMapComponent(MapComponent):
    fields = MapComponent.fields - {"arity"} | {"array"}

    def __init__(self, target_axis, target_component, array, **kwargs):
        arity = array.axes.leaf_component.count
        super().__init__(target_axis, target_component, arity, **kwargs)
        self.array = array

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
class CalledMap:
    # This function cannot be part of an index tree because it has no specialised
    # to a particular loop index path.
    def __init__(self, map, from_index):
        self.map = map
        self.from_index = from_index


class Index(LabelledNode):
    @property
    @abc.abstractmethod
    def target_paths(self):
        pass


class LoopIndex(Index):
    """Object representing a set of indices of a loop nest.

    FIXME More detail needed.

    Notes
    -----
    For the moment we assume that `LoopIndex` objects cannot be
    multi-component. This would need to produce a different set of loops
    for each component. One application of this would be to be able to iterate
    over all points of a mesh together (since cells, edges etc are different
    components).

    """

    def __init__(self, iterset, **kwargs):
        cpt_labels = [cpt_label for ax, cpt_label in iterset.leaves]
        if len(cpt_labels) > 1:
            raise NotImplementedError(
                "Multi-component loop indices are not currently supported"
            )

        super().__init__(cpt_labels, **kwargs)
        self.iterset = iterset

    @property
    def target_paths(self) -> tuple[pmap]:
        return tuple(
            self.iterset.path(leaf_axis, leaf_cpt_label)
            for leaf_axis, leaf_cpt_label in self.iterset.leaves
        )

    @functools.cached_property
    def datamap(self):
        return self.iterset.datamap


class LocalLoopIndex(Index):
    """Class representing a 'local' index."""

    def __init__(self, loop_index: LoopIndex, **kwargs):
        super().__init__(loop_index.component_labels, **kwargs)
        self.loop_index = loop_index

    @property
    def target_paths(self):
        return self.loop_index.target_paths

    @property
    def datamap(self):
        return self.loop_index.datamap


class Slice(Index):
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
        # return pmap()
        return {}


class UnrolledCalledMap(Index):
    def __init__(self, map, from_index, loop_context):
        # The degree of the called map depends on the index it acts on. If
        # the map maps from a -> {a, b, c} *and* b -> {b, c} then the degree
        # is 3 or 2 depending on from_index. If from_index is also a CalledMap
        # with multiple output components then the degree will need to be
        # determined by summing from the leaves of this prior map.
        # Note that 'degree' is equivalent to stating the number of leaves of
        # the resulting axes.
        if isinstance(from_index, LoopIndex):
            # For now maps will only work with loop indices with depth 1. Being
            # able to work with deeper loop indices is required for things like
            # ephemeral meshes to work (since mesh.cells would have base and
            # column axes).
            if from_index.iterset.depth > 1:
                raise NotImplementedError(
                    "Mapping from loop indices with depth greater than 1 is not "
                    "yet supported"
                )
            iterset_axis = from_index.iterset.root
            axis_label = iterset_axis.label
            raise NotImplementedError(
                "here need to use the loop context to get the right bit"
            )
            # just_one used since from_index can (currently) only have a
            # single component
            cpt_label = just_one(iterset_axis.components).label

            leaves = []
            for leaf in map.bits[pmap({axis_label: cpt_label})]:
                to_axis_label = leaf.target_axis
                to_cpt_label = leaf.target_component
                path = pmap({to_axis_label: to_cpt_label})
                leaves.append(path)
        else:
            assert isinstance(from_index, CalledMap)
            leaves = []
            for from_leaf in from_index.component_labels:
                for leaf in map.bits[from_leaf]:
                    to_axis_label = leaf.target_axis
                    to_cpt_label = leaf.target_component
                    path = pmap({to_axis_label: to_cpt_label})
                    leaves.append(path)

        super().__init__(leaves)
        self.map = map
        self.from_index = from_index

    @property
    def target_paths(self):
        return self.component_labels

    @functools.cached_property
    def datamap(self):
        return self.map.datamap | self.from_index.datamap

    # ick
    @property
    def bits(self):
        return self.map.bits

    @property
    def name(self):
        return self.map.name


class EnumeratedLoopIndex:
    def __init__(self, iterset: AxisTree):
        self.index = LoopIndex(iterset)
        self.count = LocalLoopIndex(self.index)


@functools.singledispatch
def as_index_tree(arg: Any, axes: AxisTree) -> IndexTree:
    # cyclic import
    from pyop3.distarray import MultiArray

    if isinstance(arg, MultiArray):
        return _index_tree_from_collection([arg], axes)
    elif isinstance(arg, Collection):
        return _index_tree_from_collection(arg, axes)
    elif arg is Ellipsis:
        return _index_tree_from_ellipsis(axes)
    else:
        raise TypeError(f"Handler is not registered for {type(arg)}")


@as_index_tree.register
def _(arg: IndexTree, axes: AxisTree) -> IndexTree:
    return arg


@as_index_tree.register
def _(arg: Index, axes: AxisTree) -> IndexTree:
    return IndexTree(arg)


@as_index_tree.register
def _(slice_: slice, axes: AxisTree) -> IndexTree:
    return _index_tree_from_collection([slice_], axes)


def _collect_datamap(index, *subdatamaps, itree):
    return index.datamap | merge_dicts(subdatamaps)


# TODO Handle axes so we can include slices in the list
# There are some rules about how we do this. Indices do not strictly need to
# follow the same ordering as the axes themselves.
def _index_tree_from_collection(
    indices: Collection[Index], axes: AxisTree, path=pmap()
):
    """Return an index tree formed by concatenating successive indices.

    If any of the indices yield multiple components then subsequent indices
    will be attached to all components.

    """
    # cyclic import
    from pyop3.distarray import MultiArray

    index, *subindices = indices

    if not isinstance(index, Index):
        # We can use a slice provided that the previous indices provide a complete path.
        # Consider an axis tree ax0 -> ax1 -> ax2. We can safely use a slice if the
        # previous indices indexed {}, {ax0} or {ax0, ax1} since the axis acted upon by
        # the slice is unambiguous. This is not the case if we have only indexed {ax1},
        # {ax2}, {ax1, ax2} or {ax0, ax2}. The following line should raise an error if
        # any of the latter cases are encountered.
        if not path:
            current_axis = axes.root
        else:
            parent_axis, parent_cpt = axes._node_from_path(path)
            current_axis = axes.child(parent_axis, parent_cpt)

        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop
            step = index.step if index.step is not None else 1

            # Slices target all components of the current axis.
            slice_cpts = []
            for cpt in current_axis.components:
                subslice = AffineSliceComponent(cpt.label, start, stop, step)
                slice_cpts.append(subslice)
        elif isinstance(index, MultiArray):
            slice_cpts = []
            for cpt in current_axis.components:
                subslice = Subset(cpt.label, index)
                slice_cpts.append(subslice)
        else:
            raise TypeError

        index = Slice(current_axis.label, slice_cpts)

    tree = IndexTree(index)
    if subindices:
        # for maps these are identical but that is not so for slices
        for component, target_path in checked_zip(
            index.component_labels, index.target_paths
        ):
            subtree = _index_tree_from_collection(subindices, axes, path | target_path)
            tree = tree.add_subtree(subtree, index, component)
    return tree


def _index_tree_from_ellipsis(
    axes: AxisTree, current_axis: Axis | None = None
) -> IndexTree:
    current_axis = current_axis or axes.root

    subslices = []
    subtrees = []
    for cpt in current_axis.components:
        subslices.append(SliceComponent(cpt.label))

        subaxis = axes.child(current_axis, cpt)
        if subaxis:
            subtrees.append(_index_tree_from_ellipsis(axes, subaxis))
        else:
            subtrees.append(None)

    slice_ = Slice(current_axis.label, subslices)
    tree = IndexTree(slice_)
    for subslice, subtree in checked_zip(subslices, subtrees):
        if subtree is not None:
            tree = tree.add_subtree(subtree, slice_, subslice.component)
    return tree


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


# it is probably a better pattern to give axis trees a "parent" option
class IndexedAxisTree:
    def __init__(self, axes: AxisTree, indices: IndexTree):
        self.axes = axes
        self.indices = as_index_tree(indices, axes)

    def index(self):
        return LoopIndex(self)

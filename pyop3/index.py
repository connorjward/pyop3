from __future__ import annotations

import abc
import collections
import functools
from typing import Any, Collection, Hashable, Sequence

import pytools
from pyrsistent import pmap

from pyop3.axis import Axis, AxisComponent
from pyop3.tree import LabelledNode, LabelledTree, postvisit
from pyop3.utils import LabelledImmutableRecord, as_tuple, just_one, merge_dicts


class IndexTree(LabelledTree):
    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
        return postvisit(self, _collect_datamap, itree=self)


IndexLabel = collections.namedtuple("IndexLabel", ["axis", "component"])


# is IndexedArray a better name? Probably
class Indexed:
    """Container representing an object that has been indexed.

    For example ``dat[index]`` would produce ``Indexed(dat, index)``.

    """

    def __init__(self, obj, itree):
        self.obj = obj
        self.itree = itree

    def __getitem__(self, indices):
        indices = as_index_tree(indices)
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


class SliceComponent(pytools.ImmutableRecord):
    fields = {"component", "start", "stop", "step"}

    def __init__(self, component, start=0, stop=None, step=1):
        super().__init__()
        self.component = component
        self.start = start
        self.stop = stop
        self.step = step


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


class Index(LabelledNode):
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

    @functools.cached_property
    def datamap(self):
        return self.iterset.datamap


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
    def datamap(self):
        # return pmap()
        return {}


class CalledMap(Index):
    def __init__(self, map, from_index):
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


@functools.singledispatch
def as_index_tree(arg: Any) -> IndexTree:
    if isinstance(arg, Collection):
        return _index_tree_from_collection(arg)
    else:
        raise TypeError(f"Handler is not registered for {type(arg)}")


@as_index_tree.register
def _(arg: IndexTree) -> IndexTree:
    return arg


@as_index_tree.register
def _(arg: Index) -> IndexTree:
    return IndexTree(arg)


def _collect_datamap(index, *subdatamaps, itree):
    return index.datamap | merge_dicts(subdatamaps)


def _index_tree_from_collection(indices: Collection[Index]):
    """Return an index tree formed by concatenating successive indices.

    If any of the indices yield multiple components then subsequent indices
    will be attached to all components.

    """
    index, *subindices = indices
    tree = IndexTree(index)
    if subindices:
        subtree = _index_tree_from_collection(subindices)
        for cpt_label in index.component_labels:
            tree = tree.add_subtree(subtree, index, cpt_label)
    return tree

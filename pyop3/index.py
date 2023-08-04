from __future__ import annotations

import abc
import collections
import functools
from typing import Any, Hashable, Sequence

import pytools
from pyrsistent import pmap

from pyop3.axis import Axis, AxisComponent
from pyop3.tree import LabelledNode, LabelledTree, NodeComponent, postvisit
from pyop3.utils import as_tuple, just_one, merge_dicts


class IndexTree(LabelledTree):
    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
        return postvisit(self, _collect_datamap, itree=self)


IndexLabel = collections.namedtuple("IndexLabel", ["axis", "component"])


class Indexed:
    """Container representing an object that has been indexed.

    For example ``dat[index]`` would produce ``Indexed(dat, index)``.

    """

    def __init__(self, obj, itree):
        self.obj = obj
        self.itree = itree

    def __getitem__(self, indices):
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


class Index(LabelledNode):
    pass


# class IndexComponent(NodeComponent, abc.ABC):
#     fields = NodeComponent.fields | {"from_axis", "from_cpt", "to_axis", "to_cpt"}
#
#     # FIXME I think that these are the wrong attributes
#     def __init__(self, from_axis, from_cpt, to_axis, to_cpt, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.from_axis = from_axis
#         self.from_cpt = from_cpt
#         self.to_axis = to_axis
#         self.to_cpt = to_cpt


# TODO
# should come from Index
# class LoopIndex(Index):
from pyop3.tree import Node


class LoopIndex(Node):
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

    # TODO this needs an ID attribute so we can differentiate p := mesh.cells.index()
    # with q := mesh.cells.index() for example.
    # fields = Index.fields | {"iterset"}

    def __init__(self, iterset, **kwargs):
        super().__init__(degree=1, **kwargs)
        self.iterset = iterset
        # FIXME
        # self.id = id(self)
        # self.degree = 1

    @functools.cached_property
    def datamap(self):
        return self.iterset.datamap


class Map(pytools.ImmutableRecord):
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
            for stuff in bit:
                data |= stuff[1].datamap
        return data


class Slice(Index):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    fields = Index.fields | {"values"}

    def __init__(self, values, axis=None, cpt=None, **kwargs):
        # cyclic import
        from pyop3.axis import Axis, AxisComponent

        # nargs = len(args)
        # if nargs == 0:
        #     start, stop, step = 0, None, 1
        # elif nargs == 1:
        #     start, stop, step = 0, args[0], 1
        # elif nargs == 2:
        #     start, stop, step = args[0], args[1], 1
        # elif nargs == 3:
        #     start, stop, step = args[0], args[1], args[2]
        # else:
        #     raise ValueError("Too many arguments")
        # the smaller space mapped from by the slice is "anonymous" so use
        # something unique here
        # FIXME this breaks copy
        # from_axis = Axis._unique_label()
        # from_cpt = AxisComponent._label_generator()
        # super().__init__(from_axis, from_cpt, axis, cpt, **kwargs)
        super().__init__(degree=len(values), **kwargs)
        self.values = values

    @property
    def datamap(self):
        # return pmap()
        return {}


class ScalarIndex(Slice):
    """Index component representing a single scalar value.

    This class is distinct from a `LoopIndex` because no loop needs to be emitted.
    We also distinguish between it and a `Slice` (even though they are really the
    same thing) because indexing with a scalar removes the axis from the resulting
    indexed array.

    """

    def __init__(self, value):
        # a scalar index is equivalent to a slice starting at value
        # and stopping at value+1.
        super().__init__(value, value + 1)


class TabulatedMap(Map):
    fields = Map.fields | {"data"}

    def __init__(
        self,
        from_axis,
        from_cpt,
        to_axis,
        to_cpt,
        arity,
        data,
        **kwargs,
    ) -> None:
        super().__init__(from_axis, from_cpt, to_axis, to_cpt, arity, **kwargs)
        self.data = data


class IdentityMap(Map):
    pass

    # TODO is this strictly needed?
    # @property
    # def label(self):
    #     assert len(self.to_labels) == 1
    #     return self.to_labels[0]


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
                to_axis_label = leaf[3]
                to_cpt_label = leaf[4]
                path = pmap({to_axis_label: to_cpt_label})
                leaves.append(path)
        else:
            assert isinstance(from_index, CalledMap)
            leaves = []
            for from_leaf in from_index.leaves:
                for leaf in map.bits[from_leaf]:
                    to_axis_label = leaf[3]
                    to_cpt_label = leaf[4]
                    path = pmap({to_axis_label: to_cpt_label})
                    leaves.append(path)

        super().__init__(degree=len(leaves))
        self.map = map
        self.from_index = from_index

        # useful for computing degree of maps that call this one
        self.leaves = leaves

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


class AffineMap(Map):
    fields = Map.fields | {"expr"}

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


@functools.singledispatch
def as_index_tree(arg: Any) -> IndexTree:
    raise TypeError


@as_index_tree.register
def _(arg: IndexTree) -> IndexTree:
    return arg


@as_index_tree.register
def _(arg: Index) -> IndexTree:
    return IndexTree(arg)


def _collect_datamap(index, *subdatamaps, itree):
    return index.datamap | merge_dicts(subdatamaps)

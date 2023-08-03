from __future__ import annotations

import abc
import collections
import functools
from typing import Any, Hashable, Sequence

import pytools

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
        try:
            key = just_one(bits.keys())
        except ValueError:
            raise AssertionError("Maps now only map from a single set of axes")

        degree = len(bits[key])

        super().__init__(**kwargs)
        self.degree = degree
        self.bits = bits
        self.name = name

    def __call__(self, index):
        return CalledMap(self, index)

    @functools.cached_property
    def datamap(self):
        try:
            key = just_one(self.bits.keys())
        except ValueError:
            raise AssertionError("Maps now only map from a single set of axes")
        return merge_dicts([bit[1].datamap for bit in self.bits[key]])


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
        super().__init__(degree=len(values))
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
        super().__init__(degree=map.degree)
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

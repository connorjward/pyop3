from __future__ import annotations

import abc
import collections
import functools
from typing import Any, Hashable, Sequence

import pytools

from pyop3.tree import LabelledNode, LabelledTree, NodeComponent, postvisit
from pyop3.utils import as_tuple, merge_dicts


class IndexTree(LabelledTree):
    fields = LabelledTree.fields | {"axes"}

    def __init__(self, *args, axes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.axes = axes

    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
        return postvisit(self, _collect_datamap, itree=self)


IndexLabel = collections.namedtuple("IndexLabel", ["axis", "component"])


class Index(LabelledNode):
    fields = LabelledNode.fields - {"degree"} | {"components"}

    def __init__(self, components: Sequence[IndexComponent] | IndexComponent, **kwargs):
        components = as_tuple(components)

        super().__init__(degree=len(components), **kwargs)
        self.components = components

    def index(self, x: Index) -> int:
        return self.components.index(x)

    # old alias
    @property
    def indices(self):
        return self.components


class IndexComponent(NodeComponent, abc.ABC):
    fields = NodeComponent.fields | {"from_axis", "from_cpt", "to_axis", "to_cpt"}

    def __init__(self, from_axis, from_cpt, to_axis, to_cpt, **kwargs) -> None:
        super().__init__(**kwargs)
        self.from_axis = from_axis
        self.from_cpt = from_cpt
        self.to_axis = to_axis
        self.to_cpt = to_cpt

    # TODO this is quite ugly
    @property
    def from_tuple(self):
        return (self.from_axis, self.from_cpt)

    @property
    def to_tuple(self):
        return (self.to_axis, self.to_cpt)


class Map(IndexComponent):
    fields = IndexComponent.fields | {"arity"}

    def __init__(self, from_axis, from_cpt, to_axis, to_cpt, arity, **kwargs) -> None:
        super().__init__(from_axis, from_cpt, to_axis, to_cpt, **kwargs)
        self.arity = arity


class Slice(IndexComponent):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    """

    fields = IndexComponent.fields | {"start", "stop", "step"}

    def __init__(self, *args, axis=None, cpt=None, **kwargs):
        # cyclic import
        from pyop3.axis import Axis, AxisComponent

        nargs = len(args)
        if nargs == 0:
            start, stop, step = 0, None, 1
        elif nargs == 1:
            start, stop, step = 0, args[0], 1
        elif nargs == 2:
            start, stop, step = args[0], args[1], 1
        elif nargs == 3:
            start, stop, step = args[0], args[1], args[2]
        else:
            raise ValueError("Too many arguments")

        # the smaller space mapped from by the slice is "anonymous" so use
        # something unique here
        # FIXME this breaks copy
        from_axis = Axis._unique_label()
        from_cpt = AxisComponent._label_generator()

        super().__init__(from_axis, from_cpt, axis, cpt, **kwargs)
        self.start = start
        self.stop = stop
        self.step = step

    # def __repr__(self) -> str:
    #     return f"{type(self)}({self.start}, {self.stop}, {self.step}, axis={self.to_axis}, to_cpt={self.to_cpt})"

    # def __str__(self) -> str:
    # return f"{type(self)}({self.start}, {self.stop}, {self.step}, from_axis={self.from_axis}, from_cpt={self.from_cpt}, to_axis={self.to_axis}, to_cpt={self.to_cpt})"


class TabulatedMap(Map):
    fields = Map.fields | {"data"}

    def __init__(
        self, from_axis, from_cpt, to_axis, to_cpt, arity, data, **kwargs
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
    datamap = {}
    for cidx, component in enumerate(index.components):
        if isinstance(component, TabulatedMap):
            datamap |= component.data.datamap
    return datamap | merge_dicts(subdatamaps)

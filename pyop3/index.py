from __future__ import annotations

import abc
import collections
import functools
from typing import Any, Hashable, Sequence

import pytools

from pyop3.tree import LabelledNode, LabelledTree, postvisit
from pyop3.utils import UniqueNameGenerator, as_tuple, merge_dicts


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


# alias
MultiIndex = Index


class IndexComponent(NodeComponent, abc.ABC):
    fields = {"from_axis", "to_axis", "from_cpt", "to_cpt", "id"}

    _lazy_id_generator = None

    def __init__(self, from_axis, to_axis, *, id: Hashable | None = None) -> None:
        super().__init__()

        self.from_axis = from_axis
        self.to_axis = to_axis
        self.id = id or next(self._id_generator)

    @classmethod
    @property
    def _id_generator(cls):
        if not cls._lazy_id_generator:
            cls._lazy_id_generator = UniqueNameGenerator(f"_{cls.__name__}_id")
        return cls._lazy_id_generator


class Slice(IndexComponent):
    fields = IndexComponent.fields | {"start", "stop", "step"}

    def __init__(self, *args, axis=None, cpt=None, **kwargs):
        nargs = len(args)
        if nargs == 0:
            start, stop, step = None, None, None
        elif nargs == 1:
            start, stop, step = None, args[0], None
        elif nargs == 2:
            start, stop, step = args[0], args[1], None
        elif nargs == 3:
            start, stop, step = args[0], args[1], args[2]
        else:
            raise ValueError("More than 3 arguments passed to Slice constructor")

        super().__init__(axis, axis, **kwargs)
        self.start = start or 0
        self.stop = stop
        self.step = step or 1


class Map(IndexComponent):
    fields = IndexComponent.fields | {"arity"}

    # in theory we can have a selector function here too so to_labels is actually bigger?
    # means we have multiple children?

    def __init__(self, from_labels, to_labels, arity, **kwargs):
        self.from_labels = tuple(from_labels)
        self.to_labels = tuple(to_labels)
        self.arity = arity
        self.selector = None  # TODO
        super().__init__(**kwargs)

    @property
    def size(self):
        return self.arity


class TabulatedMap(Map):
    fields = Map.fields | {"data"}

    def __init__(self, from_labels, to_labels, arity, data, **kwargs):
        super().__init__(from_labels, to_labels, arity, **kwargs)
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

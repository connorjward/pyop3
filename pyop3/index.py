from __future__ import annotations

import abc
import collections
import functools
from typing import Any, Hashable, Sequence

import pytools

from pyop3.tree import LabelledNode, LabelledTree, postvisit
from pyop3.utils import UniqueNameGenerator, as_tuple, merge_dicts


class IndexTree(LabelledTree):
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


class IndexComponent(pytools.ImmutableRecord, abc.ABC):
    fields = {"id"}

    _lazy_id_generator = None

    def __init__(self, *, id: Hashable | None = None) -> None:
        super().__init__()

        self.id = id or next(self._id_generator)

    @classmethod
    @property
    def _id_generator(cls):
        if not cls._lazy_id_generator:
            cls._lazy_id_generator = UniqueNameGenerator(f"_{cls.__name__}_id")
        return cls._lazy_id_generator


class Range(IndexComponent):
    # TODO: Gracefully handle start, stop, step
    # fields = IndexNode.fields | {"label", "start", "stop", "step"}
    fields = IndexComponent.fields | {"path", "stop"}

    def __init__(self, path, stop, **kwargs):
        super().__init__(**kwargs)

        if isinstance(path, Sequence) and len(path) == 2 and isinstance(path[1], int):
            self.path = path
        else:
            self.path = (path, 0)
        self.stop = stop

    # TODO: This is temporary
    @property
    def size(self):
        return self.stop

    @property
    def start(self):
        return 0  # TODO

    @property
    def step(self):
        return 1  # TODO


class Map(IndexComponent):
    fields = IndexComponent.fields | {"from_labels", "to_labels", "arity"}

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

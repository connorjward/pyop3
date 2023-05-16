import abc
import collections
from typing import Hashable, Sequence

from pyop3.tree import LabelledNode, LabelledNodeComponent, LabelledTree
from pyop3.utils import UniqueNameGenerator


class IndexTree(LabelledTree):
    pass


IndexLabel = collections.namedtuple("IndexLabel", ["axis", "component"])


class MultiIndex(LabelledNode):
    def __init__(self, label: Hashable, indices, **kwargs):
        super().__init__(indices, label, **kwargs)

    @property
    def indices(self) -> Sequence["Index"]:
        return self.components


# FIXME: not a node any more
class IndexNode(LabelledNodeComponent, abc.ABC):
    fields = LabelledNodeComponent.fields | {"id"}

    _lazy_id_generator = None

    def __init__(self, label, *, id: Hashable | None = None):
        super().__init__(label)
        self.id = id or next(self._id_generator)

    @classmethod
    @property
    def _id_generator(cls):
        if not cls._lazy_id_generator:
            cls._lazy_id_generator = UniqueNameGenerator(f"_{cls.__name__}_id")
        return cls._lazy_id_generator


# alias, better?
Index = IndexNode


class RangeNode(IndexNode):
    # TODO: Gracefully handle start, stop, step
    # fields = IndexNode.fields | {"label", "start", "stop", "step"}
    fields = IndexNode.fields | {"stop"}

    def __init__(self, label: Hashable, stop, **kwargs):
        super().__init__(label, **kwargs)
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


class MapNode(IndexNode):
    fields = IndexNode.fields | {"from_labels", "to_labels", "arity"}

    # in theory we can have a selector function here too so to_labels is actually bigger?
    # means we have multiple children?

    def __init__(self, from_labels, to_labels, arity, **kwargs):
        self.from_labels = from_labels
        self.to_labels = to_labels
        self.arity = arity
        self.selector = None  # TODO
        super().__init__(**kwargs)

    @property
    def size(self):
        return self.arity


class TabulatedMapNode(MapNode):
    fields = MapNode.fields | {"data"}

    def __init__(self, from_labels, to_labels, arity, data, **kwargs):
        self.data = data
        super().__init__(from_labels, to_labels, arity, **kwargs)


class IdentityMapNode(MapNode):
    pass

    # TODO is this strictly needed?
    # @property
    # def label(self):
    #     assert len(self.to_labels) == 1
    #     return self.to_labels[0]


class AffineMapNode(MapNode):
    fields = MapNode.fields | {"expr"}

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

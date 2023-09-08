# FIXME I am only creating this file to hold things that otherwise cause an import loop
# I should decide on the right place for them. Perhaps axis?

from __future__ import annotations

import abc
import functools

import pytools
from pyrsistent import pmap

from pyop3.utils import is_single_valued


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


# FIXME do I ever use this?
class IndexedLoopIterable(LoopIterable):
    """Class representing an indexed object that can be looped over.

    Examples include ``axes[map(p)]`` or ``map(p)[::2]``.

    """


class ContextSensitive(pytools.ImmutableRecord, abc.ABC):
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
    fields = {"values"}  # bad name

    def __init__(self, values):
        super().__init__()
        # this is terribly unclear
        if not is_single_valued([set(key.keys()) for key in values.keys()]):
            raise ValueError("Loop contexts must contain the same loop indices")

        assert all(isinstance(v, ContextFree) for v in values.values())

        self.values = pmap(values)

    @functools.cached_property
    def keys(self):
        # loop is used just for unpacking
        for context in self.values.keys():
            indices = set()
            for loop_index in context.keys():
                indices.add(loop_index)
            return frozenset(indices)

    def with_context(self, context):
        key = {}
        for loop_index, path in context.items():
            if loop_index in self.keys:
                key |= {loop_index: path}
        key = pmap(key)
        return self.values[key]


class ContextFree(ContextSensitive, abc.ABC):
    def with_context(self, context):
        return self

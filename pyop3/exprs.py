import functools
from typing import Tuple

import dtl
import dtlutils
import dtlpp

import dtlpp.monads


def loop(index, statements):
    statements = dtlutils.as_tuple(statements)

    function = functools.reduce(dtlpp.Monad.then, statements, dtlpp.NullState())

    return Loop(function, index.point_set, dtlpp.NullState())


class Loop:
    """A loop that acts on terminals or other loops.

    Parameters
    ----------
    args: ?
        Iterable of 'global' arguments to the loop.
    tmps: ?
        Iterable of temporaries instantiated at each invocation of the loop.
    stmts: ?
        Iterable of ordered statements executed by the loop.
    scope: ?, optional
        The plex op relating this loop to a surrounding one.
    """

    """A loop is a left fold. It takes:

        some initial state S
        an iterable of points
        a function that takes S and a point and returns a new state
    """

    def __init__(self, function: dtlpp.monads.MonadFunction, iterable, initializer=dtlpp.NullState()):
        super().__init__(function, iterable, initializer)

    @property
    def indices(self):
        return self.iterable.index,

    @property
    def point_set(self):
        return self.iterable

    def __str__(self):
        return (
            f"for {', '.join(index.name for index in self.indices)} ∊ {self.point_set}"
        )

    @property
    def input_expr(self):
        return self.initializer

    def reconstruct(self, *, input_expr=None):
        initializer = self.initializer if input_expr is None else input_expr
        return type(self)(self.function, self.iterable, initializer)

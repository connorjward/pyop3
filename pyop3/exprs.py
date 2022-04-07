from typing import Tuple

import dtl
import dtlpp

import dtlpp.monads

from pyop3 import utils


def loop(index, statements):
    statements = utils.as_tuple(statements)
    function = None
    # this must be done in reverse like this to get the correct dependencies
    # between states/statements
    for stmt in statements:
        function = stmt.bind(function)
    iterable = index.point_set

    return Loop(function, None, iterable)


class Loop(dtlpp.LeftFold, dtlpp.monads.StateMonad):
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
    def is_bound(self):
        return bool(self.initializer)

    def bind(self, monad_expr):
        # this should only be called once.
        assert not self.is_bound
        return type(self)(self.function, monad_expr, self.iterable)

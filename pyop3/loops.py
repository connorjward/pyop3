import abc
import collections.abc


def as_tuple(item):
    return tuple(item) if isinstance(item, collections.abc.Iterable) else (item,)


class Statement(abc.ABC):

    def __init__(self):
        ...


class Loop(Statement):
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
    def __init__(self, indices, statements=()):
        self.indices = as_tuple(indices)
        try:
            self.point_set, = set(index.point_set for index in self.indices)
        except ValueError:
            raise ValueError("Must use the same base point set")
        self.statements = as_tuple(statements)

    @property
    def arguments(self):
        return tuple(arg for stmt in self.statements for arg in stmt.arguments)


class Terminal(Statement):
    """A terminal operation."""


class Function:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, *args):
        return FunctionCall(self, args)


class FunctionCall(Terminal):
    def __init__(self, func, arguments):
        self.func = func
        self.arguments = arguments

    def __str__(self):
        return f"{self.func}({', '.join(map(str, self.arguments))})"


class Assign(Terminal):

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.arguments = lhs, rhs

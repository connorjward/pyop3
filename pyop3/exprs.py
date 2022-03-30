import abc

from pyop3 import utils


class Expression(abc.ABC):
    pass


# FIXME Do I need Expression and Statement?
class Statement(Expression):

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
        self.indices = utils.as_tuple(indices)
        try:
            self.point_set, = set(index.point_set for index in self.indices)
        except ValueError:
            raise ValueError("Must use the same base point set")
        self.statements = utils.as_tuple(statements)

    @property
    def arguments(self):
        return tuple(arg for stmt in self.statements for arg in stmt.arguments)

    def __str__(self):
        return f"for {', '.join(index.name for index in self.indices)} ∊ {self.point_set}"


class Terminal(Statement, abc.ABC):
    """A terminal operation."""


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


class Restrict(Statement):
    """Pack/unpack statement object"""

    # TODO This should be two-way (pack + unpack)
    def __init__(self, tensor, restriction, out):
        self._tensor = tensor
        self._restriction = restriction
        self._output = out

    def __str__(self):
        return f"{self._output} = {self._tensor}[{self._restriction}]"

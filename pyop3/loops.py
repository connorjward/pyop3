import abc
import functools

import graphviz

from pyop3 import utils


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


class Restrict(Statement):

    def __init__(self, tensor, restriction, out):
        ...


def visualize(expr, *, view=True):
    """Render loop expression as a DAG."""
    dot = graphviz.Digraph()
    _visualize(expr, dot)

    if view:
        dot.render("mything", view=True)

    return dot


@functools.singledispatch
def _visualize(expr: Statement, dot: graphviz.Digraph):
    raise AssertionError


@_visualize.register
def _(expr: Loop, dot: graphviz.Digraph):
    label = str(expr)
    dot.node(label)

    for stmt in expr.statements:
        child_label = _visualize(stmt, dot)
        dot.edge(label, child_label)
    return label


@_visualize.register
def _(expr: Restrict, dot: graphviz.Digraph):
    label = str(expr)
    dot.node(label)
    return label


@_visualize.register
def _(expr: FunctionCall, dot: graphviz.Digraph):
    label = str(expr)
    dot.node(label)
    return label

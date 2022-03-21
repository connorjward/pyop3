import collections.abc


def as_tuple(item):
    return tuple(item) if isinstance(item, collections.abc.Iterable) else (item,)


class Op:

    def __init__(self):
        ...


class Loop(Op):
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
    def __init__(self, domain_index, arguments=(), temporaries=(), statements=()):
        self.domain_index = domain_index
        self.arguments = as_tuple(arguments)
        self.temporaries = as_tuple(temporaries)
        self.statements = as_tuple(statements)


class Terminal(Op):
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

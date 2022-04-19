import abc
import dataclasses
import enum
import functools
from typing import Iterable, Tuple

import dtl
import dtlutils
import dtlpp

import dtlpp.monads


def loop(index, statements):
    statements = dtlutils.as_tuple(statements)

    function = functools.reduce(dtlpp.Monad.then, statements, dtlpp.NullState())

    return Loop(function, index.point_set, dtlpp.NullState())


class Expr:
    pass


class Operator(Expr):
    pass


class Loop(Operator):
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

    def __init__(self, index, statements):
        self.index = index
        self.statements = statements

    def __str__(self):
        return (
            f"for {self.index} ∊ {self.index.point_set}"
        )

    def reconstruct(self, *, input_expr=None):
        raise NotImplementedError
        initializer = self.initializer if input_expr is None else input_expr
        return type(self)(self.function, self.iterable, initializer)


class AccessDescriptor(enum.Enum):

    READ = enum.auto()
    WRITE = enum.auto()
    INC = enum.auto()


@dataclasses.dataclass(frozen=True)
class ArgumentSpec:

    access: AccessDescriptor
    space: Tuple[int]


class Function:

    def __init__(self, name: str, argspec: Iterable[ArgumentSpec]):
        self.name = name
        self.argspec = argspec

    def __call__(self, *args):
        return FunctionCall(self, args)


class Terminal(Expr):
    pass


class FunctionCall(Terminal):

    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

    @property
    def name(self):
        return self.function.name

    @property
    def argspec(self):
        return self.function.argspec

    @property
    def inputs(self):
        return tuple(arg for arg, spec in zip(self.arguments, self.argspec)
                     if spec.access == AccessDescriptor.READ)

    @property
    def outputs(self):
        return tuple(arg for arg, spec in zip(self.arguments, self.argspec)
                     if spec.access in {AccessDescriptor.WRITE, AccessDescriptor.INC})

    @property
    def output_specs(self):

        return tuple(
            filter(
                lambda spec: spec.access in {AccessDescriptor.WRITE, AccessDescriptor.INC},
                self.argspec
            )
        )


class TensorList(dtl.Node):

    @abc.abstractmethod
    def __len__(self):
        pass


# TODO Move this to DTL
class DTLFunctionCall(TensorList):

    def __init__(self, function, inputs):
        self.function = function
        self.inputs = inputs

    @property
    def hashkey(self):
        return self.function, self.inputs

    @property
    def operands(self):
        return self.inputs

    def __len__(self):
        return len(self.function.outputs)


class Take(dtl.TensorExpr):

    def __init__(self, tlist: TensorList, index: int):
        self.tlist = tlist
        self.index = index

    @property
    def tensor(self):
        return self.tlist.outputs[self.index]

    @property
    def hashkey(self):
        return self.tlist, self.index

    @property
    def indices(self):
        return self.tensor.indices

    @property
    def space(self):
        return self.tensor.space

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def operands(self):
        return self.tlist,

    def __str__(self):
        return str(self.tensor)


class FunctionOutput(dtl.TensorExpr):

    def __init__(self, function: FunctionCall, indices, shape, space):
        self.function = function
        self._indices = indices
        self._shape = shape
        self._space = space

    @property
    def hashkey(self):
        return type(self), self.function, self.indices, self.shape, self.space

    @property
    def indices(self):
        return self._indices

    @property
    def shape(self):
        return self._shape

    @property
    def space(self):
        return self._space

    @property
    def operands(self):
        return (self.function,)

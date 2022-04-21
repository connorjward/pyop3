import abc
import dataclasses
import enum
import functools
from typing import Iterable, Tuple

import dtl
import dtlutils
import dtlpp

import numpy as np


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

    @property
    def set(self):
        return self.index.set

    def __str__(self):
        return f"for {self.index} ∊ {self.index.point_set}"

    def reconstruct(self, *, input_expr=None):
        raise NotImplementedError
        initializer = self.initializer if input_expr is None else input_expr
        return type(self)(self.function, self.iterable, initializer)


class AccessDescriptor(enum.Enum):

    READ = enum.auto()
    WRITE = enum.auto()
    INC = enum.auto()
    RW = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()


READ = AccessDescriptor.READ
WRITE = AccessDescriptor.WRITE
INC = AccessDescriptor.INC
RW = AccessDescriptor.RW
MIN = AccessDescriptor.MIN
MAX = AccessDescriptor.MAX


@dataclasses.dataclass(frozen=True)
class ArgumentSpec:

    access: AccessDescriptor
    dtype: np.dtype
    space: Tuple[int]


class Function:
    def __init__(self, name: str, argspec: Iterable[ArgumentSpec], loopy_kernel):
        self.name = name
        self.argspec = argspec
        self.loopy_kernel = loopy_kernel

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
        return tuple(
            arg
            for arg, spec in zip(self.arguments, self.argspec)
            if spec.access == AccessDescriptor.READ
        )

    @property
    def outputs(self):
        return tuple(
            arg
            for arg, spec in zip(self.arguments, self.argspec)
            if spec.access in {AccessDescriptor.WRITE, AccessDescriptor.INC}
        )

    @property
    def output_specs(self):

        return tuple(
            filter(
                lambda spec: spec.access
                in {AccessDescriptor.WRITE, AccessDescriptor.INC},
                self.argspec,
            )
        )

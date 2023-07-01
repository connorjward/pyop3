"""Intermediate tensor language lowered to from pyop3.exprs.

This language is useful for breaking apart the accessor parts of the language
from the tensor evaluation components.

We can also then generically traverse this DAG to either build the loopy code
or retrieve the right data pointers for calling the thing.
"""

import abc
import itertools
from typing import Any

import pytools


class Instruction(pytools.ImmutableRecord):
    fields = set()


class Assignment(Instruction):
    fields = Instruction.fields | {"tensor", "temporary", "shape"}

    def __init__(self, tensor, temporary, shape, **kwargs):
        self.tensor = tensor
        self.temporary = temporary
        self.shape = shape
        super().__init__(**kwargs)

    # better name
    @property
    def array(self):
        return self.tensor


class Read(Assignment):
    @property
    def lhs(self):
        return self.temporary

    @property
    def rhs(self):
        return self.tensor


class Write(Assignment):
    @property
    def lhs(self):
        return self.tensor

    @property
    def rhs(self):
        return self.temporary


class Increment(Assignment):
    @property
    def lhs(self):
        return self.tensor

    @property
    def rhs(self):
        return self.temporary


class Zero(Assignment):
    @property
    def lhs(self):
        return self.temporary

    # FIXME
    @property
    def rhs(self):
        # return 0
        return self.tensor


class FunctionCall(Instruction):
    fields = Instruction.fields | {"function", "reads", "writes"}

    def __init__(self, function, reads, writes, **kwargs):
        self.function = function
        self.reads = reads
        self.writes = writes
        super().__init__(**kwargs)

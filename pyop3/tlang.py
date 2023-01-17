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

from pyop3.utils import NameGenerator


class Instruction(pytools.ImmutableRecord):
    fields = {"id", "depends_on"}

    def __init__(self, *, id=None, depends_on=frozenset()):
        # import pdb; pdb.set_trace()
        if not id:
            id = self.id_generator.next()

        self.id = id
        self.depends_on = depends_on
        super().__init__()


class Assignment(Instruction):
    fields = Instruction.fields | {"tensor", "temporary", "indices"}

    def __init__(self, tensor, temporary, indices, **kwargs):
        self.tensor = tensor
        self.temporary = temporary
        self.indices = indices
        super().__init__(**kwargs)


class Read(Assignment):
    id_generator = NameGenerator("read")

    @property
    def lhs(self):
        return self.temporary

    @property
    def rhs(self):
        return self.tensor


class Write(Assignment):
    id_generator = NameGenerator("write")

    @property
    def lhs(self):
        return self.tensor

    @property
    def rhs(self):
        return self.temporary


class Increment(Write):
    id_generator = NameGenerator("inc")


class Zero(Assignment):
    id_generator = NameGenerator("zero")

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
    id_generator = NameGenerator("func")

    def __init__(self, function, reads, writes, **kwargs):
        self.function = function
        self.reads = reads
        self.writes = writes
        super().__init__(**kwargs)

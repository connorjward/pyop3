"""Intermediate tensor language lowered to from pyop3.exprs.

This language is useful for breaking apart the accessor parts of the language
from the tensor evaluation components.

We can also then generically traverse this DAG to either build the loopy code
or retrieve the right data pointers for calling the thing.
"""

import abc
import itertools
from typing import Any


class Instruction:
    def __init__(self, *, id=None, depends_on=frozenset(), within_indices=frozenset()):
        self.id = id
        self.depends_on = depends_on
        self.within_indices = within_indices

class Assignment(Instruction):
    _count = itertools.count()
    def __init__(self, tensor, temporary, **kwargs):
        self.tensor = tensor
        self.temporary = temporary
        self.id = f"{self.prefix}{next(self._count)}"
        super().__init__(**kwargs)

    def parse_kwargs(self, **kwargs):
        return dict(
            assignee=kwargs.pop("assignee", self.assignee),
            expression=kwargs.pop("expression", self.expression),
            **super().parse_kwargs(**kwargs)
        )


class Read(Assignment):
    prefix = "read"


class Write(Assignment):
    prefix = "write"


class Increment(Assignment):
    prefix = "inc"


class Zero(Assignment):
    prefix = "zero"


class FunctionCall(Instruction):
    def __init__(self, function, reads, writes, **kwargs):
        self.function = function
        self.reads = reads
        self.writes = writes
        super().__init__(**kwargs)

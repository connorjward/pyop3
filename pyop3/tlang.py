"""Intermediate tensor language lowered to from pyop3.exprs.

This language is useful for breaking apart the accessor parts of the language
from the tensor evaluation components.

We can also then generically traverse this DAG to either build the loopy code
or retrieve the right data pointers for calling the thing.
"""

import abc


class Expression(abc.ABC):
    def __init__(self, children, id):
        self.children = children
        self.id = id

    def copy(self, **kwargs):
        return type(self)(**self.parse_kwargs(**kwargs))

    def parse_kwargs(self, **kwargs):
        children = kwargs.pop("children", self.children)
        id = kwargs.pop("id", self.id)

        if kwargs:
            raise ValueError

        return dict(children=children, id=id)


class Assignment(Expression):
    def __init__(self, assignee, expression, **kwargs):
        self.assignee = assignee
        self.expression = expression
        super().__init__(**kwargs)

    def parse_kwargs(self, **kwargs):
        return dict(
            assignee=kwargs.pop("assignee", self.assignee)
            expression=kwargs.pop("expression", self.expression)
            **super().parse_kwargs(**kwargs)
        )


class Read(Assignment):
    pass


class Write(Assignment):
    pass


class Increment(Assignment):
    pass


class Zero(Assignment):
    def __init__(self, tensor, **kwargs):
        super().__init__(tensor, 0, **kwargs)


class Loop(Expression):
    ...


class FunctionCall(Expression):
    def __init__(self, function, reads, writes):
        self.function = function
        self.reads = reads
        self.writes = writes


def apply(func, expr: Expression):
    return func(expr).copy(children=tuple(apply(func, child) for child in expr.children))

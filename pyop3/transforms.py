import functools
import itertools

import pyop3.arguments
import pyop3.exprs


def replace_restricted_tensors(expr: pyop3.exprs.Expression):
    """Replace non-temporary arguments to functions.

    (with temporaries and explicit packing instructions.)
    """
    temp_namer = pyop3.utils.NameGenerator(prefix="t")
    (expr,) = _replace_restricted_tensors(expr, temporary_namer=temp_namer)
    return expr


@functools.singledispatch
def _replace_restricted_tensors(expr: pyop3.exprs.Expression, **kwargs):
    raise AssertionError


@_replace_restricted_tensors.register
def _(expr: pyop3.exprs.Loop, **kwargs):
    statements = ()
    for stmt in expr.statements:
        statements += _replace_restricted_tensors(stmt, **kwargs)
    return (type(expr)(expr.indices, statements),)


@_replace_restricted_tensors.register
def _(expr: pyop3.exprs.Restrict, **kwargs):
    return (expr,)


@_replace_restricted_tensors.register
def _(expr: pyop3.exprs.FunctionCall, *, temporary_namer: pyop3.utils.NameGenerator):
    statements = []
    arguments = []

    for arg in expr.arguments:
        if isinstance(arg, pyop3.arguments.RestrictedTensor):
            tmp = pyop3.arguments.Dat(next(temporary_namer))
            statements.append(pyop3.exprs.Restrict(arg.parent, arg.restriction, tmp))
            arguments.append(tmp)
        else:
            arguments.append(arg)
    statements.append(type(expr)(expr.func, arguments))
    return tuple(statements)


@functools.singledispatch
def replace_restrictions_with_loops(expr):
    """Replace Restrict nodes (which correspond to linear transformations) with explicit loops."""
    raise AssertionError


@replace_restrictions_with_loops.register(pyop3.Loop)
def _(expr):
    raise NotImplementedError
    return type(self)


@replace_restrictions_with_loops.register
def _(expr: pyop3.Restrict):
    # check if indexed by a single index (and hence no loop is needed)
    if isinstance(arg.point_set, domains.Index):
        return ""

    # TODO I don't think that we can treat this the same as other loops since
    # we loop over pts AND integers.
    tmp = ctx.temporaries[arg]

    loop = loops.Loop(
        [i := arg.point_set.count_index, p := arg.point_set.point_index],
        tmp[i].assign(arg[p]),
    )

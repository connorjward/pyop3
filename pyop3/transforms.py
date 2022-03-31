import functools
import itertools

import npdtl

import pyop3.arguments
import pyop3.exprs


def replace_restricted_tensors(expr: pyop3.exprs.Expression):
    """Replace restricted tensors with a DTL expression."""
    return _replace_restricted_tensors(expr)


@functools.singledispatch
def _replace_restricted_tensors(expr: pyop3.exprs.Expression, **kwargs):
    raise AssertionError


@_replace_restricted_tensors.register
def _(expr: pyop3.exprs.Loop):
    statements = ()
    for stmt in expr.statements:
        statements += _replace_restricted_tensors(stmt, **kwargs)
    return type(expr)(expr.indices, statements)


@_replace_restricted_tensors.register
def _(expr: pyop3.exprs.FunctionCall):
    arguments = []

    for arg in expr.arguments:
        if isinstance(arg, pyop3.arguments.RestrictedTensor):
            """
            replace with a TensorExpr
            UnitSpace is a vector space containing a single one and otherwise zeros
            what about if the space doesn't contain a zero? that should also be valid
            """
            tspace = arg.parent.space * npdtl.UnitSpace() * npdtl.UnitSpace()
            permutation_tensor = dtl.TensorVariable("S", tspace)
            p, i, j = dtl.indices("p", "i", "j")
            arg = (permutation_tensor[p, i, j] * arg.parent[p]).forall(i, j)
            arguments.append(tmp)
        else:
            arguments.append(arg)
    return type(expr)(expr.func, arguments)

import dataclasses
import functools
from typing import Set

import dtl
import dtlpp
import dtlc.backends.pseudo

import pyop3.exprs

from pyop3.exprs import AccessDescriptor


def lower(expr):
    # lower an expression to DTL
    context = _CodegenContext()
    expr = _lower(expr, context)

    # expr = merge_outputs()

    return dtlc.backends.pseudo.lower(context.roots.pop())


@dataclasses.dataclass
class _CodegenContext:

    within_indices: Set[dtl.Index] = dataclasses.field(default_factory=set)
    roots: Set[dtl.Node] = dataclasses.field(default_factory=set)



@functools.singledispatch
def _lower(expr):
    raise AssertionError


@_lower.register
def _(expr: pyop3.exprs.Loop, context: _CodegenContext):
    context.within_indices.add(expr.index)

    for stmt in expr.statements:
        _lower(stmt, context)

    context.within_indices.remove(expr.index)


@_lower.register
def _(expr: pyop3.exprs.FunctionCall, context: _CodegenContext):
    function = pyop3.exprs.DTLFunctionCall(expr.function, expr.inputs)
    outputs = [pyop3.exprs.FunctionOutput(function, output.indices, output.shape, output.space) for output in expr.outputs]

    for expr_out, temp_out in zip(expr.outputs, outputs):
        # result[j] = (S[i, j, k] * t[i, k]).forall(j)
        # where i is over all cells, j over all points, and k over the arity
        if isinstance(expr_out.expr, dtl.MulBinOp):
            # this assume INC
            restriction, dat =  expr_out.expr.operands  # we discard dat for now
            i, j, k = restriction.indices
            root = (restriction * temp_out[i, k]).forall(j)
        else:
            # for output to dat[p] just a .forall(p) should be sufficient
            raise NotImplementedError
        context.roots.add(root)

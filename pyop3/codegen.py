import dataclasses
import functools
from typing import Set

import dtl
import dtlpp
import dtlc.backends.pseudo
import loopy as lp

import pyop3.exprs

from pyop3.exprs import AccessDescriptor


def lower(expr):
    context = _CodegenContext()
    expr = _lower(expr, context)
    return dtlc.backends.pseudo.lower(context.roots.pop())


def to_loopy(expr):
    context = ...
    kernels = _build_loopy_kernels(expr, context)
    return lp.fuse_kernels(kernels, data_flow=context.dependencies)


def _build_loopy_kernels(expr, context):
    raise NotImplementedError


@_build_loopy_kernels.register
def _(expr: Loop, context):
    context.within_inames.add(expr.index.name)


@_build_loopy_kernels.register
def _(expr: FunctionCall, context):
    function = lp.Callable()  # or something

    reads = [
        read_tensor(arg) for arg, spec in zip(expr.arguments, expr.argspec)
        if spec.access in {READ}
    ]

    # used for data_flow in fuse_kernels
    context.dependencies += [(arg.name, reads[arg], function) for ...]

    writes += [
        write_tensor(arg) for arg, spec in zip(expr.arguments, expr.argspec)
        if spec.access in {WRITE}
    ]


def read_tensor(tensor: IndexedTensor):
    # convert the pyop3 tensor to a DTL tensor expression
    # this is normally just len 1 since we mostly have dats
    if len(tensor.indices) != 1:
        raise NotImplementedError

    restriction = functools.reduce(operator.mul, (index.set.dtl_expr for index in tensor.indices))

    # we contract over the last index of dtl_expr as this makes things easy
    # to keep track of
    dtl_expr = restriction * tensor.dtl_tensor[tuple(index.set.dtl_expr.indices[-1]
                                               for index in expr.indices)]

    return dtlc.lower(dtl_expr, target=dtlc.targets.LOOPY)


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

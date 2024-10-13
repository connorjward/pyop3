import functools
import numbers
from typing import Any

from pyrsistent import pmap

from pyop3.array import HierarchicalArray
from pyop3.axtree.tree import AxisVar, Operator, Add, Mul
from pyop3.itree.tree import LoopIndexVar
from pyop3.utils import OrderedSet


# TODO:
# it makes more sense to traverse the loop expression and collect buffers
# (kernel arguments) there!!!
# This avoid the problem of carrying extra arguments about.
# Also it requires a 3-way tree traversal: first the loop expression, then the axis tree,
# then the subst_layouts()
# It would obvs. be nice to have generic traversal code for all of these cases, but
# I can do that later. For now do the naive datamap approach


@functools.singledispatch
def collect_datamap(expr):
    raise TypeError()


@collect_datamap.register(HierarchicalArray)
def _(dat: HierarchicalArray, /):
    return dat.datamap


@collect_datamap.register(Operator)
def _(op: Operator, /):
    return collect_datamap(op.a) | collect_datamap(op.b)


@collect_datamap.register(LoopIndexVar)
def _(loop_var: LoopIndexVar, /):
    return loop_var.index.datamap


@collect_datamap.register(AxisVar)
def _(var: AxisVar, /):
    return pmap()


@collect_datamap.register(numbers.Number)
def _(num: numbers.Number, /):
    return pmap()

# @functools.singledispatch
# def collect_arrays(expr: Any) -> OrderedSet:
#     raise TypeError(f"No handler defined for {type(expr).__name__}")
#
#
# @collect_arrays.register(HierarchicalArray)
# def _(dat: HierarchicalArray) -> OrderedSet:
#     return OrderedSet([dat]) | collect_arrays(dat.subst_layouts())
#
#
# @collect_arrays.register(Operator)
# def _(op: Operator) -> OrderedSet:
#     return collect_arrays(op.a) | collect_arrays(op.b)
#
#
# @collect_arrays.register(AxisVar)
# def _(var: AxisVar) -> OrderedSet:
#     return OrderedSet()
#
#
# @collect_arrays.register(numbers.Number)
# def _(num: numbers.Number) -> OrderedSet:
#     return OrderedSet()



# TODO: could make a postvisitor
@functools.singledispatch
def evaluate(expr: Any, *args, **kwargs):
    raise TypeError


@evaluate.register
def _(dat: HierarchicalArray, indices):
    if not dat.axes.is_linear:
        # guess this is optional at the top level, extra kwarg?
        raise NotImplementedError
    else:
        path = dat.axes.path(dat.axes.leaf)
    offset = evaluate(dat.axes.subst_layouts()[path], indices)
    return dat.buffer.data_ro_with_halos[offset]


@evaluate.register
def _(expr: Add, *args, **kwargs):
    return evaluate(expr.a, *args, **kwargs) + evaluate(expr.b, *args, **kwargs)


@evaluate.register
def _(mul: Mul, *args, **kwargs):
    return evaluate(mul.a, *args, **kwargs) * evaluate(mul.b, *args, **kwargs)


@evaluate.register
def _(num: numbers.Number, *args, **kwargs):
    return num


@evaluate.register
def _(var: AxisVar, indices):
    return indices[var.axis_label]

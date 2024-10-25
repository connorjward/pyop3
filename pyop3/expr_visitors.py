import functools
import numbers
from typing import Any

from pyrsistent import pmap

from pyop3.array import Array, Dat
from pyop3.array.petsc import AbstractMat
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


# NOTE: should get rid of .datamap entirely
@collect_datamap.register(Dat)
def _(dat: Dat, /):
    return dat.datamap


@collect_datamap.register(AbstractMat)
def _(mat: AbstractMat, /):
    return mat.datamap


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


# TODO: could make a postvisitor
@functools.singledispatch
def evaluate(expr: Any, *args, **kwargs):
    raise TypeError


@evaluate.register
def _(dat: Dat, indices):
    if dat.transform:
        raise NotImplementedError

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


@functools.singledispatch
def collect_loops(expr: Any):
    raise TypeError


@collect_loops.register(LoopIndexVar)
def _(loop_var: LoopIndexVar):
    return OrderedSet({loop_var.index})


@collect_loops.register(AxisVar)
@collect_loops.register(numbers.Number)
def _(var):
    return OrderedSet()

@collect_loops.register(Operator)
def _(op: Operator):
    return collect_loops(op.a) | collect_loops(op.b)


@collect_loops.register(Dat)
def _(dat: Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.transform:
        loop_indices |= collect_loops(dat.transform.initial)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loops(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loops.register(AbstractMat)
def _(mat: AbstractMat, /) -> OrderedSet:
    # if mat.transform:
    #     raise NotImplementedError

    loop_indices = OrderedSet()
    for cs_axes in {mat.raxes, mat.caxes}:
        for cf_axes in cs_axes.context_map.values():
            for leaf in cf_axes.leaves:
                path = cf_axes.path(leaf)
                loop_indices |= collect_loops(cf_axes.subst_layouts()[path])
    return loop_indices


@functools.singledispatch
def restrict_to_context(obj: Any, /, loop_context):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@restrict_to_context.register(AxisVar)
@restrict_to_context.register(numbers.Number)
def _(var: Any, /, loop_context) -> Any:
    return var


@restrict_to_context.register(Operator)
def _(op: Operator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context))


@restrict_to_context.register(Array)
def _(array: Array, /, loop_context):
    return array.with_context(loop_context)

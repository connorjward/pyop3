import collections
import functools
import itertools
import numbers
from collections.abc import Mapping
from typing import Any, Optional

from immutabledict import ImmutableOrderedDict
from pyrsistent import pmap, PMap

from pyop3.array import Array, Dat, _ExpressionDat
from pyop3.array.petsc import AbstractMat
from pyop3.axtree.tree import AxisVar, Expression, Operator, Add, Mul, BaseAxisTree, IndexedAxisTree, AxisTree, Axis, LoopIndexVar, merge_axis_trees, ExpressionT, Terminal
from pyop3.utils import OrderedSet, merge_dicts, just_one


# TODO: could make a postvisitor
@functools.singledispatch
def evaluate(obj: Any, /, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@evaluate.register
def _(dat: Dat, indices):
    if dat.parent:
        raise NotImplementedError

    if not dat.axes.is_linear:
        # guess this is optional at the top level, extra kwarg?
        raise NotImplementedError
    else:
        path = dat.axes.path(dat.axes.leaf)
    offset = evaluate(dat.axes.subst_layouts()[path], indices)
    return dat.buffer.data_ro_with_halos[offset]


@evaluate.register(_ExpressionDat)
def _(dat: _ExpressionDat, indices):
    offset = evaluate(dat.layout, indices)
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
def collect_loops(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


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

    if dat.parent:
        loop_indices |= collect_loops(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loops(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loops.register(_ExpressionDat)
def _(dat: _ExpressionDat, /) -> OrderedSet:
    return collect_loops(dat.layout)


@collect_loops.register(AbstractMat)
def _(mat: AbstractMat, /) -> OrderedSet:
    loop_indices = OrderedSet()
    if mat.parent:
        loop_indices |= collect_loops(mat.parent)

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


# NOTE: bad name?? something 'shape'? 'make'?
# always return an AxisTree?

# NOTE: visited_axes is more like visited_components! Only need axis labels and component information
@functools.singledispatch
def extract_axes(obj: Any, /, visited_axes) -> BaseAxisTree:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@extract_axes.register(numbers.Number)
def _(var: Any, /, visited_axes) -> AxisTree:
    return AxisTree()


@extract_axes.register(LoopIndexVar)
def _(loop_index: LoopIndexVar, /, visited_axes) -> AxisTree:
    raise NotImplementedError
    if len(collect_loops(loop_index)) > 1:
        raise NotImplementedError("Make sure to include indexed bits in the axes")

    # The idea is to return a relabelled set of axes that are unique to the
    # loop index.
    iterset = AxisTree(loop_index.index.iterset.node_map)
    return _relabel_axes(iterset, suffix=loop_index.id)


@extract_axes.register(AxisVar)
def _(var: AxisVar, /, visited_axes) -> AxisTree:
    axis, component = just_one((a, c) for a, c in visited_axes.items() if a.label == var.axis_label)
    return AxisTree(Axis(component))


@extract_axes.register(Operator)
def _(op: Operator, /, visited_axes):
    return merge_axis_trees([extract_axes(op.a, visited_axes), extract_axes(op.b, visited_axes)])


# is this needed?
# @extract_axes.register(Array)
# def _(array: Array, /, visited_axes):
#     return array.axes


@extract_axes.register(_ExpressionDat)
def _(dat, /, visited_axes):
    raise NotImplementedError
    return dat.axes


@functools.singledispatch
def relabel(obj: Any, /, suffix: str):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@relabel.register(AxisVar)
def _(var: AxisVar, /, suffix: str) -> AxisVar:
    relabelled_axis = var.axis.copy(label=var.axis.label+suffix)
    return AxisVar(relabelled_axis)


@relabel.register(Dat)
def _(dat: Dat, /, suffix: str) -> Dat:
    new_axes = _relabel_axes(dat.axes, suffix)
    # return array.with_axes(new_axes)
    return dat.reconstruct(axes=new_axes)


@functools.singledispatch
def _relabel_axes(obj: Any, suffix: str) -> BaseAxisTree:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_relabel_axes.register(AxisTree)
def _(axes: AxisTree, suffix: str) -> AxisTree:
    relabelled_node_map = _relabel_node_map(axes.node_map, suffix)
    return AxisTree(relabelled_node_map)


@_relabel_axes.register(IndexedAxisTree)
def _(axes: IndexedAxisTree, suffix: str) -> IndexedAxisTree:
    relabelled_node_map = _relabel_node_map(axes.node_map, suffix)

    # I think that I can leave unindexed the same here and just tweak the target expressions
    relabelled_targetss = tuple(
        _relabel_targets(targets, suffix)
        for targets in axes.targets
    )
    return IndexedAxisTree(relabelled_node_map, unindexed=axes.unindexed, targets=relabelled_targetss)


def _relabel_node_map(node_map: Mapping, suffix: str) -> PMap:
    relabelled_node_map = {}
    for parent, children in node_map.items():
        relabelled_children = []
        for child in children:
            if child:
                relabelled_child = child.copy(label=child.label+suffix)
                relabelled_children.append(relabelled_child)
            else:
                relabelled_children.append(None)
        relabelled_node_map[parent] = tuple(relabelled_children)
    return pmap(relabelled_node_map)


# NOTE: This only relabels the expressions. The target path is unchanged because I think that that is fine here
def _relabel_targets(targets: Mapping, suffix: str) -> PMap:
    relabelled_targets = {}
    for axis_key, (path, exprs) in targets.items():
        relabelled_exprs = {
            axis_label: relabel(expr, suffix) for axis_label, expr in exprs.items()
        }
        relabelled_targets[axis_key] = (path, relabelled_exprs)
    return pmap(relabelled_targets)


# TODO: make this a nice generic traversal
@functools.singledispatch
def replace_terminals(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace_terminals.register(Terminal)
def _(terminal: Terminal, /, replace_map) -> ExpressionT:
    return replace_map.get(terminal.terminal_key, terminal)


@replace_terminals.register(numbers.Number)
def _(var: ExpressionT, /, replace_map) -> ExpressionT:
    return var


# I don't like doing this.
@replace_terminals.register(Dat)
def _(dat: Dat, /, replace_map):
    return replace_terminals(dat._as_expression_dat(), replace_map)


@replace_terminals.register(_ExpressionDat)
def _(dat: _ExpressionDat, /, replace_map) -> _ExpressionDat:
    replaced_layout = replace_terminals(dat.layout, replace_map)
    return dat.reconstruct(layout=replaced_layout)


@replace_terminals.register(Operator)
def _(op: Operator, /, replace_map) -> Operator:
    return type(op)(replace_terminals(op.a, replace_map), replace_terminals(op.b, replace_map))


@functools.singledispatch
def replace(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace.register(AxisVar)
@replace.register(LoopIndexVar)
def _(var: Any, /, replace_map) -> ExpressionT:
    return replace_map.get(var, var)


@replace.register(numbers.Number)
def _(num: numbers.Number, /, replace_map) -> numbers.Number:
    return num


# I don't like doing this.
@replace.register(Dat)
def _(dat: Dat, /, replace_map):
    return replace(dat._as_expression_dat(), replace_map)


@replace.register(_ExpressionDat)
def _(dat: _ExpressionDat, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if dat in replace_map:
        return replace_map.get(dat, dat)
    else:
        replaced_layout = replace(dat.layout, replace_map)
        return dat.reconstruct(layout=replaced_layout)


@replace.register(Operator)
def _(op: Operator, /, replace_map) -> Operator:
    return type(op)(replace(op.a, replace_map), replace(op.b, replace_map))


# TODO: rename to concretize_array_accesses or concretize_arrays
@functools.singledispatch
def concretize_arrays(obj: Any, /, **kwargs) -> Expression:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_arrays.register(Dat)
def _(dat: Dat, /) -> _ExpressionDat:
    return dat.concretize()


@concretize_arrays.register(numbers.Number)
@concretize_arrays.register(AxisVar)
@concretize_arrays.register(LoopIndexVar)
def _(var: Any, /) -> Any:
    return var


@concretize_arrays.register(Operator)
def _(op: Operator, /) -> Operator:
    return type(op)(concretize_arrays(op.a), concretize_arrays(op.b))


# should inherit from _Dat
class _CompositeDat:
    def __init__(self, expr):
        self.expr = expr

    @property
    def axes(self):
        return self.expr.axes


INDIRECTION_PENALTY_FACTOR = 5


@functools.singledispatch
def compress_indirection_maps(obj: Any, /, visited_axes) -> tuple:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@compress_indirection_maps.register(AxisVar)
@compress_indirection_maps.register(LoopIndexVar)
@compress_indirection_maps.register(numbers.Number)
def _(var: Any, /, visited_axes) -> tuple:
    return ((var, 0),)


@compress_indirection_maps.register(Operator)
def _(op: Operator, /, visited_axes) -> tuple:
    a_result = compress_indirection_maps(op.a, visited_axes)
    b_result = compress_indirection_maps(op.b, visited_axes)

    candidates = []
    for (a_expr, a_cost), (b_expr, b_cost) in itertools.product(a_result, b_result):
        candidate_expr = type(op)(a_expr, b_expr)
        candidate_cost = a_cost + b_cost
        candidates.append((candidate_expr, candidate_cost))

    # Now also include a candidate representing the packing of the expression
    # into a Dat. The cost for this is simply the size of the resulting array.
    compressed_expr = _CompositeDat(op)
    compressed_cost = extract_axes(op, visited_axes).size
    candidates.append((compressed_expr, compressed_cost))

    return tuple(candidates)


@compress_indirection_maps.register(_ExpressionDat)
def _(dat: _ExpressionDat, /, visited_axes) -> tuple:
    candidates = []
    for layout_expr, layout_cost in compress_indirection_maps(dat.layout, visited_axes):
        candidate_expr = _ExpressionDat(dat.dat, layout_expr)
        # TODO: Undo this once I am confident things are being calculated correctly
        # candidate_cost = dat.dat.axes.size + layout_cost * INDIRECTION_PENALTY_FACTOR
        candidate_cost = dat.dat.axes.size + layout_cost
        candidates.append((candidate_expr, candidate_cost))

    candidates.append((_CompositeDat(dat), dat.axes.size))
    return tuple(candidates)

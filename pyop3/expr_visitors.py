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
from pyop3.axtree.tree import AxisVar, Expression, Operator, Add, Mul, BaseAxisTree, IndexedAxisTree, AxisTree, Axis, LoopIndexVar, merge_axis_trees
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


@functools.singledispatch
def extract_axes(obj: Any, /) -> BaseAxisTree:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@extract_axes.register(numbers.Number)
def _(var: Any, /) -> AxisTree:
    return AxisTree()


@extract_axes.register(LoopIndexVar)
def _(loop_index: LoopIndexVar, /) -> AxisTree:
    if len(collect_loops(loop_index)) > 1:
        raise NotImplementedError("Make sure to include indexed bits in the axes")
    return AxisTree(loop_index.index.iterset.node_map)


@extract_axes.register(AxisVar)
def _(var: Any, /) -> AxisTree:
    return var.axis.as_tree()


@extract_axes.register(Operator)
def _(op: Operator, /):
    # ick, move logic here
    return op.axes


@extract_axes.register(Array)
def _(array: Array, /):
    return array.axes


@extract_axes.register(_ExpressionDat)
def _(dat):
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
def replace(obj: Any, /, axes, paths_and_exprs):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace.register(AxisVar)
@replace.register(LoopIndexVar)
def _(var, axes, pathsandexprs):
    assert axes.is_linear
    # NOTE: The commented out bit will not work for None as part of the path
    # replace_map = merge_dicts(pathsandexprs[axis.id, component_label] for axis, component_label in axes.path_with_nodes(axes.leaf).items())
    replace_map = merge_dicts(e for _, e in pathsandexprs.values())
    return replace_map.get(var.axis_label, var)


@replace.register
def _(num: numbers.Number, axes, pathsandexprs):
    return num


# @replace.register(Dat)
# def _(array: Array, axes, paths_and_exprs):
#     from pyop3.itree.tree import compose_targets
#
#     if array.parent:
#         raise NotImplementedError
#     # breakpoint()
#     if not isinstance(array, Dat):
#         raise NotImplementedError
#
#     # NOTE: identical to index_axes()
#     new_targets = []
#     for orig_path in array.axes.paths_and_exprs:
#         target_path_and_exprs = compose_targets(array.axes, orig_path, axes, paths_and_exprs)
#         new_targets.append(target_path_and_exprs)
#
#     new_axes = IndexedAxisTree(axes.node_map, array.axes.unindexed, targets=new_targets)
#     # return array.with_axes(new_axes)
#     return array.reconstruct(axes=new_axes)

@replace.register(Dat)
def _(dat: Dat, *args):
    return replace(dat._as_expression_dat(), *args)


@replace.register(_ExpressionDat)
def _(dat: _ExpressionDat, axes, paths_and_exprs):
    replaced_layout = replace(dat.layout, axes, paths_and_exprs)
    return dat.reconstruct(layout=replaced_layout)


@replace.register
def _(op: Operator, *args):
    return type(op)(replace(op.a, *args), replace(op.b, *args))


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
def compress_indirection_maps(obj: Any, /) -> tuple:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@compress_indirection_maps.register(AxisVar)
@compress_indirection_maps.register(LoopIndexVar)
@compress_indirection_maps.register(numbers.Number)
def _(var: Any, /) -> tuple:
    return ((var, 0),)


@compress_indirection_maps.register(Operator)
def _(op: Operator, /) -> tuple:
    a_result = compress_indirection_maps(op.a)
    b_result = compress_indirection_maps(op.b)

    candidates = []
    for (a_expr, a_cost), (b_expr, b_cost) in itertools.product(a_result, b_result):
        candidate_expr = type(op)(a_expr, b_expr)
        candidate_cost = a_cost + b_cost
        candidates.append((candidate_expr, candidate_cost))

    # Now also include a candidate representing the packing of the expression
    # into a Dat. The cost for this is simply the size of the resulting array.
    candidates.append((_CompositeDat(op), op.axes.size))

    return tuple(candidates)


@compress_indirection_maps.register(_ExpressionDat)
def _(dat: _ExpressionDat, /) -> tuple:
    candidates = []
    for layout_expr, layout_cost in compress_indirection_maps(dat.layout):
        candidate_expr = _ExpressionDat(dat.dat, layout_expr)
        # candidate_cost = dat.dat.axes.size + layout_cost * INDIRECTION_PENALTY_FACTOR
        candidate_cost = dat.dat.axes.size + layout_cost
        candidates.append((candidate_expr, candidate_cost))

    breakpoint()
    candidates.append((_CompositeDat(dat), dat.axes.size))

    return tuple(candidates)


# NOTE: This is sort of a top-level function call - bad to include really
# TODO: For PETSc matrices this is unnecessary - or *always* necessary?
# @compress_indirection_maps.register(_ConcretizedDat)
# def _(dat: _ConcretizedDat, /) -> tuple:
#     layouts = {}
#     for leaf_path in dat.layouts.keys():
#         candidate_layouts = compress_indirection_maps(dat.layouts[leaf_path])
#
#         # Now choose the candidate layout with the lowest cost, breaking ties
#         # by choosing the left-most entry with a given cost.
#         chosen_layout = min(candidate_layouts, key=lambda item: item[1])[0]
#         layouts[leaf_path] = chosen_layout
#
#     return _ConcretizedDat(dat.dat, layouts)

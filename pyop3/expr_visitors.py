import collections
import functools
import itertools
import numbers
from collections.abc import Mapping
from typing import Any, Optional

from immutabledict import ImmutableOrderedDict
from pyrsistent import pmap, PMap

from pyop3.array import Array, Dat, _ExpressionDat, _ConcretizedDat, _ConcretizedMat
from pyop3.array.petsc import AbstractMat
from pyop3.axtree.tree import AxisVar, Expression, Operator, Add, Mul, BaseAxisTree, IndexedAxisTree, AxisTree, Axis, LoopIndexVar, merge_trees2, ExpressionT, Terminal, AxisComponent
from pyop3.dtypes import IntType
from pyop3.utils import OrderedSet, just_one


# should inherit from _Dat
# or at least be an Expression!
class _CompositeDat:
    def __init__(self, expr, visited_axes, loop_axes):
        self.expr = expr

        # we need to track these somehow so we can differentiate _CompositeDats
        # that are within different loops that have the same axis labels
        # perhaps better to not tie to it but instead return together as some sort of context
        self.visited_axes = visited_axes
        self.loop_axes = loop_axes

    def __hash__(self) -> int:
        return hash((self.expr,))

    def __eq__(self, other, /) -> bool:
        return type(self) is type(other) and other.expr == self.expr

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.expr!r})"

    def __str__(self) -> str:
        return f"acc({self.expr})"


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


@evaluate.register(LoopIndexVar)
def _(loop_var: LoopIndexVar):
    return OrderedSet({loop_var.index})


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
def extract_axes(obj: Any, /, visited_axes, loop_axes, cache) -> BaseAxisTree:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@extract_axes.register(numbers.Number)
def _(var: Any, /, visited_axes, loop_axes, cache) -> AxisTree:
    try:
        return cache[var]
    except KeyError:
        return cache.setdefault(var, AxisTree())


@extract_axes.register(LoopIndexVar)
def _(loop_var: LoopIndexVar, /, visited_axes, loop_axes, cache) -> AxisTree:
    try:
        return cache[loop_var]
    except KeyError:
        pass

    # replace LoopIndexVars in any component sizes with AxisVars
    loop_index_replace_map = {}
    for loop_id, iterset in loop_axes.items():
        for axis in iterset.nodes:
            loop_index_replace_map[(loop_id, axis.label)] = AxisVar(f"{axis.label}_{loop_id}")

    axis = just_one(axis for axis in loop_axes[loop_var.loop_id].nodes if axis.label == loop_var.axis_label)

    new_components = []
    for component in axis.components:
        if isinstance(component.count, numbers.Integral):
            new_component = component
        else:
            new_count_axes = extract_axes(just_one(component.count.leaf_layouts.values()), visited_axes, loop_axes, cache)
            new_count = Dat(new_count_axes, data=component.count.buffer)
            new_component = AxisComponent(new_count, component.label)
        new_components.append(new_component)
    new_axis = Axis(new_components, f"{axis.label}_{loop_var.loop_id}")
    return cache.setdefault(loop_var, new_axis.as_tree())


@extract_axes.register(AxisVar)
def _(var: AxisVar, /, visited_axes, loop_axes, cache) -> AxisTree:
    try:
        return cache[var]
    except KeyError:
        axis, component = just_one((a, c) for a, c in visited_axes.items() if a.label == var.axis_label)
        tree = AxisTree(Axis(component, label=axis.label))
        return cache.setdefault(var, tree)


@extract_axes.register(Operator)
def _(op: Operator, /, visited_axes, loop_axes, cache):
    return merge_trees2(extract_axes(op.a, visited_axes, loop_axes, cache), extract_axes(op.b, visited_axes, loop_axes, cache))


# is this needed?
# @extract_axes.register(Array)
# def _(array: Array, /, visited_axes):
#     return array.axes


@extract_axes.register(_ExpressionDat)
def _(dat, /, visited_axes, loop_axes, cache):
    return extract_axes(dat.layout, visited_axes, loop_axes, cache)


@extract_axes.register(_CompositeDat)
def _(dat, /, visited_axes, loop_axes, cache):
    assert visited_axes == dat.visited_axes
    assert loop_axes == dat.loop_axes
    return extract_axes(dat.expr, visited_axes, loop_axes, cache)


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
        return replace_map[dat]
    else:
        replaced_layout = replace(dat.layout, replace_map)
        return dat.reconstruct(layout=replaced_layout)


@replace.register(_CompositeDat)
def _(dat: _CompositeDat, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if dat in replace_map:
        return replace_map[dat]
    else:
        raise AssertionError("Not sure about this here...")
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
def _(dat: Dat, /) -> _ConcretizedDat:
    breakpoint()


@concretize_arrays.register(Mat)
def _(dat: Dat, /) -> _ConcretizedDat:
    breakpoint()


@concretize_arrays.register(numbers.Number)
@concretize_arrays.register(AxisVar)
@concretize_arrays.register(LoopIndexVar)
@concretize_arrays.register(_ConcretizedDat)
@concretize_arrays.register(_ConcretizedMat)
def _(var: Any, /) -> Any:
    return var


@concretize_arrays.register(Operator)
def _(op: Operator, /) -> Operator:
    return type(op)(concretize_arrays(op.a), concretize_arrays(op.b))


# TODO: account for non-affine accesses in arrays and selectively apply this
INDIRECTION_PENALTY_FACTOR = 5

MINIMUM_COST_TABULATION_THRESHOLD = 128
"""The minimum cost below which tabulation will not be considered.

Indirections with a cost below this are considered as fitting into cache and
so memory optimisations are ineffectual.

"""


# TODO: Return the minimum each time, not all candidates
# TODO: penalise non-affine accesses (inspect layout func and compare)
@functools.singledispatch
def collect_candidate_indirections(obj: Any, /, visited_axes, loop_axes) -> tuple:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_candidate_indirections.register(AxisVar)
@collect_candidate_indirections.register(LoopIndexVar)
@collect_candidate_indirections.register(numbers.Number)
def _(var: Any, /, visited_axes, loop_axes) -> tuple:
    return ((var, 0),)


@collect_candidate_indirections.register(Operator)
def _(op: Operator, /, visited_axes, loop_axes) -> tuple:
    a_result = collect_candidate_indirections(op.a, visited_axes, loop_axes)
    b_result = collect_candidate_indirections(op.b, visited_axes, loop_axes)

    candidates = []
    for (a_expr, a_cost), (b_expr, b_cost) in itertools.product(a_result, b_result):
        candidate_expr = type(op)(a_expr, b_expr)
        candidate_cost = a_cost + b_cost
        candidates.append((candidate_expr, candidate_cost))

    # Now also include a candidate representing the packing of the expression
    # into a Dat. The cost for this is simply the size of the resulting array.
    # Only do this when the cost is large as small arrays will fit in cache
    # and not benefit from the optimisation.
    if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
        compressed_expr = _CompositeDat(op, visited_axes, loop_axes)
        compressed_cost = extract_axes(op, visited_axes, loop_axes, {}).size
        candidates.append((compressed_expr, compressed_cost))

    return tuple(candidates)


@collect_candidate_indirections.register(_ExpressionDat)
def _(dat: _ExpressionDat, /, visited_axes, loop_axes) -> tuple:
    candidates = []
    for layout_expr, layout_cost in collect_candidate_indirections(dat.layout, visited_axes, loop_axes):
        candidate_expr = _ExpressionDat(dat.dat, layout_expr)
        # The cost of an expression dat (i.e. the memory volume) is given by...
        # Remember that the axes here described the outer loops that exist and that
        # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
        # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops
        dat_cost = extract_axes(dat.layout, visited_axes, loop_axes, cache={}).size
        # TODO: Only apply penalty for non-affine layouts
        candidate_cost = dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR
        candidates.append((candidate_expr, candidate_cost))

    if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
        compressed_cost = extract_axes(dat, visited_axes, loop_axes, {}).size
        candidates.append((_CompositeDat(dat, visited_axes, loop_axes), compressed_cost))
    return tuple(candidates)


@functools.singledispatch
def compute_indirection_cost(obj: Any, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@compute_indirection_cost.register(AxisVar)
@compute_indirection_cost.register(LoopIndexVar)
@compute_indirection_cost.register(numbers.Number)
def _(var: Any, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    return 0


@compute_indirection_cost.register(Operator)
def _(op: Operator, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    if seen_exprs_mut is not None:
        if op in seen_exprs_mut:
            return 0
        else:
            seen_exprs_mut.add(op)

    return (
        compute_indirection_cost(op.a, visited_axes, loop_axes, seen_exprs_mut, cache)
        + compute_indirection_cost(op.b, visited_axes, loop_axes, seen_exprs_mut, cache)
    )


@compute_indirection_cost.register(_ExpressionDat)
def _(dat: _ExpressionDat, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    if seen_exprs_mut is not None:
        if dat in seen_exprs_mut:
            return 0
        else:
            seen_exprs_mut.add(dat)

    # The cost of an expression dat (i.e. the memory volume) is given by...
    # Remember that the axes here described the outer loops that exist and that
    # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
    # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops
    # TODO: Add penalty for non-affine layouts
    layout_cost = compute_indirection_cost(dat.layout, visited_axes, loop_axes, seen_exprs_mut, cache)
    dat_cost = extract_axes(dat.layout, visited_axes, loop_axes, cache=cache).size
    return dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR


@compute_indirection_cost.register(_CompositeDat)
def _(dat: _CompositeDat, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    if seen_exprs_mut is not None:
        if dat in seen_exprs_mut:
            return 0
        else:
            seen_exprs_mut.add(dat)

    return extract_axes(dat.expr, visited_axes, loop_axes, cache).size


@functools.singledispatch
def materialize(obj: Any, /, *args, **kwargs) -> ExpressionT:
    raise TypeError


@materialize.register(AxisVar)
@materialize.register(LoopIndexVar)
@materialize.register(numbers.Number)
def _(var: Any, /, *args, **kwargs):
    return var


@materialize.register(Operator)
def _(op: Operator, /, *args, **kwargs) -> Operator:
    return type(op)(materialize(op.a, *args, **kwargs), materialize(op.b, *args, **kwargs))


@materialize.register(_CompositeDat)
def _(dat: _CompositeDat, /, visited_axes, loop_axes) -> _ExpressionDat:
    axes = extract_axes(dat, visited_axes, loop_axes)

    # dtype correct?
    result = Dat(axes, dtype=IntType)

    # replace LoopIndexVars in the expression with AxisVars
    loop_index_replace_map = {}
    for loop_id, iterset in loop_axes.items():
        for axis in iterset.nodes:
            loop_index_replace_map[(loop_id, axis.label)] = AxisVar(f"{axis.label}_{loop_id}")
    expr = replace_terminals(dat.expr, loop_index_replace_map)

    result.assign(expr, eager=True)

    # now put the loop indices back
    inv_map = {axis_var.axis_label: LoopIndexVar(loop_id, axis_label) for (loop_id, axis_label), axis_var in loop_index_replace_map.items()}
    layout = just_one(result.axes.leaf_subst_layouts.values())
    newlayout = replace_terminals(layout, inv_map)

    return _ExpressionDat(result, newlayout)

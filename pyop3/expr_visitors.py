import collections
import functools
import itertools
import numbers
from collections.abc import Mapping
from typing import Any, Optional

from immutabledict import ImmutableOrderedDict
from pyrsistent import pmap, PMap

from pyop3.array import Array, Dat
from pyop3.array.petsc import AbstractMat
from pyop3.axtree.tree import AxisVar, Expression, Operator, Add, Mul, BaseAxisTree, IndexedAxisTree, AxisTree, Axis, LoopIndexVar
from pyop3.utils import OrderedSet, merge_dicts, just_one


# TODO: Inherit from _Dat or something? YESSS
class _ConcretizedDat:
    """A dat with fixed layout functions.

    It cannot be indexed any further.

    This class is important for when we want to optimise the indirection maps.

    (Previously this was done lazily which inhibited optimisation).

    """
    def __init__(self, dat, layouts):
        self.dat = dat
        self.layouts = layouts

    @property
    def dtype(self):
        return self.dat.dtype

    @property
    def axes(self):
        return self.dat.axes

    @property
    def name(self):
        return self.dat.name

    @property
    def buffer(self):
        return self.dat.buffer

    @property
    def constant(self):
        return self.dat.constant


    @property
    def alloc_size(self):
        return self.dat.alloc_size

    def __add__(self, other):
        return Add(self, other)


class _LayoutDat(_ConcretizedDat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



# TODO: could make a postvisitor
@functools.singledispatch
def evaluate(expr: Any, *args, **kwargs):
    raise TypeError


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


@evaluate.register(_LayoutDat)
def _(dat: Dat, indices):
    offset = evaluate(just_one(dat.layouts), indices)
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

    if dat.parent:
        loop_indices |= collect_loops(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loops(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loops.register(_LayoutDat)
def _(dat: Dat, /) -> OrderedSet:
    return collect_loops(just_one(dat.layouts))


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


@extract_axes.register(AxisVar)
@extract_axes.register(LoopIndexVar)
@extract_axes.register(numbers.Number)
def _(var: Any, /) -> AxisTree:
    return AxisTree()


@extract_axes.register(Operator)
def _(op: Operator, /):
    return op.axes


@extract_axes.register(Array)
def _(array: Array, /):
    return array.axes


@functools.singledispatch
def relabel(obj: Any, /, suffix: str):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@relabel.register(AxisVar)
def _(var: AxisVar, /, suffix: str) -> AxisVar:
    return AxisVar(var.axis_label+suffix)


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


@replace.register(Dat)
def _(array: Array, axes, paths_and_exprs):
    from pyop3.itree.tree import compose_targets

    if array.parent:
        raise NotImplementedError
    # breakpoint()
    if not isinstance(array, Dat):
        raise NotImplementedError

    # NOTE: identical to index_axes()
    new_targets = []
    for orig_path in array.axes.paths_and_exprs:
        target_path_and_exprs = compose_targets(array.axes, orig_path, axes, paths_and_exprs)
        new_targets.append(target_path_and_exprs)

    new_axes = IndexedAxisTree(axes.node_map, array.axes.unindexed, targets=new_targets)
    # return array.with_axes(new_axes)
    return array.reconstruct(axes=new_axes)


@replace.register(_LayoutDat)
def _(array: Array, axes, paths_and_exprs):
    from pyop3.itree.tree import compose_targets

    new_layouts = []  # length 1 always??
    orig_layout = just_one(array.layouts)
    new_layout = replace(orig_layout, axes, paths_and_exprs)
    new_layouts.append(new_layout)

    # new_axes = IndexedAxisTree(axes.node_map, array.axes.unindexed, targets=new_targets)
    # return array.with_axes(new_axes)
    # return array.reconstruct(axes=new_axes)
    return _LayoutDat(array.dat, new_layouts)


@replace.register
def _(op: Operator, *args):
    return type(op)(replace(op.a, *args), replace(op.b, *args))


# TODO: rename to concretize_array_accesses or concretize_arrays
@functools.singledispatch
def concretize_layouts(obj: Any, /, **kwargs) -> Expression:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


# NOTE: the layouts are already concrete! Here we just mark the top level thing as fixed
@concretize_layouts.register(Dat)
def _(dat: Dat, /, *, topdat=True) -> _ConcretizedDat:
    layouts = {}
    if dat.axes.is_empty:
        leaf_layout = dat.axes.subst_layouts()[pmap()]
        # layouts[pmap()] = concretize_layouts(leaf_layout, topdat=False)
        layouts[pmap()] = leaf_layout
    else:
        for leaf_path in dat.axes.leaf_paths:
            leaf_layout = dat.axes.subst_layouts()[leaf_path]
            # layouts[leaf_path] = concretize_layouts(leaf_layout, topdat=False)
            layouts[leaf_path] = leaf_layout

    if topdat:
        return _ConcretizedDat(dat, layouts)
    else:
        # bleghghgh, dict in one place, list in another!
        layouts = [just_one(layouts.values())]
        return _LayoutDat(dat, layouts)


@concretize_layouts.register(numbers.Number)
@concretize_layouts.register(AxisVar)
@concretize_layouts.register(LoopIndexVar)
def _(var: Any, /, **kwargs) -> Any:
    return var


@concretize_layouts.register(Operator)
def _(op: Operator, /, **kwargs) -> Operator:
    return type(op)(concretize_layouts(op.a, **kwargs), concretize_layouts(op.b, **kwargs))


@functools.singledispatch
def compress_indirection_maps(obj: Any, /, **kwargs) -> tuple:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@compress_indirection_maps.register(AxisVar)
@compress_indirection_maps.register(LoopIndexVar)
@compress_indirection_maps.register(numbers.Number)
def _(var: Any, /, **kwargs) -> tuple:
    return ((var, 0),)


@compress_indirection_maps.register(Operator)
def _(op: Operator, /, *, topdat=True) -> tuple:
    a_result = compress_indirection_maps(op.a, topdat=topdat)
    b_result = compress_indirection_maps(op.b, topdat=topdat)

    if topdat:
        raise NotImplementedError

    candidates = []
    for (a_expr, a_cost), (b_expr, b_cost) in itertools.product(a_result, b_result):
        uncompressed = type(op)(a_expr, b_expr)
        uncompressed_cost = a_cost + b_cost

        candidates.append((uncompressed, uncompressed_cost))

        if op.axes:
            raise NotImplementedError
            compressed = CompressedDat(uncompressed)
            compressed_cost = uncompressed.axes.size

    return tuple(candidates)


@compress_indirection_maps.register(_LayoutDat)
def _(dat: _LayoutDat, /, *, topdat) -> tuple:
    assert not topdat

    result = compress_indirection_maps(just_one(dat.layouts), topdat=topdat)

    breakpoint()
    # for subcandidate in subcandidates:
    #     candidate = _ConcretizedDat(dat, 
    #     candidates[leaf_path].append(XXX)


    return _LayoutDat(dat.dat, layouts)


# TODO: For PETSc matrices this is unnecessary - or *always* necessary?
@compress_indirection_maps.register(_ConcretizedDat)
def _(dat: _ConcretizedDat, /, *, topdat=True) -> tuple:
    assert topdat

    layouts = {}
    for leaf_path in dat.layouts.keys():
        candidates = compress_indirection_maps(dat.layouts[leaf_path], topdat=False)

        # now choose the best candidate
        if len(candidates) > 1:
            raise NotImplementedError

        candidate_layout, _ = just_one(candidates)

        chosen_layout = candidate_layout
        layouts[leaf_path] = chosen_layout

    return _ConcretizedDat(dat.dat, layouts)

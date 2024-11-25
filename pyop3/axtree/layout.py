from __future__ import annotations

import collections
import functools
import itertools
import numbers
import operator
import sys
import typing
from collections import defaultdict
from typing import Optional

import numpy as np
import pymbolic as pym
from petsc4py import PETSc
from pyrsistent import PMap, freeze, pmap

from pyop3.array.harray import Dat
from pyop3.axtree.tree import (
    Axis,
    AxisComponent,
    AxisTree,
    AxisVar,
    component_number_from_offsets,
    component_offsets,
)
from pyop3.dtypes import IntType
from pyop3.utils import (
    StrictlyUniqueDict,
    as_tuple,
    single_valued,
    is_single_valued,
    strict_zip,
    just_one,
    OrderedSet,
    merge_dicts,
    strict_int,
    strictly_all,
)


class IntRef:
    """Pass-by-reference integer."""

    def __init__(self, value):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self


def make_layouts(axes: AxisTree, loop_vars) -> PMap:
    if axes.layout_axes.is_empty:
        return freeze({pmap(): 0})

    component_layouts = tabulate_again(axes.layout_axes)
    return _accumulate_axis_component_layouts(axes, component_layouts)


def tabulate_again(axes):
    offset = 0
    layouts = {}
    for region in _collect_regions(axes):
        offset = tabulate_again_inner(axes, region, offset, layouts)
    return layouts


def tabulate_again_inner(axes, region, offset, layouts, *, axis=None, parent_axes_acc=None):
    if axis is None:
        axis = axes.root
        parent_axes_acc = ()

    ragged = any(requires_external_index(axes, axis, c, region[axis, c]) for c in axis.components)

    if len(axis.components) > 1 and ragged:
        # Fixing this would require deciding what to do with the start variable, which
        # might need tabulating itself.
        raise NotImplementedError(
            "Cannot yet tabulate axes with multiple components if any of them are ragged"
        )

    layouts = {}
    for component in axis.components:
        parent_axes_acc_ = parent_axes_acc + (axis.copy(components=[component]),)

        mysubtree, mystep = _truncate_axis_tree(axes, axis, component)
        cache_key = ("tabulate_again", mysubtree)

        try:
            component_layout = axis.cache_get(cache_key)
        except KeyError:
            # 1. Constant stride
            if has_constant_step(axes, axis, component, "old"):
                step = step_size(axes, axis, component)

                # the subtree bits don't exactly align with what step_size gives us
                if step > 0:
                    assert step == mystep
                else:
                    mystep = step
                component_layout = AxisVar(axis.label)


            # 2. Ragged inside - must tabulate
            else:
                array_var, step = _tabulate_offsets(axes, axis, component, region, parent_axes_acc_)
                assert step == mystep
                component_layout = array_var

            axis.cache_set(cache_key, component_layout)

        layouts[(axis, component)] = mystep * component_layout + offset

        # TODO: should also do this for ragged but this breaks things currently
        if not ragged:
            offset += _axis_component_size(axes, axis, component, region)

        # Now traverse subaxes
        if subaxis := axes.child(axis, component):
            layouts |= tabulate_again_inner(axes, axis=subaxis, parent_axes_acc=parent_axes_acc_)

    return pmap(layouts), offset


def _collect_regions(axes: AxisTree, *, axis: Axis | None = None):
    """
    Examples
    --------

    Axis({"x": ["A", "B"], "y": ["B"]}, "a") -> [{"a": "A"}, {"a": "B"}]

    Axis({"x": ["A", "B"], "y": ["B", "A"]}, "a") -> [{"a": "A"}, {"a": "B"}]

    Axis({"x": ["A", "B"], "y": ["C"]}, "a") -> [{"a": "A"}, {"a": "B"}, {"a": "C"}]

    nested

    (assumes y is shared)
    Axis({"x": ["A", "B"]}, "a") / Axis({"y": ["X", "Y"]}, "b")

        ->

    [{"a": "A", "b": "X"}, {"a": "A", "b": "Y"}, {"a": "B", "b": "X"}, {"a": "B", "b": "Y"}]

    (if y is not shared)
    Axis({"x": ["A", "B"]}, "a") / Axis({"y": ["X", "Y"]}, "b")

        ->

    [{"a": "A", "b": "X"}, {"a": "A", "b": "Y"}, {"a": "B", "b": "X"}, {"a": "B", "b": "Y"}]

    """
    if axis is None:
        axis = axes.root
        axis = typing.cast(Axis, axis)  # make the type checker happy

    merged_regions = []  # NOTE: Could be an ordered set
    for component in axis.components:
        for region in component.regions:

            merged_region = {axis: region}
            if subaxis := axes.child(axis, component):
                merged_region |= _collect_regions(axes, axis=subaxis)

            if merged_region not in merged_regions:
                merged_regions.append(merged_region)
    return tuple(merged_regions)



def _tabulate_offsets(axes, axis, component, parent_axes):
    # First we build the right data structure to store the offsets. We can find the
    # axes that we need by combining the current axis with those needed by subaxes
    # along with the count of the current component (if ragged).
    axes_iter = OrderedSet()
    for ax in _collect_offset_subaxes(axes, axis, component, visited=(axis,)):
        axes_iter.add(ax)
    partial_axes = AxisTree.from_iterable(axes_iter)

    # NOTE: This code is quite unclear (partial axes made then axes_iter further modified)
    axes_iter.add(axis)
    offset_axes = AxisTree.from_iterable(axes_iter)
    offsets = Dat(offset_axes, data=np.full(offset_axes.size, -1, dtype=IntType))

    trimmed_axes, extra_step = _truncate_axis_tree(axes, axis, component)

    # this is really bloody close - just need the Python iteration to be less rubbish
    # TODO: handle iteration over empty trees
    if partial_axes.is_empty:
        offset = 0

        for axindex in axis.iter(no_index=True):  # FIXME: Should make this single component only
            offsets.set_value(axindex.source_exprs, offset)
            offset += step_size(
                trimmed_axes,
                axis,
                component,
                indices=axindex.source_exprs,
            )
    else:
        for multiindex in partial_axes.iter():
            offset = 0

            for axindex in axis.iter({multiindex}, no_index=True):  # FIXME: Should make this single component only
                offsets.set_value(multiindex.source_exprs | axindex.source_exprs, offset)
                offset += step_size(
                    trimmed_axes,
                    axis,
                    component,
                    indices=multiindex.source_exprs|axindex.source_exprs,
                )

    # if extra_step != 1:
    #     offsets *= extra_step

    return offsets, extra_step


def _truncate_axis_tree(axis_tree, axis, component) -> tuple[AxisTree, int]:
    """Return an axis tree consisting of non-constant bits below ``axis``."""
    # NOTE: dont think I need the cache here any more
    if subaxis := axis_tree.child(axis, component):
        subtree = axis_tree.subtree(subaxis)

        key = ("_truncate_axis_tree", subtree)
        try:
            return axis.cache_get(key)
        except KeyError:
            pass

        results = _truncate_axis_tree_rec(subtree, subaxis)

        # The best result has the largest step size (as the resulting tree is
        # smaller and therefore more generic).
        trimmed_tree, step = max(results, key=lambda result: result[1])

        # add current axis
        # tree = AxisTree(axis.copy(components=[component]))
        tree = AxisTree(axis)
        tree = tree.add_subtree(trimmed_tree, axis, component)

        best_result = (tree, step)

        # setdefault?
        axis.cache_set(key, best_result)
        return best_result
    else:
        # return AxisTree(axis.copy(components=[component])), 1
        return AxisTree(axis), 1

def _truncate_axis_tree_rec(axis_tree, axis) -> tuple[tuple[AxisTree, int]]:
    # NOTE: Do a post-order traversal. Need to look at the subaxes before looking
    # at this one.
    candidates_per_component = []
    for component in axis.components:
        if subaxis := axis_tree.child(axis, component):
            candidates = _truncate_axis_tree_rec(axis_tree, subaxis)
        else:
            # there is nothing below here, cannot truncate anything
            candidates = ((AxisTree(), 1),)
        candidates_per_component.append(candidates)

    axis_candidates = []
    for component_candidates in itertools.product(*candidates_per_component):
        subaxis_trees, substeps = map(tuple, zip(*component_candidates))

        # We can only consider cases where all candidates have the same step.
        try:
            substep = single_valued(substeps)
        except ValueError:
            continue

        # The new candidate consists of the per-component subtrees stuck on to the
        # current axis.
        candidate_axis_tree = AxisTree(axis)
        for component, subtree in strict_zip(axis.components, subaxis_trees):
            candidate_axis_tree = candidate_axis_tree.add_subtree(subtree, axis, component)
        axis_candidate = (candidate_axis_tree, substep)
        axis_candidates.append(axis_candidate)

    # Lastly, we can also consider the case where the entire subtree (at this
    # point) is dropped. This is only valid for constant-sized axes.
    if all(has_fixed_size(axis_tree, axis, c, "any") for c in axis.components):
        step = _axis_size(axis_tree, axis)
        axis_candidate = (AxisTree(), step)
        axis_candidates.append(axis_candidate)

    return tuple(axis_candidates)


def _collect_offset_subaxes(axes, axis, component, *, visited):
    if not isinstance(component.count, numbers.Integral):
        axes_iter = [ax for ax in sorted(component.count.axes.nodes, key=lambda ax: ax.id) if ax not in visited]
    else:
        axes_iter = []

    # needed since we don't care about "internal" axes here
    visited_ = visited + (axis,)

    if subaxis := axes.child(axis, component):
        for subcomponent in subaxis.components:
            subaxes = _collect_offset_subaxes(axes, subaxis, subcomponent, visited=visited_)
            for ax in subaxes:
                if ax not in axes_iter:
                    axes_iter.append(ax)

    return tuple(axes_iter)




    """
    There are two conditions that we need to worry about:
        1. does the axis have a fixed size (not ragged)?
            If so then we should emit a layout function and handle any inner bits.
            We don't need any external indices to fully index the array. In fact,
            if we were the use the external indices too then the resulting layout
            array would be much larger than it has to be (each index is basically
            a new dimension in the array).

        2. Does the axis have fixed size steps?

        If we have constant steps then we should index things using an affine layout.

    Care needs to be taken with the interplay of these options:

        fixed size x fixed step : affine - great
        fixed size x variable step : need to tabulate with the current axis and
                                     everything below that isn't yet handled
        variable size x fixed step : emit an affine layout but we need to tabulate above
        variable size x variable step : add an axis to the "count" tree but do nothing else
                                        not ready for tabulation as not fully indexed

    We only ever care about axes as a whole. If individual components are ragged but
    others not then we still need to index them separately as the steps are still not
    a fixed size even for the non-ragged components.
    """


def has_constant_step(axes: AxisTree, axis, cpt, inner_loop_vars, path=pmap()):
    if len(axis.components) > 1 and len(cpt.regions) > 1:
        # must interleave
        return False

    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if subaxis := axes.child(axis, cpt):
        return all(
            # not size_requires_external_index(axes, subaxis, c, path | {axis.label: cpt.label})
            not size_requires_external_index(axes, subaxis, c, inner_loop_vars, path)
            for c in subaxis.components
        )
    else:
        return True


def has_fixed_size(axes, axis, component, inner_loop_vars):
    return not size_requires_external_index(axes, axis, component, inner_loop_vars)


def requires_external_index(axtree, axis, component_index, region):
    """Return `True` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index_region(
        axtree, axis, component_index, region, set()
    )


def size_requires_external_index(axes, axis, component, inner_loop_vars, path=pmap()):
    return any(
        size_requires_external_index_region(axes, axis, component, region, inner_loop_vars, path=path)
        for region in component.regions
    )


def size_requires_external_index_region(axes, axis, component, region, inner_loop_vars, path=pmap()):
    from pyop3.array import Dat

    count = region.size
    if isinstance(count, Dat):
        if count.axes.is_empty:
            leafpath = pmap()
        else:
            leafpath = just_one(count.axes.leaf_paths)
        layout = count.axes._subst_layouts_default[leafpath]
        # no longer needed I believe
        # required_loop_vars = LoopIndexCollector(linear=False)(layout)
        # if not required_loop_vars.issubset(inner_loop_vars):
        #     return True
        # is the path sufficient? i.e. do we have enough externally provided indices
        # to correctly index the axis?
        if not count.axes.is_empty:
            for axlabel, clabel in count.axes.path(*count.axes.leaf).items():
                if axlabel in path:
                    assert path[axlabel] == clabel
                else:
                    return True

    if subaxis := axes.child(axis, component):
        for c in subaxis.components:
            # path_ = path | {subaxis.label: c.label}
            path_ = path | {axis.label: component.label}
            if size_requires_external_index(axes, subaxis, c, inner_loop_vars, path_):
                return True
    return False


def step_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    indices=pmap(),
    *,
    loop_indices=pmap(),
):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if subaxis := axes.child(axis, component):
        return _axis_size(axes, subaxis, indices, loop_indices=loop_indices)
    else:
        return 1


def has_halo(axes, axis):
    # TODO: cleanup
    return axes.comm.size > 1
    if axis.sf is not None:
        return True
    else:
        for component in axis.components:
            subaxis = axes.child(axis, component)
            if subaxis and has_halo(axes, subaxis):
                return True
        return False
    return axis.sf is not None or has_halo(axes, subaxis)


# NOTE: I am not sure that this is really required any more. We just want to
# check for loop indices in any index_exprs
# No, we need this because loop indices do not necessarily mean we need extra shape.
def collect_externally_indexed_axes(axes, axis=None, component=None, path=pmap()):
    assert False, "old code"
    from pyop3.array import Dat

    if axes.is_empty:
        return ()

    # use a dict as an ordered set
    if axis is None:
        assert component is None

        external_axes = {}
        for component in axes.root.components:
            external_axes.update(
                {
                    # NOTE: no longer axes
                    ax.id: ax
                    for ax in collect_externally_indexed_axes(
                        axes, axes.root, component
                    )
                }
            )
        return tuple(external_axes.values())

    external_axes = {}
    csize = component.count
    if isinstance(csize, Dat):
        # is the path sufficient? i.e. do we have enough externally provided indices
        # to correctly index the axis?
        loop_indices = collect_external_loops(csize.axes, csize.axes.index_exprs)
        for index in sorted(loop_indices, key=lambda i: i.id):
            external_axes[index.id] = index
    else:
        assert isinstance(csize, numbers.Integral)

    path_ = path | {axis.label: component.label}
    if subaxis := axes.child(axis, component):
        for subcpt in subaxis.components:
            external_axes.update(
                {
                    # NOTE: no longer axes
                    ax.id: ax
                    for ax in collect_externally_indexed_axes(
                        axes, subaxis, subcpt, path_
                    )
                }
            )

    return tuple(external_axes.values())


class LoopIndexCollector(pym.mapper.CombineMapper):
    def __init__(self, linear: bool):
        super().__init__()
        self._linear = linear

    def combine(self, values):
        if self._linear:
            return sum(values, start=())
        else:
            return functools.reduce(operator.or_, values, frozenset())

    def map_algebraic_leaf(self, expr):
        return () if self._linear else frozenset()

    def map_constant(self, expr):
        return () if self._linear else frozenset()

    def map_loop_index(self, index):
        rec = collect_external_loops(
            index.index.iterset, index.index.iterset.index_exprs, linear=self._linear
        )
        if self._linear:
            return rec + (index,)
        else:
            return rec | {index}

    def map_array(self, array):
        if self._linear:
            return tuple(
                item for expr in array.indices.values() for item in self.rec(expr)
            )
        else:
            return frozenset(
                {item for expr in array.indices.values() for item in self.rec(expr)}
            )


def collect_external_loops(axes, index_exprs, linear=False):
    collector = LoopIndexCollector(linear)
    keys = [None]
    if not axes.is_empty:
        nodes = (
            axes.path_with_nodes(*axes.leaf, and_components=True, ordered=True)
            if linear
            else tuple((ax, cpt) for ax in axes.nodes for cpt in ax.components)
        )
        keys.extend((ax.id, cpt.label) for ax, cpt in nodes)
    result = (
        loop
        for key in keys
        for expr in index_exprs.get(key, {}).values()
        for loop in collector(expr)
    )
    return tuple(result) if linear else frozenset(result)


def _collect_inner_loop_vars(axes: AxisTree, axis: Axis, loop_vars):
    # Terminate eagerly because axes representing loops must be outermost.
    if axis.label not in loop_vars:
        return frozenset()

    loop_var = loop_vars[axis.label]
    # Axes representing loops must be single-component.
    if subaxis := axes.child(axis, axis.component):
        return _collect_inner_loop_vars(axes, subaxis, loop_vars) | {loop_var}
    else:
        return frozenset({loop_var})


def _create_count_array_tree(
    ctree,
    loop_vars,
    axis=None,
    axes_acc=None,
    path=pmap(),
):
    if strictly_all(x is None for x in [axis, axes_acc]):
        axis = ctree.root
        axes_acc = ()

    arrays = {}
    for component in axis.components:
        path_ = path | {axis.label: component.label}
        linear_axis = Axis(component.copy(sf=None), axis.label)

        # can be None if ScalarIndex is used
        if linear_axis is not None:
            axes_acc_ = axes_acc + (linear_axis,)
        else:
            axes_acc_ = axes_acc

        if subaxis := ctree.child(axis, component):
            arrays.update(
                _create_count_array_tree(
                    ctree,
                    loop_vars,
                    subaxis,
                    axes_acc_,
                    path_,
                )
            )
        else:
            # make a multiarray here from the given sizes

            # do we have any external axes from loop indices?
            axtree = AxisTree.from_iterable(axes_acc_)

            if loop_vars:
                index_exprs = {}
                for myaxis in axes_acc_:
                    key = (myaxis.id, myaxis.component.label)
                    if myaxis.label in loop_vars:
                        loop_var = loop_vars[myaxis.label]
                        index_expr = {myaxis.label: loop_var}
                    else:
                        index_expr = {myaxis.label: AxisVar(myaxis.label)}
                    index_exprs[key] = index_expr
            else:
                index_exprs = axtree.index_exprs

            countarray = Dat(
                axtree,
                data=np.full(axtree.global_size, -1, dtype=IntType),
                prefix="offset",
            )
            arrays[path_] = countarray

    return arrays


def _tabulate_count_array_tree(
    axes,
    axis,
    loop_vars,
    count_arrays,
    offset,
    path=pmap(),  # might not be needed
    indices=pmap(),
    is_owned=True,
    setting_halo=False,
    outermost=True,
    loop_indices=pmap(),  # much nicer to combine into indices?
):
    npoints = sum(_as_int(c.count, indices) for c in axis.components)

    offsets = component_offsets(axis, indices)
    # points = axis.numbering.data_ro if axis.numbering is not None else range(npoints)
    points = range(npoints)

    counters = {c: itertools.count() for c in axis.components}
    for new_pt, old_pt in enumerate(points):
        component, _ = component_number_from_offsets(axis, old_pt, offsets)

        if component.sf is not None:
            is_owned = new_pt < component.sf.nowned

        new_strata_pt = next(counters[component])

        path_ = path | {axis.label: component.label}

        if axis.label in loop_vars:
            loop_var = loop_vars[axis.label]
            loop_indices_ = loop_indices | {loop_var.id: {loop_var.axis: new_strata_pt}}
            indices_ = indices
        else:
            loop_indices_ = loop_indices
            indices_ = indices | {axis.label: new_strata_pt}

        if path_ in count_arrays:
            if is_owned and not setting_halo or not is_owned and setting_halo:
                count_arrays[path_].set_value(
                    indices_, offset.value, loop_exprs=loop_indices_
                )
                offset += step_size(
                    axes,
                    axis,
                    component,
                    indices=indices_,
                    loop_indices=loop_indices_,
                )
        else:
            subaxis = axes.child(axis, component)
            assert subaxis
            _tabulate_count_array_tree(
                axes,
                subaxis,
                loop_vars,
                count_arrays,
                offset,
                path_,
                indices_,
                is_owned=is_owned,
                setting_halo=setting_halo,
                outermost=False,
                loop_indices=loop_indices_,
            )


def _accumulate_axis_component_layouts(
    axes: AxisTree,
    component_layouts: PMap,
    *,
    layout_axis: Optional[Axis] = None,
    layout_path: PMap=pmap(),
    path: PMap=pmap(),
    layout_expr=0,
):
    """Accumulate component-wise layouts to give a complete layout function per node.

    Parameters
    ----------
    axes
        The axis tree whose layout functions are being tabulated.
    component_layouts
        Mapping from ``(axis id, component label)`` 2-tuples to a layout expression
        for that specific node in the tree.

    Other Parameters
    ----------------
    layout_axis
        The current axis in the traversal.
    layout_path
        The current path through the layout axes (see `AxisTree.layout_axes`).
    path
        The current path through ``axes``.
    layout_expr
        The current accumulated layout expression.

    Returns
    -------
    PMap
        The accumulated layout functions per ``(axis id, component label)`` present
        in ``axes``.

    """
    if layout_axis is None:
        layout_axis = axes.layout_axes.root

    if layout_axis == axes.root:
        layouts = {pmap(): layout_expr}
    else:
        layouts = {}

    for component in layout_axis.components:
        # better not as a path here
        # layout_path_ = layout_path | {layout_axis.label: component.label}
        layout_path_ = (layout_axis, component)
        layout_expr_ = layout_expr + component_layouts.get(layout_path_, 0)

        if layout_axis in axes.nodes:
            path_ = path | {layout_axis.label: component.label}
            layouts[path_] = layout_expr_
        else:
            path_ = path

        if subaxis := axes.layout_axes.child(layout_axis, component):
            sublayouts = _accumulate_axis_component_layouts(
                axes,
                component_layouts,
                layout_axis=subaxis, layout_path=layout_path_, path=path_, layout_expr=layout_expr_
            )
            layouts.update(sublayouts)

    return freeze(layouts)


@PETSc.Log.EventDecorator()
def axis_tree_size(axes: AxisTree) -> int:
    """Return the size of an axis tree.

    The returned size represents the total number of entries in the array. For
    example, an array with shape ``(10, 3)`` will have a size of 30.

    """
    outer_loops = axes.outer_loops

    if axes.is_empty:
        return 0

    if all(
        has_fixed_size(axes, axes.root, cpt, outer_loops)
        for cpt in axes.root.components
    ):
        return _axis_size(axes, axes.root)

    sizes = []

    for idxs in my_product(outer_loops):
        source_indices = merge_dicts(idx.source_exprs for idx in idxs)
        target_indices = merge_dicts(idx.target_exprs for idx in idxs)

        size = _axis_size(axes, axes.root, target_indices)
        sizes.append(size)
    return np.asarray(sizes, dtype=IntType)


def my_product(loops):
    if len(loops) > 1:
        raise NotImplementedError(
            "Now we are nesting loops so having multiple is a "
            "headache I haven't yet tackled"
        )
    # loop, *inner_loops = loops
    (loop,) = loops

    if loop.iterset.outer_loops:
        for indices in my_product(loop.iterset.outer_loops):
            context = frozenset(indices)
            for index in loop.iter(context):
                indices_ = indices + (index,)
                yield indices_
    else:
        for index in loop.iter():
            yield (index,)


def _axis_size(
    axes: AxisTree,
    axis: Axis,
    indices=pmap(),
    *,
    loop_indices=pmap(),
):
    return sum(
        _axis_component_size(axes, axis, cpt, indices, loop_indices=loop_indices)
        for cpt in axis.components
    )


def _axis_component_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    indices=pmap(),
    *,
    loop_indices=pmap(),
):
    return sum(
        _axis_component_size_region(axes, axis, component, region, indices, loop_indices=loop_indices)
        for region in component.regions
    )


def _axis_component_size_region(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    region,
    indices=pmap(),
    *,
    loop_indices=pmap(),
):
    count = _as_int(region.size, indices, loop_indices=loop_indices)
    if subaxis := axes.child(axis, component):
        return sum(
            _axis_size(
                axes,
                subaxis,
                indices | {axis.label: i},
                loop_indices=loop_indices,
            )
            for i in range(count)
        )
    else:
        return count


@functools.singledispatch
def _as_int(arg: Any, indices, path=None, *, loop_indices=pmap()):
    from pyop3.array import Dat

    if isinstance(arg, Dat):
        # TODO this might break if we have something like [:, subset]
        # I will need to map the "source" axis (e.g. slice_label0) back
        # to the "target" axis
        # return arg.get_value(indices, target_path, index_exprs)
        return arg.get_value(indices, path, loop_exprs=loop_indices)
    else:
        raise TypeError


@_as_int.register
def _(arg: numbers.Real, *args, **kwargs):
    return strict_int(arg)


class LoopExpressionReplacer(pym.mapper.IdentityMapper):
    def __init__(self, loop_exprs):
        self._loop_exprs = loop_exprs

    def map_multi_array(self, array):
        index_exprs = {ax: self.rec(expr) for ax, expr in array.index_exprs.items()}
        return type(array)(array.array, array.target_path, index_exprs)

    def map_loop_index(self, index):
        return self._loop_exprs[index.id][index.axis]


def eval_offset(
    axes,
    layouts,
    indices,
    path=None,
    *,
    loop_exprs=pmap(),
):
    from pyop3.expr_visitors import evaluate as eval_expr

    if path is None:
        path = pmap() if axes.is_empty else just_one(axes.leaf_paths)

    # if the provided indices are not a dict then we assume that they apply in order
    # as we go down the selected path of the tree
    if not isinstance(indices, collections.abc.Mapping):
        # a single index is treated like a 1-tuple
        indices = as_tuple(indices)

        indices_ = {}
        ordered_path = iter(just_one(axes.ordered_leaf_paths))
        for index in indices:
            axis_label, _ = next(ordered_path)
            indices_[axis_label] = index
        indices = indices_

    layout_subst = layouts[freeze(path)]

    # offset = ExpressionEvaluator(indices, loop_exprs)(layout_subst)
    # offset = eval_expr(layout_subst, path, indices)
    offset = eval_expr(layout_subst, indices)
    # breakpoint()
    return strict_int(offset)

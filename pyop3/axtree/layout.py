from __future__ import annotations

import collections
import functools
import itertools
import numbers
import operator
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import pymbolic as pym
from pyrsistent import freeze, pmap

from pyop3.axtree.tree import (
    Axis,
    AxisComponent,
    AxisTree,
    ExpressionEvaluator,
    PartialAxisTree,
    UnrecognisedAxisException,
    component_number_from_offsets,
    component_offsets,
)
from pyop3.dtypes import IntType, PointerType
from pyop3.tree import LabelledTree, MultiComponentLabelledNode
from pyop3.utils import (
    PrettyTuple,
    as_tuple,
    checked_zip,
    just_one,
    merge_dicts,
    strict_int,
    strictly_all,
)


# hacky class for index_exprs to work, needs cleaning up
class AxisVariable(pym.primitives.Variable):
    init_arg_names = ("axis",)

    mapper_method = sys.intern("map_axis_variable")

    mycounter = 0

    def __init__(self, axis):
        super().__init__(f"var{self.mycounter}")
        self.__class__.mycounter += 1  # ugly
        self.axis_label = axis

    def __getinitargs__(self):
        # not very happy about this, is the name required?
        return (self.axis,)

    @property
    def axis(self):
        return self.axis_label

    @property
    def datamap(self):
        return pmap()


class IntRef:
    """Pass-by-reference integer."""

    def __init__(self, value):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self


def has_independently_indexed_subaxis_parts(axes, axis, cpt):
    """
    subaxis parts are independently indexed if they don't depend on the index from
    ``part``.

    if one sub-part needs this index to determine its extent then we need to create
    a layout function as the step sizes will differ.

    Note that we need to consider both ragged sizes and permutations here
    """
    if subaxis := axes.component_child(axis, cpt):
        return not any(
            requires_external_index(axes, subaxis, c) for c in subaxis.components
        )
    else:
        return True


def can_be_affine(axtree, axis, component, component_index):
    return (
        has_independently_indexed_subaxis_parts(
            axtree, axis, component, component_index
        )
        and component.permutation is None
    )


def has_constant_start(
    axtree, axis, component, component_index, outer_axes_are_all_indexed: bool
):
    """
    We will have an affine layout with a constant start (usually zero) if either we are not
    ragged or if we are ragged but everything above is indexed (i.e. a temporary).
    """
    assert can_be_affine(axtree, axis, component, component_index)
    return isinstance(component.count, numbers.Integral) or outer_axes_are_all_indexed


def has_constant_step(axes: AxisTree, axis, cpt, outer_loops, path=pmap()):
    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if subaxis := axes.child(axis, cpt):
        return all(
            # not size_requires_external_index(axes, subaxis, c, path | {axis.label: cpt.label})
            not size_requires_external_index(axes, subaxis, c, outer_loops, path)
            for c in subaxis.components
        )
    else:
        return True


def has_fixed_size(axes, axis, component, outer_loops):
    return not size_requires_external_index(axes, axis, component, outer_loops)


def requires_external_index(axtree, axis, component_index):
    """Return ``True`` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index(
        axtree, axis, component_index
    )  # or numbering_requires_external_index(axtree, axis, component_index)


def size_requires_external_index(axes, axis, component, outer_loops, path=pmap()):
    from pyop3.array import HierarchicalArray

    if axis.id == "_id_Axis_68":
        breakpoint()

    count = component.count
    if isinstance(count, HierarchicalArray):
        # if count.name == "size_8" and count.axes.is_empty:
        #     breakpoint()
        if not set(count.outer_loops).issubset(outer_loops):
            return True
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
            if size_requires_external_index(axes, subaxis, c, outer_loops, path_):
                return True
    return False


def step_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    outer_loops,
    indices=PrettyTuple(),
    *,
    loop_exprs=pmap(),
):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if not has_constant_step(axes, axis, component, outer_loops) and not indices:
        raise ValueError
    if subaxis := axes.child(axis, component):
        return _axis_size(axes, subaxis, indices, loop_exprs=loop_exprs)
    else:
        return 1


def has_halo(axes, axis):
    if axis.sf is not None:
        return True
    else:
        for component in axis.components:
            subaxis = axes.component_child(axis, component)
            if subaxis and has_halo(axes, subaxis):
                return True
        return False
    return axis.sf is not None or has_halo(axes, subaxis)


# NOTE: I am not sure that this is really required any more. We just want to
# check for loop indices in any index_exprs
# No, we need this because loop indices do not necessarily mean we need extra shape.
def collect_externally_indexed_axes(axes, axis=None, component=None, path=pmap()):
    assert False, "old code"
    from pyop3.array import HierarchicalArray

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
    if isinstance(csize, HierarchicalArray):
        # is the path sufficient? i.e. do we have enough externally provided indices
        # to correctly index the axis?
        loop_indices = collect_external_loops(csize.axes, csize.index_exprs)
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

    def map_multi_array(self, array):
        if self._linear:
            return tuple(
                item for expr in array.index_exprs.values() for item in self.rec(expr)
            )
        else:
            return frozenset(
                {item for expr in array.index_exprs.values() for item in self.rec(expr)}
            )

    def map_called_map_variable(self, index):
        result = (
            idx
            for index_expr in index.input_index_exprs.values()
            for idx in self.rec(index_expr)
        )
        return tuple(result) if self._linear else frozenset(result)


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


def collect_outer_loops(axes, axis, index_exprs):
    assert False, "old code"
    from pyop3.itree.tree import LoopIndexVariable

    outer_loops = []
    while axis is not None:
        if len(axis.components) > 1:
            # outer loops can only be linear
            break
        # for expr in index_exprs.get((axis.id, axis.component.label), {}):
        expr = index_exprs.get((axis.id, axis.component.label), None)
        if isinstance(expr, LoopIndexVariable):
            outer_loops.append(expr)
        axis = axes.child(axis, axis.component)
    return tuple(outer_loops)


def _compute_layouts(
    axes: AxisTree,
    loop_exprs,
    axis=None,
    layout_path=pmap(),
    index_exprs_acc=pmap(),
):
    from pyop3.array.harray import MultiArrayVariable

    if axis is None:
        assert not axes.is_empty
        axis = axes.root
        index_exprs_acc |= axes.index_exprs.get(None, {})

    layouts = {}
    steps = {}

    # Post-order traversal
    csubtrees = []
    # think I can avoid target path for now
    subindex_exprs = []  # is this needed?
    sublayoutss = []
    subloops = []
    for cpt in axis.components:
        index_exprs_acc_ = index_exprs_acc | axes.index_exprs.get(
            (axis.id, cpt.label), {}
        )

        layout_path_ = layout_path | {axis.label: cpt.label}

        if subaxis := axes.child(axis, cpt):
            (
                sublayouts,
                csubtree,
                subindex_exprs_,
                substeps,
                subloops_,
            ) = _compute_layouts(
                axes, loop_exprs, subaxis, layout_path_, index_exprs_acc_
            )
            sublayoutss.append(sublayouts)
            subindex_exprs.append(subindex_exprs_)
            csubtrees.append(csubtree)
            steps.update(substeps)
            subloops.append(subloops_)
        else:
            csubtrees.append(None)
            subindex_exprs.append(None)
            sublayoutss.append(defaultdict(list))
            subloops.append(frozenset())

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

    outer_loops_per_component = {}
    for i, cpt in enumerate(axis.components):
        # if (axis, cpt) in loop_vars:
        #     my_loops = frozenset({loop_vars[axis, cpt]}) | subloops[i]
        # else:
        #     my_loops = subloops[i]
        my_loops = subloops[i]
        outer_loops_per_component[cpt] = my_loops

    # if noouter_loops:
    # breakpoint()

    # 1. do we need to pass further up? i.e. are we variable size?
    # also if we have halo data then we need to pass to the top
    if (
        not all(
            has_fixed_size(axes, axis, cpt, outer_loops_per_component[cpt])
            # has_fixed_size(axes, axis, cpt)
            for cpt in axis.components
        )
    ) or (has_halo(axes, axis) and axis != axes.root):
        if has_halo(axes, axis) or not all(
            has_constant_step(axes, axis, c, subloops[i])
            for i, c in enumerate(axis.components)
        ):
            ctree = PartialAxisTree(axis.copy(numbering=None))

            # this doesn't follow the normal pattern because we are accumulating
            # *upwards*
            myindex_exprs = {}
            for c in axis.components:
                myindex_exprs[axis.id, c.label] = axes.index_exprs.get(
                    (axis.id, c.label), pmap()
                )
            # we enforce here that all subaxes must be tabulated, is this always
            # needed?
            if strictly_all(sub is not None for sub in csubtrees):
                for component, subtree, subindex_exprs_ in checked_zip(
                    axis.components, csubtrees, subindex_exprs
                ):
                    ctree = ctree.add_subtree(subtree, axis, component)
                    # myindex_exprs.update(subindex_exprs_)
        else:
            # we must be at the bottom of a ragged patch - therefore don't
            # add to shape of things
            # in theory if we are ragged and permuted then we do want to include this level
            ctree = None
            myindex_exprs = {}
            for c in axis.components:
                myindex_exprs[axis.id, c.label] = axes.index_exprs.get(
                    (axis.id, c.label), pmap()
                )
            for i, c in enumerate(axis.components):
                step = step_size(axes, axis, c, subloops[i], loop_exprs=loop_exprs)
                # step = step_size(axes, axis, c, index_exprs)
                # step = step_size(axes, axis, c)
                axis_var = axes.index_exprs[axis.id, c.label][axis.label]
                layouts.update({layout_path | {axis.label: c.label}: axis_var * step})

        # layouts and steps are just propagated from below
        layouts.update(merge_dicts(sublayoutss))
        return (
            layouts,
            ctree,
            myindex_exprs,
            steps,
            frozenset(x for v in outer_loops_per_component.values() for x in v),
        )

    # 2. add layouts here
    else:
        # 1. do we need to tabulate anything?
        interleaved = len(axis.components) > 1 and axis.numbering is not None
        if (
            interleaved
            or not all(
                has_constant_step(axes, axis, c, subloops[i])
                for i, c in enumerate(axis.components)
            )
            or has_halo(axes, axis)
            and axis == axes.root  # at the top
        ):
            ctree = PartialAxisTree(axis.copy(numbering=None))
            # this doesn't follow the normal pattern because we are accumulating
            # *upwards*
            # we need to keep track of this information because it will tell us, I
            # think, if we have hit all the right loop indices
            myindex_exprs = {}
            for c in axis.components:
                myindex_exprs[axis.id, c.label] = axes.index_exprs.get(
                    (axis.id, c.label), pmap()
                )
            # we enforce here that all subaxes must be tabulated, is this always
            # needed?
            if strictly_all(sub is not None for sub in csubtrees):
                for component, subtree, subiexprs in checked_zip(
                    axis.components, csubtrees, subindex_exprs
                ):
                    ctree = ctree.add_subtree(subtree, axis, component)
                    myindex_exprs.update(subiexprs)

            # myindex_exprs = index_exprs_acc

            fulltree = _create_count_array_tree(ctree, axes.index_exprs, loop_exprs)

            # now populate fulltree
            offset = IntRef(0)
            _tabulate_count_array_tree(
                axes,
                axis,
                loop_exprs,
                index_exprs_acc_,
                fulltree,
                offset,
                setting_halo=False,
            )

            # apply ghost offset stuff, the offset from the previous pass is used
            _tabulate_count_array_tree(
                axes,
                axis,
                loop_exprs,
                index_exprs_acc_,
                fulltree,
                offset,
                setting_halo=True,
            )

            # TODO think about substituting with loop_exprs
            if loop_exprs:
                breakpoint()
            for subpath, offset_data in fulltree.items():
                # offset_data must be linear so we can unroll the target paths and
                # index exprs
                source_path = offset_data.axes.path_with_nodes(*offset_data.axes.leaf)
                index_keys = [None] + [
                    (axis.id, cpt) for axis, cpt in source_path.items()
                ]
                my_target_path = merge_dicts(
                    offset_data.target_paths.get(key, {}) for key in index_keys
                )
                my_index_exprs = merge_dicts(
                    offset_data.index_exprs.get(key, {}) for key in index_keys
                )
                offset_var = MultiArrayVariable(
                    offset_data, my_target_path, my_index_exprs
                )

                layouts[layout_path | subpath] = offset_var
            ctree = None

            # bit of a hack, we can skip this if we aren't passing higher up
            if axis == axes.root:
                steps = "not used"
            else:
                steps = {layout_path: _axis_size(axes, axis)}

            layouts.update(merge_dicts(sublayoutss))
            return (
                layouts,
                ctree,
                myindex_exprs,
                steps,
                frozenset(x for v in outer_loops_per_component.values() for x in v),
            )

        # must therefore be affine
        else:
            assert all(sub is None for sub in csubtrees)
            layouts = {}
            steps = [
                # step_size(axes, axis, c, index_exprs_acc_)
                # step_size(axes, axis, c)
                step_size(axes, axis, c, subloops[i])
                for i, c in enumerate(axis.components)
            ]
            start = 0
            for cidx, step in enumerate(steps):
                mycomponent = axis.components[cidx]
                sublayouts = sublayoutss[cidx].copy()

                key = (axis.id, mycomponent.label)
                # axis_var = index_exprs[key][axis.label]
                axis_var = axes.index_exprs[key][axis.label]
                # if key in index_exprs:
                #     axis_var = index_exprs[key][axis.label]
                # else:
                #     axis_var = AxisVariable(axis.label)
                new_layout = axis_var * step + start

                sublayouts[layout_path | {axis.label: mycomponent.label}] = new_layout
                start += _axis_component_size(
                    axes, axis, mycomponent, loop_exprs=loop_exprs
                )

                layouts.update(sublayouts)
            steps = {layout_path: _axis_size(axes, axis)}
            return (
                layouts,
                None,
                None,
                steps,
                frozenset(x for v in outer_loops_per_component.values() for x in v),
            )


def _create_count_array_tree(
    ctree,
    index_exprs,
    loop_exprs,
    axis=None,
    axes_acc=None,
    index_exprs_acc=None,
    path=pmap(),
):
    from pyop3.array import HierarchicalArray

    if strictly_all(x is None for x in [axis, axes_acc, index_exprs_acc]):
        axis = ctree.root
        axes_acc = ()
        # index_exprs_acc = ()
        index_exprs_acc = pmap()

    arrays = {}
    for component in axis.components:
        path_ = path | {axis.label: component.label}
        linear_axis = axis[component.label].root
        axes_acc_ = axes_acc + (linear_axis,)
        # index_exprs_acc_ = index_exprs_acc + (index_exprs.get((axis.id, component.label), {}),)
        index_exprs_acc_ = index_exprs_acc | {
            (linear_axis.id, component.label): index_exprs.get(
                (axis.id, component.label), {}
            )
        }

        if subaxis := ctree.child(axis, component):
            arrays.update(
                _create_count_array_tree(
                    ctree,
                    index_exprs,
                    loop_exprs,
                    subaxis,
                    axes_acc_,
                    index_exprs_acc_,
                    path_,
                )
            )
        else:
            # make a multiarray here from the given sizes

            # do we have any external axes from loop indices?
            axtree = AxisTree.from_iterable(axes_acc_)
            # external_loops = collect_external_loops(
            #     axtree, index_exprs_acc_, linear=True
            # )
            # external_loops = outer_loops
            # if len(external_loops) > 0:
            #     external_axes = PartialAxisTree.from_iterable(
            #         [l.index.iterset for l in external_loops]
            #     )
            #     myaxes = external_axes.add_subtree(axtree, *external_axes.leaf)
            # else:
            #     myaxes = axtree

            # TODO some of these should be LoopIndexVariable...
            # target_paths = {}
            # layout_exprs = {}
            # for ax, clabel in myaxes.path_with_nodes(*myaxes.leaf).items():
            #     target_paths[ax.id, clabel] = {ax.label: clabel}
            #     # my_index_exprs[ax.id, cpt.label] = index_exprs.get()
            #     layout_exprs[ax.id, clabel] = {ax.label: AxisVariable(ax.label)}

            # breakpoint()
            # new_index_exprs = dict(axtree.index_exprs)
            # new_index_exprs[???] = ...

            countarray = HierarchicalArray(
                axtree,
                target_paths=axtree._default_target_paths(),
                index_exprs=index_exprs_acc_,
                outer_loops=(),
                data=np.full(axtree.global_size, -1, dtype=IntType),
                # use default layout, just tweak index_exprs
                prefix="offset",
            )
            arrays[path_] = countarray

    return arrays


def _tabulate_count_array_tree(
    axes,
    axis,
    loop_exprs,
    layout_index_exprs,
    count_arrays,
    offset,
    path=pmap(),  # might not be needed
    indices=pmap(),
    is_owned=True,
    setting_halo=False,
    outermost=True,
):
    npoints = sum(_as_int(c.count, indices) for c in axis.components)

    offsets = component_offsets(axis, indices)
    points = axis.numbering.data_ro if axis.numbering is not None else range(npoints)

    counters = {c: itertools.count() for c in axis.components}
    for new_pt, old_pt in enumerate(points):
        if axis.sf is not None:
            is_owned = new_pt < axis.sf.nowned

        component, _ = component_number_from_offsets(axis, old_pt, offsets)

        new_strata_pt = next(counters[component])

        path_ = path | {axis.label: component.label}
        indices_ = indices | {axis.label: new_strata_pt}
        if path_ in count_arrays:
            if is_owned and not setting_halo or not is_owned and setting_halo:
                count_arrays[path_].set_value(
                    indices_,
                    offset.value,
                )
                offset += step_size(
                    axes,
                    axis,
                    component,
                    outer_loops="???",
                    # index_exprs=index_exprs,
                    indices=indices_,
                    loop_exprs=loop_exprs,
                )
        else:
            subaxis = axes.component_child(axis, component)
            assert subaxis
            _tabulate_count_array_tree(
                axes,
                subaxis,
                loop_exprs,
                layout_index_exprs,
                count_arrays,
                offset,
                path_,
                indices_,
                is_owned=is_owned,
                setting_halo=setting_halo,
                outermost=False,
            )


# TODO this whole function sucks, should accumulate earlier
def _collect_at_leaves(
    axes,
    layout_axes,
    values,
    axis: Optional[Axis] = None,
    path=pmap(),
    layout_path=pmap(),
    prior=0,
):
    acc = {}
    if axis is None:
        axis = layout_axes.root

    # if axis == axes.root:
    if axis == layout_axes.root:
        acc[pmap()] = prior

    for component in axis.components:
        layout_path_ = layout_path | {axis.label: component.label}
        prior_ = prior + values.get(layout_path_, 0)

        # if axis in axes.nodes:
        if True:
            path_ = path | {axis.label: component.label}
            acc[path_] = prior_
        else:
            path_ = path

        if subaxis := layout_axes.child(axis, component):
            acc.update(
                _collect_at_leaves(
                    axes, layout_axes, values, subaxis, path_, layout_path_, prior_
                )
            )
    # if layout_axes.depth != axes.depth and len(layout_path) == 0:
    #     breakpoint()
    return acc


def axis_tree_size(axes: AxisTree) -> int:
    """Return the size of an axis tree.

    The returned size represents the total number of entries in the array. For
    example, an array with shape ``(10, 3)`` will have a size of 30.

    """
    from pyop3.array import HierarchicalArray

    # outer_loops = collect_external_loops(axes, axes.index_exprs)
    outer_loops = axes.outer_loops
    # external_axes = collect_externally_indexed_axes(axes)
    # if len(external_axes) == 0:
    if axes.is_empty:
        return 1

    if all(
        has_fixed_size(axes, axes.root, cpt, outer_loops)
        for cpt in axes.root.components
    ):
        return _axis_size(axes, axes.root)

    # axis size is now an array

    # axes_iter = []
    # index_exprs = {}
    # outer_loop_map = {}
    # for ol in outer_loops_ord:
    #     iterset = ol.index.iterset
    #     for axis in iterset.path_with_nodes(*iterset.leaf):
    #         axis_ = axis.copy(id=Axis.unique_id(), label=Axis.unique_label())
    #         # axis_ = axis
    #         axes_iter.append(axis_)
    #         index_exprs[axis_.id, axis_.component.label] = {axis.label: ol}
    #         outer_loop_map[axis_] = ol
    # size_axes = PartialAxisTree.from_iterable(axes_iter)
    #
    # # hack
    # target_paths = AxisTree(size_axes.parent_to_children)._default_target_paths()
    # layout_exprs = {}
    #
    # size_axes = AxisTree(size_axes.parent_to_children, target_paths=target_paths, index_exprs=index_exprs, outer_loops=outer_loops_ord[:-1], layout_exprs=layout_exprs)
    #
    # sizes = HierarchicalArray(
    #     size_axes,
    #     target_paths=target_paths,
    #     index_exprs=index_exprs,
    #     # outer_loops=frozenset(),  # only temporaries need this
    #     # outer_loops=axes.outer_loops,  # causes infinite recursion
    #     outer_loops=outer_loops_ord[:-1],
    #     dtype=IntType,
    #     prefix="size",
    # )
    # sizes = HierarchicalArray(AxisTree(), target_paths={}, index_exprs={}, outer_loops=outer_loops_ord[:-1])
    # breakpoint()
    # sizes = HierarchicalArray(AxisTree(outer_loops=outer_loops_ord), target_paths={}, index_exprs={}, outer_loops=outer_loops_ord)
    # sizes = HierarchicalArray(axes)
    sizes = []

    # for idxs in itertools.product(*outer_loops_iter):
    for idxs in my_product(outer_loops):
        print(idxs)
        # for idx in size_axes.iter():
        # idxs = [idx]
        source_indices = merge_dicts(idx.source_exprs for idx in idxs)
        target_indices = merge_dicts(idx.target_exprs for idx in idxs)

        # indices = {}
        # target_indices = {}
        # # myindices = {}
        # for axis in size_axes.nodes:
        #     loop_var = outer_loop_map[axis]
        #     idx = just_one(idx for idx in idxs if idx.index == loop_var.index)
        #     # myindices[axis.label] = just_one(sum(idx.source_exprs.values()))
        #
        #     axlabel = just_one(idx.index.iterset.nodes).label
        #     value = just_one(idx.target_exprs.values())
        # indices[loop_var.index.id] = {axlabel: value}

        # target_indices[just_one(idx.target_path.keys())] = just_one(idx.target_exprs.values())

        # this is a hack
        if axes.is_empty:
            size = 1
        else:
            size = _axis_size(axes, axes.root, target_indices)
        # sizes.set_value(source_indices, size)
        sizes.append(size)
    # breakpoint()
    # return sizes
    return np.asarray(sizes, dtype=IntType)


def my_product(loops, indices=(), context=frozenset()):
    loop, *inner_loops = loops

    if inner_loops:
        for index in loop.iter(context):
            indices_ = indices + (index,)
            context_ = context | {index}
            yield from my_product(inner_loops, indices_, context_)
    else:
        for index in loop.iter(context):
            yield indices + (index,)


def _axis_size(
    axes: AxisTree,
    axis: Axis,
    indices=pmap(),
    *,
    loop_exprs=pmap(),
):
    return sum(
        _axis_component_size(axes, axis, cpt, indices, loop_exprs=loop_exprs)
        for cpt in axis.components
    )


def _axis_component_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    indices=pmap(),
    *,
    loop_exprs=pmap(),
):
    count = _as_int(component.count, indices, loop_exprs=loop_exprs)
    if subaxis := axes.component_child(axis, component):
        return sum(
            _axis_size(
                axes,
                subaxis,
                indices | {axis.label: i},
                loop_exprs=loop_exprs,
            )
            for i in range(count)
        )
    else:
        return count


@functools.singledispatch
def _as_int(arg: Any, indices, path=None, *, loop_exprs=pmap()):
    from pyop3.array import HierarchicalArray

    if isinstance(arg, HierarchicalArray):
        # this shouldn't be here, but it will break things the least to do so
        # at the moment
        # if index_exprs is None:
        #     index_exprs = merge_dicts(arg.index_exprs.values())

        # TODO this might break if we have something like [:, subset]
        # I will need to map the "source" axis (e.g. slice_label0) back
        # to the "target" axis
        # return arg.get_value(indices, target_path, index_exprs)
        return arg.get_value(indices, path, loop_exprs=loop_exprs)
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
    axes, layouts, indices, target_paths, index_exprs, path=None, *, loop_exprs=pmap()
):
    from pyop3.itree.tree import IndexExpressionReplacer

    # now select target paths and index exprs from the full collection
    target_path = target_paths.get(None, {})
    index_exprs_ = index_exprs.get(None, {})

    if not axes.is_empty:
        if path is None:
            path = just_one(axes.leaf_paths)
        node_path = axes.path_with_nodes(*axes._node_from_path(path))
        for axis, component in node_path.items():
            key = axis.id, component
            if key in target_paths:
                target_path.update(target_paths[key])
            if key in index_exprs:
                index_exprs_.update(index_exprs[key])

    # if the provided indices are not a dict then we assume that they apply in order
    # as we go down the selected path of the tree
    if not isinstance(indices, collections.abc.Mapping):
        # a single index is treated like a 1-tuple
        indices = as_tuple(indices)

        indices_ = {}
        axis = axes.root
        for idx in indices:
            indices_[axis.label] = idx
            cpt_label = target_path[axis.label]
            axis = axes.child(axis, cpt_label)
        indices = indices_

    # # then any provided
    # if index_exprs is not None:
    #     replace_map_new = {}
    #     replacer = ExpressionEvaluator(indices)
    #     for axis, index_expr in index_exprs.items():
    #         try:
    #             replace_map_new[axis] = replacer(index_expr)
    #         except UnrecognisedAxisException:
    #             pass
    #     indices2 = replace_map_new
    # else:
    #     indices2 = indices
    #
    # replace_map_new = {}
    # replacer = ExpressionEvaluator(indices2)
    # for axlabel, index_expr in axes.index_exprs.get(None, {}).items():
    #     try:
    #         replace_map_new[axlabel] = replacer(index_expr)
    #     except UnrecognisedAxisException:
    #         pass
    # for axis, component in source_path_node.items():
    #     for axlabel, index_expr in axes.index_exprs.get((axis.id, component), {}).items():
    #         try:
    #             replace_map_new[axlabel] = replacer(index_expr)
    #         except UnrecognisedAxisException:
    #             pass
    # indices1 = replace_map_new

    # Substitute indices into index exprs
    # if index_exprs:

    # Replace any loop index variables in index_exprs
    # index_exprs_ = {}
    # replacer = LoopExpressionReplacer(loop_exprs)  # different class?
    # for ax, expr in index_exprs.items():
    #     # if isinstance(expr, LoopIndexVariable):
    #     #     index_exprs_[ax] = loop_exprs[expr.id][ax]
    #     # else:
    #     index_exprs_[ax] = replacer(expr)

    # # Substitute something TODO with indices
    # if indices:
    #     breakpoint()
    # else:
    #     indices_ = index_exprs_

    # replacer = IndexExpressionReplacer(index_exprs_, loop_exprs)
    replacer = IndexExpressionReplacer(index_exprs_, loop_exprs)
    layout_orig = layouts[freeze(target_path)]
    layout_subst = replacer(layout_orig)

    # if loop_exprs:
    #     breakpoint()

    # offset = pym.evaluate(layouts[target_path], indices_, ExpressionEvaluator)
    offset = ExpressionEvaluator(indices, loop_exprs)(layout_subst)
    return strict_int(offset)

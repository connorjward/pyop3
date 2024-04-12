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
    AxisVariable,
    ExpressionEvaluator,
    IndexedAxisTree,
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
# class AxisVariable(pym.primitives.Variable):
#     init_arg_names = ("axis",)
#
#     mapper_method = sys.intern("map_axis_variable")
#
#     mycounter = 0
#
#     def __init__(self, axis):
#         super().__init__(f"var{self.mycounter}")
#         self.__class__.mycounter += 1  # ugly
#         self.axis_label = axis
#
#     def __getinitargs__(self):
#         # not very happy about this, is the name required?
#         return (self.axis,)
#
#     @property
#     def axis(self):
#         return self.axis_label
#
#     @property
#     def datamap(self):
#         return pmap()


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


def has_constant_step(axes: AxisTree, axis, cpt, inner_loop_vars, path=pmap()):
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


def requires_external_index(axtree, axis, component_index):
    """Return `True` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index(
        axtree, axis, component_index
    )  # or numbering_requires_external_index(axtree, axis, component_index)


def size_requires_external_index(axes, axis, component, inner_loop_vars, path=pmap()):
    from pyop3.array import HierarchicalArray

    count = component.count
    if isinstance(count, HierarchicalArray):
        if count.axes.is_empty:
            leafpath = pmap()
        else:
            leafpath = just_one(count.axes.leaf_paths)
        layout = count.subst_layouts[leafpath]
        required_loop_vars = LoopIndexCollector(linear=False)(layout)
        if not required_loop_vars.issubset(inner_loop_vars):
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
            if size_requires_external_index(axes, subaxis, c, inner_loop_vars, path_):
                return True
    return False


def step_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    indices=PrettyTuple(),
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

    # def map_called_map_variable(self, index):
    #     result = (
    #         idx
    #         for index_expr in index.input_index_exprs.values()
    #         for idx in self.rec(index_expr)
    #     )
    #     return tuple(result) if self._linear else frozenset(result)


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


# TODO: If an axis has size 1 then we don't need a variable for it.
def _compute_layouts(
    axes: AxisTree,
    loop_vars,
    axis=None,
    layout_path=pmap(),
    index_exprs_acc=pmap(),
):
    """
    Parameters
    ----------
    axes
        The axis tree to construct a layout for.
    loop_vars
        Mapping from axis label to loop index variable. Needed for tabulating
        indexed layouts because, as we go up the tree, we can identify which
        loop indices are materialised.
    """

    from pyop3.array.harray import ArrayVar

    if axis is None:
        assert not axes.is_empty
        axis = axes.root
        # get rid of this
        index_exprs_acc |= axes.index_exprs.get(None, {})

    # Collect the loop variables that are captured by this axis and those below
    # it. This lets us determine whether or not something that is indexed is
    # sufficiently "within" loops for us to tabulate.
    if len(axis.components) == 1 and (subaxis := axes.child(axis, axis.component)):
        inner_loop_vars = _collect_inner_loop_vars(axes, subaxis, loop_vars)
    else:
        inner_loop_vars = frozenset()
    inner_loop_vars_with_self = _collect_inner_loop_vars(axes, axis, loop_vars)

    layouts = {}
    steps = {}

    # Post-order traversal
    csubtrees = []
    sublayoutss = []
    for cpt in axis.components:
        index_exprs_acc_ = index_exprs_acc | axes.index_exprs.get(
            (axis.id, cpt.label), {}
        )

        layout_path_ = layout_path | {axis.label: cpt.label}

        if subaxis := axes.child(axis, cpt):
            (
                sublayouts,
                csubtree,
                substeps,
            ) = _compute_layouts(
                axes, loop_vars, subaxis, layout_path_, index_exprs_acc_
            )
            sublayoutss.append(sublayouts)
            csubtrees.append(csubtree)
            steps.update(substeps)
        else:
            csubtrees.append(None)
            sublayoutss.append(defaultdict(list))

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

    # 1. do we need to pass further up? i.e. are we variable size?
    # also if we have halo data then we need to pass to the top
    if (
        not all(
            has_fixed_size(axes, axis, cpt, inner_loop_vars_with_self)
            for cpt in axis.components
        )
    ) or (has_halo(axes, axis) and axis != axes.root):
        if has_halo(axes, axis) or not all(
            has_constant_step(axes, axis, c, inner_loop_vars)
            for i, c in enumerate(axis.components)
        ):
            ctree = AxisTree(axis.copy(numbering=None))

            # we enforce here that all subaxes must be tabulated, is this always
            # needed?
            if strictly_all(sub is not None for sub in csubtrees):
                for component, subtree in checked_zip(axis.components, csubtrees):
                    ctree = ctree.add_subtree(subtree, axis, component)
        else:
            # we must be at the bottom of a ragged patch - therefore don't
            # add to shape of things
            # in theory if we are ragged and permuted then we do want to include this level
            ctree = None
            for i, c in enumerate(axis.components):
                step = step_size(axes, axis, c)
                if (axis.id, c.label) in loop_vars:
                    axis_var = loop_vars[axis.id, c.label][axis.label]
                else:
                    axis_var = AxisVariable(axis.label)
                layouts.update({layout_path | {axis.label: c.label}: axis_var * step})

        # layouts and steps are just propagated from below
        layouts.update(merge_dicts(sublayoutss))
        return (
            layouts,
            ctree,
            steps,
        )

    # 2. add layouts here
    else:
        # 1. do we need to tabulate anything?
        interleaved = len(axis.components) > 1 and axis.numbering is not None
        if (
            interleaved
            or not all(
                has_constant_step(axes, axis, c, inner_loop_vars)
                for i, c in enumerate(axis.components)
            )
            or has_halo(axes, axis)
            and axis == axes.root  # at the top
        ):
            ctree = AxisTree(axis.copy(numbering=None))
            # we enforce here that all subaxes must be tabulated, is this always
            # needed?
            if strictly_all(sub is not None for sub in csubtrees):
                for component, subtree, subiexprs in checked_zip(
                    axis.components, csubtrees
                ):
                    ctree = ctree.add_subtree(subtree, axis, component)

            fulltree = _create_count_array_tree(ctree, loop_vars)

            # now populate fulltree
            offset = IntRef(0)
            _tabulate_count_array_tree(
                axes,
                axis,
                loop_vars,
                fulltree,
                offset,
                setting_halo=False,
            )

            # apply ghost offset stuff, the offset from the previous pass is used
            _tabulate_count_array_tree(
                axes,
                axis,
                loop_vars,
                fulltree,
                offset,
                setting_halo=True,
            )

            for subpath, offset_data in fulltree.items():
                # offset_data must be linear so we can unroll the indices
                # flat_indices = {
                #     ax: expr
                # }
                source_path = offset_data.axes.path_with_nodes(*offset_data.axes.leaf)
                index_keys = [None] + [
                    (axis.id, cpt) for axis, cpt in source_path.items()
                ]
                mytargetpath = merge_dicts(
                    offset_data.target_paths.get(key, {}) for key in index_keys
                )
                myindices = merge_dicts(
                    offset_data.index_exprs.get(key, {}) for key in index_keys
                )
                offset_var = ArrayVar(offset_data, myindices, mytargetpath)

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
                steps,
            )

        # must therefore be affine
        else:
            assert all(sub is None for sub in csubtrees)
            layouts = {}
            steps = [
                # step_size(axes, axis, c, index_exprs_acc_)
                step_size(axes, axis, c)
                # step_size(axes, axis, c, subloops[i])
                for i, c in enumerate(axis.components)
            ]
            start = 0
            for cidx, step in enumerate(steps):
                mycomponent = axis.components[cidx]
                sublayouts = sublayoutss[cidx].copy()

                # key = (axis.id, mycomponent.label)
                # axis_var = index_exprs[key][axis.label]
                axis_var = AxisVariable(axis.label)
                # axis_var = axes.index_exprs[key][axis.label]
                # if key in index_exprs:
                #     axis_var = index_exprs[key][axis.label]
                # else:
                #     axis_var = AxisVariable(axis.label)
                new_layout = axis_var * step + start

                sublayouts[layout_path | {axis.label: mycomponent.label}] = new_layout
                start += _axis_component_size(axes, axis, mycomponent)

                layouts.update(sublayouts)
            steps = {layout_path: _axis_size(axes, axis)}
            return (
                layouts,
                None,
                steps,
            )


def _create_count_array_tree(
    ctree,
    loop_vars,
    axis=None,
    axes_acc=None,
    path=pmap(),
):
    from pyop3.array import HierarchicalArray
    from pyop3.itree.tree import IndexExpressionReplacer

    if strictly_all(x is None for x in [axis, axes_acc]):
        axis = ctree.root
        axes_acc = ()

    arrays = {}
    for component in axis.components:
        path_ = path | {axis.label: component.label}
        # This causes an infinite recursion because axis[component.label]
        # returns an IndexedAxisTree instead of an AxisTree. This should be
        # avoidable.
        # linear_axis = axis[component.label].root
        # current workaround:
        if len(axis.components) > 1:
            linear_axis = axis[component.label].root
        else:
            # discard SF since the tabulating arrays are not parallel
            linear_axis = axis.copy(sf=None)
        axes_acc_ = axes_acc + (linear_axis,)

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
                        index_expr = {myaxis.label: AxisVariable(myaxis.label)}
                    index_exprs[key] = index_expr
            else:
                index_exprs = axtree.index_exprs

            # think this is completely unnecessary - just use the AxisTree
            # iaxtree = IndexedAxisTree(
            #     axtree.node_map,
            #     target_paths=axtree.target_paths,
            #     index_exprs=index_exprs,
            #     outer_loops=(),  # ???
            #     layout_exprs=axtree.layout_exprs,
            #     layouts=axtree.layouts,
            #     sf=axtree.sf,
            # )

            countarray = HierarchicalArray(
                # iaxtree,
                axtree,
                # target_paths=axtree.target_paths,
                # index_exprs=index_exprs,
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
    points = axis.numbering.data_ro if axis.numbering is not None else range(npoints)

    counters = {c: itertools.count() for c in axis.components}
    for new_pt, old_pt in enumerate(points):
        if axis.sf is not None:
            is_owned = new_pt < axis.sf.nowned

        component, _ = component_number_from_offsets(axis, old_pt, offsets)

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
            subaxis = axes.component_child(axis, component)
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
    if axis is None:
        axis = layout_axes.root

    acc = {pmap(): prior} if axis == axes.root else {}
    for component in axis.components:
        layout_path_ = layout_path | {axis.label: component.label}
        prior_ = prior + values.get(layout_path_, 0)

        if axis in axes.nodes:
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
    return acc


def axis_tree_size(axes: AxisTree) -> int:
    """Return the size of an axis tree.

    The returned size represents the total number of entries in the array. For
    example, an array with shape ``(10, 3)`` will have a size of 30.

    """
    outer_loops = axes.outer_loops

    if axes.is_empty:
        return 1

    if all(
        has_fixed_size(axes, axes.root, cpt, outer_loops)
        for cpt in axes.root.components
    ):
        # if not outer_loops:
        # return _axis_size(axes, axes.root, loop_exprs=loop_exprs)
        return _axis_size(axes, axes.root)

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
    # return sizes
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
    count = _as_int(component.count, indices, loop_indices=loop_indices)
    if subaxis := axes.component_child(axis, component):
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
    # axes, layouts, indices, target_paths, index_exprs, path=None, *, loop_exprs=pmap()
    axes,
    layouts,
    indices,
    path=None,
    *,
    loop_exprs=pmap(),
):
    from pyop3.itree.tree import IndexExpressionReplacer

    # layout_axes = axes.layout_axes
    layout_axes = axes

    # now select target paths and index exprs from the full collection
    # target_path = target_paths.get(None, {})
    # index_exprs_ = index_exprs.get(None, {})

    # if not layout_axes.is_empty:
    #     if path is None:
    #         path = just_one(layout_axes.leaf_paths)
    #     node_path = layout_axes.path_with_nodes(*layout_axes._node_from_path(path))
    #     for axis, component in node_path.items():
    #         key = axis.id, component
    #         if key in target_paths:
    #             target_path.update(target_paths[key])
    #         if key in index_exprs:
    #             index_exprs_.update(index_exprs[key])

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

    # replacer = IndexExpressionReplacer(index_exprs_, loop_exprs)
    # layout_orig = layouts[freeze(target_path)]
    # layout_subst = replacer(layout_orig)

    layout_subst = layouts[freeze(path)]

    offset = ExpressionEvaluator(indices, loop_exprs)(layout_subst)
    return strict_int(offset)

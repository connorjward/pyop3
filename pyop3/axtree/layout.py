from __future__ import annotations

import functools
import numbers
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import pymbolic as pym
from pyrsistent import freeze, pmap

from pyop3.axtree.tree import Axis, AxisComponent, AxisTree
from pyop3.dtypes import IntType, PointerType
from pyop3.tree import LabelledTree, MultiComponentLabelledNode
from pyop3.utils import PrettyTuple, merge_dicts, strict_int, strictly_all


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


def has_fixed_size(axes, axis, component):
    return not size_requires_external_index(axes, axis, component)


def step_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    path=pmap(),
    indices=PrettyTuple(),
):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if not has_constant_step(axes, axis, component) and not indices:
        raise ValueError
    if subaxis := axes.component_child(axis, component):
        return _axis_size(axes, subaxis, path, indices)
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


def requires_external_index(axtree, axis, component_index):
    """Return ``True`` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index(
        axtree, axis, component_index
    )  # or numbering_requires_external_index(axtree, axis, component_index)


def size_requires_external_index(axes, axis, component, path=pmap()):
    return len(collect_externally_indexed_axes(axes, axis, component, path)) > 0


def collect_externally_indexed_axes(axes, axis=None, component=None, path=pmap()):
    from pyop3.array import HierarchicalArray

    if axes.is_empty:
        return ()

    # use a dict as an ordered set
    external_axes = {}
    if axis is None:
        assert component is None
        for component in axes.root.components:
            external_axes.update(
                collect_externally_indexed_axes(axes, axes.root, component)
            )
    else:
        csize = component.count
        if isinstance(csize, HierarchicalArray):
            if csize.axes.is_empty:
                pass
            else:
                # is the path sufficient? i.e. do we have enough externally provided indices
                # to correctly index the axis?
                for caxis, ccpt in csize.axes.path_with_nodes(*csize.axes.leaf).items():
                    if caxis.label in path:
                        assert path[caxis.label] == ccpt, "Paths do not match"
                    else:
                        external_axes[caxis] = None
        else:
            assert isinstance(csize, numbers.Integral)
            if subaxis := axes.child(axis, component):
                path_ = path | {axis.label: component.label}
                for subcpt in subaxis.components:
                    external_axes.update(
                        collect_externally_indexed_axes(axes, subaxis, subcpt, path_)
                    )

    # top level return is a tuple
    if not path:
        return tuple(external_axes.keys())
    else:
        return external_axes


def has_constant_step(axes: AxisTree, axis, cpt):
    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if subaxis := axes.child(axis, cpt):
        return all(
            # not size_requires_external_index(axes, subaxis, c, freeze({subaxis.label: c.label}))
            not size_requires_external_index(axes, subaxis, c)
            for c in subaxis.components
        )
    else:
        return True


# use this to build a tree of sizes that we use to construct
# the right count arrays
class CustomNode(MultiComponentLabelledNode):
    fields = MultiComponentLabelledNode.fields | {"counts", "component_labels"}

    def __init__(self, counts, *, component_labels=None, **kwargs):
        super().__init__(counts, **kwargs)
        self.counts = tuple(counts)
        self._component_labels = component_labels

    @property
    def component_labels(self):
        if self._component_labels is None:
            self._component_labels = tuple(self.unique_label() for _ in self.counts)
        return self._component_labels


def _compute_layouts(
    axes: AxisTree,
    axis=None,
    path=pmap(),
):
    from pyop3.array.harray import MultiArrayVariable

    axis = axis or axes.root
    layouts = {}
    steps = {}

    # Post-order traversal
    # make sure to catch children that are None
    csubroots = []
    csubtrees = []
    sublayoutss = []
    for cpt in axis.components:
        if subaxis := axes.component_child(axis, cpt):
            sublayouts, csubroot, csubtree, substeps = _compute_layouts(
                axes, subaxis, path | {axis.label: cpt.label}
            )
            sublayoutss.append(sublayouts)
            csubroots.append(csubroot)
            csubtrees.append(csubtree)
            steps.update(substeps)
        else:
            csubroots.append(None)
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
    if (not all(has_fixed_size(axes, axis, cpt) for cpt in axis.components)) or (
        has_halo(axes, axis) and axis != axes.root
    ):
        if has_halo(axes, axis) or not all(
            has_constant_step(axes, axis, c) for c in axis.components
        ):
            croot = CustomNode(
                [(cpt.count, axis.label, cpt.label) for cpt in axis.components]
            )
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = pmap(
                    {croot.id: [sub for sub in csubroots]}
                ) | merge_dicts(sub for sub in csubtrees)
            else:
                cparent_to_children = {}
            ctree = cparent_to_children
        else:
            # we must be at the bottom of a ragged patch - therefore don't
            # add to shape of things
            # in theory if we are ragged and permuted then we do want to include this level
            croot = None
            ctree = None
            for c in axis.components:
                step = step_size(axes, axis, c)
                layouts.update(
                    {path | {axis.label: c.label}: AxisVariable(axis.label) * step}
                )

        # layouts and steps are just propagated from below
        layouts.update(merge_dicts(sublayoutss))
        return layouts, croot, ctree, steps

    # 2. add layouts here
    else:
        # 1. do we need to tabulate anything?
        interleaved = len(axis.components) > 1 and axis.numbering is not None
        if (
            interleaved
            or not all(has_constant_step(axes, axis, c) for c in axis.components)
            or has_halo(axes, axis)
            and axis == axes.root
        ):
            # super ick
            bits = []
            for cpt in axis.components:
                axlabel, clabel = axis.label, cpt.label
                bits.append((cpt.count, axlabel, clabel))
            croot = CustomNode(bits)
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = pmap(
                    {croot.id: [sub for sub in csubroots]}
                ) | merge_dicts(sub for sub in csubtrees)
            else:
                cparent_to_children = {}

            cparent_to_children |= {None: (croot,)}
            ctree = LabelledTree(cparent_to_children)

            fulltree = _create_count_array_tree(ctree)

            # now populate fulltree
            offset = IntRef(0)
            _tabulate_count_array_tree(axes, axis, fulltree, offset, setting_halo=False)

            # apply ghost offset stuff, the offset from the previous pass is used
            _tabulate_count_array_tree(axes, axis, fulltree, offset, setting_halo=True)

            for subpath, offset_data in fulltree.items():
                # TODO avoid copy paste stuff, this is the same as in itree/tree.py

                offset_axes = offset_data.axes

                # must be single component
                source_path = offset_axes.path(*offset_axes.leaf)
                index_keys = [None] + [
                    (axis.id, cpt.label)
                    for axis, cpt in offset_axes.detailed_path(source_path).items()
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

                layouts[path | subpath] = offset_var
            ctree = None
            steps = {path: _axis_size(axes, axis)}

            layouts.update(merge_dicts(sublayoutss))
            return layouts, None, ctree, steps

        # must therefore be affine
        else:
            assert all(sub is None for sub in csubtrees)
            ctree = None
            layouts = {}
            steps = [step_size(axes, axis, c) for c in axis.components]
            start = 0
            for cidx, step in enumerate(steps):
                mycomponent = axis.components[cidx]
                sublayouts = sublayoutss[cidx].copy()

                new_layout = AxisVariable(axis.label) * step + start
                sublayouts[path | {axis.label: mycomponent.label}] = new_layout
                start += _axis_component_size(axes, axis, mycomponent)

                layouts.update(sublayouts)
            steps = {path: _axis_size(axes, axis)}
            return layouts, None, None, steps


# I don't think that this actually needs to be a tree, just return a dict
# TODO I need to clean this up a lot now I'm using component labels
def _create_count_array_tree(
    ctree, current_node=None, counts=PrettyTuple(), path=pmap()
):
    from pyop3.array import HierarchicalArray

    current_node = current_node or ctree.root
    arrays = {}

    for cidx in range(current_node.degree):
        count, axis_label, cpt_label = current_node.counts[cidx]

        child = ctree.children(current_node)[cidx]
        new_path = path | {axis_label: cpt_label}
        if child is None:
            # make a multiarray here from the given sizes
            axes = [
                Axis([(ct, clabel)], axlabel)
                for (ct, axlabel, clabel) in counts | current_node.counts[cidx]
            ]
            root = axes[0]
            parent_to_children = {None: (root,)}
            for parent, child in zip(axes, axes[1:]):
                parent_to_children[parent.id] = (child,)
            axtree = AxisTree.from_node_map(parent_to_children)
            countarray = HierarchicalArray(
                axtree,
                data=np.full(axis_tree_size(axtree), -1, dtype=IntType),
            )
            arrays[new_path] = countarray
        else:
            arrays.update(
                _create_count_array_tree(
                    ctree,
                    child,
                    counts | current_node.counts[cidx],
                    new_path,
                )
            )

    return arrays


def _tabulate_count_array_tree(
    axes,
    axis,
    count_arrays,
    offset,
    path=pmap(),
    indices=pmap(),
    is_owned=True,
    setting_halo=False,
):
    npoints = sum(_as_int(c.count, path, indices) for c in axis.components)

    point_to_component_id = np.empty(npoints, dtype=np.int8)
    point_to_component_num = np.empty(npoints, dtype=PointerType)
    *strata_offsets, _ = [0] + list(
        np.cumsum([_as_int(c.count, path, indices) for c in axis.components])
    )
    pos = 0
    point = 0
    # TODO this is overkill, we can just inspect the ranges?
    for cidx, component in enumerate(axis.components):
        # can determine this once above
        csize = _as_int(component.count, path, indices)
        for i in range(csize):
            point_to_component_id[point] = cidx
            # this is now just the identity with an offset?
            point_to_component_num[point] = i
            point += 1
        pos += csize

    counters = np.zeros(len(axis.components), dtype=int)
    points = axis.numbering.data_ro if axis.numbering is not None else range(npoints)
    for new_pt, old_pt in enumerate(points):
        if axis.sf is not None:
            # more efficient outside of loop
            _, ilocal, _ = axis.sf._graph
            is_owned = new_pt < npoints - len(ilocal)

        # equivalent to plex strata
        selected_component_id = point_to_component_id[old_pt]
        # selected_component_num = point_to_component_num[old_pt]
        selected_component_num = old_pt - strata_offsets[selected_component_id]
        selected_component = axis.components[selected_component_id]

        new_strata_pt = counters[selected_component_id]
        counters[selected_component_id] += 1

        new_path = path | {axis.label: selected_component.label}
        new_indices = indices | {axis.label: new_strata_pt}
        if new_path in count_arrays:
            if is_owned and not setting_halo or not is_owned and setting_halo:
                count_arrays[new_path].set_value(new_path, new_indices, offset.value)
                offset += step_size(
                    axes,
                    axis,
                    selected_component,
                    new_path,
                    new_indices,
                )
        else:
            subaxis = axes.component_child(axis, selected_component)
            assert subaxis
            _tabulate_count_array_tree(
                axes,
                subaxis,
                count_arrays,
                offset,
                new_path,
                new_indices,
                is_owned=is_owned,
                setting_halo=setting_halo,
            )


# TODO this whole function sucks, should accumulate earlier
def _collect_at_leaves(
    axes,
    values,
    axis: Optional[Axis] = None,
    path=pmap(),
    prior=0,
):
    axis = axis or axes.root
    acc = {}

    for cpt in axis.components:
        new_path = path | {axis.label: cpt.label}
        if new_path in values:
            # prior_ = prior | {axis.label: values[new_path]}
            prior_ = prior + values[new_path]
        else:
            prior_ = prior
        if subaxis := axes.component_child(axis, cpt):
            acc.update(_collect_at_leaves(axes, values, subaxis, new_path, prior_))
        else:
            acc[new_path] = prior_

    return acc


def axis_tree_size(axes: AxisTree) -> int:
    """Return the size of an axis tree.

    The returned size represents the total number of entries in the array. For
    example, an array with shape ``(10, 3)`` will have a size of 30.

    """
    if axes.is_empty:
        return 1
    return _axis_size(axes, axes.root, pmap(), pmap())


def _axis_size(
    axes: AxisTree,
    axis: Axis,
    path=pmap(),
    indices=pmap(),
) -> int:
    return sum(
        _axis_component_size(axes, axis, cpt, path, indices) for cpt in axis.components
    )


def _axis_component_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    path=pmap(),
    indices=pmap(),
):
    if size_requires_external_index(axes, axis, component, path):
        raise NotImplementedError

    count = _as_int(component.count, path, indices)
    if subaxis := axes.component_child(axis, component):
        return sum(
            _axis_size(
                axes,
                subaxis,
                path | {axis.label: component.label},
                indices | {axis.label: i},
            )
            for i in range(count)
        )
    else:
        return count


@functools.singledispatch
def _as_int(arg: Any, path, indices):
    from pyop3.array import HierarchicalArray

    if isinstance(arg, HierarchicalArray):
        # TODO this might break if we have something like [:, subset]
        # I will need to map the "source" axis (e.g. slice_label0) back
        # to the "target" axis
        return arg.get_value(path, indices, allow_unused=True)
    else:
        raise TypeError


@_as_int.register
def _(arg: numbers.Real, path, indices):
    return strict_int(arg)


def collect_sizes(axes: AxisTree) -> pmap:  # TODO value-type of returned pmap?
    return _collect_sizes_rec(axes, axes.root)


def _collect_sizes_rec(axes, axis) -> pmap:
    sizes = {}
    for cpt in axis.components:
        sizes[axis.label, cpt.label] = cpt.count

        if subaxis := axes.component_child(axis, cpt):
            subsizes = _collect_sizes_rec(axes, subaxis)
            for loc, size in subsizes.items():
                # make sure that sizes always match for duplicates
                if loc not in sizes:
                    sizes[loc] = size
                else:
                    if sizes[loc] != size:
                        raise RuntimeError
    return pmap(sizes)

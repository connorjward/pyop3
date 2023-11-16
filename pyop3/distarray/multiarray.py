from __future__ import annotations

import collections
import functools
import itertools
import numbers
import operator
import sys
import threading
from functools import cached_property
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pymbolic as pym
import pytools
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.axtree import (
    Axis,
    AxisComponent,
    AxisTree,
    ContextFree,
    ContextSensitive,
    as_axis_tree,
)
from pyop3.axtree.tree import AxisVariable, FrozenAxisTree, MultiArrayCollector
from pyop3.distarray2 import DistributedArray
from pyop3.distarray.base import Tensor
from pyop3.dtypes import IntType, ScalarType, get_mpi_dtype
from pyop3.extras.debug import print_if_rank, print_with_rank
from pyop3.itree import IndexedAxisTree, IndexTree, as_index_forest, index_axes
from pyop3.itree.tree import CalledMapVariable, collect_loop_indices
from pyop3.utils import (
    PrettyTuple,
    UniqueNameGenerator,
    as_tuple,
    deprecated,
    is_single_valued,
    just_one,
    merge_dicts,
    readonly,
    single_valued,
    strict_int,
    strictly_all,
)


class IncompatibleShapeError(Exception):
    """TODO, also bad name"""


# should be elsewhere, this is copied from loopexpr2loopy VariableReplacer
class IndexExpressionReplacer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        self._replace_map = replace_map

    def map_axis_variable(self, expr):
        # print_if_rank(0, "replace map ", self._replace_map)
        # return self._replace_map[expr.axis_label]
        return self._replace_map.get(expr.axis_label, expr)

    def map_multi_array(self, expr):
        # print_if_rank(0, self._replace_map)
        # print_if_rank(0, expr.indices)
        indices = {axis: self.rec(index) for axis, index in expr.indices.items()}
        return MultiArrayVariable(expr.array, indices)

    def map_called_map(self, expr):
        array = expr.function.map_component.array

        # should the following only exist at eval time?

        # the inner_expr tells us the right mapping for the temporary, however,
        # for maps that are arrays the innermost axis label does not always match
        # the label used by the temporary. Therefore we need to do a swap here.
        # I don't like this.
        # inner_axis = array.axes.leaf_axis
        # print_if_rank(0, self._replace_map)
        # print_if_rank(0, expr.parameters)
        indices = {axis: self.rec(idx) for axis, idx in expr.parameters.items()}
        # indices[inner_axis.label] = indices.pop(expr.function.full_map.name)

        return CalledMapVariable(expr.function, indices)

    def map_loop_index(self, expr):
        # this is hacky, if I make this raise a KeyError then we fail in indexing
        return self._replace_map.get((expr.name, expr.axis), expr)


class MultiArrayVariable(pym.primitives.Variable):
    mapper_method = sys.intern("map_multi_array")

    def __init__(self, array, indices):
        super().__init__(array.name)
        self.array = array
        self.indices = freeze(indices)

    def __repr__(self) -> str:
        return f"MultiArrayVariable({self.array!r}, {self.indices!r})"

    def __getinitargs__(self):
        return self.array, self.indices

    @property
    def datamap(self):
        return self.array.datamap | merge_dicts(
            idx.datamap for idx in self.indices.values()
        )


class Dat(Tensor, ContextFree):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------
    sf : ???
        PETSc star forest connecting values (offsets) in the local array with
        remote equivalents.

    """

    DEFAULT_DTYPE = DistributedArray.DEFAULT_DTYPE

    def __init__(
        self,
        axes: AxisTree,
        dtype=None,
        *,
        data=None,
        max_value=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # TODO This is ugly
        temporary_axes = as_axis_tree(axes).freeze()  # used for the temporary
        axes = layout_axes(axes)

        if data is None and dtype is None:
            dtype = ScalarType

        if isinstance(data, DistributedArray):
            # disable for now, temporaries hit this in an annoying way
            # if data.sf is not axes.sf:
            #     raise ValueError("Star forests do not match")
            if dtype is not None:
                raise ValueError(
                    "If data is a DistributedArray, dtype should not be provided"
                )
            pass
        elif isinstance(data, np.ndarray):
            data = DistributedArray(data, dtype, name=self.name, sf=axes.sf)
        else:
            data = DistributedArray(axes.size, dtype, name=self.name, sf=axes.sf)
        self.array = data

        self.temporary_axes = temporary_axes
        self.axes = axes
        self.layout_axes = axes  # used? likely don't need all these

        self.max_value = max_value

    def __str__(self):
        return self.name

    def __getitem__(self, indices) -> Union[MultiArray, ContextSensitiveMultiArray]:
        from pyop3.itree.tree import (
            _compose_bits,
            _index_axes,
            as_index_tree,
            collect_loop_contexts,
            index_axes,
        )

        loop_contexts = collect_loop_contexts(indices)
        if not loop_contexts:
            index_tree = just_one(as_index_forest(indices, axes=self.layout_axes))
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.layout_axes, index_tree, pmap())
            target_paths, index_exprs, layout_exprs = _compose_bits(
                self.layout_axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = IndexedAxisTree(
                indexed_axes.root,
                indexed_axes.parent_to_children,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            new_layouts = substitute_layouts(
                self.layout_axes,
                new_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
            )
            layout_axes = FrozenAxisTree(
                new_axes.root,
                new_axes.parent_to_children,
                target_paths=target_paths,
                index_exprs=index_exprs,
                layouts=new_layouts,
            )
            return self._with_axes(layout_axes)

        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=self.layout_axes):
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.layout_axes, index_tree, loop_context)

            (
                target_paths,
                index_exprs,
                layout_exprs,
            ) = _compose_bits(
                self.layout_axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = IndexedAxisTree(
                indexed_axes.root,
                indexed_axes.parent_to_children,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            new_layouts = substitute_layouts(
                self.layout_axes,
                new_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
            )
            layout_axes = FrozenAxisTree(
                new_axes.root,
                new_axes.parent_to_children,
                target_paths=target_paths,
                index_exprs=index_exprs,
                layouts=new_layouts,
            )
            array_per_context[loop_context] = self._with_axes(layout_axes)
        return ContextSensitiveMultiArray(array_per_context)

    # TODO remove this
    @property
    def layouts(self):
        return self.axes.layouts

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def sf(self) -> StarForest:
        return self.array.sf

    @property
    @deprecated(".data_rw")
    def data(self):
        return self.data_rw

    @property
    def data_rw(self):
        return self.array.data_rw

    @property
    def data_ro(self):
        return self.array.data_ro

    @property
    def data_wo(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should call
        `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        return self.array.data_wo

    @property
    def sf(self):
        return self.array.sf

    @cached_property
    def datamap(self):
        datamap_ = {self.name: self}
        datamap_.update(self.axes.datamap)
        datamap_.update(self.layout_axes.datamap)
        return freeze(datamap_)

    def assemble(self):
        """Ensure that stored values are up-to-date.

        This function is typically only required when accessing the `Dat` in a
        write-only mode (`Access.WRITE`, `Access.MIN_WRITE` or `Access.MAX_WRITE`)
        and only setting a subset of the values. Without `Dat.assemble` the non-subset
        entries in the array would hold undefined values.

        """
        self.array._reduce_then_broadcast()

    def _with_axes(self, axes):
        """Return a new `Dat` with new axes pointing to the same data."""
        return type(self)(
            axes,
            data=self.array,
            max_value=self.max_value,
            name=self.name,
            orig_array=self.orig_array,
        )

    def as_var(self):
        # must not be branched...
        indices = freeze(
            {
                axis: AxisVariable(axis)
                for axis, _ in self.axes.path(*self.axes.leaf).items()
            }
        )
        return MultiArrayVariable(self, indices)

    @property
    def alloc_size(self):
        return self.axes.alloc_size() if not self.axes.is_empty else 1

    @classmethod
    def from_list(cls, data, axis_labels, name=None, dtype=ScalarType, inc=0):
        """Return a multi-array formed from a list of lists.

        The returned array must have one axis component per axis. These are
        permitted to be ragged.

        """
        flat, count = cls._get_count_data(data)
        flat = np.array(flat, dtype=dtype)

        if isinstance(count, Sequence):
            count = cls.from_list(count, axis_labels[:-1], name, dtype, inc + 1)
            subaxis = Axis(count, axis_labels[-1])
            axes = count.axes.add_subaxis(subaxis, count.axes.leaf)
        else:
            axes = AxisTree(Axis(count, axis_labels[-1]))

        assert axes.depth == len(axis_labels)
        return cls(axes, data=flat, dtype=dtype)

    @classmethod
    def _get_count_data(cls, data):
        # recurse if list of lists
        if not strictly_all(isinstance(d, collections.abc.Iterable) for d in data):
            return data, len(data)
        else:
            flattened = []
            count = []
            for d in data:
                x, y = cls._get_count_data(d)
                flattened.extend(x)
                count.append(y)
            return flattened, count

    def get_value(self, *args, **kwargs):
        return self.data[self.axes.get_offset(*args, **kwargs)]

    def set_value(self, path, indices, value):
        self.data[self.axes.get_offset(path, indices)] = value

    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)


# Needs to be subclass for isinstance checks to work
# TODO Delete
class MultiArray(Dat):
    @deprecated("Dat")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Now ContextSensitiveDat
class ContextSensitiveMultiArray(ContextSensitive):
    def __getitem__(self, indices) -> ContextSensitiveMultiArray:
        from pyop3.itree.tree import (
            _compose_bits,
            _index_axes,
            as_index_tree,
            collect_loop_contexts,
            index_axes,
        )

        loop_contexts = collect_loop_contexts(indices)
        if not loop_contexts:
            raise NotImplementedError("code path untested")

        # FIXME for now assume that there is only one context
        context, array = just_one(self.context_map.items())

        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=array.layout_axes):
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(array.layout_axes, index_tree, loop_context)

            (
                target_paths,
                index_exprs,
                layout_exprs,
            ) = _compose_bits(
                array.layout_axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = IndexedAxisTree(
                indexed_axes.root,
                indexed_axes.parent_to_children,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            new_layouts = substitute_layouts(
                array.layout_axes,
                new_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
            )
            layout_axes = FrozenAxisTree(
                new_axes.root,
                new_axes.parent_to_children,
                target_paths,
                index_exprs,
                layouts=new_layouts,
            )
            array_per_context[loop_context] = array._with_axes(layout_axes)
        return ContextSensitiveMultiArray(array_per_context)

    # don't like this name
    # FIXME This function returns dats, the "array" function returns a DistributedArray,
    # this is confusing and should be cleaned up
    @property
    def orig_array(self):
        return single_valued(dat.orig_array for dat in self.context_map.values())

    @property
    def array(self):
        return single_valued(dat.array for dat in self.context_map.values())

    @property
    def dtype(self):
        return single_valued(array.dtype for array in self.context_map.values())

    @property
    def name(self):
        return single_valued(array.name for array in self.context_map.values())

    @functools.cached_property
    def datamap(self):
        return merge_dicts(array.datamap for array in self.context_map.values())


def replace_layout(orig_layout, replace_map):
    return IndexExpressionReplacer(replace_map)(orig_layout)


@functools.singledispatch
def layout_axes(axes) -> FrozenAxisTree:
    if isinstance(axes, FrozenAxisTree):
        return axes
    else:
        return as_axis_tree(axes).freeze()


@layout_axes.register
def _(axes: IndexedAxisTree) -> FrozenAxisTree:
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if axes.is_empty:
        raise NotImplementedError
        # return LayoutAxisTree(axes, freeze({pmap(): 0}))
        return LayoutAxisTree(axes, NotImplemented)

    # relabel the axis tree, note that the strong statements (i.e. just_one(...), etc)
    # are *required* rather than me being lazy. If these conditions do not hold then
    # it is not valid to use this axis tree as a layout.
    new_root_labels = set()
    new_root_cpts = []
    for orig_cpt in axes.root.components:
        axlabel, clabel = just_one(
            axes.target_paths[axes.root.id, orig_cpt.label].items()
        )
        new_root_labels.add(axlabel)
        new_root_cpts.append(orig_cpt.copy(label=clabel))
    new_root_label = just_one(new_root_labels)
    new_root = axes.root.copy(label=new_root_label, components=new_root_cpts)

    new_parent_to_children = {}
    for axis_id, subaxes in axes.parent_to_children.items():
        new_subaxes = []
        for subaxis in subaxes:
            if subaxis is None:
                new_subaxes.append(None)
                continue
            axis_labels = set()
            cpts = []
            for orig_cpt in subaxis.components:
                axlabel, clabel = just_one(
                    axes.target_paths[subaxis.id, orig_cpt.label].items()
                )
                axis_labels.add(axlabel)
                cpts.append(orig_cpt.copy(label=clabel))
            axis_label = just_one(axis_labels)
            new_subaxes.append(subaxis.copy(label=axis_label, components=cpts))
        new_parent_to_children[axis_id] = new_subaxes

    new_axes = FrozenAxisTree(root=new_root, parent_to_children=new_parent_to_children)

    assert axes.layout_exprs
    layouts_ = {}
    for leaf in axes.leaves:
        orig_path = axes.path(*leaf)
        new_path = {}
        replace_map = {}
        for axis, cpt in axes.path_with_nodes(*leaf).items():
            new_path.update(axes.target_paths[axis.id, cpt])
            replace_map.update(axes.layout_exprs[axis.id, cpt])
        new_path = freeze(new_path)

        print(axes.layout_exprs.values())
        print(replace_map)

        orig_layout = axes.restore().layouts[orig_path]
        new_layout = IndexExpressionReplacer(replace_map)(orig_layout)
        assert new_layout != orig_layout
        layouts_[new_path] = new_layout
    return FrozenAxisTree(new_axes.root, new_axes.parent_to_children, layouts=layouts_)


def make_sparsity(
    iterindex,
    lmap,
    rmap,
    llabels=PrettyTuple(),
    rlabels=PrettyTuple(),
    lindices=PrettyTuple(),
    rindices=PrettyTuple(),
):
    if iterindex:
        if iterindex.children:
            raise NotImplementedError(
                "Need to think about what to do when we have more complicated "
                "iteration sets that have multiple indices (e.g. extruded cells)"
            )

        if not isinstance(iterindex, Range):
            raise NotImplementedError(
                "Need to think about whether maps are reasonable here"
            )

        if not is_single_valued(idx.id for idx in [iterindex, lmap, rmap]):
            raise ValueError("Indices must share common roots")

        sparsity = collections.defaultdict(set)
        for i in range(iterindex.size):
            subsparsity = make_sparsity(
                None,
                lmap.child,
                rmap.child,
                llabels | iterindex.label,
                rlabels | iterindex.label,
                lindices | i,
                rindices | i,
            )
            for labels, indices in subsparsity.items():
                sparsity[labels].update(indices)
        return sparsity
    elif lmap:
        if not isinstance(lmap, TabulatedMap):
            raise NotImplementedError("Need to think about other index types")
        if len(lmap.children) not in [0, 1]:
            raise NotImplementedError("Need to think about maps forking")

        new_labels = list(llabels)
        # first pop the old things
        for lbl in lmap.from_labels:
            if lbl != new_labels[-1]:
                raise ValueError("from_labels must match existing labels")
            new_labels.pop()
        # then append the new ones - only do the labels here, indices are
        # done inside the loop
        new_labels.extend(lmap.to_labels)
        new_labels = PrettyTuple(new_labels)

        sparsity = collections.defaultdict(set)
        for i in range(lmap.size):
            new_indices = PrettyTuple([lmap.data.get_value(lindices | i)])
            subsparsity = make_sparsity(
                None, lmap.child, rmap, new_labels, rlabels, new_indices, rindices
            )
            for labels, indices in subsparsity.items():
                sparsity[labels].update(indices)
        return sparsity
    elif rmap:
        if not isinstance(rmap, TabulatedMap):
            raise NotImplementedError("Need to think about other index types")
        if len(rmap.children) not in [0, 1]:
            raise NotImplementedError("Need to think about maps forking")

        new_labels = list(rlabels)
        # first pop the old labels
        for lbl in rmap.from_labels:
            if lbl != new_labels[-1]:
                raise ValueError("from_labels must match existing labels")
            new_labels.pop()
        # then append the new ones
        new_labels.extend(rmap.to_labels)
        new_labels = PrettyTuple(new_labels)

        sparsity = collections.defaultdict(set)
        for i in range(rmap.size):
            new_indices = PrettyTuple([rmap.data.get_value(rindices | i)])
            subsparsity = make_sparsity(
                None, lmap, rmap.child, llabels, new_labels, lindices, new_indices
            )
            for labels, indices in subsparsity.items():
                sparsity[labels].update(indices)
        return sparsity
    else:
        # at the bottom, record an entry
        # return {(llabels, rlabels): {(lindices, rindices)}}
        # TODO: For now assume single values for each of these
        llabel, rlabel = map(single_valued, [llabels, rlabels])
        lindex, rindex = map(single_valued, [lindices, rindices])
        return {(llabel, rlabel): {(lindex, rindex)}}


def distribute_sparsity(sparsity, ax1, ax2, owner="row"):
    if any(ax.nparts > 1 for ax in [ax1, ax2]):
        raise NotImplementedError("Only dealing with single-part multi-axes for now")

    # how many points need to get sent to other processes?
    # how many points do I get from other processes?
    new_sparsity = collections.defaultdict(set)
    points_to_send = collections.defaultdict(set)
    for lindex, rindex in sparsity[ax1.part.label, ax2.part.label]:
        if owner == "row":
            olabel = ax1.part.overlap[lindex]
            if is_owned_by_process(olabel):
                new_sparsity[ax1.part.label, ax2.part.label].add((lindex, rindex))
            else:
                points_to_send[olabel.root.rank].add(
                    (ax1.part.lgmap[lindex], ax2.part.lgmap[rindex])
                )
        else:
            raise NotImplementedError

    # send points

    # first determine how many new points we are getting from each rank
    comm = single_valued([ax1.sf.comm, ax2.sf.comm]).tompi4py()
    npoints_to_send = np.array(
        [len(points_to_send[rank]) for rank in range(comm.size)], dtype=IntType
    )
    npoints_to_recv = np.empty_like(npoints_to_send)
    comm.Alltoall(npoints_to_send, npoints_to_recv)

    # communicate the offsets back
    from_offsets = np.cumsum(npoints_to_recv)
    to_offsets = np.empty_like(from_offsets)
    comm.Alltoall(from_offsets, to_offsets)

    # now send the globally numbered row, col values for each point that
    # needs to be sent. This is easiest with an SF.

    # nroots is the number of points to send
    nroots = sum(npoints_to_send)
    local_points = None  # contiguous storage

    idx = 0
    remote_points = []
    for rank in range(comm.size):
        for i in range(npoints_to_recv[rank]):
            remote_points.extend([rank, to_offsets[idx]])
            idx += 1

    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, local_points, remote_points)

    # create a buffer to hold the new values
    # x2 since we are sending row and column numbers
    new_points = np.empty(sum(npoints_to_recv) * 2, dtype=IntType)
    rootdata = np.array(
        [
            num
            for rank in range(comm.size)
            for lnum, rnum in points_to_send[rank]
            for num in [lnum, rnum]
        ],
        dtype=new_points.dtype,
    )

    mpi_dtype, _ = get_mpi_dtype(np.dtype(IntType))
    mpi_op = MPI.REPLACE
    args = (mpi_dtype, rootdata, new_points, mpi_op)
    sf.bcastBegin(*args)
    sf.bcastEnd(*args)

    for i in range(sum(npoints_to_recv)):
        new_sparsity[ax1.part.label, ax2.part.label].add(
            (new_points[2 * i], new_points[2 * i + 1])
        )

    # import pdb; pdb.set_trace()
    return new_sparsity


def substitute_layouts(orig_axes, new_axes, target_paths, index_exprs):
    # replace layout bits that disappear with loop index
    if new_axes.is_empty:
        new_layouts = {}
        orig_path = target_paths[None]
        new_path = pmap()

        orig_layout = orig_axes.layouts[orig_path]
        new_layout = IndexExpressionReplacer(index_exprs[None])(orig_layout)
        new_layouts[new_path] = new_layout
        # don't silently do nothing
        assert new_layout != orig_layout
    else:
        new_layouts = {}
        for leaf_axis, leaf_cpt in new_axes.leaves:
            orig_path = dict(target_paths.get(None, {}))
            replace_map = dict(index_exprs.get(None, {}))
            for myaxis, mycpt in new_axes.path_with_nodes(leaf_axis, leaf_cpt).items():
                orig_path.update(target_paths.get((myaxis.id, mycpt), {}))
                replace_map.update(index_exprs.get((myaxis.id, mycpt), {}))

            orig_layout = orig_axes.layouts[freeze(orig_path)]
            new_layout = IndexExpressionReplacer(replace_map)(orig_layout)
            new_layouts[new_axes.path(leaf_axis, leaf_cpt)] = new_layout
            # TODO, this sometimes fails, is that valid?
            # don't silently do nothing
            # assert new_layout != orig_layout

    # TODO not sure if target paths etc needed to pass through
    return new_layouts
    # return FrozenAxisTree(new_axes.root, new_axes.parent_to_children, layouts=new_layouts)

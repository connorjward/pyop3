from __future__ import annotations

import collections
import functools
import itertools
import numbers
import operator
import sys
import threading
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pymbolic as pym
import pytools
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3 import utils
from pyop3.axes import (
    Axis,
    AxisComponent,
    AxisTree,
    ContextFree,
    ContextSensitive,
    as_axis_tree,
)
from pyop3.axes.tree import AxisVariable, MultiArrayCollector
from pyop3.distarray.base import DistributedArray
from pyop3.dtypes import IntType, ScalarType, get_mpi_dtype
from pyop3.indices import IndexedAxisTree, IndexTree, as_index_forest, index_axes
from pyop3.indices.tree import collect_loop_indices
from pyop3.utils import (
    PrettyTuple,
    UniqueNameGenerator,
    as_tuple,
    just_one,
    merge_dicts,
    single_valued,
    strict_int,
    strictly_all,
)


# should be elsewhere, this is copied from loopexpr2loopy VariableReplacer
class IndexExpressionReplacer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        self._replace_map = replace_map

    def map_axis_variable(self, expr):
        return self._replace_map.get(expr.axis_label, expr)

    def map_multi_array(self, expr):
        indices = tuple(self.rec(index) for index in expr.index_tuple)
        return MultiArrayVariable(expr.array, indices)


class MultiArrayVariable(pym.primitives.Subscript):
    mapper_method = sys.intern("map_multi_array")

    def __init__(self, array, indices):
        super().__init__(pym.var(array.name), indices)
        self.array = array

        # alias
        self.indices = indices

    @property
    def datamap(self):
        return self.array.datamap | merge_dicts(idx.datamap for idx in self.indices)


class MultiArray(DistributedArray, ContextFree):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------
    sf : ???
        PETSc star forest connecting values (offsets) in the local array with
        remote equivalents.

    """

    fields = DistributedArray.fields | {
        "axes",
        "dtype",
        "name",
        "data",
        "max_value",
        "sf",
    }

    prefix = "array"
    name_generator = UniqueNameGenerator()

    def __init__(
        self,
        axes: AxisTree,
        dtype=None,
        *,
        name: str = None,
        prefix: str = None,
        data=None,
        max_value=None,
        sf=None,
    ):
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")

        # if not isinstance(axes, (AxisTree, IndexedAxisTree)):
        #     # must be context-free
        #     raise TypeError()

        if isinstance(data, np.ndarray):
            if dtype:
                data = np.asarray(data, dtype=dtype)
            else:
                dtype = data.dtype
        elif isinstance(data, Sequence):
            data = np.asarray(data, dtype=dtype)
            dtype = data.dtype
        elif data is None:
            if not dtype:
                raise ValueError("Must either specify a dtype or provide an array")
            dtype = np.dtype(dtype)
            data = np.zeros(axes.size, dtype=dtype)
        else:
            raise TypeError("data argument not recognised")

        # add prefix as an existing name so it is a true prefix
        if prefix:
            self.name_generator.add_name(prefix, conflicting_ok=True)
        name = name or self.name_generator(prefix or self.prefix)

        DistributedArray.__init__(self, name)
        # pym.primitives.Variable.__init__(self, name)

        self._data = data
        self.dtype = dtype

        self.axes = layout_axes(axes)

        self.max_value = max_value
        self.sf = sf

        self._pending_write_op = None
        self._halo_modified = False
        self._halo_valid = True

        self._sync_thread = None

    def __getitem__(
        self, indices
    ) -> Union[IndexedMultiArray, ContextSensitiveMultiArray]:
        from pyop3.indices.tree import (
            _compose_bits,
            _index_axes,
            as_index_tree,
            collect_loop_contexts,
            index_axes,
        )

        loop_contexts = collect_loop_contexts(indices)
        if not loop_contexts:
            index_tree = just_one(as_index_forest(indices, axes=self.axes))
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.axes, index_tree, pmap())
            target_paths, index_exprs, layout_exprs, *oldleafdata = _compose_bits(
                self.axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = IndexedAxisTree(
                indexed_axes,
                target_path_per_indexed_cpt,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            layout_axes = substitute_layouts(
                self.axes,
                new_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
            )
            return self.copy_record(axes=layout_axes)

        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=self.axes):
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.axes, index_tree, loop_context)

            (
                target_paths,
                index_exprs,
                layout_exprs,
                leaf_target_paths,
                leaf_index_exprs,
            ) = _compose_bits(
                self.axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = IndexedAxisTree(
                indexed_axes,
                target_path_per_indexed_cpt,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            layout_axes = substitute_layouts(
                self.axes,
                new_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
            )
            array_per_context[loop_context] = self.copy_record(axes=layout_axes)
        return ContextSensitiveMultiArray(array_per_context)

    # TODO remove this
    @property
    def layouts(self):
        return self.axes.layouts

    @property
    def data(self):
        import warnings

        warnings.warn("deprecate this")
        return self.data_rw

    @property
    def data_rw(self):
        return self._data

    @property
    def data_ro(self):
        # TODO
        return self._data

    @property
    def data_wo(self):
        # TODO
        return self._data

    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
        datamap_ = {self.name: self}
        datamap_.update(self.axes.datamap)
        for layout_expr in self.layouts.values():
            if isinstance(layout_expr, pym.primitives.NaN):
                continue
            for array in MultiArrayCollector()(layout_expr):
                datamap_.update(array.datamap)
        return freeze(datamap_)

    def as_var(self):
        # must not be branched...
        indices = tuple(
            AxisVariable(axis)
            for axis, _ in self.axes.path(*self.axes.leaf, ordered=True)
        )
        return MultiArrayVariable(self, indices)

    @property
    def alloc_size(self):
        return self.axes.alloc_size() if not self.axes.is_empty else 1

    # ???
    # @property
    # def pym_var(self) -> pym.primitives.Variable:
    #     return MultiArray

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

    @property
    def reduction_ops(self):
        from pyop3.loopexprs import INC

        return {
            INC: MPI.SUM,
        }

    def reduce_leaves_to_roots(self, sf, pending_write_op):
        mpi_dtype, _ = get_mpi_dtype(self.data.dtype)
        mpi_op = self.reduction_ops[pending_write_op]
        args = (mpi_dtype, self.data, self.data, mpi_op)
        sf.reduceBegin(*args)
        sf.reduceEnd(*args)

    def broadcast_roots_to_leaves(self, sf):
        mpi_dtype, _ = get_mpi_dtype(self.data.dtype)
        mpi_op = MPI.REPLACE
        args = (mpi_dtype, self.data, self.data, mpi_op)
        sf.bcastBegin(*args)
        sf.bcastEnd(*args)

    def sync_begin(self, need_halo_values=False):
        """Begin synchronizing shared data."""
        self._sync_thread = threading.Thread(
            target=self.__class__.sync,
            args=(self,),
            kwargs={"need_halo_values": need_halo_values},
        )
        self._sync_thread.start()

    def sync_end(self):
        """Finish synchronizing shared data."""
        if not self._sync_thread:
            raise RuntimeError(
                "Cannot call sync_end without a prior call to sync_begin"
            )
        self._sync_thread.join()
        self._sync_thread = None

    # TODO create Synchronizer object for encapsulation?
    def sync(self, need_halo_values=False):
        """Perform halo exchanges to ensure that all ranks store up-to-date values.

        Parameters
        ----------
        need_halo_values : bool
            Whether or not halo values also need to be synchronized.

        Notes
        -----
        This is a blocking operation. F.labelor the non-blocking alternative use
        :meth:`sync_begin` and :meth:`sync_end` (FIXME)

        Note that this method should only be called when one needs to read from
        the array.
        """
        # 1. Reduce leaf values to roots if they have been written to.
        # (this is basically local-to-global)
        if self._pending_write_op:
            assert (
                not self._halo_valid
            ), "If a write is pending the halo cannot be valid"
            # If halo entries have also been written to then we need to use the
            # full SF containing both shared and halo points. If the halo has not
            # been modified then we only need to reduce with shared points.
            if self._halo_modified:
                self.reduce_leaves_to_roots(self.root.sf, self._pending_write_op)
            else:
                # only reduce with values from owned points
                self.reduce_leaves_to_roots(self.root.shared_sf, self._pending_write_op)

        # implicit barrier? can only broadcast reduced values

        # 3. at this point only one of the owned points knows the correct result which
        # now needs to be scattered back to some (but not necessarily all) of the other ranks.
        # (this is basically global-to-local)

        # only send to halos if we want to read them and they are out-of-date
        if need_halo_values and not self._halo_valid:
            # send the root value back to all the points
            self.broadcast_roots_to_leaves(self.root.sf)
            self._halo_valid = True  # all the halo points are now up-to-date
        else:
            # only need to update owned points if we did a reduction earlier
            if self._pending_write_op:
                # send the root value back to just the owned points
                self.broadcast_roots_to_leaves(self.root.shared_sf)
                # (the halo is still dirty)

        # set self.last_op to None here? what if halo is still off?
        # what if we read owned values and then owned+halo values?
        # just perform second step
        self._pending_write_op = None
        self._halo_modified = False

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

    def __str__(self):
        return self.name


class IndexedMultiArray(pytools.ImmutableRecord, ContextFree):
    fields = {"array", "axes"}

    def __init__(self, array, axes):
        super().__init__()
        self.array = array
        self.axes = axes

    def __getitem__(
        self, indices
    ) -> Union[IndexedMultiArray, ContextSensitiveMultiArray]:
        raise NotImplementedError("TODO")

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def name(self):
        return self.array.name

    @functools.cached_property
    def datamap(self):
        datamap_ = dict(self.array.datamap)
        datamap_.update(self.axes.datamap)
        for layout_expr in self.layouts.values():
            for array in MultiArrayCollector()(layout_expr):
                datamap_.update(array.datamap)
        return freeze(datamap_)


class ContextSensitiveMultiArray(ContextSensitive):
    def __getitem__(self, indices) -> ContextSensitiveMultiArray:
        from pyop3.indices.tree import (
            _compose_bits,
            _index_axes,
            as_index_tree,
            collect_loop_contexts,
            index_axes,
        )

        loop_contexts = collect_loop_contexts(indices)
        if not loop_contexts:
            raise NotImplementedError("code path untested")
            (
                indexed_axes,
                target_paths_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.axes, indices, pmap())
            target_paths, index_exprs, layout_exprs = _compose_bits(
                self.axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = indexed_axes.copy(
                target_paths=target_paths,
                index_exprs=index_exprs,
                layout_exprs=layout_exprs,
            )

            new_layouts = IndexExpressionReplacer(leaf_index_exprs)(NotImplemented)
            return self.copy(axes=new_axes, layouts=new_layouts)

        # FIXME for now assume that there is only one context
        context, array = just_one(self.context_map.items())

        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=array.axes):
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(array.axes, index_tree, loop_context)

            (
                target_paths,
                index_exprs,
                layout_exprs,
                leaf_target_paths,
                leaf_index_exprs,
            ) = _compose_bits(
                array.axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = IndexedAxisTree(
                indexed_axes,
                target_path_per_indexed_cpt,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            layout_axes = substitute_layouts(
                array.axes,
                new_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
            )
            array_per_context[loop_context] = array.copy_record(axes=layout_axes)
        return ContextSensitiveMultiArray(array_per_context)

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


class LayoutAxisTree(pytools.ImmutableRecord):
    fields = {"axes", "layouts"}

    def __init__(self, axes, layouts):
        super().__init__()
        self.axes = axes
        self.layouts = layouts

    # TODO I think inheritance should allow us to undo a lot of the property passthrough stuff
    def __getattr__(self, name):
        return getattr(self.axes, name)


@functools.singledispatch
def layout_axes(axes) -> LayoutAxisTree:
    if isinstance(axes, LayoutAxisTree):
        return axes
    else:
        axes = as_axis_tree(axes)
        return LayoutAxisTree(axes, axes._default_layouts())


@layout_axes.register
def _(axes: IndexedAxisTree) -> LayoutAxisTree:
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if axes.is_empty:
        # return LayoutAxisTree(axes, freeze({pmap(): 0}))
        breakpoint()
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

    new_axes = axes.axes.copy(root=new_root, parent_to_children=new_parent_to_children)

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

        orig_layout = axes.axes._default_layouts()[orig_path]
        new_layout = IndexExpressionReplacer(replace_map)(orig_layout)
        assert new_layout != orig_layout
        layouts_[new_path] = new_layout
    return LayoutAxisTree(new_axes, layouts_)


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

        if not utils.is_single_valued(idx.id for idx in [iterindex, lmap, rmap]):
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


def overlap_axes(ax1, ax2, sparsity=None):
    """Combine two multiaxes, possibly sparsely."""
    if ax1.depth != 1 or ax2.depth != 1:
        raise NotImplementedError(
            "Need to think about composition rules for nested axes"
        )

    new_parts = []
    for pt1 in ax1.parts:
        new_subparts = []
        for pt2 in ax2.parts:
            # some initial checks
            if any(not isinstance(pt.count, numbers.Integral) for pt in [pt1, pt2]):
                raise NotImplementedError(
                    "Need to think about non-integral sized axis parts"
                )

            # now do the real work
            count = []
            indices = []
            for i1 in range(pt1.count):
                indices_per_row = []
                for i2 in range(pt2.count):
                    # if ((i1,), (i2,)) in sparsity[(pt1.label,), (pt2.label,)]:
                    if (i1, i2) in sparsity[(pt1.label, pt2.label)]:
                        indices_per_row.append(i2)
                count.append(len(indices_per_row))
                indices.append(indices_per_row)

            count = MultiArray.from_list(
                count, labels=[pt1.label], name="count", dtype=IntType
            )
            indices = MultiArray.from_list(
                indices, labels=[pt1.label, "any"], name="indices", dtype=IntType
            )

            # FIXME: I think that the inner axis count should be "full", not
            # the number of indices. This means that we need to use the
            # number of indices when computing layouts.
            new_subpart = pt2.copy(count=count, indices=indices)
            new_subparts.append(new_subpart)

        subaxis = MultiAxis(new_subparts)
        new_part = pt1.copy(subaxis=subaxis)
        new_parts.append(new_part)

    return MultiAxis(new_parts).set_up(with_sf=False)


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
            for myaxis, mycpt in new_axes.path_with_nodes(leaf_axis, leaf_cpt).items():
                orig_path.update(target_paths.get((myaxis.id, mycpt), {}))

            orig_layout = orig_axes.layouts[freeze(orig_path)]
            new_layout = IndexExpressionReplacer(index_exprs.get(None, {}))(orig_layout)
            new_layouts[new_axes.path(leaf_axis, leaf_cpt)] = new_layout
            # TODO, this sometimes fails, is that valid?
            # don't silently do nothing
            # assert new_layout != orig_layout

    return LayoutAxisTree(new_axes, new_layouts)

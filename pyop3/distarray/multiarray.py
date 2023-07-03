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

from pyop3 import utils
from pyop3.axis import (
    Axis,
    AxisComponent,
    AxisTree,
    as_axis_tree,
    fill_shape,
    get_bottom_part,
)
from pyop3.distarray.base import DistributedArray
from pyop3.dtypes import IntType, ScalarType, get_mpi_dtype
from pyop3.index import IndexTree, TabulatedMap, as_index_tree
from pyop3.utils import PrettyTuple, just_one, single_valued, strict_int, strictly_all


class MultiArray(DistributedArray, pym.primitives.Variable):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------
    sf : ???
        PETSc star forest connecting values (offsets) in the local array with
        remote equivalents.

    """

    fields = DistributedArray.fields | {
        "dim",
        "dtype",
        "name",
        "data",
        "max_value",
        "sf",
        "index",
    }

    mapper_method = sys.intern("map_multi_array")

    prefix = "ten"
    name_generator = pytools.UniqueNameGenerator()

    def __init__(
        self,
        dim,
        dtype=None,
        *,
        name: str = None,
        prefix: str = None,
        data=None,
        max_value=None,
        sf=None,
        index=None,
    ):
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")
        dim = as_axis_tree(dim)

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
            data = np.zeros(dim.calc_size(dim.root), dtype=dtype)
        else:
            raise TypeError("data argument not recognised")

        # add prefix as an existing name so it is a true prefix
        if prefix:
            self.name_generator.add_name(prefix, conflicting_ok=True)
        name = name or self.name_generator(prefix or self.prefix)

        DistributedArray.__init__(self, name)
        pym.primitives.Variable.__init__(self, name)

        self._data = data
        self.dtype = dtype

        self.index = index

        self.dim = dim

        self.max_value = max_value
        self.sf = sf

        self._pending_write_op = None
        self._halo_modified = False
        self._halo_valid = True

        self._sync_thread = None

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
        return {self.name: self} | self.axes.datamap

    @property
    def alloc_size(self):
        return self.axes.alloc_size() if self.axes else 1

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
        This is a blocking operation. For the non-blocking alternative use
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

    def get_value(self, indices):
        offset = self.root.get_offset(indices)
        return self.data[int(offset)]

    def set_value(self, indices, value):
        offset = strict_int(self.root.get_offset(indices))
        self.data[offset] = value

    # aliases, this is preferred to dim
    @property
    def axes(self):
        return self.dim

    @property
    def root(self):
        return self.dim

    def __getitem__(self, index: IndexTree | Index | IndexComponent):
        if index is Ellipsis:
            new_index = fill_shape(self.axes, self.index)
        else:
            index = as_index_tree(index)
            if not self.index:
                new_index = fill_shape(self.axes, index)
            else:
                # TODO add_subtree is broken but would be a nice alternative here
                root = self.index.root
                # convert children to a list so I can mutate it
                parent_to_children = {
                    k: list(v) for k, v in self.index.parent_to_children.items()
                }

                # FIXME uniquify
                if len(self.index.leaves) > 1:
                    raise NotImplementedError("need to uniquify")

                for leaf, icpt in self.index.leaves:
                    cidx = [c.label for c in leaf.components].index(icpt.label)
                    parent_to_children[leaf.id][cidx] = index.root
                    parent_to_children |= index.parent_to_children

                new_index = IndexTree(root, parent_to_children)
        return self.copy(index=new_index)

    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)

    def __str__(self):
        return self.name

    @property
    def is_indexed(self):
        assert False, "not touched"
        return all(self._check_indexed(self.dim, idxs) for idxs in self.indicess)

    def _check_indexed(self, dim, indices):
        assert False, "not touched"
        for label, size in zip(dim.labels, dim.sizes):
            try:
                ((index, subindices),) = [
                    (idx, subidxs) for idx, subidxs in indices if idx.label == label
                ]

                npart = dim.labels.index(index.label)

                if subdims := self.dim.get_children(dim):
                    subdim = subdims[npart]
                    return self._check_indexed(subdim, subindices)
                else:
                    return index.size != size
            except:
                return True

    @property
    def indexed_shape(self):
        assert False, "not touched"
        try:
            (sh,) = self.indexed_shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def indexed_shapes(self):
        assert False, "not touched"
        return indexed_shapes(self)

    @property
    def indexed_size(self):
        assert False, "not touched"
        return functools.reduce(operator.mul, self.indexed_shape, 1)

    @property
    def shape(self):
        assert False, "not touched"
        try:
            (sh,) = self.shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def shapes(self):
        assert False, "not touched"
        return self._compute_shapes(self.dim)

    @property
    def size(self):
        assert False, "not touched"
        return functools.reduce(operator.mul, self.shape, 1)


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

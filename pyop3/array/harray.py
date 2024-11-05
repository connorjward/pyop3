from __future__ import annotations

import abc
import collections
import contextlib
import sys
from functools import cached_property
from typing import Any, Sequence

import numpy as np
import pymbolic as pym
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.array.base import Array
from pyop3.axtree import (
    Axis,
    ContextSensitive,
    AxisTree,
    as_axis_tree,
)
from pyop3.axtree.tree import ContextSensitiveAxisTree, subst_layouts
from pyop3.buffer import Buffer, DistributedBuffer
from pyop3.dtypes import ScalarType
from pyop3.exceptions import Pyop3Exception
from pyop3.lang import KernelArgument, Assignment
from pyop3.log import warning
from pyop3.utils import (
    Record,
    deprecated,
    just_one,
    strictly_all,
)


# is this used?
class IncompatibleShapeError(Exception):
    """TODO, also bad name"""


class AxisMismatchException(Pyop3Exception):
    pass


class FancyIndexWriteException(Exception):
    pass


class _Dat(Array, KernelArgument, Record, abc.ABC):

    DEFAULT_DTYPE = Buffer.DEFAULT_DTYPE

    @classmethod
    def _parse_buffer(cls, data, dtype, size, name):
        if data is not None:
            if isinstance(data, Buffer):
                return data


            assert isinstance(data, np.ndarray)
            assert dtype is None or dtype == data.dtype

            dtype = data.dtype

            # always deal with flattened data
            if len(data.shape) > 1:
                data = data.flatten()
        elif dtype is None:
            data = np.zeros(size, dtype=dtype)
        else:
            dtype = cls.DEFAULT_DTYPE
            data = np.zeros(size, dtype=dtype)

        buffer = DistributedBuffer(
            data.size,  # not a useful property anymore
            dtype,
            name=name,
            data=data,
        )

        return buffer

    @property
    def alloc_size(self):
        return self.axes.alloc_size

    @property
    def size(self):
        return self.axes.size

    @property
    def kernel_dtype(self):
        # TODO Think about the fact that the dtype refers to either to dtype of the
        # array entries (e.g. double), or the dtype of the whole thing (double*)
        return self.dtype

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
            axes = count.axes.add_axis(subaxis, count.axes.leaf)
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

    def get_value(self, indices, path=None, *, loop_exprs=pmap()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        return self.buffer.data_ro[offset]

    def set_value(self, indices, value, path=None, *, loop_exprs=pmap()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        self.buffer.data_wo[offset] = value

    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)

    # better to call copy
    def copy2(self):
        assert False, "old?"
        return type(self)(
            self.axes,
            data=self.buffer.copy(),
            max_value=self.max_value,
            name=f"{self.name}_copy",
            constant=self.constant,
        )

    # assign is a much better name for this
    def copy(self, other, subset=Ellipsis):
        """Copy the contents of the array into another."""
        # NOTE: Is copy_to/copy_into a clearer name for this?
        # TODO: Check that self and other are compatible, should have same axes and dtype
        # for sure
        # TODO: We can optimise here and copy the private data attribute and set halo
        # validity. Here we do the simple but hopefully correct thing.
        # None is an old potential argument here.
        if subset is Ellipsis or subset is None:
            other.data_wo[...] = self.data_ro
        else:
            self[subset].assign(other[subset])

    def zero(self, *, subset=Ellipsis, eager=False):
        # old Firedrake code may hit this, should probably raise a warning
        if subset is None:
            subset = Ellipsis

        expr = Assignment(self[subset], 0, "write")
        return expr() if eager else expr


class Dat(_Dat):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------

    """

    def __init__(
        self,
        axes,
        dtype=None,
        *,
        data=None,
        max_value=None,
        name=None,
        prefix=None,
        constant=False,
        parent=None,
    ):
        super().__init__(name=name, prefix=prefix, parent=parent)

        axes = as_axis_tree(axes)

        self.buffer = self._parse_buffer(data, dtype, axes.size, self.name)
        self.axes = axes
        self.max_value = max_value

        # TODO This attr really belongs to the buffer not the array
        self.constant = constant

        # self._cache = {}

    @property
    def _record_fields(self) -> frozenset:
        return frozenset({"axes", "buffer", "max_value", "name", "constant", "parent"})

    def __str__(self) -> str:
        return "\n".join(
            f"{self.name}[{self.axes.subst_layouts()[self.axes.path(leaf)]}]"
            for leaf in self.axes.leaves
        )

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    # TODO: redo now that we have Record?
    def __hash__(self) -> int:
        return hash(
            (
                type(self), self.axes, self.dtype, self.buffer, self.max_value, self.name, self.constant)
        )

    def getitem(self, indices, *, strict=False):
        from pyop3.itree import as_index_forest, index_axes

        if indices is Ellipsis:
            return self

        # key = (indices, strict)
        # if key in self._cache:
        #     return self._cache[key]

        index_forest = as_index_forest(indices, axes=self.axes, strict=strict)

        if len(index_forest) == 1:
            # There is no outer loop context to consider. Needn't return a
            # context sensitive object.
            index_trees = just_one(index_forest.values())

            # Loop over "restricted" index trees. This is necessary because maps
            # can yield multiple equivalent indexed axis trees. For example,
            # closure(cell) can map any of:
            #
            #   "points"  ->  {"points"}
            #   "points"  ->  {"cells", "edges", "vertices"}
            #   "cells"   ->  {"points"}
            #   "cells"   ->  {"cells", "edges", "vertices"}
            #
            # In each case the required arrays are different from each other and the
            # resulting axis tree is also different. Hence in order for things to work
            # we need to consider each of these separately and produce an axis *forest*.
            indexed_axess = []
            for restricted_index_tree in index_trees:
                indexed_axes = index_axes(restricted_index_tree, pmap(), self.axes)
                indexed_axess.append(indexed_axes)

            if len(indexed_axess) > 1:
                raise NotImplementedError("Need axis forests")
            else:
                indexed_axes = just_one(indexed_axess)
                dat = self.reconstruct(axes=indexed_axes)
        else:
            raise NotImplementedError
            context_sensitive_axes = {}
            for loop_context, index_tree in index_forest.items():
                indexed_axes = index_axes(index_tree, loop_context, self.axes)
                breakpoint()
                axes = compose_axes(indexed_axes, self.axes)
                context_sensitive_axes[loop_context] = axes
            context_sensitive_axes = ContextSensitiveAxisTree(context_sensitive_axes)

            dat = self.reconstruct(axes=context_sensitive_axes)
        # self._cache[key] = dat
        return dat

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    def with_context(self, context):
        return self.reconstruct(axes=self.axes.with_context(context))

    @property
    def context_free(self):
        return self.reconstruct(axes=self.axes.context_free)

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    @deprecated(".data_rw")
    def data(self):
        return self.data_rw

    @property
    def data_rw(self):
        self._check_no_copy_access()
        return self.buffer.data_rw[self.axes._buffer_indices]

    @property
    def data_ro(self):
        if not isinstance(self.axes._buffer_indices, slice):
            warning(
                "Read-only access to the array is provided with a copy, "
                "consider avoiding if possible."
            )
        return self.buffer.data_ro[self.axes._buffer_indices]

    @property
    def data_wo(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should
        call `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        self._check_no_copy_access()
        return self.buffer.data_wo[self.axes._buffer_indices]

    @property
    @deprecated(".data_rw_with_halos")
    def data_with_halos(self):
        return self.data_rw_with_halos

    @property
    def data_rw_with_halos(self):
        self._check_no_copy_access(include_ghost_points=True)
        return self.buffer.data_rw_with_halos[self.axes._buffer_indices_ghost]

    @property
    def data_ro_with_halos(self):
        if not isinstance(self.axes._buffer_indices_ghost, slice):
            warning(
                "Read-only access to the array is provided with a copy, "
                "consider avoiding if possible."
            )
        return self.buffer.data_ro_with_halos[self.axes._buffer_indices_ghost]

    @property
    def data_wo_with_halos(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should
        call `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        self._check_no_copy_access(include_ghost_points=True)
        return self.buffer.data_wo_with_halos[self.axes._buffer_indices_ghost]

    def _check_no_copy_access(self, *, include_ghost_points=False):
        if include_ghost_points:
            buffer_indices = self.axes._buffer_indices_ghost
        else:
            buffer_indices = self.axes._buffer_indices

        if not isinstance(buffer_indices, slice):
            raise FancyIndexWriteException(
                "Writing to the array directly is not supported for "
                "non-trivially indexed (i.e. sliced) arrays."
            )

    # TODO: It is inefficient (I think) to create a new vec every time, even
    # if we are reusing the underlying array. Care must be taken though because
    # sometimes we cannot create write-able vectors and use a copy (when fancy
    # indexing is required).
    @property
    @contextlib.contextmanager
    def vec_rw(self):
        self._check_vec_dtype()
        yield PETSc.Vec().createWithArray(self.data_rw, comm=self.comm)

    @property
    @contextlib.contextmanager
    def vec_ro(self):
        self._check_vec_dtype()
        yield PETSc.Vec().createWithArray(self.data_ro, comm=self.comm)

    @property
    @contextlib.contextmanager
    def vec_wo(self):
        self._check_vec_dtype()
        yield PETSc.Vec().createWithArray(self.data_wo, comm=self.comm)

    @property
    @deprecated(".vec_rw")
    def vec(self):
        return self.vec_rw

    def concretize(self):
        return _ConcretizedDat(self.axes, self.axes.leaf_subst_layouts, data=self.buffer, name=self.name)

    def _check_vec_dtype(self):
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a Vec with data type {self.dtype}, "
                f"must be {PETSc.ScalarType}"
            )

    @property
    def outer_loops(self):
        return self._outer_loops

    @property
    def sf(self):
        return self.buffer.sf

    @property
    def comm(self):
        return self.buffer.comm

    # TODO update docstring
    # TODO is this a property of the buffer?
    def assemble(self, update_leaves=False):
        """Ensure that stored values are up-to-date.

        This function is typically only required when accessing the `Dat` in a
        write-only mode (`Access.WRITE`, `Access.MIN_WRITE` or `Access.MAX_WRITE`)
        and only setting a subset of the values. Without `Dat.assemble` the non-subset
        entries in the array would hold undefined values.

        """
        if update_leaves:
            self.buffer._reduce_then_broadcast()
        else:
            self.buffer._reduce_leaves_to_roots()

    def materialize(self) -> Dat:
        """Return a new "unindexed" array with the same shape."""
        assert False, "old code"
        return type(self)(self.axes.materialize(), dtype=self.dtype)


    def reshape(self, axes: AxisTree) -> Dat:
        """Return a reshaped view of the `Dat`.

        TODO

        """
        assert isinstance(axes, AxisTree), "not indexed"

        return self.reconstruct(axes=axes, parent=self)

    # NOTE: should this only accept AxisTrees, or are IndexedAxisTrees fine also?
    # is this ever used?
    def with_axes(self, axes) -> Dat:
        """Return a view of the current `Dat` with new axes.

        Parameters
        ----------
        axes
            XXX (type?)

        Returns
        -------
        Dat
            XXX

        """
        if axes.size != self.axes.size:
            raise AxisMismatchException(
                "New axis tree is a different size to the existing one."
            )

        return self.reconstruct(axes=axes)


class _ConcretizedDat(_Dat):
    """A dat with fixed (?) layout.

    It cannot be indexed.

    This class is useful for describing arrays used in index expressions, at which
    point it has a fixed set of axes.

    """
    def __init__(self, axes, layouts=None, *, data=None, dtype=None, name: str = None, prefix: str = None, constant: bool = False):
        if layouts is None:
            layouts = {
                leaf_path: axes.layouts[leaf_path] for leaf_path in axes.leaf_paths
            }
        layouts = pmap(layouts)


        super().__init__(name=name, prefix=prefix, parent=None)

        buffer = self._parse_buffer(data, dtype, axes.size, self.name)

        self.axes = axes
        self.layouts = layouts
        self.buffer = buffer
        self.constant = constant

    # TODO: redo now that we have Record?
    def __hash__(self) -> int:
        return hash((type(self), self.axes, self.layouts, self.buffer, self.constant))

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def layout(self):
        return just_one(self.layouts.values())

    def with_context(self, context):
        assert False, "not appropriate"

    @property
    def context_free(self):
        assert False, "not appropriate"

    @property
    def _record_fields(self) -> frozenset:
        return frozenset({"axes", "layouts", "buffer", "name", "constant"})

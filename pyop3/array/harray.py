from __future__ import annotations

import collections
import contextlib
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
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.array.base import Array
from pyop3.axtree import (
    Axis,
    AxisTree,
    ContextFree,
    as_axis_tree,
)
from pyop3.axtree.tree import IndexedAxisTree, MultiArrayCollector, ContextSensitiveAxisTree
from pyop3.buffer import Buffer, DistributedBuffer
from pyop3.dtypes import ScalarType
from pyop3.lang import KernelArgument, ReplaceAssignment
from pyop3.log import warning
from pyop3.sf import serial_forest
from pyop3.utils import (
    PrettyTuple,
    UniqueNameGenerator,
    as_tuple,
    debug_assert,
    deprecated,
    is_single_valued,
    just_one,
    merge_dicts,
    readonly,
    single_valued,
    some_but_not_all,
    strict_int,
    strictly_all,
)


class IncompatibleShapeError(Exception):
    """TODO, also bad name"""


class ArrayVar(pym.primitives.AlgebraicLeaf):
    mapper_method = sys.intern("map_array")

    def __init__(self, array, indices, path=None):
        assert path is not None
        if path is None:
            if array.axes.is_empty:
                path = pmap()
            else:
                path = just_one(array.axes.leaf_paths)

        super().__init__()
        self.array = array
        self.indices = freeze(indices)
        self.path = freeze(path)

    def __getinitargs__(self):
        return (self.array, self.indices, self.path)


from pymbolic.mapper.stringifier import PREC_CALL, PREC_NONE, StringifyMapper


# This was adapted from pymbolic's map_subscript
def stringify_array(self, array, enclosing_prec, *args, **kwargs):
    index_str = self.join_rec(
        ", ", array.index_exprs.values(), PREC_NONE, *args, **kwargs
    )

    return self.parenthesize_if_needed(
        self.format("%s[%s]", array.name, index_str), enclosing_prec, PREC_CALL
    )


pym.mapper.stringifier.StringifyMapper.map_array = stringify_array


CalledMapVariable = ArrayVar


class FancyIndexWriteException(Exception):
    pass


class HierarchicalArray(Array, KernelArgument):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------

    """

    DEFAULT_DTYPE = Buffer.DEFAULT_DTYPE

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
    ):
        super().__init__(name=name, prefix=prefix)

        axes = as_axis_tree(axes)

        if isinstance(data, Buffer):
            # disable for now, temporaries hit this in an annoying way
            # if data.sf is not axes.sf:
            #     raise ValueError("Star forests do not match")
            if dtype is not None:
                raise ValueError("If data is a Buffer, dtype should not be provided")
            pass
        else:
            if isinstance(data, np.ndarray):
                dtype = dtype or data.dtype
            else:
                dtype = dtype or self.DEFAULT_DTYPE

            if data is not None:
                data = np.asarray(data, dtype=dtype)

                # always deal with flattened data
                if len(data.shape) > 1:
                    data = data.flatten()
                if data.size != axes.alloc_size:
                    raise ValueError("Data shape does not match axes")

            # IndexedAxisTrees do not currently have SFs, so create a dummy one here
            if isinstance(axes, AxisTree):
                sf = axes.sf
            else:
                assert isinstance(axes, (ContextSensitiveAxisTree, IndexedAxisTree))
                # not sure this is the right thing to do
                sf = serial_forest(axes.alloc_size)

            data = DistributedBuffer(
                axes.alloc_size,  # not a useful property anymore
                sf,
                dtype,
                name=self.name,
                data=data,
            )

        self.buffer = data
        self._axes = axes
        self.max_value = max_value

        # TODO This attr really belongs to the buffer not the array
        self.constant = constant

        # self._cache = {}

    def __str__(self):
        return self.name

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    def getitem(self, indices, *, strict=False):
        from pyop3.itree.tree import as_index_forest, compose_axes, index_axes

        if indices is Ellipsis:
            return self

        # key = (indices, strict)
        # if key in self._cache:
        #     return self._cache[key]

        index_forest = as_index_forest(indices, axes=self.axes, strict=strict)
        if index_forest.keys() == {pmap()}:
            index_tree = index_forest[pmap()]
            indexed_axes = index_axes(index_tree, pmap(), self.axes)
            axes = compose_axes(indexed_axes, self.axes)
            dat = HierarchicalArray(
                axes, data=self.buffer, max_value=self.max_value, name=self.name
            )
            # self._cache[key] = dat
            return dat

        context_sensitive_axes = {}
        for loop_context, index_tree in index_forest.items():
            indexed_axes = index_axes(index_tree, loop_context, self.axes)
            axes = compose_axes(indexed_axes, self.axes)
            context_sensitive_axes[loop_context] = axes
        context_sensitive_axes = ContextSensitiveAxisTree(context_sensitive_axes)

        dat = HierarchicalArray(
            context_sensitive_axes, data=self.buffer, name=self.name, max_value=self.max_value
        )
        # self._cache[key] = dat
        return dat

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    def with_context(self, context):
        return type(self)(
            self.axes.with_context(context),
            name=self.name,
            data=self.buffer,
            max_value=self.max_value,
            constant=self.constant,
        )

    @property
    def context_free(self, context):
        return type(self)(
            self.axes.context_free,
            name=self.name,
            data=self.buffer,
            max_value=self.max_value,
            constant=self.constant,
        )

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def kernel_dtype(self):
        # TODO Think about the fact that the dtype refers to either to dtype of the
        # array entries (e.g. double), or the dtype of the whole thing (double*)
        return self.dtype

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

    def _check_vec_dtype(self):
        if self.dtype != PETSc.ScalarType:
            raise RuntimeError(
                f"Cannot create a Vec with data type {self.dtype}, "
                f"must be {PETSc.ScalarType}"
            )

    @property
    def axes(self):
        return self._axes

    @property
    def outer_loops(self):
        return self._outer_loops

    @property
    def sf(self):
        return self.buffer.sf

    @property
    def comm(self):
        return self.buffer.comm

    @cached_property
    def datamap(self):
        datamap_ = {}
        datamap_.update(self.buffer.datamap)
        datamap_.update(self.axes.datamap)

        # FIXME, deleting this breaks stuff...
        for index_exprs in self.axes.index_exprs.values():
            for expr in index_exprs.values():
                for array in MultiArrayCollector()(expr):
                    datamap_.update(array.datamap)
        for layout_expr in self.axes.layouts.values():
            for array in MultiArrayCollector()(layout_expr):
                datamap_.update(array.datamap)
        return freeze(datamap_)

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

    def materialize(self) -> HierarchicalArray:
        """Return a new "unindexed" array with the same shape."""
        # "unindexed" axis tree
        # strip parallel semantics (in a bad way)
        parent_to_children = collections.defaultdict(list)
        for p, cs in self.axes.parent_to_children.items():
            for c in cs:
                if c is not None and c.sf is not None:
                    c = c.copy(sf=None)
                parent_to_children[p].append(c)

        axes = AxisTree(parent_to_children)
        return type(self)(axes, dtype=self.dtype)

    def iter_indices(self, outer_map):
        from pyop3.itree.tree import iter_axis_tree

        return iter_axis_tree(
            self.axes.index(),
            self.axes,
            self.target_paths,
            self.index_exprs,
            outer_map,
        )

    def _with_axes(self, axes):
        """Return a new `Dat` with new axes pointing to the same data."""
        assert False, "do not use, it's wrong"
        return type(self)(
            axes,
            data=self.buffer,
            max_value=self.max_value,
            name=self.name,
        )

    @property
    def alloc_size(self):
        return self.axes.alloc_size if not self.axes.is_empty else 1

    @property
    def size(self):
        return self.axes.size

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

    def zero(self, *, subset=Ellipsis, eager=True):
        # old Firedrake code may hit this, should probably raise a warning
        if subset is None:
            subset = Ellipsis

        expr = ReplaceAssignment(self[subset], 0)
        return expr() if eager else expr


# Needs to be subclass for isinstance checks to work
# TODO Delete
class MultiArray(HierarchicalArray):
    @deprecated("HierarchicalArray")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

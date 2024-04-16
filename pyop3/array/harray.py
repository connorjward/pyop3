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
import pytools
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.array.base import Array
from pyop3.axtree import (
    Axis,
    AxisComponent,
    AxisTree,
    ContextFree,
    ContextSensitive,
    as_axis_tree,
)
from pyop3.axtree.layout import eval_offset
from pyop3.axtree.tree import Indexed, IndexedAxisTree, MultiArrayCollector
from pyop3.buffer import Buffer, DistributedBuffer
from pyop3.dtypes import IntType, ScalarType
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

    # def __str__(self) -> str:
    #     return f"{self.array.name}[{{{', '.join(f'{i[0]}: {i[1]}' for i in self.indices.items())}}}]"
    #
    # def __repr__(self) -> str:
    #     return f"MultiArrayVariable({self.array!r}, {self.indices!r})"


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


# does not belong here!
# class CalledMapVariable(ArrayVar):
#     mapper_method = sys.intern("map_called_map_variable")
#
#     def __init__(self, array, path, input_index_exprs, shape_index_exprs):
#         super().__init__(array, {**input_index_exprs, **shape_index_exprs}, path)
#         self.input_index_exprs = freeze(input_index_exprs)
#         self.shape_index_exprs = freeze(shape_index_exprs)
#
#     def __getinitargs__(self):
#         return (
#             self.array,
#             self.target_path,
#             self.input_index_exprs,
#             self.shape_index_exprs,
#         )


class FancyIndexWriteException(Exception):
    pass


class HierarchicalArray(Array, Indexed, ContextFree, KernelArgument):
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
        # layouts=None,
        # target_paths=None,
        # index_exprs=None,
        # outer_loops=None,
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
                shape = data.shape
            else:
                shape = axes.global_size

            # IndexedAxisTrees do not currently have SFs, so create a dummy one here
            if isinstance(axes, AxisTree):
                sf = axes.sf
            else:
                assert isinstance(axes, IndexedAxisTree)
                sf = serial_forest(axes.global_size)

            data = DistributedBuffer(
                shape,
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

        # if some_but_not_all(x is None for x in [target_paths, index_exprs]):
        #     raise ValueError

        # if target_paths is None:
        #     target_paths = axes._default_target_paths()
        # if index_exprs is None:
        #     index_exprs = axes._default_index_exprs()
        #
        # self._target_paths = freeze(target_paths)
        # self._index_exprs = freeze(index_exprs)
        # self._outer_loops = outer_loops or ()
        #
        # self._layouts = layouts if layouts is not None else axes.layouts

    @property
    @deprecated()
    def target_paths(self):
        return self.axes.target_paths

    @property
    @deprecated()
    def index_exprs(self):
        return self.axes.index_exprs

    @property
    @deprecated()
    def layouts(self):
        return self.axes.layouts

    def __str__(self):
        return self.name

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    def getitem(self, indices, *, strict=False):
        from pyop3.itree.tree import as_index_forest, compose_axes, index_axes

        if indices is Ellipsis:
            return self

        index_forest = as_index_forest(indices, axes=self.axes, strict=strict)
        if index_forest.keys() == {pmap()}:
            index_tree = index_forest[pmap()]
            indexed_axes = index_axes(index_tree, pmap(), self.axes)
            axes = compose_axes(indexed_axes, self.axes)
            return HierarchicalArray(
                axes, data=self.buffer, max_value=self.max_value, name=self.name
            )

        array_per_context = {}
        for loop_context, index_tree in index_forest.items():
            indexed_axes = index_axes(index_tree, loop_context, self.axes)
            axes = compose_axes(indexed_axes, self.axes)
            array_per_context[loop_context] = HierarchicalArray(
                axes, data=self.buffer, name=self.name, max_value=self.max_value
            )

        return ContextSensitiveMultiArray(array_per_context)

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    @property
    @deprecated("buffer")
    def array(self):
        return self.buffer

    @property
    def dtype(self):
        return self.array.dtype

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
        return self.buffer.data_rw[self._buffer_indices]

    @property
    def data_ro(self):
        if not isinstance(self._buffer_indices, slice):
            warning(
                "Read-only access to the array is provided with a copy, "
                "consider avoiding if possible."
            )
        return self.buffer.data_ro[self._buffer_indices]

    @property
    def data_wo(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should
        call `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        self._check_no_copy_access()
        return self.buffer.data_wo[self._buffer_indices]

    @property
    @deprecated(".data_rw_with_halos")
    def data_with_halos(self):
        return self.data_rw_with_halos

    @property
    def data_rw_with_halos(self):
        self._check_no_copy_access()
        return self.buffer.data_rw[self._buffer_indices_ghost]

    @property
    def data_ro_with_halos(self):
        if not isinstance(self._buffer_indices_ghost, slice):
            warning(
                "Read-only access to the array is provided with a copy, "
                "consider avoiding if possible."
            )
        return self.buffer.data_ro[self._buffer_indices_ghost]

    @property
    def data_wo_with_halos(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should
        call `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        self._check_no_copy_access()
        return self.buffer.data_wo[self._buffer_indices_ghost]

    @cached_property
    def _buffer_indices(self):
        return self._collect_buffer_indices(ghost=False)

    @cached_property
    def _buffer_indices_ghost(self):
        return self._collect_buffer_indices(ghost=True)

    def _collect_buffer_indices(self, *, ghost: bool):
        # TODO: This method is inefficient as for affine things we still tabulate
        # everything first. It would be best to inspect index_exprs to determine
        # if a slice is sufficient, but this is hard.
        # TODO: This should be more widely cached, don't want to tabulate more often
        # than required.

        size = self.axes.size if ghost else self.axes.owned.size
        assert size > 0

        indices = np.full(size, -1, dtype=IntType)
        # TODO: Handle any outer loops.
        # TODO: Generate code for this.
        for i, p in enumerate(self.axes.iter()):
            indices[i] = self.axes.offset(p.source_exprs, p.source_path)
        debug_assert(lambda: (indices >= 0).all())

        # The packed indices are collected component-by-component so, for
        # numbered multi-component axes, they are not in ascending order.
        # We sort them so we can test for "affine-ness".
        indices.sort()

        # See if we can represent these indices as a slice. This is important
        # because slices enable no-copy access to the array.
        steps = np.unique(indices[1:] - indices[:-1])
        if len(steps) == 0:
            start = just_one(indices)
            return slice(start, start + 1, 1)
        elif len(steps) == 1:
            start = indices[0]
            stop = indices[-1] + 1
            (step,) = steps
            return slice(start, stop, step)
        else:
            return indices

    def _check_no_copy_access(self):
        if not isinstance(self._buffer_indices, slice):
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
        return self.array.sf

    @property
    def comm(self):
        return self.buffer.comm

    @cached_property
    def datamap(self):
        datamap_ = {}
        datamap_.update(self.buffer.datamap)
        datamap_.update(self.axes.datamap)
        for index_exprs in self.index_exprs.values():
            for expr in index_exprs.values():
                for array in MultiArrayCollector()(expr):
                    datamap_.update(array.datamap)
        for layout_expr in self.layouts.values():
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
            self.array._reduce_then_broadcast()
        else:
            self.array._reduce_leaves_to_roots()

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
            data=self.array,
            max_value=self.max_value,
            name=self.name,
        )

    @property
    def alloc_size(self):
        return self.axes.alloc_size() if not self.axes.is_empty else 1

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

    def get_value(self, indices, path=None, *, loop_exprs=pmap()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        return self.buffer.data_ro[offset]

    def set_value(self, indices, value, path=None, *, loop_exprs=pmap()):
        offset = self.axes.offset(indices, path, loop_exprs=loop_exprs)
        self.buffer.data_wo[offset] = value

    # def offset(self, indices, path=None, *, loop_exprs=pmap()):
    #     return eval_offset(
    #         self.axes,
    #         self.subst_layouts,
    #         indices,
    #         path,
    #         loop_exprs=loop_exprs,
    #     )

    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)

    def copy(self, other, subset=Ellipsis):
        """Copy the contents of the array into another."""
        # NOTE: Is copy_to/copy_into a clearer name for this?
        # TODO: Check that self and other are compatible, should have same axes and dtype
        # for sure
        # TODO: We can optimise here and copy the private data attribute and set halo
        # validity. Here we do the simple but hopefully correct thing.
        if subset is Ellipsis:
            other.data_wo[...] = self.data_ro
        else:
            self[subset].assign(other[subset])

    # symbolic
    def zero(self, *, subset=Ellipsis):
        return ReplaceAssignment(self[subset], 0)

    def eager_zero(self, *, subset=Ellipsis):
        self.zero(subset=subset)()


# Needs to be subclass for isinstance checks to work
# TODO Delete
class MultiArray(HierarchicalArray):
    @deprecated("HierarchicalArray")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Now ContextSensitiveDat
class ContextSensitiveMultiArray(Array, ContextSensitive):
    def __init__(self, arrays):
        name = single_valued(a.name for a in arrays.values())

        Array.__init__(self, name)
        ContextSensitive.__init__(self, arrays)

    def __getitem__(self, indices) -> ContextSensitiveMultiArray:
        raise NotImplementedError("Untested")
        # from pyop3.itree.tree import _compose_bits, _index_axes, as_index_forest
        #
        # # FIXME for now assume that there is only one context
        # context, array = just_one(self.context_map.items())
        #
        # index_forest = as_index_forest(indices, axes=array.axes)
        #
        # if len(index_forest) == 1 and pmap() in index_forest:
        #     raise NotImplementedError("code path untested")
        #
        # array_per_context = {}
        # for loop_context, index_tree in index_forest.items():
        #     indexed_axes = _index_axes(index_tree, loop_context, array.axes)
        #
        #     (
        #         target_paths,
        #         index_exprs,
        #         layout_exprs,
        #     ) = _compose_bits(
        #         array.axes,
        #         array.target_paths,
        #         array.index_exprs,
        #         None,
        #         indexed_axes,
        #         indexed_axes.target_paths,
        #         indexed_axes.index_exprs,
        #         indexed_axes.layout_exprs,
        #     )
        #     array_per_context[loop_context] = HierarchicalArray(
        #         indexed_axes,
        #         data=self.array,
        #         max_value=self.max_value,
        #         target_paths=target_paths,
        #         index_exprs=index_exprs,
        #         outer_loops=indexed_axes.outer_loops,
        #         layouts=self.layouts,
        #         name=self.name,
        #     )
        #
        # return ContextSensitiveMultiArray(array_per_context)

    @property
    def array(self):
        return self._shared_attr("array")

    @property
    def buffer(self):
        return self._shared_attr("buffer")

    @property
    def dtype(self):
        return self._shared_attr("dtype")

    @property
    def kernel_dtype(self):
        # TODO Think about the fact that the dtype refers to either to dtype of the
        # array entries (e.g. double), or the dtype of the whole thing (double*)
        return self.dtype

    @property
    def max_value(self):
        return self._shared_attr("max_value")

    @property
    def layouts(self):
        return self._shared_attr("layouts")

    @functools.cached_property
    def datamap(self):
        return merge_dicts(array.datamap for array in self.context_map.values())

    def _shared_attr(self, attr: str):
        return single_valued(getattr(a, attr) for a in self.context_map.values())

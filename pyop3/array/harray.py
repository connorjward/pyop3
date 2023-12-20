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

from pyop3.array.base import Array
from pyop3.axtree import (
    Axis,
    AxisComponent,
    AxisTree,
    ContextFree,
    ContextSensitive,
    as_axis_tree,
)
from pyop3.axtree.tree import (
    AxisVariable,
    ExpressionEvaluator,
    Indexed,
    MultiArrayCollector,
    PartialAxisTree,
    _path_and_indices_from_index_tuple,
    _trim_path,
)
from pyop3.buffer import Buffer, DistributedBuffer
from pyop3.dtypes import IntType, ScalarType, get_mpi_dtype
from pyop3.itree import IndexTree, as_index_forest
from pyop3.itree.tree import iter_axis_tree
from pyop3.lang import KernelArgument
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
    some_but_not_all,
    strict_int,
    strictly_all,
)


class IncompatibleShapeError(Exception):
    """TODO, also bad name"""


class MultiArrayVariable(pym.primitives.Expression):
    mapper_method = sys.intern("map_multi_array")

    def __init__(self, array, target_path, index_exprs):
        super().__init__()
        self.array = array
        self.target_path = freeze(target_path)
        self.index_exprs = freeze(index_exprs)

    def __getinitargs__(self):
        return (self.array, self.target_path, self.index_exprs)

    # def __str__(self) -> str:
    #     return f"{self.array.name}[{{{', '.join(f'{i[0]}: {i[1]}' for i in self.indices.items())}}}]"
    #
    # def __repr__(self) -> str:
    #     return f"MultiArrayVariable({self.array!r}, {self.indices!r})"


# does not belong here!
class CalledMapVariable(MultiArrayVariable):
    mapper_method = sys.intern("map_called_map_variable")

    def __init__(self, array, target_path, input_index_exprs, shape_index_exprs):
        super().__init__(array, target_path, {**input_index_exprs, **shape_index_exprs})
        self.input_index_exprs = freeze(input_index_exprs)
        self.shape_index_exprs = freeze(shape_index_exprs)

    def __getinitargs__(self):
        return (
            self.array,
            self.target_path,
            self.input_index_exprs,
            self.shape_index_exprs,
        )


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
        layouts=None,
        target_paths=None,
        index_exprs=None,
        domain_index_exprs=pmap(),
        name=None,
        prefix=None,
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
                shape = axes.size

            data = DistributedBuffer(
                shape, dtype, name=self.name, data=data, sf=axes.sf
            )

        self.buffer = data

        # instead implement "materialize"
        self.axes = axes

        self.max_value = max_value

        if some_but_not_all(x is None for x in [target_paths, index_exprs]):
            raise ValueError

        self._target_paths = target_paths or axes._default_target_paths()
        self._index_exprs = index_exprs or axes._default_index_exprs()
        self.domain_index_exprs = domain_index_exprs

        self.layouts = layouts or axes.layouts

    def __str__(self):
        return self.name

    def __getitem__(self, indices) -> Union[MultiArray, ContextSensitiveMultiArray]:
        from pyop3.itree.tree import _compose_bits, _index_axes

        index_forest = as_index_forest(indices, axes=self.axes)
        if len(index_forest) == 1 and pmap() in index_forest:
            index_tree = just_one(index_forest.values())
            indexed_axes = _index_axes(index_tree, pmap(), self.axes)

            target_paths, index_exprs, layout_exprs = _compose_bits(
                self.axes,
                self.target_paths,
                self.index_exprs,
                None,
                indexed_axes,
                indexed_axes.target_paths,
                indexed_axes.index_exprs,
                indexed_axes.layout_exprs,
            )

            return HierarchicalArray(
                indexed_axes,
                data=self.array,
                max_value=self.max_value,
                target_paths=target_paths,
                index_exprs=index_exprs,
                domain_index_exprs=indexed_axes.domain_index_exprs,
                layouts=self.layouts,
                name=self.name,
            )

        array_per_context = {}
        for loop_context, index_tree in index_forest.items():
            indexed_axes = _index_axes(index_tree, loop_context, self.axes)

            (
                target_paths,
                index_exprs,
                layout_exprs,
            ) = _compose_bits(
                self.axes,
                self.target_paths,
                self.index_exprs,
                None,
                indexed_axes,
                indexed_axes.target_paths,
                indexed_axes.index_exprs,
                indexed_axes.layout_exprs,
            )

            array_per_context[loop_context] = HierarchicalArray(
                indexed_axes,
                data=self.array,
                layouts=self.layouts,
                target_paths=target_paths,
                index_exprs=index_exprs,
                domain_index_exprs=indexed_axes.domain_index_exprs,
                name=self.name,
                max_value=self.max_value,
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
    def target_paths(self):
        return self._target_paths

    @property
    def index_exprs(self):
        return self._index_exprs

    @property
    def sf(self):
        return self.array.sf

    @cached_property
    def datamap(self):
        datamap_ = {self.name: self}
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
        axes = AxisTree(self.axes.parent_to_children)
        return type(self)(axes, dtype=self.dtype)

    def offset(self, *args, allow_unused=False, insert_zeros=False):
        nargs = len(args)
        if nargs == 2:
            path, indices = args[0], args[1]
        else:
            assert nargs == 1
            path, indices = _path_and_indices_from_index_tuple(self.axes, args[0])

        if allow_unused:
            path = _trim_path(self.axes, path)

        if insert_zeros:
            # extend the path by choosing the zero offset option every time
            # this is needed if we don't have all the internal bits available
            while path not in self.layouts:
                axis, clabel = self.axes._node_from_path(path)
                subaxis = self.axes.child(axis, clabel)
                # choose the component that is first in the renumbering
                if subaxis.numbering:
                    cidx = subaxis._component_index_from_axis_number(
                        subaxis.numbering.data_ro[0]
                    )
                else:
                    cidx = 0
                subcpt = subaxis.components[cidx]
                path |= {subaxis.label: subcpt.label}
                indices |= {subaxis.label: 0}

        offset = pym.evaluate(self.layouts[path], indices, ExpressionEvaluator)
        return strict_int(offset)

    def simple_offset(self, path, indices):
        offset = pym.evaluate(self.layouts[path], indices, ExpressionEvaluator)
        return strict_int(offset)

    def iter_indices(self, outer_map):
        return iter_axis_tree(self.axes, self.target_paths, self.index_exprs, outer_map)

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

    def get_value(self, *args, **kwargs):
        return self.data[self.offset(*args, **kwargs)]

    def set_value(self, path, indices, value):
        self.data[self.simple_offset(path, indices)] = value

    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)


# Needs to be subclass for isinstance checks to work
# TODO Delete
class MultiArray(HierarchicalArray):
    @deprecated("HierarchicalArray")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Now ContextSensitiveDat
class ContextSensitiveMultiArray(ContextSensitive, KernelArgument):
    def __getitem__(self, indices) -> ContextSensitiveMultiArray:
        from pyop3.itree.tree import _compose_bits, _index_axes

        # FIXME for now assume that there is only one context
        context, array = just_one(self.context_map.items())

        index_forest = as_index_forest(indices, axes=array.axes)

        if len(index_forest) == 1 and pmap() in index_forest:
            raise NotImplementedError("code path untested")

        array_per_context = {}
        for loop_context, index_tree in index_forest.items():
            indexed_axes = _index_axes(index_tree, loop_context, array.axes)

            (
                target_paths,
                index_exprs,
                layout_exprs,
            ) = _compose_bits(
                array.axes,
                array.target_paths,
                array.index_exprs,
                None,
                indexed_axes,
                indexed_axes.target_paths,
                indexed_axes.index_exprs,
                indexed_axes.layout_exprs,
            )
            array_per_context[loop_context] = HierarchicalArray(
                indexed_axes,
                data=self.array,
                max_value=self.max_value,
                target_paths=target_paths,
                index_exprs=index_exprs,
                domain_index_exprs=indexed_axes.domain_index_exprs,
                layouts=self.layouts,
                name=self.name,
            )

        return ContextSensitiveMultiArray(array_per_context)

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
    def max_value(self):
        return self._shared_attr("max_value")

    @property
    def name(self):
        return self._shared_attr("name")

    @property
    def layouts(self):
        return self._shared_attr("layouts")

    @functools.cached_property
    def datamap(self):
        return merge_dicts(array.datamap for array in self.context_map.values())

    def _shared_attr(self, attr: str):
        return single_valued(getattr(a, attr) for a in self.context_map.values())

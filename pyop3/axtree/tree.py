from __future__ import annotations

import abc
import bisect
import collections
import copy
import dataclasses
import enum
import functools
import itertools
import numbers
import operator
import sys
import threading
from functools import cached_property
from itertools import chain
from typing import Any, FrozenSet, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pymbolic as pym
import pyrsistent
import pytools
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze, pmap, thaw

from pyop3.dtypes import IntType
from pyop3.sf import StarForest, serial_forest
from pyop3.tree import (
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    MutableLabelledTreeMixin,
    as_component_label,
    postvisit,
    previsit,
)
from pyop3.utils import (
    checked_zip,
    debug_assert,
    deprecated,
    invert,
    just_one,
    merge_dicts,
    pairwise,
    single_valued,
    steps,
    strict_int,
    strictly_all,
)


class ExpectedLinearAxisTreeException(Exception):
    ...


class Indexed(abc.ABC):
    @property
    @abc.abstractmethod
    def axes(self):
        pass

    @property
    @abc.abstractmethod
    def target_paths(self):
        pass

    @property
    @abc.abstractmethod
    def index_exprs(self):
        pass

    @property
    @abc.abstractmethod
    def outer_loops(self):
        pass

    @property
    @abc.abstractmethod
    def layouts(self):
        pass


class ContextAware(abc.ABC):
    @abc.abstractmethod
    def with_context(self, context):
        pass


class ContextSensitive(ContextAware, abc.ABC):
    #     """Container of `IndexTree`s distinguished by outer loop information.
    #
    #     This class is required because multi-component outer loops can lead to
    #     ambiguity in the shape of the resulting `IndexTree`. Consider the loop:
    #
    #     .. code:: python
    #
    #         loop(p := mesh.points, kernel(dat0[closure(p)]))
    #
    #     In this case, assuming ``mesh`` to be at least 1-dimensional, ``p`` will
    #     loop over multiple components (cells, edges, vertices, etc) and each
    #     component will have a differently sized temporary. This is because
    #     vertices map to themselves whereas, for example, edges map to themselves
    #     *and* the incident vertices.
    #
    #     A `SplitIndexTree` is therefore useful as it allows the description of
    #     an `IndexTree` *per possible configuration of relevant loop indices*.
    #
    #     """
    #
    def __init__(self, context_map):
        if isinstance(context_map, pyrsistent.PMap):
            raise TypeError("context_map must be deterministically ordered")
        self.context_map = context_map

    @cached_property
    def keys(self):
        # loop is used just for unpacking
        for context in self.context_map.keys():
            indices = set()
            for loop_index in context.keys():
                indices.add(loop_index)
            return frozenset(indices)

    def with_context(self, context):
        return self.context_map[self.filter_context(context)]

    def filter_context(self, context):
        key = {}
        for loop_index, path in context.items():
            if loop_index in self.keys:
                key.update({loop_index: freeze(path)})
        return freeze(key)


# this is basically just syntactic sugar, might not be needed
# avoids the need for
# if isinstance(obj, ContextSensitive):
#     obj = obj.with_context(...)
class ContextFree(ContextAware, abc.ABC):
    def with_context(self, context):
        return self

    def filter_context(self, context):
        return pmap()


class LoopIterable(abc.ABC):
    """Class representing something that can be looped over.

    In order for an object to be loop-able over it needs to have shape
    (``axes``) and an index expression per leaf of the shape. The simplest
    case is `AxisTree` since the index expression is just identity. This
    contrasts with something like an `IndexedLoopIterable` or `CalledMap`.
    For the former the index expression for ``axes[::2]`` would be ``2*i``
    and for the latter ``map(p)`` would be something like ``map[i, j]``.

    """

    @abc.abstractmethod
    def __getitem__(self, indices) -> Union[LoopIterable, ContextSensitiveLoopIterable]:
        raise NotImplementedError

    # not iterable in the Python sense
    __iter__ = None

    @abc.abstractmethod
    def index(self) -> LoopIndex:
        pass


class ContextFreeLoopIterable(LoopIterable, ContextFree, abc.ABC):
    pass


class ContextSensitiveLoopIterable(LoopIterable, ContextSensitive, abc.ABC):
    pass


class UnrecognisedAxisException(ValueError):
    pass


class ExpressionEvaluator(pym.mapper.evaluator.EvaluationMapper):
    def __init__(self, context, loop_exprs):
        super().__init__(context)
        self._loop_exprs = loop_exprs

    def map_axis_variable(self, expr):
        try:
            return self.context[expr.axis_label]
        except KeyError as e:
            raise UnrecognisedAxisException from e

    def map_array(self, array_var):
        from pyop3.itree.tree import ExpressionEvaluator, IndexExpressionReplacer

        array = array_var.array

        indices = {ax: self.rec(idx) for ax, idx in array_var.indices.items()}
        # replacer = IndexExpressionReplacer(indices, self._loop_exprs)
        # layout_orig = array.layouts[freeze(array_var.target_path)]
        # layout_subst = replacer(layout_orig)
        layout_subst = array.axes.subst_layouts[array_var.path]

        # offset = ExpressionEvaluator(indices, self._loop_exprs)(layout_subst)
        # offset = ExpressionEvaluator(self.context | indices, self._loop_exprs)(layout_subst)
        offset = ExpressionEvaluator(indices, self._loop_exprs)(layout_subst)
        offset = strict_int(offset)

        # return array_var.array.get_value(
        #     self.context,
        #     array_var.target_path,  # should be source path
        #     # index_exprs=array_var.index_exprs,
        #     loop_exprs=self._loop_exprs,
        # )
        return array.data_ro[offset]

    def map_loop_index(self, expr):
        return self._loop_exprs[expr.id][expr.axis]


def _collect_datamap(axis, *subdatamaps, axes):
    from pyop3.array import HierarchicalArray

    datamap = {}
    for cidx, component in enumerate(axis.components):
        if isinstance(count := component.count, HierarchicalArray):
            datamap.update(count.datamap)

    datamap.update(merge_dicts(subdatamaps))
    return datamap


class AxisComponent(LabelledNodeComponent):
    """
    Parameters
    ----------
    indexed : bool
        Is this axis indexed (as part of a temporary) - used to generate the right layouts

    indices
        If the thing is sparse then we need to specify the indices of the sparsity here.
        This is like CSR. This is normally a nested/ragged thing.

        E.g. a diagonal matrix would be 3 x [1, 1, 1] with indices being [0, 1, 2]. The
        CSR row pointers are [0, 1, 2] (we already calculate this), but when we look up
        the values we use [0, 1, 2] instead of [0, 0, 0]. A binary search of all the
        indices is required to find the right offset.

        Note that this is an entirely separate concept to the numbering. Imagine a
        sparse matrix where the row and column axes are renumbered. The indices are
        still sorted. The indices gives us a mapping from "dense" indices to "sparse"
        ones. This is normally inverted (via binary search) to get the "dense" index
        from the "sparse" one. The numbering then concerns the lookup from dense
        indices to an offset. This means, for example, that the numbering of a sparse
        thing is dense and contains the numbers [0, ..., ndense).

    """

    fields = LabelledNodeComponent.fields | {"count", "unit"}

    def __init__(
        self,
        count,
        label=None,
        *,
        indices=None,
        indexed=False,
        lgmap=None,
        unit=False,
    ):
        from pyop3.array import HierarchicalArray

        if not isinstance(count, (numbers.Integral, HierarchicalArray)):
            raise TypeError("Invalid count type")
        if unit and count != 1:
            raise ValueError(
                "Components may only be marked as 'unit' if they have length 1"
            )

        super().__init__(label=label)
        self.count = count
        self.unit = unit

    # TODO this is just a traversal - clean up
    def alloc_size(self, axtree, axis):
        from pyop3.array import HierarchicalArray

        if isinstance(self.count, HierarchicalArray):
            npoints = self.count.max_value
        else:
            assert isinstance(self.count, numbers.Integral)
            npoints = self.count

        assert npoints is not None

        if subaxis := axtree.component_child(axis, self):
            size = npoints * axtree.alloc_size(subaxis)
        else:
            size = npoints

        # TODO: May be excessive
        # Cast to an int as numpy integers cause loopy to break
        return strict_int(size)


class Axis(LoopIterable, MultiComponentLabelledNode):
    fields = MultiComponentLabelledNode.fields | {
        "components",
        "numbering",
        "sf",
    }

    def __init__(
        self,
        components,
        label=None,
        *,
        numbering=None,
        sf=None,
        id=None,
    ):
        components = self._parse_components(components)
        numbering = self._parse_numbering(numbering)

        if numbering is not None:
            if not all(isinstance(c.count, numbers.Integral) for c in components):
                raise NotImplementedError(
                    "Axis numberings are only supported for axes with fixed component sizes"
                )
            if sum(c.count for c in components) != numbering.size:
                raise ValueError

        super().__init__(label=label, id=id)

        self.components = components
        self.numbering = numbering
        self.sf = sf

    def __getitem__(self, indices):
        # NOTE: This *must* return an axis tree because that is where we attach
        # index expression information. Just returning as_axis_tree(self).root
        # here will break things.
        # Actually this is not the case for "identity" slices since index_exprs
        # and labels are unchanged (AxisTree vs IndexedAxisTree)
        # TODO return a flat axis in these cases
        return self._tree[indices]

    def __call__(self, *args):
        return as_axis_tree(self)(*args)

    def __str__(self) -> str:
        return (
            self.__class__.__name__
            + f"({{{', '.join(f'{c.label}: {c.count}' for c in self.components)}}}, {self.label})"
        )

    @classmethod
    def from_serial(cls, serial: Axis, sf):
        # FIXME
        from pyop3.axtree.parallel import partition_ghost_points

        if serial.sf is not None:
            raise RuntimeError("serial axis is not serial")

        if isinstance(sf, PETSc.SF):
            sf = StarForest(sf, serial.size)

        # renumber the serial axis to store ghost entries at the end of the vector
        numbering = partition_ghost_points(serial, sf)
        return cls(serial.components, serial.label, numbering=numbering, sf=sf)

    @property
    def component_labels(self):
        return tuple(c.label for c in self.components)

    @property
    def component(self):
        return just_one(self.components)

    def component_index(self, component) -> int:
        clabel = as_component_label(component)
        return self.component_labels.index(clabel)

    @property
    def comm(self):
        return self.sf.comm if self.sf else MPI.COMM_SELF

    @property
    def size(self):
        return self._tree.size

    @property
    def count(self):
        """Return the total number of entries in the axis across all axis parts.
        Will fail if axis parts do not have integer counts.
        """
        # hacky but right (no inner shape)
        return self.size

    # @parallel_only  # TODO
    @cached_property
    def owned_count(self):
        return self.count - self.sf.nleaves

    @cached_property
    def count_per_component(self):
        return freeze({c: c.count for c in self.components})

    @cached_property
    # @parallel_only
    def owned_count_per_component(self):
        return freeze(
            {
                cpt: count - self.ghost_count_per_component[cpt]
                for cpt, count in self.count_per_component.items()
            }
        )

    @cached_property
    # @parallel_only
    def ghost_count_per_component(self):
        counts = np.zeros_like(self.components, dtype=int)
        for leaf_index in self.sf.ileaf:
            counts[self._axis_number_to_component_index(leaf_index)] += 1
        return freeze(
            {cpt: count for cpt, count in checked_zip(self.components, counts)}
        )

    @cached_property
    def owned(self):
        from pyop3.itree import AffineSliceComponent, Slice

        if self.comm.size == 1:
            return self

        slices = [
            AffineSliceComponent(
                c.label,
                stop=self.owned_count_per_component[c],
            )
            for c in self.components
        ]
        slice_ = Slice(self.label, slices)
        return self[slice_].root

    def index(self):
        return self._tree.index()

    def iter(self):
        return self._tree.iter()

    @property
    def target_path_per_component(self):
        return self._tree.target_path_per_component

    @property
    def index_exprs_per_component(self):
        return self._tree.index_exprs_per_component

    @property
    def layout_exprs_per_component(self):
        return self._tree.layout_exprs_per_component

    @deprecated("as_tree")
    @property
    def axes(self):
        return self.as_tree()

    @property
    def index_exprs(self):
        return self._tree.index_exprs

    def as_tree(self) -> AxisTree:
        """Convert the axis to a tree that contains it.

        Returns
        -------
        Axis Tree
            TODO

        Notes
        -----
        The result of this function is cached because `AxisTree`s are immutable
        and we want to cache expensive computations on them.

        """
        return self._tree

    # Ideally I want to cythonize a lot of these methods
    def component_numbering(self, component):
        cidx = self.component_index(component)
        return self._default_to_applied_numbering[cidx]

    def component_permutation(self, component):
        cidx = self.component_index(component)
        return self._default_to_applied_permutation[cidx]

    def default_to_applied_component_number(self, component, number):
        cidx = self.component_index(component)
        return self._default_to_applied_numbering[cidx][number]

    def applied_to_default_component_number(self, component, number):
        cidx = self.component_index(component)
        return self._applied_to_default_numbering[cidx][number]

    def axis_to_component_number(self, number):
        # return axis_to_component_number(self, number)
        cidx = self._axis_number_to_component_index(number)
        return self.components[cidx], number - self._component_offsets[cidx]

    def component_offset(self, component):
        cidx = self.component_index(component)
        return self._component_offsets[cidx]

    def component_to_axis_number(self, component, number):
        cidx = self.component_index(component)
        return self._component_offsets[cidx] + number

    def renumber_point(self, component, point):
        renumbering = self.component_numbering(component)
        return renumbering[point]

    @cached_property
    def _tree(self):
        return AxisTree(self)

    @cached_property
    def _component_offsets(self):
        return (0,) + tuple(np.cumsum([c.count for c in self.components], dtype=int))

    @cached_property
    def _default_to_applied_numbering(self):
        renumbering = [np.empty(c.count, dtype=IntType) for c in self.components]
        counters = [itertools.count() for _ in range(self.degree)]
        for pt in self.numbering.data_ro:
            cidx = self._axis_number_to_component_index(pt)
            old_cpt_pt = pt - self._component_offsets[cidx]
            renumbering[cidx][old_cpt_pt] = next(counters[cidx])
        assert all(next(counters[i]) == c.count for i, c in enumerate(self.components))
        return tuple(renumbering)

    @cached_property
    def _default_to_applied_permutation(self):
        # is this right?
        return self._applied_to_default_numbering

    # same as the permutation...
    @cached_property
    def _applied_to_default_numbering(self):
        return tuple(invert(num) for num in self._default_to_applied_numbering)

    def _axis_number_to_component_index(self, number):
        off = self._component_offsets
        for i, (min_, max_) in enumerate(zip(off, off[1:])):
            if min_ <= number < max_:
                return i
        raise ValueError(f"{number} not found")

    @staticmethod
    def _parse_components(components):
        if isinstance(components, collections.abc.Mapping):
            return tuple(
                AxisComponent(count, clabel) for clabel, count in components.items()
            )
        elif isinstance(components, collections.abc.Iterable):
            return tuple(as_axis_component(c) for c in components)
        else:
            return (as_axis_component(components),)

    @staticmethod
    def _parse_numbering(numbering):
        from pyop3.array import HierarchicalArray

        if numbering is None:
            return None
        elif isinstance(numbering, HierarchicalArray):
            return numbering
        elif isinstance(numbering, collections.abc.Collection):
            return HierarchicalArray(len(numbering), data=numbering, dtype=IntType)
        else:
            raise TypeError(
                f"{type(numbering).__name__} is not a supported type for numbering"
            )


# Do I ever want this? component_offsets is expensive so we don't want to
# do it every time
def axis_to_component_number(axis, number, context=pmap()):
    offsets = component_offsets(axis, context)
    return component_number_from_offsets(axis, number, offsets)


# TODO move into layout.py
def component_number_from_offsets(axis, number, offsets):
    cidx = None
    for i, (min_, max_) in enumerate(pairwise(offsets)):
        if min_ <= number < max_:
            cidx = i
            break
    assert cidx is not None
    return axis.components[cidx], number - offsets[cidx]


# TODO move into layout.py
def component_offsets(axis, context):
    from pyop3.axtree.layout import _as_int

    return steps([_as_int(c.count, context) for c in axis.components])


class LoopIndexReplacer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        super().__init__()
        self._replace_map = replace_map

    def map_axis_variable(self, var):
        try:
            return self._replace_map[var.axis]
        except KeyError:
            return var

    def map_array(self, array_var):
        indices = {ax: self(expr) for ax, expr in array_var.indices.items()}
        return type(array_var)(array_var.array, indices, array_var.path)


class MultiArrayCollector(pym.mapper.Collector):
    def map_array(self, array_var):
        return {array_var.array}.union(
            *(self.rec(expr) for expr in array_var.indices.values())
        )

    def map_nan(self, nan):
        return set()


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


class BaseAxisTree(ContextFreeLoopIterable, LabelledTree):
    # TODO: Cache this function.
    def __getitem__(self, indices):
        from pyop3.itree.tree import _compose_bits, _index_axes, as_index_forest

        if indices is Ellipsis:
            return self

        axis_trees = {}
        for context, index_tree in as_index_forest(indices, axes=self).items():
            indexed_axes = _index_axes(index_tree, context, self)

            # NOTE: If this works combine into _index_axes
            # itarget_paths, iindex_exprs, ilayout_exprs = acc_bits(indexed_axes, indexed_axes.target_paths, indexed_axes.index_exprs, indexed_axes.layout_exprs)

            target_paths, index_exprs = _compose_bits(
                indexed_axes,
                indexed_axes.target_paths,
                indexed_axes.index_exprs,
                self,
                self.target_paths,
                self.index_exprs,
            )
            axis_tree = IndexedAxisTree(
                indexed_axes.node_map,
                self.unindexed,
                target_paths=target_paths,
                index_exprs=index_exprs,
                layout_exprs={},
                outer_loops=indexed_axes.outer_loops,
            )
            axis_trees[context] = axis_tree

        if len(axis_trees) == 1 and just_one(axis_trees.keys()) == pmap():
            return axis_trees[pmap()]
        else:
            return ContextSensitiveAxisTree(axis_trees)

    @property
    @abc.abstractmethod
    def unindexed(self):
        pass

    @property
    @abc.abstractmethod
    def target_paths(self):
        pass

    @property
    @abc.abstractmethod
    def index_exprs(self):
        pass

    @property
    @abc.abstractmethod
    def layout_exprs(self):
        pass

    @property
    @abc.abstractmethod
    def layouts(self):
        pass

    @property
    @abc.abstractmethod
    def outer_loops(self):
        pass

    @property
    @abc.abstractmethod
    def subst_layouts(self):
        pass

    def index(self, ghost=False):
        from pyop3.itree.tree import ContextFreeLoopIndex, LoopIndex

        iterset = self if ghost else self.owned
        # If the iterset is linear (single-component for every axis) then we
        # can consider the loop to be "context-free".
        if len(iterset.leaves) == 1:
            path = iterset.path(*iterset.leaf)
            target_path = {}
            for ax, cpt in iterset.path_with_nodes(*iterset.leaf).items():
                target_path.update(iterset.target_paths.get((ax.id, cpt), {}))
            return ContextFreeLoopIndex(iterset, path, target_path)
        else:
            return LoopIndex(iterset)

    def iter(self, outer_loops=(), loop_index=None, include=False, ghost=False):
        from pyop3.itree.tree import iter_axis_tree

        iterset = self if ghost else self.owned

        return iter_axis_tree(
            # hack because sometimes we know the right loop index to use
            loop_index or self.index(),
            iterset,
            self.target_paths,
            self.index_exprs,
            outer_loops,
            include,
        )

    @property
    def axes(self):
        return self

    @cached_property
    def datamap(self):
        if self.is_empty:
            dmap = {}
        else:
            dmap = postvisit(self, _collect_datamap, axes=self)

        for cleverdict in [self.index_exprs]:
            for exprs in cleverdict.values():
                for expr in exprs.values():
                    for array in MultiArrayCollector()(expr):
                        dmap.update(array.datamap)
        return pmap(dmap)

    @cached_property
    def owned(self):
        """Return the owned portion of the axis tree."""
        from pyop3.itree import AffineSliceComponent, Slice

        if self.comm.size == 1:
            return self

        paraxes = [axis for axis in self.nodes if axis.sf is not None]
        if len(paraxes) == 0:
            return self

        # assumes that there is at most one parallel axis (can appear multiple times
        # if mixed)
        paraxis = paraxes[0]
        slices = [
            AffineSliceComponent(
                c.label,
                stop=paraxis.owned_count_per_component[c],
                # this feels like a hack, generally don't want this ambiguity
                label=c.label,
            )
            for c in paraxis.components
        ]
        # this feels like a hack, generally don't want this ambiguity
        slice_ = Slice(paraxis.label, slices, label=paraxis.label)
        return self[slice_]

    def as_tree(self):
        return self

    def offset(self, indices, path=None, *, loop_exprs=pmap()):
        from pyop3.axtree.layout import eval_offset

        return eval_offset(
            self,
            self.subst_layouts,
            indices,
            path,
            loop_exprs=loop_exprs,
        )

    @cached_property
    def owned_size(self):
        nghost = self.sf.nleaves if self.sf is not None else 0
        return self.size - nghost

    def _check_labels(self):
        def check(node, prev_labels):
            if node == self.root:
                return prev_labels
            if node.label in prev_labels:
                raise ValueError("shouldn't have the same label as above")
            return prev_labels | {node.label}

        previsit(self, check, self.root, frozenset())

    # should be a cached property?
    def global_numbering(self):
        if self.comm.size == 1:
            return np.arange(self.size, dtype=IntType)

        numbering = np.full(self.size, -1, dtype=IntType)

        start = self.sf.comm.tompi4py().exscan(self.owned.size, MPI.SUM)
        if start is None:
            start = 0

        # TODO do I need to account for numbering/layouts? The SF should probably
        # manage this.
        numbering[: self.owned.size] = np.arange(
            start, start + self.owned.size, dtype=IntType
        )
        # numbering[self.numbering.data_ro[: self.owned.size]] = np.arange(
        #     start, start + self.owned.size, dtype=IntType
        # )

        # print_with_rank("before", numbering)

        self.sf.broadcast(numbering, MPI.REPLACE)

        # print_with_rank("after", numbering)
        debug_assert(lambda: (numbering >= 0).all())
        return numbering

    @property
    def comm(self):
        paraxes = [axis for axis in self.nodes if axis.sf is not None]
        if not paraxes:
            return MPI.COMM_SELF
        else:
            return single_valued(ax.comm for ax in paraxes)

    @cached_property
    def leaf_target_paths(self):
        return tuple(
            merge_dicts(
                self.target_paths.get((ax.id, clabel), {})
                for ax, clabel in self.path_with_nodes(*leaf, ordered=True)
            )
            for leaf in self.leaves
        )

    @property
    def leaf_axis(self):
        return self.leaf[0]

    @property
    def leaf_component(self):
        leaf_axis, leaf_clabel = self.leaf
        leaf_cidx = leaf_axis.component_index(leaf_clabel)
        return leaf_axis.components[leaf_cidx]

    @cached_property
    def size(self):
        from pyop3.axtree.layout import axis_tree_size

        return axis_tree_size(self)

    @cached_property
    def global_size(self):
        from pyop3.array import HierarchicalArray
        from pyop3.axtree.layout import _axis_size, my_product

        if not self.outer_loops:
            return self.size

        mysize = 0
        for idxs in my_product(self.outer_loops):
            loop_exprs = {idx.index.id: idx.source_exprs for idx in idxs}
            # target_indices = merge_dicts(idx.target_exprs for idx in idxs)
            # this is a hack
            if self.is_empty:
                mysize += 1
            else:
                mysize += _axis_size(self, self.root, loop_indices=loop_exprs)
        return mysize

        if isinstance(self.size, HierarchicalArray):
            # does this happen any more?
            return np.sum(self.size.data_ro, dtype=IntType)
        if isinstance(self.size, np.ndarray):
            return np.sum(self.size, dtype=IntType)
        else:
            assert isinstance(self.size, numbers.Integral)
            return self.size

    # rename to local_size?
    def alloc_size(self, axis=None):
        if self.is_empty:
            return 1
        axis = axis or self.root
        return sum(cpt.alloc_size(self, axis) for cpt in axis.components)


class AxisTree(MutableLabelledTreeMixin, BaseAxisTree):
    @classmethod
    def from_iterable(cls, subaxes):
        tree = AxisTree()
        for subaxis in subaxes:
            subaxis = as_axis(subaxis)

            if len(subaxis.components) > 1:
                raise ExpectedLinearAxisTreeException(
                    "Cannot construct multi-component axis trees from an iterable."
                )

            tree = tree.append_axis(subaxis)
        return tree

    @classmethod
    def from_nest(cls, nest) -> AxisTree:
        # Is this generic to LabelledTree?
        root, node_map = cls._from_nest(nest)
        node_map.update({None: [root]})
        return cls(node_map)

    @property
    def unindexed(self):
        return self

    @cached_property
    def target_paths(self):
        if self.is_empty:
            return pmap()
        else:
            return self._collect_target_paths()

    def _collect_target_paths(self, axis=None, target_path_acc=None):
        assert not self.is_empty

        if strictly_all(x is None for x in {axis, target_path_acc}):
            axis = self.root
            target_path_acc = pmap()

        target_paths = {}
        for component in axis.components:
            target_path_acc_ = target_path_acc | {axis.label: component.label}
            target_paths[axis.id, component.label] = target_path_acc_

            if subaxis := self.child(axis, component):
                target_paths.update(
                    self._collect_target_paths(subaxis, target_path_acc_)
                )
        return freeze(target_paths)

    @cached_property
    def index_exprs(self):
        if self.is_empty:
            return pmap()
        else:
            return self._collect_index_exprs()

    # NOTE: This function is very similar to _collect_target_paths
    def _collect_index_exprs(self, axis=None, index_exprs_acc=None):
        assert not self.is_empty

        if strictly_all(x is None for x in {axis, index_exprs_acc}):
            axis = self.root
            index_exprs_acc = pmap()

        index_exprs = {}
        for component in axis.components:
            index_exprs_acc_ = index_exprs_acc | {axis.label: AxisVariable(axis.label)}
            index_exprs[axis.id, component.label] = index_exprs_acc_

            if subaxis := self.child(axis, component):
                index_exprs.update(self._collect_index_exprs(subaxis, index_exprs_acc_))
        return freeze(index_exprs)

    @property
    def layout_exprs(self):
        return self.index_exprs

    @property
    def outer_loops(self):
        return ()

    @cached_property
    def sf(self) -> StarForest:
        from pyop3.axtree.parallel import collect_sf_graphs

        if self.is_empty:
            # no, this is probably not right. Could have a global
            return serial_forest(self.global_size)

        graphs = collect_sf_graphs(self)
        if len(graphs) == 0:
            return serial_forest(self.global_size)
        else:
            # merge the graphs
            nroots = 0
            ilocals = []
            iremotes = []
            for graph in graphs:
                nr, ilocal, iremote = graph
                nroots += nr
                ilocals.append(ilocal)
                iremotes.append(iremote)
            ilocal = np.concatenate(ilocals)
            iremote = np.concatenate(iremotes)
            return StarForest.from_graph(self.size, nroots, ilocal, iremote, self.comm)

    @cached_property
    def datamap(self):
        if self.is_empty:
            dmap = {}
        else:
            dmap = postvisit(self, _collect_datamap, axes=self)
        return freeze(dmap)

    def add_axis(self, axis, parent_axis, parent_component=None, *, uniquify=False):
        parent_axis = self._as_node(parent_axis)
        if parent_component is not None:
            parent_component = (
                parent_component.label
                if isinstance(parent_component, AxisComponent)
                else parent_component
            )
        return super().add_node(axis, parent_axis, parent_component, uniquify=uniquify)

    def append_axis(self, axis, *, uniquify=False):
        if self.is_empty:
            return self.add_axis(axis, None, uniquify=uniquify)
        else:
            if len(self.leaves) == 1:
                leaf_axis, leaf_component = self.leaf
                return self.add_axis(axis, leaf_axis, leaf_component, uniquify=uniquify)
            else:
                raise ExpectedLinearAxisTreeException(
                    "Can only append axes to trees with one leaf."
                )

    @deprecated("add_axis")
    def add_subaxis(self, *args, **kwargs):
        return self.add_axis(*args, **kwargs)

    @property
    def layout_axes(self):
        return self

    @cached_property
    def layouts(self):
        """Initialise the multi-axis by computing the layout functions."""
        from pyop3.axtree.layout import (
            _collect_at_leaves,
            _compute_layouts,
            collect_externally_indexed_axes,
        )
        from pyop3.itree.tree import IndexExpressionReplacer, LoopIndexVariable

        if self.layout_axes.is_empty:
            return freeze({pmap(): 0})

        loop_vars = self.outer_loop_bits[1] if self.outer_loops else {}
        layouts, check_none, _ = _compute_layouts(self.layout_axes, loop_vars)

        assert check_none is None

        layoutsnew = _collect_at_leaves(self, self.layout_axes, layouts)
        layouts = freeze(dict(layoutsnew))

        if self.outer_loops:
            _, loop_vars = self.outer_loop_bits

            layouts_ = {}
            for k, layout in layouts.items():
                layouts_[k] = IndexExpressionReplacer(loop_vars)(layout)
            layouts = freeze(layouts_)

        # for now
        return freeze(layouts)

        # Have not considered how to do sparse things with external loops
        if self.layout_axes.depth > self.depth:
            return layouts

        layouts_ = {pmap(): 0}
        for axis in self.nodes:
            for component in axis.components:
                orig_path = self.path(axis, component)
                new_path = {}
                replace_map = {}
                for ax, cpt in self.path_with_nodes(axis, component).items():
                    new_path.update(self.target_paths.get((ax.id, cpt), {}))
                    replace_map.update(self.layout_exprs.get((ax.id, cpt), {}))
                new_path = freeze(new_path)

                orig_layout = layouts[orig_path]
                new_layout = IndexExpressionReplacer(replace_map, loop_exprs)(
                    orig_layout
                )
                layouts_[new_path] = new_layout
        return freeze(layouts_)

    @property
    def subst_layouts(self):
        return self.layouts


# are all of these necessary?
# class IndexedAxisTree(Indexed, BaseAxisTree):
class IndexedAxisTree(BaseAxisTree):
    def __init__(
        self,
        node_map,
        unindexed,
        *,
        target_paths,
        index_exprs,
        layout_exprs,
        outer_loops,
    ):
        if outer_loops is None:
            outer_loops = ()
        else:
            assert isinstance(outer_loops, tuple)

        super().__init__(node_map)
        self._unindexed = unindexed
        self._target_paths = target_paths
        self._index_exprs = index_exprs
        self._layout_exprs = layout_exprs
        self._outer_loops = tuple(outer_loops)

    @property
    def unindexed(self):
        return self._unindexed

    @property
    def target_paths(self):
        return self._target_paths

    @property
    def index_exprs(self):
        return self._index_exprs

    @property
    def layout_exprs(self):
        return self._layout_exprs

    @property
    def layouts(self):
        return self.unindexed.layouts

    @property
    def outer_loops(self):
        return self._outer_loops

    @cached_property
    def layout_axes(self) -> AxisTree:
        if not self.outer_loops:
            return self
        loop_axes, _ = self.outer_loop_bits
        return loop_axes.add_subtree(self, *loop_axes.leaf)

    # This could easily be two functions
    @cached_property
    def outer_loop_bits(self):
        from pyop3.itree.tree import LocalLoopIndexVariable

        if len(self.outer_loops) > 1:
            # We do not yet support something like dat[p, q] if p and q
            # are independent (i.e. q != f(p) ).
            raise NotImplementedError(
                "Multiple independent outer loops are not supported."
            )
        loop = just_one(self.outer_loops)

        # TODO: Don't think this is needed
        # Since loop itersets must be linear, we can unpack target_paths
        # and index_exprs from
        #
        #     {(axis_id, component_label): {axis_label: expr}}
        #
        # to simply
        #
        #     {axis_label: expr}
        flat_target_paths = {}
        flat_index_exprs = {}
        for axis in loop.iterset.nodes:
            key = (axis.id, axis.component.label)
            flat_target_paths.update(loop.iterset.target_paths.get(key, {}))
            flat_index_exprs.update(loop.iterset.index_exprs.get(key, {}))

        # Make sure that the layout axes are uniquely labelled.
        suffix = f"_{loop.id}"
        loop_axes = relabel_axes(loop.iterset, suffix)

        # Nasty hack: loop_axes need to be a PartialAxisTree so we can add to it.
        loop_axes = AxisTree(loop_axes.parent_to_children)

        # When we tabulate the layout, the layout expressions will contain
        # axis variables that we actually want to be loop index variables. Here
        # we construct the right replacement map.
        loop_vars = {
            axis.label + suffix: LocalLoopIndexVariable(loop, axis.label)
            for axis in loop.iterset.nodes
        }

        # Recursively fetch other outer loops and make them the root of
        # the current axes.
        if loop.iterset.outer_loops:
            ax_rec, lv_rec = loop.iterset.outer_loop_bits
            loop_axes = ax_rec.add_subtree(loop_axes, *ax_rec.leaf)
            loop_vars.update(lv_rec)

        return loop_axes, freeze(loop_vars)

    @cached_property
    def subst_layouts(self):
        return subst_layouts(self, self.target_paths, self.index_exprs, self.layouts)


class ContextSensitiveAxisTree(ContextSensitiveLoopIterable):
    def __getitem__(self, indices) -> ContextSensitiveAxisTree:
        raise NotImplementedError
        # TODO think harder about composing context maps
        # answer is something like:
        # new_context_map = {}
        # for context, axes in self.context_map.items():
        #     for context_, axes_ in index_axes(axes, indices).items():
        #         new_context_map[context | context_] = axes_
        # return ContextSensitiveAxisTree(new_context_map)

    def index(self) -> LoopIndex:
        from pyop3.itree import LoopIndex

        # TODO
        # return LoopIndex(self.owned)
        return LoopIndex(self)

    @cached_property
    def datamap(self):
        return merge_dicts(axes.datamap for axes in self.context_map.values())


@functools.singledispatch
def as_axis_tree(arg: Any) -> AxisTree:
    axis = as_axis(arg)
    return as_axis_tree(axis)


@as_axis_tree.register
def _(axes: AxisTree) -> AxisTree:
    return axes


@as_axis_tree.register
def _(axes: IndexedAxisTree) -> IndexedAxisTree:
    return axes


@as_axis_tree.register
def _(axis: Axis) -> AxisTree:
    return AxisTree(axis)


@functools.singledispatch
def as_axis(arg: Any) -> Axis:
    component = as_axis_component(arg)
    return as_axis(component)


@as_axis.register
def _(axis: Axis) -> Axis:
    return axis


@as_axis.register
def _(component: AxisComponent) -> Axis:
    return Axis(component)


@functools.singledispatch
def as_axis_component(arg: Any) -> AxisComponent:
    from pyop3.array import HierarchicalArray  # cyclic import

    if isinstance(arg, HierarchicalArray):
        return AxisComponent(arg)
    else:
        raise TypeError(f"No handler defined for {type(arg).__name__}")


@as_axis_component.register
def _(component: AxisComponent) -> AxisComponent:
    return component


@as_axis_component.register
def _(arg: numbers.Integral) -> AxisComponent:
    return AxisComponent(arg)


def relabel_axes(axes: AxisTree, suffix: str) -> AxisTree:
    # comprehension?
    parent_to_children = {}
    for parent_id, children in axes.parent_to_children.items():
        children_ = []
        for axis in children:
            if axis is not None:
                axis_ = axis.copy(label=axis.label + suffix)
            else:
                axis_ = None
            children_.append(axis_)
        parent_to_children[parent_id] = children_
    return AxisTree(parent_to_children)


def subst_layouts(
    axes,
    target_paths,
    index_exprs,
    layouts,
    axis=None,
    path=None,
    target_path_acc=None,
    index_exprs_acc=None,
):
    from pyop3 import HierarchicalArray
    from pyop3.itree.tree import IndexExpressionReplacer

    if isinstance(axes, HierarchicalArray):
        assert axis is None
        axes = axes.axes

    # TODO Don't do this every time this function is called
    loop_exprs = {}
    # for outer_loop in self.outer_loops:
    #     loop_exprs[outer_loop.id] = {}
    #     for ax in outer_loop.iterset.nodes:
    #         key = (ax.id, ax.component.label)
    #         for ax_, expr in outer_loop.iterset.index_exprs.get(key, {}).items():
    #             loop_exprs[outer_loop.id][ax_] = expr

    layouts_subst = {}
    if strictly_all(x is None for x in [axis, path, target_path_acc, index_exprs_acc]):
        path = pmap()
        target_path_acc = target_paths.get(None, pmap())
        index_exprs_acc = index_exprs.get(None, pmap())

        replacer = IndexExpressionReplacer(index_exprs_acc, loop_exprs=loop_exprs)
        layouts_subst[path] = replacer(layouts.get(target_path_acc, 0))

        if not axes.is_empty:
            layouts_subst.update(
                subst_layouts(
                    axes,
                    target_paths,
                    index_exprs,
                    layouts,
                    axes.root,
                    path,
                    target_path_acc,
                    index_exprs_acc,
                )
            )
    else:
        for component in axis.components:
            path_ = path | {axis.label: component.label}
            target_path_acc_ = target_path_acc | target_paths.get(
                (axis.id, component.label), {}
            )
            index_exprs_acc_ = index_exprs_acc | index_exprs.get(
                (axis.id, component.label), {}
            )

            replacer = IndexExpressionReplacer(index_exprs_acc_)
            layouts_subst[path_] = replacer(layouts.get(target_path_acc_, 0))

            if subaxis := axes.child(axis, component):
                layouts_subst.update(
                    subst_layouts(
                        axes,
                        target_paths,
                        index_exprs,
                        layouts,
                        subaxis,
                        path_,
                        target_path_acc_,
                        index_exprs_acc_,
                    )
                )
    return freeze(layouts_subst)

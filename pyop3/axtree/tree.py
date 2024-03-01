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
from pyrsistent import freeze, pmap

from pyop3 import utils
from pyop3.dtypes import IntType, PointerType, get_mpi_dtype
from pyop3.extras.debug import print_with_rank
from pyop3.sf import StarForest, serial_forest
from pyop3.tree import (
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    as_component_label,
    postvisit,
    previsit,
)
from pyop3.utils import (
    PrettyTuple,
    as_tuple,
    checked_zip,
    debug_assert,
    deprecated,
    flatten,
    frozen_record,
    has_unique_entries,
    invert,
    is_single_valued,
    just_one,
    merge_dicts,
    pairwise,
    single_valued,
    some_but_not_all,
    steps,
    strict_int,
    strictly_all,
    unique,
)


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

    @cached_property
    def subst_layouts(self):
        return self._subst_layouts()

    def _subst_layouts(self, axis=None, path=None, target_path=None, index_exprs=None):
        from pyop3.itree.tree import IndexExpressionReplacer

        layouts = {}
        if strictly_all(x is None for x in [axis, path, target_path, index_exprs]):
            path = pmap()  # or None?
            target_path = self.target_paths.get(None, pmap())
            index_exprs = self.index_exprs.get(None, pmap())

            replacer = IndexExpressionReplacer(index_exprs)
            layouts[path] = replacer(self.layouts.get(target_path, 0))

            if not self.axes.is_empty:
                layouts.update(
                    self._subst_layouts(self.axes.root, path, target_path, index_exprs)
                )
        else:
            for component in axis.components:
                path_ = path | {axis.label: component.label}
                target_path_ = target_path | self.target_paths.get(
                    (axis.id, component.label), {}
                )
                index_exprs_ = index_exprs | self.index_exprs.get(
                    (axis.id, component.label), {}
                )

                replacer = IndexExpressionReplacer(index_exprs_)
                layouts[path_] = replacer(self.layouts.get(target_path_, 0))

                if subaxis := self.axes.child(axis, component):
                    layouts.update(
                        self._subst_layouts(subaxis, path_, target_path_, index_exprs_)
                    )
        return freeze(layouts)


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


class ExpressionEvaluator(pym.mapper.evaluator.EvaluationMapper):
    def map_axis_variable(self, expr):
        return self.context[expr.axis_label]

    def map_multi_array(self, array_var):
        # indices = {ax: self.rec(idx) for ax, idx in array_var.index_exprs.items()}
        return array_var.array.get_value(
            self.context, array_var.target_path, array_var.index_exprs
        )

    def map_loop_index(self, expr):
        from pyop3.itree.tree import LocalLoopIndexVariable, LoopIndexVariable

        if isinstance(expr, LocalLoopIndexVariable):
            return self.context[expr.id][0][expr.axis]
        else:
            assert isinstance(expr, LoopIndexVariable)
            return self.context[expr.id][1][expr.axis]


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

    fields = LabelledNodeComponent.fields | {
        "count",
    }

    def __init__(
        self,
        count,
        label=None,
        *,
        indices=None,
        indexed=False,
        lgmap=None,
    ):
        from pyop3.array import HierarchicalArray

        if not isinstance(count, (numbers.Integral, HierarchicalArray)):
            raise TypeError("Invalid count type")

        super().__init__(label=label)
        self.count = count

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
            return npoints * axtree.alloc_size(subaxis)
        else:
            return npoints


class Axis(MultiComponentLabelledNode, LoopIterable):
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
        return as_axis_tree(self)[indices]

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
        return as_axis_tree(self).size

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
            return tuple(_as_axis_component(c) for c in components)
        else:
            return (_as_axis_component(components),)

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


class MultiArrayCollector(pym.mapper.Collector):
    def map_multi_array(self, array_var):
        return {array_var.array} | {
            arr for iexpr in array_var.index_exprs.values() for arr in self.rec(iexpr)
        }

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


class PartialAxisTree(LabelledTree):
    def __init__(
        self,
        parent_to_children=None,
    ):
        super().__init__(parent_to_children)

        # TODO Move check to generic LabelledTree
        self._check_node_labels_unique_in_paths(self.parent_to_children)

        # makea  cached property, then delete this method
        self._layout_exprs = AxisTree._default_index_exprs(self)

    @classmethod
    def from_iterable(cls, iterable):
        # NOTE: This currently only works for linear trees
        item, *iterable = iterable
        tree = PartialAxisTree(as_axis_tree(item).parent_to_children)
        for item in iterable:
            tree = tree.add_subtree(as_axis_tree(item), *tree.leaf)
        return tree

    @classmethod
    def _check_node_labels_unique_in_paths(
        cls, node_map, node=None, seen_labels=frozenset()
    ):
        from pyop3.tree import InvalidTreeException

        if not node_map:
            return

        if node is None:
            node = just_one(node_map[None])

        if node.label in seen_labels:
            raise InvalidTreeException("Duplicate labels found along a path")

        for subnode in filter(None, node_map.get(node.id, [])):
            cls._check_node_labels_unique_in_paths(
                node_map, subnode, seen_labels | {node.label}
            )

    def set_up(self):
        return AxisTree.from_partial_tree(self)

    @deprecated("set_up")
    def freeze(self):
        return self.set_up()

    @cached_property
    def datamap(self):
        if self.is_empty:
            dmap = {}
        else:
            dmap = postvisit(self, _collect_datamap, axes=self)
        return freeze(dmap)

    def add_node(
        self,
        axis,
        parent,
        parent_component=None,
        **kwargs,
    ):
        parent = self._as_node(parent)
        if parent_component is None:
            if len(parent.components) == 1:
                parent_cpt_label = parent.components[0].label
            else:
                raise ValueError("Must specify parent component")
        else:
            parent_cpt_label = _as_axis_component_label(parent_component)
        return super().add_node(axis, parent, parent_cpt_label, **kwargs)

    def add_subaxis(self, subaxis, *loc):
        return self.add_node(subaxis, *loc)

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
            target_indices = merge_dicts(idx.target_exprs for idx in idxs)
            # this is a hack
            if self.is_empty:
                mysize += 1
            else:
                mysize += _axis_size(self, self.root, target_indices)
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
        axis = axis or self.root
        return sum(cpt.alloc_size(self, axis) for cpt in axis.components)


@frozen_record
class AxisTree(PartialAxisTree, Indexed, ContextFreeLoopIterable):
    fields = PartialAxisTree.fields | {
        "target_paths",
        "index_exprs",
        "outer_loops",
        "layout_exprs",
    }

    def __init__(
        self,
        parent_to_children=pmap(),
        target_paths=None,
        index_exprs=None,
        outer_loops=None,
        layout_exprs=None,
    ):
        # if some_but_not_all(
        #     arg is None
        #     for arg in [target_paths, index_exprs, outer_loops, layout_exprs]
        # ):
        #     raise ValueError

        if outer_loops is None:
            outer_loops = ()
        else:
            assert isinstance(outer_loops, tuple)

        super().__init__(parent_to_children)
        self._target_paths = target_paths or self._default_target_paths()
        self._index_exprs = index_exprs or self._default_index_exprs()
        self.layout_exprs = layout_exprs or self._default_layout_exprs()
        self._outer_loops = tuple(outer_loops)

    def __getitem__(self, indices):
        from pyop3.itree.tree import _compose_bits, _index_axes, as_index_forest

        if indices is Ellipsis:
            raise NotImplementedError("TODO")
            indices = index_tree_from_ellipsis(self)

        axis_trees = {}
        for context, index_tree in as_index_forest(indices, axes=self).items():
            indexed_axes = _index_axes(index_tree, context, self)

            target_paths, index_exprs, layout_exprs = _compose_bits(
                self,
                self.target_paths,
                self.index_exprs,
                self.layout_exprs,
                indexed_axes,
                indexed_axes.target_paths,
                indexed_axes.index_exprs,
                indexed_axes.layout_exprs,
            )
            axis_tree = AxisTree(
                indexed_axes.parent_to_children,
                target_paths,
                index_exprs,
                outer_loops=indexed_axes.outer_loops,
                layout_exprs=layout_exprs,
            )
            axis_trees[context] = axis_tree

        if len(axis_trees) == 1 and just_one(axis_trees.keys()) == pmap():
            return axis_trees[pmap()]
        else:
            return ContextSensitiveAxisTree(axis_trees)

    @classmethod
    def from_nest(cls, nest) -> AxisTree:
        root, node_map = cls._from_nest(nest)
        node_map.update({None: [root]})
        return cls.from_node_map(node_map)

    @classmethod
    def from_iterable(
        cls, iterable, *, target_paths=None, index_exprs=None, layout_exprs=None
    ) -> AxisTree:
        tree = PartialAxisTree.from_iterable(iterable)
        return AxisTree(
            tree.parent_to_children,
            target_paths=target_paths,
            index_exprs=index_exprs,
            layout_exprs=layout_exprs,
        )

    @classmethod
    def from_node_map(cls, node_map):
        tree = PartialAxisTree(node_map)
        return cls.from_partial_tree(tree)

    @classmethod
    def from_partial_tree(cls, tree: PartialAxisTree) -> AxisTree:
        target_paths = cls._default_target_paths(tree)
        index_exprs = cls._default_index_exprs(tree)
        layout_exprs = index_exprs
        outer_loops = ()
        return cls(
            tree.parent_to_children,
            target_paths,
            index_exprs,
            outer_loops=outer_loops,
            layout_exprs=layout_exprs,
        )

    def index(self):
        from pyop3.itree import LoopIndex

        return LoopIndex(self.owned)

    def iter(self, outer_loops=(), loop_index=None):
        from pyop3.itree.tree import iter_axis_tree

        return iter_axis_tree(
            # hack because sometimes we know the right loop index to use
            loop_index or self.index(),
            self,
            self.target_paths,
            self.index_exprs,
            outer_loops,
        )

    @property
    def axes(self):
        return self

    @property
    def target_paths(self):
        return self._target_paths

    @property
    def index_exprs(self):
        return self._index_exprs

    @property
    def outer_loops(self):
        return self._outer_loops

    @cached_property
    def layout_axes(self):
        # TODO same loop as in AxisTree.layouts
        axes_iter = []
        for ol in self.outer_loops:
            for axis in ol.iterset.nodes:
                # FIXME relabelling here means that paths are not propagated properly
                # when we tabulate.
                # axis_ = axis.copy(id=Axis.unique_id(), label=Axis.unique_label())
                axis_ = axis
                axes_iter.append(axis_)
        return AxisTree.from_iterable([*axes_iter, self])

    @cached_property
    def layouts(self):
        """Initialise the multi-axis by computing the layout functions."""
        from pyop3.axtree.layout import (
            _collect_at_leaves,
            _compute_layouts,
            collect_externally_indexed_axes,
        )
        from pyop3.itree.tree import IndexExpressionReplacer, LocalLoopIndexVariable

        index_exprs = {}
        loop_vars = {}
        for ol in self.outer_loops:
            for axis in ol.iterset.nodes:
                # FIXME relabelling here means that paths are not propagated properly
                # when we tabulate.
                # axis_ = axis.copy(id=Axis.unique_id(), label=Axis.unique_label())
                axis_ = axis
                index_exprs[axis_.id, axis_.component.label] = {
                    axis.label: LocalLoopIndexVariable(ol, axis_.label)
                }
                loop_vars[axis_, axis.component] = ol

        for axis in self.nodes:
            for component in axis.components:
                index_exprs[axis.id, component.label] = {
                    axis.label: AxisVariable(axis.label)
                }

        layout_axes = self.layout_axes

        if layout_axes.is_empty:
            return freeze({pmap(): 0})

        layouts, _, _, _, _ = _compute_layouts(
            layout_axes, self.index_exprs | index_exprs, loop_vars
        )

        layoutsnew = _collect_at_leaves(self, layout_axes, layouts)
        layouts = freeze(dict(layoutsnew))

        # Have not considered how to do sparse things with external loops
        if layout_axes.depth > self.depth:
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
                new_layout = IndexExpressionReplacer(replace_map)(orig_layout)
                layouts_[new_path] = new_layout
        return freeze(layouts_)

    @cached_property
    def leaf_target_paths(self):
        return tuple(
            merge_dicts(
                self.target_paths.get((ax.id, clabel), {})
                for ax, clabel in self.path_with_nodes(*leaf, ordered=True)
            )
            for leaf in self.leaves
        )

    @cached_property
    def sf(self):
        return self._default_sf()

    # @property
    # def lgmap(self):
    #     if not hasattr(self, "_lazy_lgmap"):
    #         # if self.sf.nleaves == 0 then some assumptions are broken in
    #         # ISLocalToGlobalMappingCreateSF, but we need to be careful things are done
    #         # collectively
    #         self.sf.sf.view()
    #         lgmap = PETSc.LGMap().createSF(self.sf.sf, PETSc.DECIDE)
    #         lgmap.setType(PETSc.LGMap.Type.BASIC)
    #         self._lazy_lgmap = lgmap
    #         lgmap.view()
    #     return self._lazy_lgmap

    @property
    def comm(self):
        paraxes = [axis for axis in self.nodes if axis.sf is not None]
        if not paraxes:
            return MPI.COMM_SELF
        else:
            return single_valued(ax.comm for ax in paraxes)

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
            )
            for c in paraxis.components
        ]
        slice_ = Slice(paraxis.label, slices)
        return self[slice_]

    def freeze(self):
        return self

    def as_tree(self):
        return self

    def offset(self, indices, target_path=None, index_exprs=None):
        from pyop3.axtree.layout import eval_offset

        return eval_offset(self, self.layouts, indices, target_path, index_exprs)

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

    def _default_target_paths(self):
        if self.is_empty:
            return pmap()

        return pmap(
            {
                (axis.id, cpt.label): pmap({axis.label: cpt.label})
                for axis in self.nodes
                for cpt in axis.components
            }
        )

    def _default_index_exprs(self):
        if self.is_empty:
            return pmap()

        return pmap(
            {
                (axis.id, cpt.label): pmap({axis.label: AxisVariable(axis.label)})
                for axis in self.nodes
                for cpt in axis.components
            }
        )

    def _default_layout_exprs(self):
        return self._default_index_exprs()

    def _default_sf(self):
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
def as_axis_tree(arg: Any):
    from pyop3.array import HierarchicalArray  # cyclic import

    if isinstance(arg, HierarchicalArray):
        return as_axis_tree(AxisComponent(arg))
    raise TypeError


@as_axis_tree.register
def _(arg: PartialAxisTree):
    return arg


@as_axis_tree.register
def _(arg: AxisTree):
    return arg


@as_axis_tree.register
def _(arg: Axis):
    return arg.as_tree()


@as_axis_tree.register
def _(arg: AxisComponent):
    return as_axis_tree(Axis([arg]))


@as_axis_tree.register(numbers.Integral)
def _(arg: numbers.Integral):
    return as_axis_tree(AxisComponent(arg))


@functools.singledispatch
def _as_axis(arg) -> Axis:
    return Axis(_as_axis_component(arg))


@_as_axis.register
def _(arg: Axis):
    return arg


@_as_axis.register
def _(arg: AxisComponent):
    return Axis(arg)


@functools.singledispatch
def _as_axis_component(arg: Any) -> AxisComponent:
    from pyop3.array import HierarchicalArray  # cyclic import

    if isinstance(arg, HierarchicalArray):
        return AxisComponent(arg)
    else:
        raise TypeError


@_as_axis_component.register
def _(arg: AxisComponent) -> AxisComponent:
    return arg


@_as_axis_component.register
def _(arg: numbers.Integral) -> AxisComponent:
    return AxisComponent(arg)


@_as_axis_component.register
def _(arg: tuple) -> AxisComponent:
    return AxisComponent(*arg)


@functools.singledispatch
def _as_axis_component_label(arg: Any):
    if isinstance(arg, str):
        return arg
    else:
        raise TypeError(f"No handler registered for {type(arg).__name__}")


@_as_axis_component_label.register
def _(component: AxisComponent):
    return component.label

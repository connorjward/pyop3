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
from pyop3.extras.debug import print_if_rank, print_with_rank
from pyop3.sf import StarForest
from pyop3.tree import (
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    postvisit,
    previsit,
)
from pyop3.utils import (
    PrettyTuple,
    as_tuple,
    checked_zip,
    deprecated,
    flatten,
    has_unique_entries,
    is_single_valued,
    just_one,
    merge_dicts,
    single_valued,
    some_but_not_all,
    strict_int,
    strictly_all,
    unique,
)


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
    def __init__(self, context_map: pmap[pmap[LoopIndex, pmap[str, str]], ContextFree]):
        self.context_map = pmap(context_map)

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
                key.update({loop_index: path})
        return pmap(key)


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
    @property
    @abc.abstractmethod
    def target_paths(self):
        pass

    @property
    @abc.abstractmethod
    def index_exprs(self):
        pass


class ContextSensitiveLoopIterable(LoopIterable, ContextSensitive, abc.ABC):
    pass


class ExpressionEvaluator(pym.mapper.evaluator.EvaluationMapper):
    def map_axis_variable(self, expr):
        return self.context[expr.axis_label]

    def map_multi_array(self, expr):
        # path = _trim_path(array.axes, self.context[0])
        # not multi-component for now, is that useful to add?
        path = expr.array.axes.path(*expr.array.axes.leaf)
        # context = []
        # for keyval in self.context.items():
        #     context.append(keyval)
        # return expr.array.get_value(path, self.context[1])
        replace_map = {axis: self.rec(idx) for axis, idx in expr.indices.items()}
        return expr.array.get_value(path, replace_map)

    def map_loop_index(self, expr):
        return self.context[expr.name, expr.axis]

    def map_called_map(self, expr):
        array = expr.function.map_component.array
        indices = {axis: self.rec(idx) for axis, idx in expr.parameters.items()}

        path = array.axes.path(*array.axes.leaf)

        # the inner_expr tells us the right mapping for the temporary, however,
        # for maps that are arrays the innermost axis label does not always match
        # the label used by the temporary. Therefore we need to do a swap here.
        # I don't like this.
        # print_if_rank(0, repr(array.axes))
        # print_if_rank(0, "before: ",indices)
        inner_axis = array.axes.leaf_axis
        indices[inner_axis.label] = indices.pop(expr.function.full_map.name)

        # print_if_rank(0, "after:",indices)
        # print_if_rank(0, repr(expr))
        # print_if_rank(0, self.context)
        return array.get_value(path, indices)


class IntRef:
    """Pass-by-reference integer."""

    def __init__(self, value):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self


def get_bottom_part(axis):
    # must be linear
    return just_one(axis.leaves)


def as_multiaxis(axis):
    if isinstance(axis, MultiAxis):
        return axis
    elif isinstance(axis, AxisPart):
        return MultiAxis(axis)
    else:
        raise TypeError


# def is_set_up(axtree, axis=None):
#     """Return ``True`` if all parts (recursively) of the multi-axis have an associated
#     layout function.
#     """
#     axis = axis or axtree.root
#     return all(
#         part_is_set_up(axtree, axis, cpt, cidx)
#         for cidx, cpt in enumerate(axis.components)
#     )


# # this would be an easy place to start with writing a tree visitor instead
# def part_is_set_up(axtree, axis, cpt):
#     if (subaxis := axtree.child(axis, cpt)) and not is_set_up(
#         axtree, subaxis
#     ):
#         return False
#     if (axis.id, component_index) not in axtree._layouts:
#         return False
#     return True


def only_linear(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_linear:
            raise RuntimeError(f"{func.__name__} only admits linear multi-axes")
        return func(self, *args, **kwargs)

    return wrapper


# --------------------- \/ lifted from halo.py \/ -------------------------


from pyop3.dtypes import as_numpy_dtype


def reduction_op(op, invec, inoutvec, datatype):
    dtype = as_numpy_dtype(datatype)
    invec = np.frombuffer(invec, dtype=dtype)
    inoutvec = np.frombuffer(inoutvec, dtype=dtype)
    inoutvec[:] = op(invec, inoutvec)


_contig_min_op = MPI.Op.Create(
    functools.partial(reduction_op, np.minimum), commute=True
)
_contig_max_op = MPI.Op.Create(
    functools.partial(reduction_op, np.maximum), commute=True
)

# --------------------- ^ lifted from halo.py ^ -------------------------


def _collect_datamap(axis, *subdatamaps, axes):
    from pyop3.distarray import Dat

    datamap = {}
    for cidx, component in enumerate(axis.components):
        if isinstance(count := component.count, Dat):
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
        "overlap",
        "indexed",
        "indices",
        "lgmap",
    }

    def __init__(
        self,
        count,
        label=None,
        *,
        indices=None,
        overlap=None,
        indexed=False,
        lgmap=None,
    ):
        super().__init__(label=label)
        self.count = count
        self.indices = indices
        self.overlap = overlap
        self.indexed = indexed
        self.lgmap = lgmap
        """
        this property is required because we can hit situations like the following:

            sizes = 3 -> [2, 1, 2] -> [[2, 1], [1], [3, 2]]

        this yields a layout that looks like

            [[0, 2], [3], [4, 7]]

        however, if we have a temporary where we only want the inner two dimensions
        then we need a layout that looks like the following:

            [[0, 2], [0], [0, 3]]

        This effectively means that we need to zero the offset as we traverse the
        tree to produce the layout. This is why we need this ``indexed`` flag.
        """

    @property
    def is_distributed(self):
        return self.overlap is not None

    @property
    def has_integer_count(self):
        return isinstance(self.count, numbers.Integral)

    @property
    def is_ragged(self):
        from pyop3.distarray import MultiArray

        return isinstance(self.count, MultiArray)

    # TODO this is just a traversal - clean up
    def alloc_size(self, axtree, axis):
        from pyop3.distarray import MultiArray

        if isinstance(self.count, MultiArray):
            npoints = self.count.max_value
        else:
            assert isinstance(self.count, numbers.Integral)
            npoints = self.count

        assert npoints is not None

        if subaxis := axtree.component_child(axis, self):
            return npoints * axtree.alloc_size(subaxis)
        else:
            return npoints

    @property
    def has_partitioned_halo(self):
        if self.overlap is None:
            return True

        remaining = itertools.dropwhile(lambda o: is_owned_by_process(o), self.overlap)
        return all(isinstance(o, Halo) for o in remaining)

    @property
    def num_owned(self) -> int:
        from pyop3.distarray import MultiArray

        """Return the number of owned points."""
        if isinstance(self.count, MultiArray):
            # TODO: Might we ever want this to work?
            raise RuntimeError("nowned is only valid for non-ragged axes")

        if self.overlap is None:
            return self.count
        else:
            return sum(1 for o in self.overlap if is_owned_by_process(o))

    @property
    def nowned(self):
        # alias, what is the best name?
        return self.num_owned


class _AxisComponent(LabelledNodeComponent, abc.ABC):
    fields = {"_component", "target_path", "index_expr", "layout_expr"}

    def __init__(self, _component, target_path, index_expr, layout_expr):
        super().__init__(_component.label)
        self._component = _component
        self.target_path = target_path
        self.index_expr = index_expr
        self.layout_expr = layout_expr

    # TODO: use specific @property methods
    def __getattr__(self, name):
        return getattr(self._component, name)


class _InternalAxisComponent(_AxisComponent):
    pass


class _LeafAxisComponent(_AxisComponent):
    fields = _AxisComponent.fields | {"layout"}

    def __init__(self, _component, target_path, index_expr, layout_expr, layout):
        super().__init__(_component, target_path, index_expr, layout_expr)
        self.layout = layout


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

        super().__init__(components, label=label, id=id)

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
            counts[self._component_index_from_axis_number(leaf_index)] += 1
        return freeze(
            {cpt: count for cpt, count in checked_zip(self.components, counts)}
        )

    def index(self):
        return as_axis_tree(self).index()

    @property
    def target_path_per_component(self):
        return as_axis_tree(self).target_path_per_component

    @property
    def index_exprs_per_component(self):
        return as_axis_tree(self).index_exprs_per_component

    @property
    def layout_exprs_per_component(self):
        return as_axis_tree(self).layout_exprs_per_component

    # cached?
    @property
    def axes(self):
        return as_axis_tree(self)

    # cached?
    @property
    def index_exprs(self):
        return as_axis_tree(self).index_exprs

    # Note: these functions assume that the numbering follows the plex convention
    # of numbering each strata contiguously. I think (?) that I effectively also do this.
    # actually this might well be wrong. we have a renumbering after all - this gives us
    # the original numbering only
    def component_number_to_axis_number(self, component, num):
        component_index = self.components.index(component)
        canonical = self._component_numbering_offsets[component_index] + num
        return self._to_renumbered(canonical)

    def axis_number_to_component(self, num):
        # guess, is this the right map (from new numbering to original)?
        # I don't think so because we have a funky point SF. can we get rid?
        # num = self.numbering[num]
        component_index = self._component_index_from_axis_number(num)
        component_num = num - self._component_numbering_offsets[component_index]
        # return self.components[component_index], component_num
        return self.components[component_index], component_num

    def _component_index_from_axis_number(self, num):
        offsets = self._component_numbering_offsets
        for i, (min_, max_) in enumerate(zip(offsets, offsets[1:])):
            if min_ <= num < max_:
                return i
        raise ValueError(f"Axis number {num} not found.")

    @cached_property
    def _component_numbering_offsets(self):
        return (0,) + tuple(np.cumsum([c.count for c in self.components], dtype=int))

    # FIXME bad name
    def _to_renumbered(self, num):
        """Convert a flat/canonical/unpermuted axis number to its renumbered equivalent."""
        if self.numbering is None:
            return num
        else:
            return self._inverse_numbering[num]

    @cached_property
    def _inverse_numbering(self):
        # put in utils.py
        from pyop3.axtree.parallel import invert

        if self.numbering is None:
            return np.arange(self.count, dtype=IntType)
        else:
            return invert(self.numbering.data_ro)

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
        from pyop3.distarray import Dat

        if numbering is None:
            return None
        elif isinstance(numbering, Dat):
            return numbering
        elif isinstance(numbering, collections.abc.Collection):
            return Dat(len(numbering), data=numbering, dtype=IntType)
        else:
            raise TypeError(
                f"{type(numbering).__name__} is not a supported type for numbering"
            )


class MultiArrayCollector(pym.mapper.Collector):
    def map_called_map(self, expr):
        return self.rec(expr.function) | set.union(
            *(self.rec(idx) for idx in expr.parameters.values())
        )

    def map_map_variable(self, expr):
        return {expr.map_component.array}

    def map_multi_array(self, expr):
        return {expr}

    def map_nan(self, expr):
        return set()


class AxisTreeMixin(abc.ABC):
    @property
    def leaf_axis(self):
        return self.leaf[0]

    @property
    def leaf_component(self):
        return self.leaf[1]

    @cached_property
    def size(self):
        from pyop3.axtree.layout import axis_tree_size

        return axis_tree_size(self)

    def alloc_size(self, axis=None):
        axis = axis or self.root
        return sum(cpt.alloc_size(self, axis) for cpt in axis.components)

    def offset(self, *args, allow_unused=False, insert_zeros=False):
        nargs = len(args)
        if nargs == 2:
            path, indices = args[0], args[1]
        else:
            assert nargs == 1
            path, indices = _path_and_indices_from_index_tuple(self, args[0])

        if allow_unused:
            path = _trim_path(self, path)

        if insert_zeros:
            # extend the path by choosing the zero offset option every time
            # this is needed if we don't have all the internal bits available
            while path not in self.layouts:
                axis, clabel = self._node_from_path(path)
                subaxis = self.component_child(axis, clabel)
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

    @deprecated("offset")
    def get_offset(self, *args, **kwargs):
        return self.offset(*args, **kwargs)


class AxisTree(AxisTreeMixin, LabelledTree, ContextFreeLoopIterable):
    def __init__(
        self,
        parent_to_children=None,
        *,
        target_path=None,
        index_expr=None,
        layout_expr=None,
        layout=None,
    ):
        super().__init__(parent_to_children)

        # these are just the "fully indexed" bits, the rest remains attached to components
        self.target_path = target_path
        self.index_expr = index_expr
        self.layout_expr = layout_expr
        self.layout = layout

        # makea  cached property, then delete this method
        self._layout_exprs = "is this used???"
        # self._layout_exprs = FrozenAxisTree._default_index_exprs(self)

    def __getitem__(self, indices) -> IndexedAxisTree:
        return self.freeze()[indices]

    @classmethod
    def from_nest(cls, nest) -> AxisTree:
        root, node_map = cls._from_nest(nest)
        node_map |= {None: (root,)}
        tree = LabelledTree(node_map)
        return cls.from_tree(tree)

    @classmethod
    def from_tree(cls, tree: LabelledTree) -> AxisTree:
        """Parse a "partial" tree of axes to a "complete" one.

        This method is useful for when one has a tree of axes but no
        indexing/layout information. This method performs this tabulation
        to generate a "complete" tree.

        """
        from pyop3.axtree.layout import _collect_at_leaves, _compute_layouts

        if tree.is_empty:
            return cls()
        else:
            layouts, _, _, _ = _compute_layouts(tree, tree.root)
            layoutsnew = _collect_at_leaves(tree, layouts)

            root, node_map = cls._from_tree(tree, layoutsnew)
            node_map |= {None: (root,)}
            return cls(node_map)

    def index(self) -> LoopIndex:
        return self.freeze().index()

    # TODO is this the right way to deal with these properties?
    @property
    def target_paths(self):
        raise RuntimeError("Should already be frozen")

    @property
    def index_exprs(self):
        raise RuntimeError("Should already be frozen")

    def freeze(self) -> FrozenAxisTree:
        return FrozenAxisTree(self.parent_to_children)

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

    # alias
    add_subaxis = add_node

    # currently untested but should keep
    @classmethod
    def from_layout(cls, layout: Sequence[ConstrainedMultiAxis]) -> Any:  # TODO
        return order_axes(layout)

    # TODO this is just a regular tree search
    @deprecated(internal=True)  # I think?
    def get_part_from_path(self, path, axis=None):
        axis = axis or self.root

        label, *sublabels = path

        (component, component_index) = just_one(
            [
                (cpt, cidx)
                for cidx, cpt in enumerate(axis.components)
                if (axis.label, cidx) == label
            ]
        )
        if sublabels:
            return self.get_part_from_path(
                sublabels, self.component_child(axis, component)
            )
        else:
            return axis, component

    @property
    @deprecated(internal=True)
    def is_linear(self):
        """Return ``True`` if the multi-axis contains no branches at any level."""
        if self.nparts == 1:
            return self.part.subaxis.is_linear if self.part.subaxis else True
        else:
            return False

    @deprecated()
    def add_subaxis(self, subaxis, *loc):
        return self.add_node(subaxis, *loc)

    @classmethod
    def _from_tree(cls, tree, layouts, axis=None):
        from pyop3.axtree.layout import AxisVariable

        if axis is None:
            axis = tree.root

        components = []
        node_map = {axis.id: [None] * axis.degree}
        for cidx, cpt in enumerate(axis.components):
            target_path = freeze({axis.label: cpt.label})
            index_expr = freeze({axis.label: AxisVariable(axis.label)})
            layout_expr = index_expr

            subaxis = tree.child(axis, cpt)
            if subaxis:
                cpt_ = _InternalAxisComponent(cpt, target_path, index_expr, layout_expr)
                subaxis_, subnode_map = cls._from_tree(tree, layouts, subaxis)
                node_map[axis.id][cidx] = subaxis_
                node_map.update(subnode_map)
            else:
                layout = layouts[axis.id, cpt.label]
                cpt_ = _LeafAxisComponent(
                    cpt, target_path, index_expr, layout_expr, layout
                )
            components.append(cpt_)

        axis_ = axis.copy(components=components)
        return axis_, freeze(node_map)

    @staticmethod
    def _is_valid_branch(node: _AxisComponent) -> bool:
        return isinstance(node, _InternalAxisComponent)

    @staticmethod
    def _is_valid_leaf(node: _AxisComponent) -> bool:
        return isinstance(node, _LeafAxisComponent)


# TODO: Inherit things from AxisTree, StaticAxisTree?
# class IndexedAxisTree(StrictLabelledTree, AxisTreeMixin, ContextFreeLoopIterable):
class IndexedAxisTree(AxisTreeMixin, LabelledTree, ContextFreeLoopIterable):
    def __init__(
        self,
        parent_to_children,
        target_paths,
        index_exprs,
        layout_exprs,
    ):
        super().__init__(parent_to_children)
        self._target_paths = target_paths
        self._index_exprs = index_exprs
        self.layout_exprs = layout_exprs

    def __getitem__(self, indices):
        from pyop3.itree.tree import (
            as_index_forest,
            collect_loop_contexts,
            index_axes,
            index_tree_from_ellipsis,
        )

        if indices is Ellipsis:
            indices = index_tree_from_ellipsis(self)

        if not collect_loop_contexts(indices):
            index_tree = just_one(as_index_forest(indices, axes=self))
            return index_axes(self, index_tree)

        axis_trees = {}
        for index_tree in as_index_forest(indices, axes=self):
            axis_trees[index_tree.loop_context] = index_axes(self, index_tree)
        return ContextSensitiveAxisTree(axis_trees)

    @property
    def sf(self):
        # FIXME
        return None

    # hacky
    def restore(self):
        return FrozenAxisTree(self.parent_to_children)

    def index(self) -> LoopIndex:
        from pyop3.itree import LoopIndex

        # TODO
        # return LoopIndex(self.owned)
        return LoopIndex(self)

    @property
    def target_paths(self):
        return self._target_paths

    @property
    def index_exprs(self):
        return self._index_exprs

    # TODO refactor
    @property
    def datamap(self):
        if self.is_empty:
            datamap_ = {}
        else:
            datamap_ = postvisit(self, _collect_datamap, axes=self)
        for index_exprs in self.index_exprs.values():
            for index_expr in index_exprs.values():
                for array in MultiArrayCollector()(index_expr):
                    datamap_.update(array.datamap)
        for layout_exprs in self.layout_exprs.values():
            for layout_expr in layout_exprs.values():
                for array in MultiArrayCollector()(layout_expr):
                    datamap_.update(array.datamap)
        return freeze(datamap_)

    def freeze(self):
        return self


# TODO Inherit from FrozenLabelledTree
# TODO The order of inheritance is annoying here, mixin class currently needs to come first
# class FrozenAxisTree(StrictLabelledTree, AxisTreeMixin, ContextFreeLoopIterable):
class FrozenAxisTree(AxisTreeMixin, LabelledTree, ContextFreeLoopIterable):
    def __init__(
        self,
        parent_to_children=None,
        sf=None,
    ):
        super().__init__(parent_to_children)
        # factory method?
        self.sf = sf or self._sf()

    def __getitem__(self, indices) -> Union[IndexedAxisTree, ContextSensitiveAxisTree]:
        from pyop3.itree.tree import as_index_forest, collect_loop_contexts, index_axes

        if indices is Ellipsis:
            indices = index_tree_from_ellipsis(self)

        if not collect_loop_contexts(indices):
            index_tree = just_one(as_index_forest(indices, axes=self))
            return index_axes(self, index_tree)

        axis_trees = {}
        for index_tree in as_index_forest(indices, axes=self):
            axis_trees[index_tree.loop_context] = index_axes(self, index_tree)
        return ContextSensitiveAxisTree(axis_trees)

    # hacky
    def restore(self):
        return FrozenAxisTree(self.parent_to_children)

    def index(self):
        from pyop3.itree import LoopIndex

        return LoopIndex(self.owned)

    # TODO directly address the components
    @cached_property
    def target_paths(self):
        return {
            (ax.id, c.label): c.target_path for ax in self.nodes for c in ax.components
        }

    # TODO directly address the components
    @cached_property
    def index_exprs(self):
        return {
            (ax.id, c.label): c.index_expr for ax in self.nodes for c in ax.components
        }

    @cached_property
    def datamap(self):
        if self.is_empty:
            dmap = {}
        else:
            dmap = postvisit(self, _collect_datamap, axes=self)

        # for cleverdict in [self.layouts, self.orig_layout_fn]:
        #     for layout in cleverdict.values():
        #         for layout_expr in layout.values():
        #             # catch invalid layouts
        #             if isinstance(layout_expr, pym.primitives.NaN):
        #                 continue
        #             for array in MultiArrayCollector()(layout_expr):
        #                 dmap.update(array.datamap)

        # TODO
        # for cleverdict in [self.index_exprs, self.layout_exprs]:
        for cleverdict in [self.index_exprs]:
            for exprs in cleverdict.values():
                for expr in exprs.values():
                    for array in MultiArrayCollector()(expr):
                        dmap.update(array.datamap)
        for layout_expr in self.layouts.values():
            for array in MultiArrayCollector()(layout_expr):
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

    def freeze(self) -> FrozenAxisTree:
        return self

    def _sf(self):
        from pyop3.axtree.parallel import collect_sf_graphs

        if self.is_empty:
            return None

        graphs = collect_sf_graphs(self)
        if len(graphs) == 0:
            return None
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
            # fixme, get the right comm (and ensure consistency)
            return StarForest.from_graph(self.size, nroots, ilocal, iremote)

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


@dataclasses.dataclass(frozen=True)
class Path:
    # TODO Make a persistent dict?
    from_axes: Tuple[Any]  # axis part IDs I guess (or labels)
    to_axess: Tuple[Any]  # axis part IDs I guess (or labels)
    arity: int
    selector: Optional[Any] = None
    """The thing that chooses between the different possible output axes at runtime."""

    @property
    def degree(self):
        return len(self.to_axess)

    @property
    def to_axes(self):
        if self.degree != 1:
            raise RuntimeError("Only for degree 1 paths")
        return self.to_axess[0]


@functools.singledispatch
def as_axis_tree(arg: Any):
    from pyop3.distarray import Dat  # cyclic import
    from pyop3.itree import IndexedAxisTree

    if isinstance(arg, IndexedAxisTree):
        return arg
    if isinstance(arg, Dat):
        return as_axis_tree(AxisComponent(arg))
    raise TypeError


@as_axis_tree.register
def _(arg: AxisTreeMixin):
    return arg


@as_axis_tree.register
def _(arg: Axis):
    return AxisTree.from_nest(arg)


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
    from pyop3.distarray import Dat  # cyclic import

    if isinstance(arg, Dat):
        return AxisComponent(arg)
    else:
        raise TypeError


@_as_axis_component.register
def _(arg: AxisComponent) -> AxisComponent:
    return arg


# not sure about this
@_as_axis_component.register
def _(arg: _AxisComponent) -> _AxisComponent:
    return arg


@_as_axis_component.register
def _(arg: numbers.Integral) -> AxisComponent:
    return AxisComponent(arg)


@_as_axis_component.register
def _(arg: tuple) -> AxisComponent:
    return AxisComponent(*arg)


@functools.singledispatch
def _as_axis_component_label(arg: Any) -> ComponentLabel:
    if isinstance(arg, ComponentLabel):
        return arg
    else:
        raise TypeError(f"No handler registered for {type(arg).__name__}")


@_as_axis_component_label.register
def _(component: AxisComponent):
    return component.label

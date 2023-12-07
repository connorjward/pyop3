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
    frozen_record,
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


class Indexed(abc.ABC):
    @property
    @abc.abstractmethod
    def target_paths(self):
        pass

    @property
    @abc.abstractmethod
    def index_exprs(self):
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


def has_halo(axes, axis):
    if axis.sf is not None:
        return True
    else:
        for component in axis.components:
            subaxis = axes.component_child(axis, component)
            if subaxis and has_halo(axes, subaxis):
                return True
        return False
    return axis.sf is not None or has_halo(axes, subaxis)


def has_independently_indexed_subaxis_parts(axes, axis, cpt):
    """
    subaxis parts are independently indexed if they don't depend on the index from
    ``part``.

    if one sub-part needs this index to determine its extent then we need to create
    a layout function as the step sizes will differ.

    Note that we need to consider both ragged sizes and permutations here
    """
    if subaxis := axes.component_child(axis, cpt):
        return not any(
            requires_external_index(axes, subaxis, c) for c in subaxis.components
        )
    else:
        return True


def only_linear(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_linear:
            raise RuntimeError(f"{func.__name__} only admits linear multi-axes")
        return func(self, *args, **kwargs)

    return wrapper


def can_be_affine(axtree, axis, component, component_index):
    return (
        has_independently_indexed_subaxis_parts(
            axtree, axis, component, component_index
        )
        and component.permutation is None
    )


def has_constant_start(
    axtree, axis, component, component_index, outer_axes_are_all_indexed: bool
):
    """
    We will have an affine layout with a constant start (usually zero) if either we are not
    ragged or if we are ragged but everything above is indexed (i.e. a temporary).
    """
    assert can_be_affine(axtree, axis, component, component_index)
    return isinstance(component.count, numbers.Integral) or outer_axes_are_all_indexed


def has_fixed_size(axes, axis, component):
    return not size_requires_external_index(axes, axis, component)


def step_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    path=pmap(),
    indices=PrettyTuple(),
):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if not has_constant_step(axes, axis, component) and not indices:
        raise ValueError
    if subaxis := axes.component_child(axis, component):
        return _axis_size(axes, subaxis, path, indices)
    else:
        return 1


def make_star_forest_per_axis_part(part, comm):
    if part.is_distributed:
        # we have a root if a point is shared but doesn't point to another rank
        nroots = len(
            [pt for pt in part.overlap if isinstance(pt, Shared) and not pt.root]
        )

        # which local points are leaves?
        local_points = [
            i for i, pt in enumerate(part.overlap) if not is_owned_by_process(pt)
        ]

        # roots of other processes (rank, index)
        remote_points = utils.flatten(
            [pt.root.as_tuple() for pt in part.overlap if not is_owned_by_process(pt)]
        )

        # import pdb; pdb.set_trace()

        sf = PETSc.SF().create(comm)
        sf.setGraph(nroots, local_points, remote_points)
        return sf
    else:
        raise NotImplementedError(
            "Need to think about concatenating star forests. This will happen if mixed."
        )


def attach_owned_star_forest(axis):
    raise NotImplementedError


@dataclasses.dataclass
class RemotePoint:
    rank: numbers.Integral
    index: numbers.Integral

    def as_tuple(self):
        return (self.rank, self.index)


@dataclasses.dataclass
class PointOverlapLabel(abc.ABC):
    pass


@dataclasses.dataclass
class Owned(PointOverlapLabel):
    pass


@dataclasses.dataclass
class Shared(PointOverlapLabel):
    root: Optional[RemotePoint] = None


@dataclasses.dataclass
class Halo(PointOverlapLabel):
    root: RemotePoint


def is_owned_by_process(olabel):
    return isinstance(olabel, Owned) or isinstance(olabel, Shared) and not olabel.root


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


class PointLabel(abc.ABC):
    """Container associating points in an :class:`AxisPart` with a enumerated label."""


# TODO: Maybe could make this a little more descriptive a la star forest so we could
# then automatically generate an SF for the multi-axis.
class PointOwnershipLabel(PointLabel):
    """Label indicating parallel point ownership semantics (i.e. owned or halo)."""

    # TODO: Write a factory function/constructor that takes advantage of the fact that
    # the majority of the points are OWNED and there are only two options so a set is
    # an efficient choice of data structure.
    def __init__(self, owned_points, halo_points):
        owned_set = set(owned_points)
        halo_set = set(halo_points)

        if len(owned_set) != len(owned_points) or len(halo_set) != len(halo_points):
            raise ValueError("Labels cannot contain duplicate values")
        if owned_set.intersection(halo_set):
            raise ValueError("Points cannot appear with different values")

        self._owned_points = owned_points
        self._halo_points = halo_points

    def __len__(self):
        return len(self._owned_points) + len(self._halo_points)


# this isn't really a thing I should be caring about - it's just a multi-axis!
class Sparsity:
    def __init__(self, maps):
        if isinstance(maps, collections.abc.Sequence):
            rmap, cmap = maps
        else:
            rmap, cmap = maps, maps

        ...

        raise NotImplementedError


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
        return self._tree.index()

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

    @cached_property
    def _tree(self):
        return AxisTree(self)

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

        # makea  cached property, then delete this method
        self._layout_exprs = AxisTree._default_index_exprs(self)

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

    @deprecated(internal=True)
    def drop_last(self):
        """Remove the last subaxis"""
        if not self.part.subaxis:
            return None
        else:
            return self.copy(
                parts=[self.part.copy(subaxis=self.part.subaxis.drop_last())]
            )

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


@frozen_record
class AxisTree(PartialAxisTree, Indexed, ContextFreeLoopIterable):
    fields = PartialAxisTree.fields | {
        "target_paths",
        "index_exprs",
        "layout_exprs",
        "layouts",
        "sf",
    }

    def __init__(
        self,
        parent_to_children=pmap(),
        target_paths=None,
        index_exprs=None,
        layout_exprs=None,
        sf=None,
    ):
        if some_but_not_all(
            arg is None for arg in [target_paths, index_exprs, layout_exprs]
        ):
            raise ValueError

        super().__init__(parent_to_children)
        self._target_paths = target_paths or self._default_target_paths()
        self._index_exprs = index_exprs or self._default_index_exprs()
        self.layout_exprs = layout_exprs or self._default_layout_exprs()
        self.sf = sf or self._default_sf()

    def __getitem__(self, indices):
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

    @classmethod
    def from_nest(cls, nest) -> AxisTree:
        root, node_map = cls._from_nest(nest)
        node_map.update({None: [root]})
        return cls.from_node_map(node_map)

    @classmethod
    def from_node_map(cls, node_map):
        tree = PartialAxisTree(node_map)
        return cls.from_partial_tree(tree)

    @classmethod
    def from_partial_tree(cls, tree: PartialAxisTree) -> AxisTree:
        target_paths = cls._default_target_paths(tree)
        index_exprs = cls._default_index_exprs(tree)
        layout_exprs = index_exprs
        return cls(
            tree.parent_to_children,
            target_paths,
            index_exprs,
            layout_exprs,
        )

    def index(self):
        from pyop3.itree import LoopIndex

        return LoopIndex(self.owned)

    @property
    def target_paths(self):
        return self._target_paths

    @property
    def index_exprs(self):
        return self._index_exprs

    @cached_property
    def layouts(self):
        """Initialise the multi-axis by computing the layout functions."""
        from pyop3.axtree.layout import _collect_at_leaves, _compute_layouts
        from pyop3.distarray.multiarray import IndexExpressionReplacer

        if self.is_empty:
            return pmap({pmap(): 0})

        layouts, _, _, _ = _compute_layouts(self, self.root)
        layoutsnew = _collect_at_leaves(self, layouts)
        layouts = freeze(dict(layoutsnew))

        layouts_ = {}
        for leaf in self.leaves:
            orig_path = self.path(*leaf)
            new_path = {}
            replace_map = {}
            for axis, cpt in self.path_with_nodes(*leaf).items():
                new_path.update(self.target_paths[axis.id, cpt])
                replace_map.update(self.layout_exprs[axis.id, cpt])
            new_path = freeze(new_path)

            orig_layout = layouts[orig_path]
            new_layout = IndexExpressionReplacer(replace_map)(orig_layout)
            # assert new_layout != orig_layout
            layouts_[new_path] = new_layout
        return freeze(layouts_)

    @cached_property
    def sf(self):
        return cls._default_sf(tree)

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

    def freeze(self):
        return self

    # needed here? or just for the Dat? perhaps a free function?
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

    if isinstance(arg, Dat):
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
    from pyop3.distarray import Dat  # cyclic import

    if isinstance(arg, Dat):
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
def _as_axis_component_label(arg: Any) -> ComponentLabel:
    if isinstance(arg, ComponentLabel):
        return arg
    else:
        raise TypeError(f"No handler registered for {type(arg).__name__}")


@_as_axis_component_label.register
def _(component: AxisComponent):
    return component.label


def _path_and_indices_from_index_tuple(axes, index_tuple):
    from pyop3.axtree.layout import _as_int

    path = pmap()
    indices = pmap()
    axis = axes.root
    for index in index_tuple:
        if axis is None:
            raise IndexError("Too many indices provided")
        if isinstance(index, numbers.Integral):
            if axis.degree > 1:
                raise IndexError(
                    "Cannot index multi-component array with integers, a "
                    "2-tuple of (component index, index value) is needed"
                )
            cpt_label = axis.components[0].label
        else:
            cpt_label, index = index

        cpt_index = axis.component_labels.index(cpt_label)

        if index < 0:
            # In theory we could still get this to work...
            raise IndexError("Cannot use negative indices")
        # TODO need to pass indices here for ragged things
        if index >= _as_int(axis.components[cpt_index].count, path, indices):
            raise IndexError("Index is too large")

        indices |= {axis.label: index}
        path |= {axis.label: cpt_label}
        axis = axes.component_child(axis, cpt_label)

    if axis is not None:
        raise IndexError("Insufficient number of indices given")

    return path, indices


def _trim_path(axes: AxisTree, path) -> pmap:
    """Drop unused axes from the axis path."""
    new_path = {}
    axis = axes.root
    while axis:
        cpt_label = path[axis.label]
        new_path[axis.label] = cpt_label
        axis = axes.component_child(axis, cpt_label)
    return pmap(new_path)

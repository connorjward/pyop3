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
from pyop3.tree import (
    ComponentLabel,
    LabelledNode,
    LabelledTree,
    NodeId,
    StrictLabelledNode,
    StrictLabelledTree,
    postvisit,
    previsit,
)
from pyop3.utils import (
    LabelledImmutableRecord,
    PrettyTuple,
    UniquelyIdentifiedImmutableRecord,
    as_tuple,
    checked_zip,
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

    @functools.cached_property
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
    def index(self) -> LoopIndex:
        pass

    @abc.abstractmethod
    def __getitem__(self, indices) -> Union[LoopIterable, ContextSensitiveLoopIterable]:
        raise NotImplementedError


class ContextFreeLoopIterable(LoopIterable, ContextFree):
    pass


class ContextSensitiveLoopIterable(LoopIterable, ContextSensitive):
    pass


class ExpressionEvaluator(pym.mapper.evaluator.EvaluationMapper):
    def map_axis_variable(self, expr):
        return self.context[1][expr.axis_label]

    def map_multi_array(self, array):
        path = _trim_path(array.axes, self.context[0])
        # context = []
        # for keyval in self.context.items():
        #     context.append(keyval)
        return array.get_value(path, self.context[1])


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


def has_independently_indexed_subaxis_parts(axes, axis, cpt):
    """
    subaxis parts are independently indexed if they don't depend on the index from
    ``part``.

    if one sub-part needs this index to determine its extent then we need to create
    a layout function as the step sizes will differ.

    Note that we need to consider both ragged sizes and permutations here
    """
    if subaxis := axes.child(axis, cpt):
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
    if subaxis := axes.child(axis, component):
        return _axis_size(axes, subaxis, path, indices)
    else:
        return 1


def attach_star_forest(axis, with_halo_points=True):
    raise NotImplementedError("TODO")
    comm = MPI.COMM_WORLD

    # 1. construct the point-to-point SF per axis part
    if len(axis.children(axis.root)) != 1:
        raise NotImplementedError
    for part in axis.children(axis.root):
        part_sf = make_star_forest_per_axis_part(part, comm)

    # for now, will want to concat or something
    point_sf = part_sf

    # 2. broadcast the root offset to all leaves
    # TODO use a single buffer
    part = just_one(axis.children(axis.root))
    from_buffer = np.zeros(part.count, dtype=PointerType)
    to_buffer = np.zeros(part.count, dtype=PointerType)

    for pt, label in enumerate(part.overlap):
        # only need to broadcast offsets for roots
        if isinstance(label, Shared) and not label.root:
            from_buffer[pt] = axis.get_offset((pt,))

    # TODO: It's quite bad to allocate a massive buffer when not much of it gets
    # moved. Perhaps good to use some sort of map and create a minimal SF.

    cdim = axis.calc_size(part, 0) if axis.children(part) else 1
    dtype, _ = get_mpi_dtype(np.dtype(PointerType), cdim)
    bcast_args = dtype, from_buffer, to_buffer, MPI.REPLACE
    point_sf.bcastBegin(*bcast_args)
    point_sf.bcastEnd(*bcast_args)

    # 3. construct a new SF with these offsets
    nroots, _local, _remote = part_sf.getGraph()

    local_offsets = []
    remote_offsets = []
    i = 0
    for pt, label in enumerate(part.overlap):
        # TODO not a nice check (is_leaf?)
        cond1 = not is_owned_by_process(label)
        if cond1:
            if with_halo_points or (not with_halo_points and isinstance(label, Shared)):
                local_offsets.append(axis.get_offset((pt,)))
                remote_offsets.append((_remote[i, 0], to_buffer[pt]))
            i += 1

    local_offsets = np.array(local_offsets, dtype=IntType)
    remote_offsets = np.array(remote_offsets, dtype=IntType)

    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, local_offsets, remote_offsets)

    if with_halo_points:
        axis.sf = sf
    else:
        axis.shared_sf = sf

    return axis


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
    from pyop3.distarray import MultiArray

    datamap = {}
    for cidx, component in enumerate(axis.components):
        if isinstance(count := component.count, MultiArray):
            datamap.update(count.datamap)

    datamap.update(merge_dicts(subdatamaps))
    return datamap


class AxisComponent(LabelledImmutableRecord):
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

    fields = {
        "count",
        "overlap",
        "indexed",
        "indices",
        "lgmap",
    } | LabelledImmutableRecord.fields

    def __init__(
        self,
        count,
        label: Optional[Hashable] = None,
        *,
        indices=None,
        overlap=None,
        indexed=False,
        lgmap=None,
        **kwargs,
    ):
        super().__init__(label=label, **kwargs)
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

    def __str__(self) -> str:
        return f"{{count={self.count}}}"

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

    # deprecated alias, permutation is a better name as it is easier to reason
    # about sending points vs where they map to.
    @property
    def numbering(self):
        return self.permutation

    # TODO this is just a traversal - clean up
    def alloc_size(self, axtree, axis):
        from pyop3.distarray import MultiArray

        if axis.indexed:
            npoints = 1
        elif isinstance(self.count, MultiArray):
            npoints = self.count.max_value
        else:
            assert isinstance(self.count, numbers.Integral)
            npoints = self.count

        assert npoints is not None

        if subaxis := axtree.child(axis, self):
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


class Axis(StrictLabelledNode, LoopIterable):
    fields = StrictLabelledNode.fields - {"component_labels"} | {
        "components",
        "permutation",
    }

    def __init__(
        self,
        components: Union[Sequence[AxisComponent], AxisComponent, int],
        label: Optional[Hashable] = None,
        *,
        permutation: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        components = tuple(_as_axis_component(cpt) for cpt in as_tuple(components))

        if permutation is not None and not all(
            isinstance(cpt.count, numbers.Integral) for cpt in components
        ):
            raise NotImplementedError(
                "Axis permutations are only supported for axes with fixed component sizes"
            )
        # TODO could also check sizes here

        super().__init__([c.label for c in components], label=label, **kwargs)
        self.components = components

        # FIXME: permutation should be something hashable but not quite like this...
        self.permutation = tuple(permutation) if permutation is not None else None

        # dead code, remove
        self.indexed = False

    def __getitem__(self, indices):
        return as_axis_tree(self)[indices]

    def __call__(self, *args):
        return as_axis_tree(self)(*args)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}([{', '.join(str(cpt) for cpt in self.components)}], label={self.label})"

    @property
    def size(self):
        return as_axis_tree(self).size

    @property
    def count(self):
        """Return the total number of entries in the axis across all axis parts.
        Will fail if axis parts do not have integer counts.
        """
        if not all(cpt.has_integer_count for cpt in self.components):
            raise RuntimeError("non-int counts present, cannot sum")
        return sum(cpt.find_integer_count() for cpt in self.components)

    def index(self):
        return as_axis_tree(self).index()

    def enumerate(self):
        return as_axis_tree(self).enumerate()

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


class MultiArrayCollector(pym.mapper.Collector):
    def map_map_variable(self, expr):
        return {expr.map_component.array}

    def map_multi_array(self, expr):
        return {expr}

    def map_nan(self, expr):
        return set()


# hacky class for index_exprs to work, needs cleaning up
class AxisVariable(pym.primitives.Variable):
    mapper_method = sys.intern("map_axis_variable")

    mycounter = 0

    def __init__(self, axis_label):
        super().__init__(f"var{self.mycounter}")
        self.__class__.mycounter += 1  # ugly
        self.axis_label = axis_label


class AxisTree(StrictLabelledTree, ContextFreeLoopIterable):
    # FIXME this causes a recursive hash error...
    fields = StrictLabelledTree.fields | {"sf", "shared_sf", "comm"}

    def __init__(
        self,
        root: Optional[MultiAxis] = None,
        parent_to_children: Optional[Dict] = None,
        *,
        sf=None,
        shared_sf=None,
        comm=None,
    ):
        super().__init__(root, parent_to_children)

        self.sf = sf
        self.shared_sf = shared_sf
        self.comm = comm  # FIXME DTRT with internal comms

        self._target_paths = self._default_target_path_per_component()
        self._index_exprs = self._default_index_exprs_per_component()
        self._layout_exprs = self._default_layout_exprs()

    def __getitem__(self, indices) -> Union[IndexedAxisTree, ContextSensitiveAxisTree]:
        from pyop3.indices.tree import index_axes

        return index_axes(self, indices)

    @property
    def axis_trees(self):
        return pmap({pmap(): self})

    def index(self):
        from pyop3.indices import LoopIndex

        return LoopIndex(self)

    @property
    def target_paths(self):
        return self._target_paths

    # deprecated alias
    @property
    def target_path_per_component(self):
        return self.target_paths

    @property
    def index_exprs(self):
        return self._index_exprs

    # deprecated alias
    @property
    def index_exprs_per_component(self):
        return self.index_exprs

    @property
    def axes(self):
        return self

    def _default_target_path_per_component(self):
        if self.is_empty:
            return pmap()

        return pmap(
            {
                (axis.id, cpt.label): pmap({axis.label: cpt.label})
                for axis in self.nodes
                for cpt in axis.components
            }
        )

    def _default_index_exprs_per_component(self):
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
        return self._default_index_exprs_per_component()

    @property
    def layout_exprs(self):
        if not self._layout_exprs:
            self._layout_exprs = self._make_index_exprs()
        return self._layout_exprs

    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
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
        for cleverdict in [self.index_exprs_per_component]:
            for exprs in cleverdict.values():
                for expr in exprs.values():
                    for array in MultiArrayCollector()(expr):
                        dmap.update(array.datamap)
        return pmap(dmap)

    def _make_target_paths(self):
        return tuple(self.path(ax, cpt) for ax, cpt in self.leaves)

    @property
    def layouts(self):
        # FIXME function not property
        return self._default_layouts()

    def _default_layouts(self):
        # ick
        return self.set_up()

    def find_part(self, label):
        return self._parts_by_label[label]

    def offset(self, *args, allow_unused=False):
        nargs = len(args)
        if nargs == 2:
            path, indices = args[0], args[1]
        else:
            assert nargs == 1
            path, indices = _path_and_indices_from_index_tuple(self, args[0])

        if allow_unused:
            path = _trim_path(self, path)

        offset = pym.evaluate(self.layouts[path], (path, indices), ExpressionEvaluator)
        return strict_int(offset)

    # old alias
    get_offset = offset

    def find_component(self, node_label, cpt_label, also_node=False):
        """Return the first component in the tree matching the given labels.

        Notes
        -----
        This will return the first component matching the labels. Multiple may exist
        but we assume that they are identical.

        """
        for node in self.nodes:
            if node.label == node_label:
                for cpt in node.components:
                    if cpt.label == cpt_label:
                        if also_node:
                            return node, cpt
                        else:
                            return cpt
        raise ValueError("Matching component not found")

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

    def path(self, axis: Axis, component: AxisComponent, **kwargs):
        cpt_label = _as_axis_component_label(component)
        return super().path(axis, cpt_label, **kwargs)

    def path_with_nodes(self, axis: Axis, component: AxisComponent, **kwargs):
        cpt_label = _as_axis_component_label(component)
        return super().path_with_nodes(axis, cpt_label, **kwargs)

    # bad name
    def detailed_path(self, path):
        node = self._node_from_path(path)
        if node is None:
            return pmap()
        else:
            return self.path_with_nodes(*node, and_components=True)

    def is_valid_path(self, path):
        try:
            self._node_from_path(path)
            return True
        except:
            return False

    @property
    def leaf(self):
        leaf_axis, leaf_cpt_label = super().leaf
        leaf_cpt = leaf_axis.components[
            leaf_axis.component_labels.index(leaf_cpt_label)
        ]
        return leaf_axis, leaf_cpt

    @property
    def leaf_axis(self):
        return self.leaf[0]

    @property
    def leaf_component(self):
        return self.leaf[1]

    def child(
        self, parent: Axis, component: Union[AxisComponent, ComponentLabel]
    ) -> Optional[Axis]:
        cpt_label = _as_axis_component_label(component)
        return super().child(parent, cpt_label)

    @functools.cached_property
    def size(self):
        return axis_tree_size(self)

    def alloc_size(self, axis=None):
        axis = axis or self.root
        return sum(cpt.alloc_size(self, axis) for cpt in axis.components)

    def _check_labels(self):
        def check(node, prev_labels):
            if node == self.root:
                return prev_labels
            if node.label in prev_labels:
                raise ValueError("shouldn't have the same label as above")
            return prev_labels | {node.label}

        previsit(self, check, self.root, frozenset())

    def set_up(self, with_sf=True):
        """Initialise the multi-axis by computing the layout functions."""
        # TODO: put somewhere better
        # self._check_labels()

        # catch empyt axis tree
        if self.root is None:
            return pmap({pmap(): 0})

        layouts, _, _ = _compute_layouts(self, self.root)
        layoutsnew = _collect_at_leaves(self, layouts)
        # self.apply_layouts(layouts)
        return pyrsistent.freeze(dict(layoutsnew))
        # assert is_set_up(self)

        # FIXME reinsert this code
        # set the .sf and .owned_sf properties of new_axis
        # if with_sf and is_distributed(self):
        #     attach_star_forest(self)
        #     attach_star_forest(self, with_halo_points=False)
        #
        #     # attach a local to global map
        #     if len(self.children(self.root)) > 1:
        #         raise NotImplementedError(
        #             "Currently only compute lgmaps for a single part, shouldn't "
        #             "be hard to fix"
        #         )
        #     lgmap = create_lgmap(self)
        #     new_part = just_one(self.children(self.root))
        #     self.replace_node(new_part.copy(lgmap=lgmap))
        # self.copy(parts=[new_axis.part.copy(lgmap=lgmap)])
        # new_axis = attach_owned_star_forest(new_axis)

    @classmethod
    def from_layout(cls, layout: Sequence[ConstrainedMultiAxis]) -> Any:  # TODO
        return order_axes(layout)

    # TODO this is just a regular tree search
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
            return self.get_part_from_path(sublabels, self.child(axis, component))
        else:
            return axis, component

    def drop_last(self):
        """Remove the last subaxis"""
        if not self.part.subaxis:
            return None
        else:
            return self.copy(
                parts=[self.part.copy(subaxis=self.part.subaxis.drop_last())]
            )

    @property
    def is_linear(self):
        """Return ``True`` if the multi-axis contains no branches at any level."""
        if self.nparts == 1:
            return self.part.subaxis.is_linear if self.part.subaxis else True
        else:
            return False

    def add_subaxis(self, subaxis, *loc):
        import warnings

        warnings.warn("deprecated")
        return self.add_node(subaxis, *loc)


# TODO: Inherit things from AxisTree, StaticAxisTree?
class IndexedAxisTree(ContextFreeLoopIterable, pytools.ImmutableRecord):
    fields = {"axes", "target_paths", "index_exprs", "layout_exprs"}

    def __init__(self, axes, target_paths, index_exprs, layout_exprs):
        self.axes = axes
        self.target_paths = target_paths
        self.index_exprs = index_exprs
        self.layout_exprs = layout_exprs

    def __getitem__(self, indices):
        from pyop3.indices.tree import index_axes

        return index_axes(self, indices)

    def index(self) -> LoopIndex:
        from pyop3.indices import LoopIndex

        return LoopIndex(self)

    # TODO Is this a property? _default_layouts?
    @property
    def layouts(self):
        return self.axes.layouts

    @property
    def datamap(self):
        datamap_ = dict(self.axes.datamap)
        for index_exprs in self.index_exprs.values():
            for index_expr in index_exprs.values():
                for array in MultiArrayCollector()(index_expr):
                    datamap_.update(array.datamap)
        for layout_exprs in self.layout_exprs.values():
            for layout_expr in layout_exprs.values():
                for array in MultiArrayCollector()(layout_expr):
                    datamap_.update(array.datamap)
        return freeze(datamap_)

    @property
    def size(self):
        return self.axes.size

    @property
    def alloc_size(self):
        return self.axes.alloc_size

    @property
    def root(self):
        return self.axes.root

    @property
    def leaves(self):
        return self.axes.leaves

    def child(self, axis, component):
        return self.axes.child(axis, component)

    def path(self, axis, component):
        return self.axes.path(axis, component)

    def path_with_nodes(self, axis, component, **kwargs):
        return self.axes.path_with_nodes(axis, component, **kwargs)

    def detailed_path(self, path):
        return self.axes.detailed_path(path)

    def is_valid_path(self, path):
        return self.axes.is_valid_path(path)

    @property
    def is_empty(self):
        return self.axes.is_empty

    ### deprecated aliases
    @property
    def target_path_per_component(self):
        return self.target_paths

    @property
    def index_exprs_per_component(self):
        return self.index_exprs


class ContextSensitiveAxisTree(ContextSensitiveLoopIterable):
    def __getitem__(self, indices) -> ContextSensitiveAxisTree:
        from pyop3.indices import index_axes

        # TODO think harder about composing context maps
        new_context_map = {}
        for context, axes in self.context_map.items():
            for context_, axes_ in index_axes(axes, indices).items():
                new_context_map[context | context_] = axes_
        return ContextSensitiveAxisTree(new_context_map)

    def index(self) -> LoopIndex:
        from pyop3.indices import LoopIndex

        return LoopIndex(self)

    @functools.cached_property
    def datamap(self):
        return merge_dicts(axes.datamap for axes in self.context_map.values())


def get_slice_bounds(array, indices):
    raise NotImplementedError("used for sparse things I believe")
    from pyop3.distarray import MultiArray

    part = just_one(array.axes.children(array.axes.root))
    for _ in indices:
        part = just_one(array.axes.children(part))

    layout = array._layouts[part]
    if isinstance(layout, AffineLayoutFunction):
        if isinstance(layout.start, MultiArray):
            start = layout.start.get_value(indices)
        else:
            start = layout.start
        size = part.calc_size(array.axes, indices)
    else:
        # I don't think that this ever happens. We only use IndirectLayoutFunctions when
        # we have numbering and that is not permitted with sparsity
        raise NotImplementedError

    return strict_int(start), strict_int(start + size)


def requires_external_index(axtree, axis, component_index):
    """Return ``True`` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index(
        axtree, axis, component_index
    )  # or numbering_requires_external_index(axtree, axis, component_index)


def size_requires_external_index(axes, axis, component, depth=0):
    from pyop3.distarray import MultiArray

    count = component.count
    if not component.has_integer_count and count.axes.depth > depth:
        return True
    else:
        if subaxis := axes.child(axis, component):
            for c in subaxis.components:
                if size_requires_external_index(axes, subaxis, c, depth + 1):
                    return True
    return False


def has_constant_step(axes: AxisTree, axis, cpt, depth=0):
    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if subaxis := axes.child(axis, cpt):
        return all(
            not size_requires_external_index(axes, subaxis, c, depth)
            for c in subaxis.components
        )
    else:
        return True


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


# i.e. maps and layouts (that take in indices and write to things)
# In theory we don't need to track the component here as layouts only expect
# a particular component per axis
class IndexFunction(pytools.ImmutableRecord, abc.ABC):
    fields = set()


# from indices to offsets
class LayoutFunction(IndexFunction, abc.ABC):
    pass


# class AffineLayout(LayoutFunction):
#     fields = {"axis", "cpt", "step", "start"}
#
#     def __init__(self, axis, cpt, step, start=0):
#         assert False, "old code"
#         super().__init__()
#         self.axis = axis
#         self.cpt = cpt
#         self.step = step
#         self.start = start


# FIXME I don't think that layout functions generically need to record which axes
# they work over (we do for affine) since the map already knows.
# class TabulatedLayout(LayoutFunction):
#     fields = {"data"}
#
#     def __init__(self, data):
#         super().__init__()
#         self.data = data


@dataclasses.dataclass
class SyncStatus:
    pending_write_op: Optional[Any] = None
    halo_valid: bool = True
    halo_modified: bool = False


def create_lgmap(axes):
    if len(axes.children(axes.root)) > 1:
        raise NotImplementedError
    axes_part = just_one(axes.children(axes.root))
    if axes_part.overlap is None:
        raise ValueError("axes is expected to have a specified overlap")
    if not isinstance(axes_part.count, numbers.Integral):
        raise NotImplementedError("Expecting an integral axis size")

    # 1. Globally number all owned processes
    sendbuf = np.array([axes_part.nowned], dtype=PETSc.IntType)
    recvbuf = np.zeros_like(sendbuf)
    axes.sf.comm.tompi4py().Exscan(sendbuf, recvbuf)
    global_num = single_valued(recvbuf)
    indices = np.full(axes_part.count, -1, dtype=PETSc.IntType)
    for i, olabel in enumerate(axes_part.overlap):
        if is_owned_by_process(olabel):
            indices[i] = global_num
            global_num += 1

    # 2. Broadcast the global numbering to SF leaves
    mpi_dtype, _ = get_mpi_dtype(indices.dtype)
    mpi_op = MPI.REPLACE
    args = (mpi_dtype, indices, indices, mpi_op)
    axes.sf.bcastBegin(*args)
    axes.sf.bcastEnd(*args)

    assert not any(indices == -1)

    # return PETSc.LGMap().create(indices, comm=axes.sf.comm)
    return indices


@functools.singledispatch
def as_axis_tree(arg: Any):
    from pyop3.indices import IndexedAxisTree

    if isinstance(arg, IndexedAxisTree):
        return arg
    raise TypeError


@as_axis_tree.register
def _(arg: AxisTree):
    return arg


@as_axis_tree.register
def _(arg: Axis):
    return AxisTree(arg)


@as_axis_tree.register
def _(arg: AxisComponent):
    return AxisTree(Axis([arg]))


@functools.singledispatch
def _as_axis_component(arg: Any) -> AxisComponent:
    from pyop3.distarray import MultiArray

    # Needed to avoid cyclic import
    if isinstance(arg, MultiArray):
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


# use this to build a tree of sizes that we use to construct
# the right count arrays
class CustomNode(StrictLabelledNode):
    fields = LabelledNode.fields - {"component_labels"} | {"counts"}

    def __init__(self, counts, **kwargs):
        super().__init__(counts, **kwargs)
        self.counts = tuple(counts)


def _compute_layouts(
    axes: AxisTree,
    axis=None,
    path=pmap(),
):
    axis = axis or axes.root
    layouts = {}
    steps = {}

    # Post-order traversal
    # make sure to catch children that are None
    csubtrees = []
    sublayoutss = []
    for cpt in axis.components:
        if subaxis := axes.child(axis, cpt):
            sublayouts, csubtree, substeps = _compute_layouts(
                axes, subaxis, path | {axis.label: cpt.label}
            )
            sublayoutss.append(sublayouts)
            csubtrees.append(csubtree)
            steps.update(substeps)
        else:
            csubtrees.append(None)
            sublayoutss.append(collections.defaultdict(list))

    """
    There are two conditions that we need to worry about:
        1. does the axis have a fixed size (not ragged)?
            If so then we should emit a layout function and handle any inner bits.
            We don't need any external indices to fully index the array. In fact,
            if we were the use the external indices too then the resulting layout
            array would be much larger than it has to be (each index is basically
            a new dimension in the array).

        2. Does the axis have fixed size steps?

        If we have constant steps then we should index things using an affine layout.

    Care needs to be taken with the interplay of these options:

        fixed size x fixed step : affine - great
        fixed size x variable step : need to tabulate with the current axis and
                                     everything below that isn't yet handled
        variable size x fixed step : emit an affine layout but we need to tabulate above
        variable size x variable step : add an axis to the "count" tree but do nothing else
                                        not ready for tabulation as not fully indexed

    We only ever care about axes as a whole. If individual components are ragged but
    others not then we still need to index them separately as the steps are still not
    a fixed size even for the non-ragged components.
    """

    # 1. do we need to pass further up?
    if not all(has_fixed_size(axes, axis, cpt) for cpt in axis.components):
        # 0. We ignore things if they are indexed. They don't contribute
        if axis.indexed:
            ctree = None
        elif all(has_constant_step(axes, axis, c) for c in axis.components):
            # we must be at the bottom of a ragged patch - therefore don't
            # add to shape of things
            # in theory if we are ragged and permuted then we do want to include this level
            ctree = None
            for c in axis.components:
                step = step_size(axes, axis, c)
                layouts.update(
                    {
                        path
                        # | {axis.label: c.label}: AffineLayout(axis.label, c.label, step)
                        | {axis.label: c.label}: AxisVariable(axis.label) * step
                    }
                )

        else:
            croot = CustomNode(
                [(cpt.count, axis.label, cpt.label) for cpt in axis.components]
            )
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = pmap(
                    {croot.id: [sub.root for sub in csubtrees]}
                ) | merge_dicts(sub.parent_to_children for sub in csubtrees)
            else:
                cparent_to_children = {}
            ctree = StrictLabelledTree(croot, cparent_to_children)

        # layouts and steps are just propagated from below
        layouts.update(merge_dicts(sublayoutss))
        return layouts, ctree, steps

    # 2. add layouts here
    else:
        # 1. do we need to tabulate anything?
        if axis.permutation is not None or not all(
            has_constant_step(axes, axis, c) for c in axis.components
        ):
            # If this axis is indexed and there is no inner shape to index then do nothing
            if axis.indexed and strictly_all(sub is None for sub in csubtrees):
                return layouts | merge_dicts(sublayoutss), None, steps

            # super ick
            bits = []
            for cpt in axis.components:
                # axlabel, clabel = just_one(axes.target_paths[((axis, cpt),)].items())
                axlabel, clabel = axis.label, cpt.label
                bits.append((cpt.count, axlabel, clabel))
            croot = CustomNode(bits)
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = pmap(
                    {croot.id: [sub.root for sub in csubtrees]}
                ) | merge_dicts(sub.parent_to_children for sub in csubtrees)
            else:
                cparent_to_children = {}
            ctree = StrictLabelledTree(croot, cparent_to_children)

            fulltree = _create_count_array_tree(ctree)

            # now populate fulltree
            offset = IntRef(0)
            _tabulate_count_array_tree(axes, axis, fulltree, offset)

            for subpath, offset_data in fulltree.items():
                layouts[path | subpath] = offset_data
            ctree = None
            steps = {path: _axis_size(axes, axis)}

            layouts.update(merge_dicts(sublayoutss))
            return layouts, ctree, steps

        # must therefore be affine
        else:
            # 0. We ignore things if they are indexed. They don't contribute
            if axis.indexed:
                return layouts | merge_dicts(sublayoutss), None, steps
            ctree = None
            layouts = {}
            steps = [step_size(axes, axis, c) for c in axis.components]
            start = 0
            for cidx, step in enumerate(steps):
                mycomponent = axis.components[cidx]
                sublayouts = sublayoutss[cidx].copy()

                # new_layout = AffineLayout(axis.label, mycomponent.label, step, start)
                new_layout = AxisVariable(axis.label) * step + start
                sublayouts[path | {axis.label: mycomponent.label}] = new_layout
                start += _axis_component_size(axes, axis, mycomponent)

                layouts.update(sublayouts)
            steps = {path: _axis_size(axes, axis)}
            return layouts, None, steps


# I don't think that this actually needs to be a tree, just return a dict
# TODO I need to clean this up a lot now I'm using component labels
def _create_count_array_tree(
    ctree, current_node=None, counts=PrettyTuple(), path=pmap()
):
    from pyop3.distarray import MultiArray

    current_node = current_node or ctree.root
    arrays = {}

    for cidx in range(current_node.degree):
        count, axis_label, cpt_label = current_node.counts[cidx]

        child = ctree.children(current_node)[cidx]
        new_path = path | {axis_label: cpt_label}
        if child is None:
            # make a multiarray here from the given sizes
            axes = [
                Axis([(ct, clabel)], axlabel)
                for (ct, axlabel, clabel) in counts | current_node.counts[cidx]
            ]
            root = axes[0]
            parent_to_children = {}
            for parent, child in zip(axes, axes[1:]):
                parent_to_children[parent.id] = child
            axtree = AxisTree(root, parent_to_children)
            countarray = MultiArray(
                axtree,
                data=np.full(axis_tree_size(axtree), -1, dtype=IntType),
            )
            arrays[new_path] = countarray
        else:
            arrays.update(
                _create_count_array_tree(
                    ctree,
                    child,
                    counts | current_node.counts[cidx],
                    new_path,
                )
            )

    return arrays


def _tabulate_count_array_tree(
    axes,
    axis,
    count_arrays,
    offset,
    path=pmap(),
    # other_path=pmap(),
    indices=pyrsistent.pmap(),
):
    npoints = sum(_as_int(c.count, path, indices) for c in axis.components)
    permutation = (
        axis.permutation
        if axis.permutation is not None
        else np.arange(npoints, dtype=IntType)
    )

    point_to_component_id = np.empty(npoints, dtype=np.int8)
    point_to_component_num = np.empty(npoints, dtype=PointerType)
    pos = 0
    for cidx, component in enumerate(axis.components):
        # can determine this once above
        csize = _as_int(component.count, path, indices)
        for i in range(csize):
            point = permutation[pos + i]
            point_to_component_id[point] = cidx
            point_to_component_num[point] = i
        pos += csize

    for pt in range(npoints):
        selected_component_id = point_to_component_id[pt]
        selected_component_num = point_to_component_num[pt]
        selected_component = axis.components[selected_component_id]

        new_path = (
            path
            # | axes.target_path_per_leaf[pmap({axis.label: selected_component.label})]
            | {axis.label: selected_component.label}
        )
        # new_other_path = other_path | {axis.label: selected_component.label}
        new_indices = indices | {
            # just_one(
            #     axes.target_path_per_leaf[
            #         pmap({axis.label: selected_component.label})
            #     ].keys()
            # ): selected_component_num
            axis.label: selected_component_num
        }
        # new_indices = indices | {axis.label: selected_component_num}
        if new_path in count_arrays:
            count_arrays[new_path].set_value(new_path, new_indices, offset.value)
            if not axis.indexed:
                offset += step_size(
                    axes,
                    axis,
                    selected_component,
                    new_path,
                    # new_other_path,
                    new_indices,
                )
        else:
            if axis.indexed:
                saved_offset = offset.value

            subaxis = axes.child(axis, selected_component)
            assert subaxis
            _tabulate_count_array_tree(
                axes,
                subaxis,
                count_arrays,
                offset,
                new_path,
                new_indices,
            )

            if axis.indexed:
                offset.value = saved_offset


# TODO this whole function sucks, should accumulate earlier
def _collect_at_leaves(
    axes,
    values,
    axis: Optional[Axis] = None,
    path=pmap(),
    prior=0,
):
    axis = axis or axes.root
    acc = {}

    for cpt in axis.components:
        new_path = path | {axis.label: cpt.label}
        if new_path in values:
            # prior_ = prior | {axis.label: values[new_path]}
            prior_ = prior + values[new_path]
        else:
            prior_ = prior
        if subaxis := axes.child(axis, cpt):
            acc.update(_collect_at_leaves(axes, values, subaxis, new_path, prior_))
        else:
            acc[new_path] = prior_

    return acc


def axis_tree_size(axes: AxisTree) -> int:
    """Return the size of an axis tree.

    The returned size represents the total number of entries in the array. For
    example, an array with shape ``(10, 3)`` will have a size of 30.

    """
    if not axes.root:
        return 1
    return _axis_size(axes, axes.root, pmap(), pmap())


def _axis_size(
    axes: AxisTree,
    axis: Axis,
    path: pmap[tuple[Label, Label]] = pmap(),
    indices: Mapping = pyrsistent.pmap(),
) -> int:
    return sum(
        _axis_component_size(axes, axis, cpt, path, indices) for cpt in axis.components
    )


def _axis_component_size(
    axes: AxisTree,
    axis: Axis,
    component: AxisComponent,
    path=pmap(),
    indices: Mapping = pyrsistent.pmap(),
):
    count = _as_int(component.count, path, indices)
    if subaxis := axes.child(axis, component):
        return sum(
            _axis_size(
                axes,
                subaxis,
                path | {axis.label: component.label},
                indices | {axis.label: i},
            )
            for i in range(count)
        )
    else:
        return count


@functools.singledispatch
def _as_int(arg: Any, path: Mapping, indices: Mapping):
    from pyop3.distarray import MultiArray

    # cyclic import
    if isinstance(arg, MultiArray):
        # TODO this might break if we have something like [:, subset]
        # I will need to map the "source" axis (e.g. slice_label0) back
        # to the "target" axis
        return arg.get_value(path, indices)
    else:
        raise TypeError


@_as_int.register
def _(arg: numbers.Real, path: Mapping, indices: Mapping):
    return strict_int(arg)


def _path_and_indices_from_index_tuple(
    axes, index_tuple
) -> Tuple[pmap[Label, Label], pmap[Label, int]]:
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
        axis = axes.child(axis, cpt_label)

    if axis is not None:
        raise IndexError("Insufficient number of indices given")

    return path, indices


def _trim_path(axes: AxisTree, path: Mapping) -> pyrsistent.pmap:
    """Drop unused axes from the axis path."""
    new_path = {}
    axis = axes.root
    while axis:
        cpt_label = path[axis.label]
        new_path[axis.label] = cpt_label
        axis = axes.child(axis, cpt_label)
    return pyrsistent.pmap(new_path)


def collect_sizes(axes: AxisTree) -> pmap:  # TODO value-type of returned pmap?
    return _collect_sizes_rec(axes, axes.root)


def _collect_sizes_rec(axes, axis) -> pmap:
    sizes = {}
    for cpt in axis.components:
        sizes[axis.label, cpt.label] = cpt.count

        if subaxis := axes.child(axis, cpt):
            subsizes = _collect_sizes_rec(axes, subaxis)
            for loc, size in subsizes.items():
                # make sure that sizes always match for duplicates
                if loc not in sizes:
                    sizes[loc] = size
                else:
                    if sizes[loc] != size:
                        raise RuntimeError
    return pmap(sizes)

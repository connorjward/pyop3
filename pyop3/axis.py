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
import threading
from typing import Any, FrozenSet, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pymbolic as pym
import pyrsistent
import pytools
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from pyop3.dtypes import IntType, PointerType, get_mpi_dtype
from pyop3.index import AffineMap, Index, IndexTree, Map, Slice, TabulatedMap
from pyop3.tree import (
    LabelledNode,
    LabelledTree,
    Node,
    NodeComponent,
    postvisit,
    previsit,
)
from pyop3.utils import (
    PrettyTuple,
    as_tuple,
    checked_zip,
    flatten,
    has_unique_entries,
    just_one,
    merge_dicts,
    single_valued,
    some_but_not_all,
    strict_int,
    strictly_all,
    unique,
)

# def is_distributed(axtree, axis=None):
#     """Return ``True`` if any part of a :class:`MultiAxis` is distributed across ranks."""
#     axis = axis or axtree.root
#     for cpt in axis.components:
#         if (
#             cpt.is_distributed
#             or (subaxis := axtree.child(axis, cpt))
#             and is_distributed(axtree, subaxis)
#         ):
#             return True
#     return False


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


def prepare_layouts(axtree, axis, component, path=None):
    """Make a magic nest of dictionaries for storing intermediate results."""
    # path is a tuple key for holding the different axis parts
    if not path:
        path = ((axis.label, component.label),)

    # import pdb; pdb.set_trace()
    layouts = {path: None}

    if child := axtree.child(axis, component):
        for i, subcpt in enumerate(child.components):
            layouts |= prepare_layouts(
                axtree, child, subcpt, path + ((child.label, subcpt.label),)
            )

    return layouts


def set_null_layouts(layouts, axtree, path=(), axis=None):
    """
    we have null layouts whenever the step is non-const (the variability is captured by
    the start property of the affine layout below it).

    We also get them when the axis is "indexed", that is, not ever actually used by
    temporaries.
    """
    axis = axis or axtree.root
    for cpt in axis.components:
        new_path = path + ((axis.label, cpt.label),)
        if (
            not has_constant_step(axtree, axis, part) or part.indexed
        ):  # I think indexed is gone
            layouts[new_path] = "null layout"

        if child := axtree.child(axis, cpt):
            set_null_layouts(layouts, axtree, new_path, child)


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


def step_size(axtree, axis, component, indices=PrettyTuple()):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if not has_constant_step(axtree, axis, component) and not indices:
        raise ValueError
    if subaxis := axtree.child(axis, component):
        return axtree.calc_size(subaxis, indices)
    else:
        return 1


def attach_star_forest(axis, with_halo_points=True):
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
    from pyop3.distarray import IndexedMultiArray, MultiArray

    datamap = {}
    for cidx, component in enumerate(axis.components):
        if isinstance(count := component.count, (IndexedMultiArray, MultiArray)):
            datamap |= count.datamap

    return datamap | merge_dicts(subdatamaps)


class AxisTree(LabelledTree):
    fields = LabelledTree.fields | {"within_axes"}

    def __init__(
        self,
        root: MultiAxis | None = None,
        parent_to_children: dict | None = None,
        *,
        within_axes=None,
        sf=None,
        shared_sf=None,
        comm=None,
    ):
        super().__init__(root, parent_to_children)

        # this is a map from axis labels to their extents. This is useful for
        # things like temporaries where the axis tree does not itself fully characterise
        # the shapes of things
        self.within_axes = within_axes or {}
        self.sf = sf
        self.shared_sf = shared_sf
        self.comm = comm  # FIXME DTRT with internal comms

        self._layouts = {}

    @property
    def index(self):
        return fill_shape(self, extra_kwargs={"axes": self})

    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
        dmap = postvisit(self, _collect_datamap, axes=self)
        for layout in flatten(list(self.layouts.values())):
            if isinstance(layout, TabulatedLayout):
                dmap |= layout.data.datamap
        return dmap

    @property
    def part(self):
        try:
            (pt,) = self.parts
            return pt
        except ValueError:
            raise RuntimeError

    @functools.cached_property
    def layouts(self):
        if not self._layouts:
            self.set_up()
        return self._layouts

    def find_part(self, label):
        return self._parts_by_label[label]

    # TODO indices can be either a list of integers, a list of 2-tuples (cidx, idx), or
    # a mapping from axis label to (cidx, idx) (or just integers I guess)
    def get_offset(self, indices):
        if isinstance(indices, Mapping):
            # indices may be unordered
            raise NotImplementedError
        else:
            # assume that indices track axes in order

            from pyop3.distarray import MultiArray

            # parse the indices to get the right path and do some bounds checking
            axis = self.root
            path = []
            indices_map = {}  # map from (axis, component) to integer
            # TODO if indices is not ordered then we should traverse the axis tree instead
            # else we would not be able to index ragged things if the indices were:
            # [nnz, outer]
            for index in indices:
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

                cpt_index = [c.label for c in axis.components].index(cpt_label)

                if index < 0:
                    # In theory we could still get this to work...
                    raise IndexError("Cannot use negative indices")
                # TODO need to pass indices here for ragged things
                if index >= axis.components[cpt_index].find_integer_count():
                    raise IndexError("Index is too large")

                indices_map[axis.label, cpt_label] = index
                path.append(cpt_label)
                axis = self.child(axis, cpt_label)

            if axis is not None:
                raise IndexError("Insufficient number of indices given")

            offset = 0
            layouts = self.layouts[tuple(path)]
            for layout in layouts:
                if isinstance(layout, TabulatedLayout):
                    offset += layout.data.get_value(indices_map)
                else:
                    assert isinstance(layout, AffineLayout)

                    index = indices_map[layout.axis, layout.cpt]
                    start = layout.start

                    # handle sparsity
                    # if component.indices is not None:
                    #     bstart, bend = get_slice_bounds(component.indices, prior_indices)
                    #
                    #     last_index = bisect.bisect_left(
                    #         component.indices.data, last_index, bstart, bend
                    #     )
                    #     last_index -= bstart

                    offset += index * layout.step + start

            return strict_int(offset)

    def mul(self, other, sparsity=None):
        """Compute the outer product with another :class:`MultiAxis`.

        Parameters
        ----------
        other : :class:`MultiAxis`
            The other :class:`MultiAxis` used to compute the product.
        sparsity : ???
            The sparsity of the resulting product (produced by combining maps). If
            ``None`` then the resulting axis will be dense.
        """
        # NOTE: As discussed in the message below, composing star forests is really hard.
        # In particular it is difficult to prescibe the ownership of the points that are
        # owned in one axis and halo in the other. This effectively corresponds to making
        # the off-diagonal portions of the matrices stored on a process either share the
        # row or the column.
        # Simply making a policy decision here (of distributed along rows) is not enough
        # because for the Real space we need to distribute the dense column between
        # processes (so distributed along rows), but the dense row also needs to be
        # distributed (so distribute along the columns).
        # Once this works we can start implementing our own matrices in pyop3 which
        # would be good for things like PCPATCH. We could also play around with COO and
        # CSC layouts by swapping the axes around.
        raise NotImplementedError(
            "Computing the outer product of multi-axes is difficult in parallel "
            "since we need to compose star forests and decide upon the ownership of the "
            "off-diagonal components."
        )

    def find_part_from_indices(self, indices):
        """Traverse axis to find things

        indices is a list of integers"""
        index, *rest = indices

        if not rest:
            return self.parts[index]
        else:
            return self.parts[index].subaxis.find_part_from_indices(rest)

    # TODO I think I would prefer to subclass tuple here s.t. indexing with
    # None works iff len(self.parts) == 1
    def get_part(self, npart):
        if npart is None:
            if len(self.parts) != 1:
                raise RuntimeError
            return self.parts[0]
        else:
            return self.parts[npart]

    @property
    def nparts(self):
        return len(self.parts)

    @property
    def local_view(self):
        assert False, "dont think i touch this"
        if self.nparts > 1:
            raise NotImplementedError
        if not self.part.has_partitioned_halo:
            raise NotImplementedError

        if not self.part.is_distributed:
            new_part = self.part.copy(overlap=None)
        else:
            new_part = self.part.copy(
                count=self.part.num_owned, max_count=None, overlap=None
            )
        return self.copy(parts=[new_part])

    def calc_size(self, axis=None, indices=PrettyTuple()) -> int:
        """Return the size of a multi-axis by summing the sizes of its components."""
        # NOTE: this works because the size array cannot be multi-part, therefore integer
        # indices (as opposed to typed ones) are valid.
        axis = axis or self.root
        if not isinstance(indices, PrettyTuple):
            indices = PrettyTuple(indices)
        return sum(cpt.calc_size(self, axis, indices) for cpt in axis.components)

    @property
    def size(self):
        return self.calc_size()

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

        layouts, _, _ = _compute_layouts(self, self.root)
        layoutsnew = _collect_at_leaves(self, layouts)
        # self.apply_layouts(layouts)
        self._layouts = pyrsistent.freeze(dict(layoutsnew))
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
        return self

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

    def add_subaxis(self, subaxis, loc):
        return self.put_node(subaxis, loc)


class Axis(LabelledNode):
    fields = LabelledNode.fields - {"degree"} | {"components", "permutation", "indexed"}

    def __init__(
        self,
        components: Sequence[AxisComponent] | AxisComponent | int,
        label: Hashable | None = None,
        *,
        permutation: Sequence[int] | None = None,
        indexed: bool = False,
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

        super().__init__(label, degree=len(components), **kwargs)
        self.components = components

        # FIXME: permutation should be something hashable but not quite like this...
        self.permutation = tuple(permutation) if permutation is not None else None
        self.indexed = indexed

    def __str__(self) -> str:
        return f"{self.__class__.__name__}([{', '.join(str(cpt) for cpt in self.components)}], label={self.label})"

    @property
    def index(self):
        return as_axis_tree(self).index

    @property
    def count(self):
        """Return the total number of entries in the axis across all axis parts.
        Will fail if axis parts do not have integer counts.
        """
        if not all(cpt.has_integer_count for cpt in self.components):
            raise RuntimeError("non-int counts present, cannot sum")
        return sum(cpt.find_integer_count() for cpt in self.components)


class MyNode(Node):
    fields = Node.fields | {"data"}

    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data


def get_slice_bounds(array, indices):
    assert False, "not touched"
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
    from pyop3.distarray import IndexedMultiArray

    count = component.count
    if isinstance(count, IndexedMultiArray):
        count = count.data
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


class AxisComponent(NodeComponent):
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

    fields = NodeComponent.fields | {
        "count",
        "overlap",
        "indexed",
        "indices",
        "lgmap",
    }

    def __init__(
        self,
        count,
        label: Hashable | None = None,
        *,
        indices=None,
        overlap=None,
        indexed=False,
        lgmap=None,
        **kwargs,
    ):
        super().__init__(label, **kwargs)
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
        from pyop3.distarray import IndexedMultiArray, MultiArray

        if axis.indexed:
            npoints = 1
        elif isinstance(self.count, MultiArray):
            npoints = self.count.max_value
        elif isinstance(self.count, IndexedMultiArray):
            npoints = self.count.data.max_value
        else:
            assert isinstance(self.count, numbers.Integral)
            npoints = self.count

        assert npoints is not None

        if subaxis := axtree.child(axis, self):
            return npoints * axtree.alloc_size(subaxis)
        else:
            return npoints

    # TODO make a free function or something - this is horrible
    def calc_size(self, axtree, axis, indices=PrettyTuple()):
        extent = self.find_integer_count(indices)
        if subaxis := axtree.child(axis, self):
            return sum(axtree.calc_size(subaxis, indices | i) for i in range(extent))
        else:
            return extent

    def find_integer_count(self, indices=PrettyTuple()):
        from pyop3.distarray import IndexedMultiArray, MultiArray

        if isinstance(self.count, IndexedMultiArray):
            return self.count.data.get_value(indices)
        elif isinstance(self.count, MultiArray):
            return self.count.get_value(indices)
        else:
            assert isinstance(self.count, numbers.Integral)
            return self.count

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


class AffineLayout(LayoutFunction):
    fields = {"axis", "cpt", "step", "start"}

    def __init__(self, axis, cpt, step, start=0):
        super().__init__()
        self.axis = axis
        self.cpt = cpt
        self.step = step
        self.start = start


# FIXME I don't think that layout functions generically need to record which axes
# they work over (we do for affine) since the map already knows.
class TabulatedLayout(LayoutFunction):
    fields = {"data"}

    def __init__(self, data):
        super().__init__()
        self.data = data


@dataclasses.dataclass
class SyncStatus:
    pending_write_op: Optional[Any] = None
    halo_valid: bool = True
    halo_modified: bool = False


# TODO This algorithm is pretty much identical to quite a few others
def fill_shape(axes, indices=None, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    if not indices:
        return fill_missing_shape(axes, {}).copy(**extra_kwargs)

    new_indices = indices
    for leaf_index, leaf_cpt in indices.leaves:
        axis_path = {}
        for idx, cpt in indices.path(leaf_index, leaf_cpt):
            if cpt.from_axis in axis_path:
                axis_path.pop(cpt.from_axis)
            axis_path |= {cpt.to_axis: cpt.to_cpt}

        extra_slices = fill_missing_shape(axes, axis_path)

        if extra_slices:
            new_indices = new_indices.add_subtree(extra_slices, leaf_index, leaf_cpt)

    return new_indices.copy(**extra_kwargs)


def fill_missing_shape(
    axes: AxisTree, indexed: dict[Hashable, int], current_axis: Axis | None = None
) -> IndexTree | None:
    """Return the indices required to fully index the axes.

    Parameters
    ----------
    axes
        The axis tree requiring indexing.
    indexed
        Mapping from axis labels to axis components. These axes have already
        been indexed so encountering them will not produce a new slice.

    Returns
    -------
    indices
        Tree of indices required to fully index ``axes``. `None` if axes is
        already fully indexed.

    """
    current_axis = current_axis or axes.root

    # 1. Axis is already indexed, select the appropriate subaxis and continue
    if current_axis.label in indexed:
        subaxis = axes.child(current_axis, indexed[current_axis.label])
        if subaxis:
            return fill_missing_shape(axes, indexed, subaxis)
        else:
            return None

    # 2. Axis is not indexed, emit slices for each component
    else:
        components = []
        subtrees = []
        for cpt in current_axis.components:
            components.append(Slice(axis=current_axis.label, cpt=cpt.label))
            subaxis = axes.child(current_axis, cpt)
            if subaxis:
                subtree = fill_missing_shape(axes, indexed, subaxis)
            else:
                subtree = None
            subtrees.append(subtree)

        root = Index(components)
        parent_to_children = collections.defaultdict(list)
        for subtree in subtrees:
            if subtree:
                parent_to_children[root.id].append(subtree.root)
                parent_to_children |= subtree.parent_to_children
            else:
                parent_to_children[root.id].append(None)
        return IndexTree(root, parent_to_children)


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
    from pyop3.distarray import IndexedMultiArray, MultiArray

    # Needed to avoid cyclic import
    if isinstance(arg, (IndexedMultiArray, MultiArray)):
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


# use this to build a tree of sizes that we use to construct
# the right count arrays
class CustomNode(Node):
    fields = Node.fields - {"degree"} | {"counts"}

    def __init__(self, counts, **kwargs):
        super().__init__(len(counts), **kwargs)
        self.counts = tuple(counts)


def _compute_layouts(
    axes: AxisTree,
    axis=None,
    path=PrettyTuple(),
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
                axes, subaxis, path | cpt.label
            )
            sublayoutss.append(sublayouts)
            csubtrees.append(csubtree)
            steps |= substeps
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
                layouts |= {path | c.label: AffineLayout(axis.label, c.label, step)}

        else:
            croot = CustomNode([(cpt.count, axis.label) for cpt in axis.components])
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = {
                    croot.id: [sub.root for sub in csubtrees]
                } | merge_dicts(sub.parent_to_children for sub in csubtrees)
            else:
                cparent_to_children = {}
            ctree = LabelledTree(croot, cparent_to_children)

        # layouts and steps are just propagated from below
        return layouts | merge_dicts(sublayoutss), ctree, steps

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
            croot = CustomNode(
                [(cpt.count, axis.label, cpt.label) for cpt in axis.components]
            )
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = {
                    croot.id: [sub.root for sub in csubtrees]
                } | merge_dicts(sub.parent_to_children for sub in csubtrees)
            else:
                cparent_to_children = {}
            ctree = LabelledTree(croot, cparent_to_children)

            fulltree = _create_count_array_tree(ctree)

            # now populate fulltree
            offset = IntRef(0)
            _tabulate_count_array_tree(axes, axis, fulltree, offset)

            for subpath, offset_data in fulltree.items():
                axis_ = axis
                labels = []
                for cpt in subpath:
                    labels.append(axis_.label)
                    axis_ = axes.child(axis_, cpt)

                layouts[path + subpath] = TabulatedLayout(offset_data)
            ctree = None
            steps = {path: axes.calc_size(axis)}

            return layouts | merge_dicts(sublayoutss), ctree, steps

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

                new_layout = AffineLayout(axis.label, mycomponent.label, step, start)
                sublayouts[path | mycomponent.label] = new_layout
                start += mycomponent.calc_size(axes, axis)

                layouts |= sublayouts
            steps = {path: axes.calc_size(axis)}
            return layouts, None, steps


# I don't think that this actually needs to be a tree, just return a dict
# TODO I need to clean this up a lot now I'm using component labels
def _create_count_array_tree(
    ctree, current_node=None, counts=PrettyTuple(), component_path=PrettyTuple()
):
    from pyop3.distarray import MultiArray

    current_node = current_node or ctree.root
    arrays = {}

    for cidx in range(current_node.degree):
        cpt_label = current_node.counts[cidx][2]
        child = ctree.children(current_node)[cidx]
        if child is None:
            # make a multiarray here from the given sizes
            axes = [
                Axis([(count, cpt_label)], axis_label)
                for (count, axis_label, cpt_label) in counts | current_node.counts[cidx]
            ]
            root = axes[0]
            parent_to_children = {}
            for parent, child in zip(axes, axes[1:]):
                parent_to_children[parent.id] = child
            axtree = AxisTree(root, parent_to_children)
            countarray = MultiArray(
                axtree,
                data=np.full(axtree.calc_size(axtree.root), -1, dtype=IntType),
            )
            arrays[component_path | cpt_label] = countarray
        else:
            arrays |= _create_count_array_tree(
                ctree,
                child,
                counts | current_node.counts[cidx],
                component_path | cpt_label,
            )

    return arrays


def _tabulate_count_array_tree(
    axes,
    axis,
    count_arrays,
    offset,
    path=PrettyTuple(),
    indices=PrettyTuple(),
):
    from pyop3.distarray import IndexedMultiArray, MultiArray

    npoints = 0
    for component in axis.components:
        if isinstance(component.count, IndexedMultiArray):
            count = strict_int(component.count.data.get_value(indices))
        elif isinstance(component.count, MultiArray):
            count = strict_int(component.count.get_value(indices))
        else:
            assert isinstance(component.count, numbers.Integral)
            count = component.count
        npoints += count

    permutation = (
        axis.permutation
        if axis.permutation is not None
        else np.arange(npoints, dtype=IntType)
    )

    point_to_component_id = np.empty(npoints, dtype=np.int8)
    point_to_component_num = np.empty(npoints, dtype=PointerType)
    pos = 0
    for cidx, component in enumerate(axis.components):
        if isinstance(component.count, IndexedMultiArray):
            csize = strict_int(component.count.data.get_value(indices))
        elif isinstance(component.count, MultiArray):
            csize = strict_int(component.count.get_value(indices))
        else:
            assert isinstance(component.count, numbers.Integral)
            csize = component.count

        for i in range(csize):
            point = permutation[pos + i]
            point_to_component_id[point] = cidx
            point_to_component_num[point] = i
        pos += csize

    for pt in range(npoints):
        selected_component_id = point_to_component_id[pt]
        selected_component_num = point_to_component_num[pt]
        selected_component = axis.components[selected_component_id]

        if path | selected_component.label in count_arrays:
            count_arrays[path | selected_component.label].set_value(
                indices | selected_component_num, offset.value
            )
            if not axis.indexed:
                offset += step_size(
                    axes,
                    axis,
                    selected_component,
                    indices | selected_component_num,
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
                path | selected_component.label,
                indices | selected_component_num,
            )

            if axis.indexed:
                offset.value = saved_offset


def _collect_at_leaves(
    axes,
    values,
    axis: Axis | None = None,
    component_path=PrettyTuple(),
    prior=PrettyTuple(),
):
    axis = axis or axes.root
    acc = {}

    for cpt in axis.components:
        if component_path | cpt.label in values:
            prior_ = prior | values[component_path | cpt.label]
        else:
            prior_ = prior
        if subaxis := axes.child(axis, cpt):
            acc |= _collect_at_leaves(
                axes, values, subaxis, component_path | cpt.label, prior_
            )
        else:
            acc[component_path | cpt.label] = prior_

    return acc

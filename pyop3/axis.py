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
from typing import Any, FrozenSet, Hashable, Optional, Sequence, Tuple, Union

import numpy as np
import pymbolic as pym
import pyrsistent
import pytools
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from pyop3.dtypes import IntType, PointerType, get_mpi_dtype
from pyop3.index import (
    AffineMap,
    Index,
    IndexTree,
    Map,
    MultiIndex,
    Range,
    Slice,
    TabulatedMap,
)
from pyop3.tree import (
    FixedAryTree,
    NodeComponent,
    LabelledNode,
    LabelledTree,
    Node,
    postvisit,
    previsit,
)
from pyop3.utils import NameGenerator  # TODO delete
from pyop3.utils import (
    PrettyTuple,
    UniqueNameGenerator,
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


def is_distributed(axtree, axis=None):
    """Return ``True`` if any part of a :class:`MultiAxis` is distributed across ranks."""
    axis = axis or axtree.root
    for cidx, cpt in enumerate(axis.components):
        if (
            cpt.is_distributed
            or (subaxis := axtree.find_node((axis.id, cidx)))
            and is_distributed(axtree, subaxis)
        ):
            return True
    return False


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


def compute_offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)[:-1]], dtype=np.int32)


def is_set_up(axtree, axis=None):
    """Return ``True`` if all parts (recursively) of the multi-axis have an associated
    layout function.
    """
    axis = axis or axtree.root
    return all(
        part_is_set_up(axtree, axis, cpt, cidx)
        for cidx, cpt in enumerate(axis.components)
    )


# this would be an easy place to start with writing a tree visitor instead
def part_is_set_up(axtree, axis, component, component_index):
    if (subaxis := axtree.find_node((axis.id, component_index))) and not is_set_up(
        axtree, subaxis
    ):
        return False
    if (axis.id, component_index) not in axtree._layouts:
        return False
    return True


def has_independently_indexed_subaxis_parts(axtree, axis, component, component_index):
    """
    subaxis parts are independently indexed if they don't depend on the index from
    ``part``.

    if one sub-part needs this index to determine its extent then we need to create
    a layout function as the step sizes will differ.

    Note that we need to consider both ragged sizes and permutations here
    """
    if child := axtree.find_node((axis.id, component_index)):
        return not any(
            requires_external_index(axtree, child, i)
            for i, cpt in enumerate(child.components)
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


def prepare_layouts(axtree, axis, component, component_index, path=None):
    """Make a magic nest of dictionaries for storing intermediate results."""
    # path is a tuple key for holding the different axis parts
    if not path:
        path = ((axis.label, component_index),)
    # path += ((axis.name, component.label),)

    # import pdb; pdb.set_trace()
    layouts = {path: None}

    if child := axtree.find_node((axis.id, component_index)):
        for i, subcpt in enumerate(child.components):
            layouts |= prepare_layouts(
                axtree, child, subcpt, i, path + ((child.label, i),)
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
    for i, part in enumerate(axis.components):
        new_path = path + ((axis.label, i),)
        if (
            not has_constant_step(axtree, axis, part, i) or part.indexed
        ):  # I think indexed is gone
            layouts[new_path] = "null layout"

        if child := axtree.find_node((axis.id, i)):
            set_null_layouts(layouts, axtree, new_path, child)


def can_be_affine(axtree, axis, component, component_index):
    return (
        has_independently_indexed_subaxis_parts(
            axtree, axis, component, component_index
        )
        and component.permutation is None
    )


def handle_const_starts(
    axtree,
    layouts,
    path=PrettyTuple(),
    outer_axes_are_all_indexed=True,
    axis=None,
):
    axis = axis or axtree.root
    offset = 0
    for i, part in enumerate(axis.components):
        # catch already set null layouts
        if layouts[path | (axis.label, i)] is not None:
            continue
        # need to track if ragged permuted below
        # check for None here in case we have already set this to a null layout
        if can_be_affine(axtree, axis, part, i) and has_constant_start(
            axtree, axis, part, i, outer_axes_are_all_indexed
        ):
            step = step_size(axtree, axis, part, i)
            layouts[path | (axis.label, i)] = AffineLayoutFunction(step, offset)

        if not has_fixed_size(axtree, axis, part, i):
            # can't do any more as things to the right of this will also move around
            break
        else:
            offset += part.calc_size(axtree, axis, i)

    for i, part in enumerate(axis.components):
        if child := axtree.find_node((axis.id, i)):
            handle_const_starts(
                axtree,
                layouts,
                path | (axis.label, i),
                outer_axes_are_all_indexed and (part.indexed or part.count == 1),
                child,
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


def has_fixed_size(axes, axis, component_index):
    return not size_requires_external_index(axes, axis, component_index)


def step_size(axtree, axis, component_index, indices=PrettyTuple()):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if not has_constant_step(axtree, axis, component_index) and not indices:
        raise ValueError
    if subaxis := axtree.find_node((axis.id, component_index)):
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
    def __init__(
        self,
        root: MultiAxis | None = None,
        parent_to_children: dict | None = None,
        *,
        sf=None,
        shared_sf=None,
        comm=None,
    ):
        super().__init__(root, parent_to_children)
        self.sf = sf
        self.shared_sf = shared_sf
        self.comm = comm  # FIXME DTRT with internal comms

        self._layouts = {}

        # FIXME this is probably silly as an attribute
        self.index = fill_shape(self)

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
        from pyop3.distarray import MultiArray

        # parse the indices to get the right path and do some bounds checking
        offset = 0
        depth = 0
        axis = self.root
        path = []
        new_indices = []
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
                cidx = 0
            else:
                cidx, index = index

            if index < 0:
                # In theory we could still get this to work...
                raise IndexError("Cannot use negative indices")
            # TODO need to pass indices here for ragged things
            if index >= axis.components[cidx].find_integer_count():
                raise IndexError("Index is too large")

            new_indices.append(index)
            path.append(cidx)
            axis = self.find_node((axis.id, cidx))

        if axis is not None:
            raise IndexError("Insufficient number of indices given")

        indices = new_indices

        layouts = self.layouts[tuple(path)]
        remaining_indices = list(indices)
        for layout in layouts:
            if isinstance(layout, IndirectLayoutFunction):
                # if component.indices:
                #     raise NotImplementedError(
                #         "Does not make sense for indirect layouts to have sparsity "
                #         "(I think) since the the indices must be ordered..."
                #     )
                offset += layout.data.get_value(indices[: depth + 1])
            else:
                assert isinstance(layout, AffineLayoutFunction)

                prior_indices = PrettyTuple(indices[:depth])
                last_index = indices[depth]

                if isinstance(layout.start, MultiArray):
                    start = layout.start.get_value(prior_indices)
                else:
                    start = layout.start

                # handle sparsity
                # if component.indices is not None:
                #     bstart, bend = get_slice_bounds(component.indices, prior_indices)
                #
                #     last_index = bisect.bisect_left(
                #         component.indices.data, last_index, bstart, bend
                #     )
                #     last_index -= bstart

                offset += last_index * layout.step + start

                # update indices to include the modications for sparsity
                indices = PrettyTuple(
                    prior_indices + (last_index,) + tuple(indices[depth + 1 :])
                )

            remaining_indices = remaining_indices[len(layout.consumed_labels) :]
            # delete please
            depth += len(layout.consumed_labels)

        assert not remaining_indices
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

    def calc_size(self, axis, indices=PrettyTuple()) -> int:
        """Return the size of a multi-axis by summing the sizes of its components."""
        # NOTE: this works because the size array cannot be multi-part, therefore integer
        # indices (as opposed to typed ones) are valid.
        if not isinstance(indices, PrettyTuple):
            indices = PrettyTuple(indices)
        return sum(
            cpt.calc_size(self, axis, i, indices)
            for i, cpt in enumerate(axis.components)
        )

    def alloc_size(self, axis=None):
        axis = axis or self.root
        return sum(
            cpt.alloc_size(self, axis, i) for i, cpt in enumerate(axis.components)
        )

    # TODO ultimately remove this as it leads to duplication of data
    layout_namer = NameGenerator("layout")

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

    def finish_off_layouts(
        self,
        path=PrettyTuple(),
        layout_path=PrettyTuple(),
        axis=None,
    ):
        from pyop3.distarray import MultiArray

        layouts = {}
        axis = axis or self.root

        # if the axes are not permuted then they are stored contiguously
        if axis.permutation is None:
            for cidx, component in enumerate(axis.components):
                if (axis.id, cidx) not in existing_layouts | layouts.keys():
                    if has_constant_step(self, axis, component, cidx):
                        # elif has_independently_indexed_subaxis_parts(self, axis, component, cidx):
                        step = step_size(self, axis, component, cidx)
                        # affine stuff
                        layouts[(axis.id, cidx)] = AffineLayout(step, offset.value)
                        offset += component.calc_size(self, axis, cidx)
                    else:
                        # we must be ragged
                        new_tree = self.make_ragged_tree(path, layout_path, axis)
                        self.create_layout_lists(
                            path,
                            PrettyTuple(),
                            offset,
                            axis,
                            new_tree,
                        )

                        # also insert intermediary bits into layouts

                        # but step is for the inner dimension - I am making the wrong tree
                        raise NotImplementedError
                        for node in new_tree.nodes:
                            for loc, layout_data in node.data:
                                if layout_data is not None:
                                    layouts[loc] = AffineLayout(step, start=layout_data)
                                else:
                                    layouts[loc] = None

        else:
            for cidx, component in enumerate(axis.components):
                if (axis.id, cidx) not in existing_layouts | layouts.keys():
                    if has_constant_step(self, axis, component, cidx):
                        # elif has_independently_indexed_subaxis_parts(self, axis, component, cidx):
                        step = step_size(self, axis, component, cidx)
                        # I dont want to recurse here really, just want to do the top level one
                        new_tree = self.make_ragged_tree(path, layout_path, axis)
                        assert new_tree.depth == 1
                        self.create_layout_lists(
                            path,
                            PrettyTuple(),
                            offset,
                            axis,
                            new_tree,
                        )

                        for loc, layout_data in new_tree.root.data:
                            layouts[loc] = TabulatedLayout(layout_data)
                    else:
                        # we must be ragged
                        new_tree = self.make_ragged_tree(path, layout_path, axis)
                        self.create_layout_lists(
                            path,
                            PrettyTuple(),
                            offset,
                            axis,
                            new_tree,
                        )

                        # also insert intermediary bits into layouts

                        for node in new_tree.nodes:
                            for loc, layout_data in node.data:
                                if layout_data is not None:
                                    layouts[loc] = TabulatedLayout(layout_data)
                                else:
                                    layouts[loc] = None

        for cidx, component in enumerate(axis.components):
            if subaxis := self.find_node((axis.id, cidx)):
                # if (axis.id, cidx) in existing_layouts | set(layouts.keys()):
                if has_independently_indexed_subaxis_parts(self, axis, component, cidx):
                    # saved_offset = offset.value
                    # offset.value = 0
                    layouts |= self.finish_off_layouts(
                        offset,
                        path | (axis.label, cidx),
                        PrettyTuple(),
                        subaxis,
                        existing_layouts | set(layouts.keys()),
                    )
                    # offset += saved_offset
                    # not sure about this...
                    # offset += component.calc_size(self, axis, cidx)
                else:
                    layouts |= self.finish_off_layouts(
                        offset,
                        path | (axis.label, cidx),
                        layout_path | (axis.label, cidx),
                        subaxis,
                        existing_layouts | set(layouts.keys()),
                    )

        return layouts

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
            return self.get_part_from_path(
                sublabels, self.find_node((axis.id, component_index))
            )
        else:
            return axis, component

    _my_layout_namer = NameGenerator("layout")

    def turn_lists_into_layout_functions(self, layouts):
        from pyop3.distarray import MultiArray

        for path, layout in layouts.items():
            axis, part = self.get_part_from_path(path)
            component_index = axis.components.index(part)
            if can_be_affine(self, axis, part, component_index):
                if isinstance(layout, tuple):
                    layout_path, layout = layout
                    name = self._my_layout_namer.next()

                    # layout_path contains component indices, we don't want those here
                    # so drop them
                    layout_path = [layout[0] for layout in layout_path]

                    starts = MultiArray.from_list(
                        layout, layout_path, name, PointerType
                    )
                    step = step_size(self, axis, part)
                    layouts[path] = AffineLayoutFunction(step, starts)
                else:
                    # FIXME weird code structure
                    assert isinstance(layout, AffineLayoutFunction)
            else:
                if layout == "null layout":
                    continue
                layout_path, layout = layout
                name = self._my_layout_namer.next()

                # layout_path contains component indices, we don't want those here
                # so drop them
                layout_path = [layout[0] for layout in layout_path]

                data = MultiArray.from_list(layout, layout_path, name, PointerType)
                layouts[path] = IndirectLayoutFunction(data)

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
    ) or numbering_requires_external_index(axtree, axis, component_index)


def size_requires_external_index(axes, axis, component_index, depth=0):
    from pyop3.distarray import IndexedMultiArray

    component = axis.components[component_index]
    count = component.count
    if isinstance(count, IndexedMultiArray):
        count = count.data
    if not component.has_integer_count and count.axes.depth > depth:
        return True
    else:
        if child := axes.find_node((axis.id, component_index)):
            for i, cpt in enumerate(child.components):
                if size_requires_external_index(axes, child, i, depth + 1):
                    return True
    return False


def numbering_requires_external_index(axtree, axis, component_index, depth=1):
    # I don't allow this
    return False
    if component.numbering is not None and component.numbering.dim.depth > depth:
        return True
    else:
        if subaxis := axtree.find_node((axis.id, component_index)):
            for i, cpt in enumerate(subaxis.components):
                if numbering_requires_external_index(
                    axtree, subaxis, cpt, i, depth + 1
                ):
                    return True
    return False


def has_constant_step(axes: AxisTree, axis, component_index, depth=0):
    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if child := axes.find_node((axis.id, component_index)):
        return all(
            not size_requires_external_index(axes, child, i, depth)
            for i, cpt in enumerate(child.components)
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
        "name",
        "overlap",
        "indexed",
        "indices",
        "lgmap",
    }

    def __init__(
        self,
        count,
        *,
        name: str | None = None,
        indices=None,
        overlap=None,
        indexed=False,
        lgmap=None,
    ):
        super().__init__()

        self.count = count
        self.name = name
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
    def alloc_size(self, axtree, axis, component_index):
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

        if subaxis := axtree.find_node((axis.id, component_index)):
            return npoints * axtree.alloc_size(subaxis)
        else:
            return npoints

    # TODO make a free function or something - this is horrible
    def calc_size(self, axtree, axis, component_index, indices=PrettyTuple()):
        extent = self.find_integer_count(indices)
        if subaxis := axtree.find_node((axis.id, component_index)):
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
class IndexFunction(pytools.ImmutableRecord, abc.ABC):
    fields = {"consumed_labels"}

    def __init__(self, consumed_labels):
        super().__init__()
        self.consumed_labels = frozenset(consumed_labels)


# from indices to offsets
class LayoutFunction(IndexFunction, abc.ABC):
    pass


class AffineLayoutFunction(LayoutFunction):
    fields = LayoutFunction.fields | {"step", "start"}

    def __init__(self, consumed_label, step, start=0):
        super().__init__({consumed_label})
        self.step = step
        self.start = start

    @property
    def consumed_label(self):
        return just_one(self.consumed_labels)


class IndirectLayoutFunction(LayoutFunction):
    fields = LayoutFunction.fields | {"data"}

    def __init__(self, consumed_components, data):
        if len(consumed_components) != data.axes.depth:
            raise ValueError
        super().__init__(consumed_components)
        self.data = data


# aliases
AffineLayout = AffineLayoutFunction
TabulatedLayout = IndirectLayoutFunction


@dataclasses.dataclass
class SyncStatus:
    pending_write_op: Optional[Any] = None
    halo_valid: bool = True
    halo_modified: bool = False


# TODO This algorithm is pretty much identical to _axes_from_index_tree
def fill_shape(axes, visited=None, current_axis=None):
    from pyop3.distarray import MultiArray

    current_axis = current_axis or axes.root
    visited = visited or {}

    indices = []
    subindex_trees = []

    # register a full slice if we haven't indexed it yet.
    if current_axis.label not in visited:
        for cidx, component in enumerate(current_axis.components):
            new_visited = visited.copy()
            new_visited[current_axis.label] = cidx

            indices.append(Slice((current_axis.label, cidx), None))

            if subaxis := axes.find_node((current_axis.id, cidx)):
                subindex_tree = fill_shape(axes, visited, subaxis)
                subindex_trees.append(subindex_tree)
            else:
                subindex_trees.append(None)

        root = MultiIndex(indices)
        subroots = [
            subindex_tree.root if subindex_tree else None
            for subindex_tree in subindex_trees
        ]
        others = functools.reduce(
            operator.or_,
            [dict(sit.parent_to_children) if sit else {} for sit in subindex_trees],
            {},
        )

        return IndexTree(root, {root.id: subroots} | others)

    else:
        cidx = visited[current_axis.label]
        if axes.find_node((current_axis.id, cidx)):
            return fill_shape(axes, new_axis_path, prev_indices | new_index)
        else:
            return IndexTree(None)


def expand_indices_to_fill_empty_shape(
    axis,
    itree,
    multi_index=None,
    visited=None,
):
    multi_index = multi_index or itree.root

    visited = visited or {}

    subroots = []
    subnodes = {}
    for i, index in enumerate(multi_index.indices):
        # TODO: this bit is very similar to myinnerfunc in loopy.py
        new_visited = visited.copy()
        new_visited.pop(index.from_axis[0], None)
        new_visited[index.from_axis[0]] = index.from_axis[1]

        if submidx := itree.find_node((multi_index.id, i)):
            subitree = expand_indices_to_fill_empty_shape(
                axis,
                itree,
                submidx,
                new_visited,
            )
            subroots.append(subitree.root)
            subnodes |= subitree.parent_to_children
        else:
            subitree = fill_shape(axis, new_visited)
            subroots.append(subitree.root)
            subnodes |= subitree.parent_to_children

    return IndexTree(multi_index, {multi_index.id: subroots} | subnodes)


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
    for cidx, subaxis in enumerate(axes.children(axis)):
        if subaxis is not None:
            sublayouts, csubtree, substeps = _compute_layouts(
                axes, subaxis, path | cidx
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
    if not all(has_fixed_size(axes, axis, cidx) for cidx in range(axis.degree)):
        # 0. We ignore things if they are indexed. They don't contribute
        if axis.indexed:
            ctree = None
        elif all(has_constant_step(axes, axis, cidx) for cidx in range(axis.degree)):
            # we must be at the bottom of a ragged patch - therefore don't
            # add to shape of things
            # in theory if we are ragged and permuted then we do want to include this level
            ctree = None
            for cidx in range(axis.degree):
                step = step_size(axes, axis, cidx)
                layouts |= {path | cidx: AffineLayout(axis.label, step)}

        else:
            croot = CustomNode([(cpt.count, axis.label) for cpt in axis.components])
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = {
                    croot.id: [sub.root for sub in csubtrees]
                } | merge_dicts(sub.parent_to_children for sub in csubtrees)
            else:
                cparent_to_children = {}
            ctree = FixedAryTree(croot, cparent_to_children)

        # layouts and steps are just propagated from below
        return layouts | merge_dicts(sublayoutss), ctree, steps

    # 2. add layouts here
    else:
        # 1. do we need to tabulate anything?
        if axis.permutation is not None or not all(
            has_constant_step(axes, axis, cidx) for cidx in range(axis.degree)
        ):
            # If this axis is indexed and there is no inner shape to index then do nothing
            if axis.indexed and strictly_all(sub is None for sub in csubtrees):
                return layouts | merge_dicts(sublayoutss), None, steps

            croot = CustomNode([(cpt.count, axis.label) for cpt in axis.components])
            if strictly_all(sub is not None for sub in csubtrees):
                cparent_to_children = {
                    croot.id: [sub.root for sub in csubtrees]
                } | merge_dicts(sub.parent_to_children for sub in csubtrees)
            else:
                cparent_to_children = {}
            ctree = FixedAryTree(croot, cparent_to_children)

            fulltree = _create_count_array_tree(ctree)

            # now populate fulltree
            offset = IntRef(0)
            _tabulate_count_array_tree(axes, axis, fulltree, offset)

            for subpath, offset_data in fulltree.items():
                axis_ = axis
                labels = []
                for cidx in subpath:
                    labels.append(axis_.label)
                    axis_ = axes.find_node((axis_.id, cidx))

                layouts[path + subpath] = TabulatedLayout(labels, offset_data)
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
            steps = [step_size(axes, axis, cidx) for cidx in range(axis.degree)]
            start = 0
            for cidx, step in enumerate(steps):
                sublayouts = sublayoutss[cidx].copy()

                new_layout = AffineLayout(axis.label, step, start)
                sublayouts[path | cidx] = new_layout
                start += axis.components[cidx].calc_size(axes, axis, cidx)

                layouts |= sublayouts
            steps = {path: axes.calc_size(axis)}
            return layouts, None, steps


# I don't think that this actually needs to be a tree, just return a dict
def _create_count_array_tree(
    ctree, current_node=None, counts=PrettyTuple(), component_path=PrettyTuple()
):
    from pyop3.distarray import MultiArray

    current_node = current_node or ctree.root
    arrays = {}

    for cidx in range(current_node.degree):
        child = ctree.children(current_node)[cidx]
        if child is None:
            # make a multiarray here from the given sizes
            axes = [
                Axis(count, label)
                for (count, label) in counts | current_node.counts[cidx]
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
            arrays[component_path | cidx] = countarray
        else:
            arrays |= _create_count_array_tree(
                ctree, child, counts | current_node.counts[cidx], component_path | cidx
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

        if path | selected_component_id in count_arrays:
            count_arrays[path | selected_component_id].set_value(
                indices | selected_component_num, offset.value
            )
            if not axis.indexed:
                offset += step_size(
                    axes,
                    axis,
                    selected_component_id,
                    indices | selected_component_num,
                )
        else:
            if axis.indexed:
                saved_offset = offset.value

            subaxis = axes.find_node((axis.id, selected_component_id))
            assert subaxis
            _tabulate_count_array_tree(
                axes,
                subaxis,
                count_arrays,
                offset,
                path | selected_component_id,
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

    for cidx in range(axis.degree):
        if component_path | cidx in values:
            prior_ = prior | values[component_path | cidx]
        else:
            prior_ = prior
        if subaxis := axes.find_node((axis.id, cidx)):
            acc |= _collect_at_leaves(
                axes, values, subaxis, component_path | cidx, prior_
            )
        else:
            acc[component_path | cidx] = prior_

    return acc

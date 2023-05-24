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
    TabulatedMap,
)
from pyop3.tree import (
    FixedAryTree,
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
    has_unique_entries,
    just_one,
    merge_dicts,
    single_valued,
    some_but_not_all,
    strict_int,
    strictly_all,
    unique,
)

DEFAULT_PRIORITY = 100


class InvalidConstraintsException(Exception):
    pass


class ConstrainedAxis(pytools.ImmutableRecord):
    fields = {"axis", "priority", "within_labels"}
    # TODO We could use 'label' to set the priority
    # via commandline options

    def __init__(
        self,
        axis: Axis,
        *,
        priority: int = DEFAULT_PRIORITY,
        within_labels: FrozenSet[Hashable] = frozenset(),
    ):
        self.axis = axis
        self.priority = priority
        self.within_labels = frozenset(within_labels)
        super().__init__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(axis=({', '.join(str(axis_cpt) for axis_cpt in self.axis)}), priority={self.priority}, within_labels={self.within_labels})"


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
            requires_external_index(axtree, child, cpt, i)
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


def has_fixed_size(axtree, axis, component, component_index):
    return not size_requires_external_index(axtree, axis, component, component_index)


def step_size(axtree, axis, component, component_index, indices=PrettyTuple()):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if not has_constant_step(axtree, axis, component, component_index) and not indices:
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
    from pyop3.distarray import MultiArray

    datamap = {}
    for cidx, component in enumerate(axis.components):
        if isinstance(layout := axes.layouts[(axis.id, cidx)], IndirectLayoutFunction):
            datamap |= layout.data.datamap
        if isinstance(count := component.count, MultiArray):
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
    ):
        super().__init__(root, parent_to_children)
        self.sf = sf
        self.shared_sf = shared_sf

        self._layouts = {}

    @functools.cached_property
    def datamap(self) -> dict[str:DistributedArray]:
        return postvisit(self, _collect_datamap, axes=self)

    @property
    def part(self):
        try:
            (pt,) = self.parts
            return pt
        except ValueError:
            raise RuntimeError

    @functools.cached_property
    def layouts(self):
        self.set_up()
        return self._layouts

    def find_part(self, label):
        return self._parts_by_label[label]

    def get_offset(self, indices):
        from pyop3.distarray import MultiArray

        # use layout functions to access the right thing
        # indices here are integers, so this will only work for multi-arrays that
        # are not multi-part
        # if self.is_multi_part:
        #   raise Exception("cannot index with integers here")
        # accumulate offsets from the layout functions
        offset = 0
        depth = 0
        axis = self.root

        # effectively loop over depth
        while True:
            component = just_one(axis.components)

            layout = self.layouts[axis.id, 0]
            if isinstance(layout, IndirectLayoutFunction):
                if component.indices:
                    raise NotImplementedError(
                        "Does not make sense for indirect layouts to have sparsity "
                        "(I think) since the the indices must be ordered..."
                    )
                offset += layout.data.get_value(indices[: depth + 1])
            elif layout == "null layout":
                pass
            else:
                assert isinstance(layout, AffineLayoutFunction)

                prior_indices = PrettyTuple(indices[:depth])
                last_index = indices[depth]

                if isinstance(layout.start, MultiArray):
                    start = layout.start.get_value(prior_indices)
                else:
                    start = layout.start

                # handle sparsity
                if component.indices is not None:
                    bstart, bend = get_slice_bounds(component.indices, prior_indices)

                    last_index = bisect.bisect_left(
                        component.indices.data, last_index, bstart, bend
                    )
                    last_index -= bstart

                offset += last_index * layout.step + start

                # update indices to include the modications for sparsity
                indices = PrettyTuple(
                    prior_indices + (last_index,) + tuple(indices[depth + 1 :])
                )

            depth += 1
            axis = self.find_node((axis.id, 0))

            if not axis:
                break

        # make sure its an int
        assert offset - int(offset) == 0

        return offset

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

    @property
    def count(self):
        """Return the total number of entries in the axis across all axis parts.
        Will fail if axis parts do not have integer counts.
        """
        if self.nparts == 1:
            return self.part.count
        if not all(isinstance(pt.count, numbers.Integral) for pt in self.parts):
            raise RuntimeError("non-int counts present, cannot sum")
        return sum(pt.count for pt in self.parts)

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

        layouts = self._set_up()
        # import pdb; pdb.set_trace()
        # self.apply_layouts(layouts)
        self._layouts = layouts
        assert is_set_up(self)
        # breakpoint()

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

    def apply_layouts(self, layouts, path=PrettyTuple(), axis=None):
        """Attach a complicated dict of multiple parts and layouts functions to the
        right axis parts.

        Input is a dict of the form

            {
                (0,): func,
                (1,): func,
                (0, 0): func,
                (0, 1): func
            }

        i.e. the dicts keys are 'paths' directing which axis parts to apply the layout
        functions to.
        This function just traverses the axis tree and attaches the right thing.
        """
        axis = axis or self.root
        for cidx, part in enumerate(axis.components):
            pth = path | (axis.label, cidx)

            layout = layouts[pth]
            if subaxis := self.find_node((axis.id, cidx)):
                self.apply_layouts(layouts, pth, subaxis)
            self._layouts[(axis.id, cidx)] = layout

    def finish_off_layouts(
        self,
        offset,
        path=PrettyTuple(),
        layout_path=PrettyTuple(),
        axis=None,
        existing_layouts=None,
    ):
        from pyop3.distarray import MultiArray

        layouts = {}
        axis = axis or self.root
        existing_layouts = existing_layouts or frozenset()

        for cidx, component in enumerate(axis.components):
            if (axis.id, cidx) not in existing_layouts:
                if has_constant_step(self, axis, component, cidx):
                    # elif has_independently_indexed_subaxis_parts(self, axis, component, cidx):
                    step = step_size(self, axis, component, cidx)
                    if axis.permutation is None:
                        # affine stuff
                        layouts[(axis.id, cidx)] = AffineLayout(step, offset.value)
                    else:
                        # I dont want to recurse here really, just want to do the top level one
                        layout_labels = self.determine_layout_axis_labels(
                            path, layout_path, axis
                        )
                        sublayout_bits = self.create_layout_lists(
                            path,
                            PrettyTuple(),
                            offset,
                            axis,
                            layout_labels,
                        )
                        for labels, (loc, (layout_data, _)) in checked_zip(
                            layout_labels, sublayout_bits.items()
                        ):
                            layout_data = MultiArray.from_list(
                                layout_data, labels, dtype=IntType
                            )
                            layouts[loc] = TabulatedLayout(layout_data)
                else:
                    # we must be ragged
                    new_tree = self.make_ragged_tree(path, layout_path, axis)
                    # breakpoint()
                    self.create_layout_lists(
                        path,
                        PrettyTuple(),
                        offset,
                        axis,
                        new_tree,
                    )
                    # breakpoint()

                    # also insert intermediary bits into layouts

                    for node in new_tree.nodes:
                        for loc, layout_data in node.data:
                            if layout_data is not None:
                                layouts[loc] = TabulatedLayout(layout_data)
                            else:
                                layouts[loc] = None

            # breakpoint()

            # next bit, recurse
            if subaxis := self.find_node((axis.id, cidx)):
                if (axis.id, cidx) in layouts:
                    saved_offset = offset.value
                    offset.value = 0
                    layouts |= self.finish_off_layouts(
                        offset,
                        path | (axis.label, cidx),
                        PrettyTuple(),
                        subaxis,
                        existing_layouts | set(layouts.keys()),
                    )
                    offset += saved_offset
                    offset += component.calc_size(self, axis, cidx)
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

    def create_layout_lists(
        self,
        path,
        layout_path,
        offset,
        axis,
        layout_tree,
        mynode=None,
        indices=PrettyTuple(),
    ):
        mynode = mynode or layout_tree.root
        layout_data = []
        layout_bits = {}

        npoints = 0
        for part in axis.components:
            if part.has_integer_count:
                count = part.count
            else:
                count = strict_int(part.count.get_value(indices))
            npoints += count

        if axis.permutation is not None:
            permutation = axis.permutation
        else:
            permutation = np.arange(npoints, dtype=IntType)

        point_to_component_id = np.empty(npoints, dtype=np.int8)
        point_to_component_num = np.empty(npoints, dtype=PointerType)
        pos = 0
        for cidx, component in enumerate(axis.components):
            if component.has_integer_count:
                csize = component.count
            else:
                csize = strict_int(component.count.get_value(indices))

            for i in range(csize):
                point = permutation[pos + i]
                point_to_component_id[point] = cidx
                point_to_component_num[point] = i
            # layout_bits[(target[0], cidx)] = [np.empty(csize, dtype=object), None]
            # layout_data.append(np.empty(csize, dtype=object))
            pos += csize

        for pt in range(npoints):
            selected_component_id = point_to_component_id[pt]
            selected_component = axis.components[selected_component_id]
            selected_component_num = point_to_component_num[pt]

            # if axis.id == mynode.data[0]:
            if mynode in layout_tree.leaves:
                # layout_bits[(axis.id, selected_component_id)][0][
                #     selected_component_num
                # ] = offset.value
                mynode.data[selected_component_id][1].set_value(
                    indices | selected_component_num, offset.value
                )
                # layout_data[selected_component_id][
                #     selected_component_num
                # ] = offset.value
                offset += step_size(
                    self,
                    axis,
                    selected_component,
                    selected_component_id,
                    indices | selected_component_num,
                )
            else:
                # subaxis = self.find_node((axis.id, selected_component_id))
                # if all(has_constant_step(self, subaxis, subcomponent, subcidx) for (subcidx, subcomponent) in enumerate(subaxis.components)) :
                #     layout_bits[(axis.id, selected_component_id)][0][
                #         selected_component_num
                #     ] = offset.value
                #     offset += step_size(
                #         self,
                #         axis,
                #         selected_component,
                #         selected_component_id,
                #         indices | selected_component_num,
                #     )
                # else:
                subaxis = self.find_node((axis.id, selected_component_id))
                assert subaxis
                self.create_layout_lists(
                    path | (axis.label, selected_component_id),
                    layout_path | axis.label,
                    offset,
                    subaxis,
                    layout_tree,
                    layout_tree.find_node((mynode.id, selected_component_id)),
                    indices | selected_component_num,
                )

                # breakpoint()

                # for subloc, (sublayout_data, sublayout_path) in subdata.items():
                #     layout_bits[(axis.id, selected_component_id)][0][
                #         selected_component_num
                #     ] = sublayout_data
                # layout_bits[(axis.id, selected_component_id)][1] = sublayout_path

        # catch zero-sized sub-bits
        # for n in range(self.nparts):
        #     path_ = path | n
        #     if isinstance(new_layouts[path_], dict) and len(new_layouts[path_]) != npoints:
        #         assert len(new_layouts[path_]) < npoints
        #         for i in range(npoints):
        #             if i not in new_layouts[path_]:
        #                 new_layouts[path_][i] = []

        # breakpoint()
        # return layout_data
        #
        # return layout_bits

    def make_ragged_tree(self, path, layout_path, axis, sizes=PrettyTuple()):
        from pyop3.distarray import MultiArray

        root_id = next(MyNode._id_generator)
        nodedata = []
        subtrees = []

        for cidx, component in enumerate(axis.components):
            mylayout_path = layout_path | axis.label
            mysizes = sizes | component.count

            subaxis = self.find_node((axis.id, cidx))
            if all(
                has_constant_step(self, subaxis, subcomponent, subcidx)
                for (subcidx, subcomponent) in enumerate(subaxis.components)
            ):
                # make a multiarray here from the given sizes
                axes = [
                    Axis(count, label)
                    for (count, label) in checked_zip(mysizes, mylayout_path)
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
                nodedata.append(((axis.id, cidx), countarray))
            else:
                subaxis = self.find_node((axis.id, cidx))
                assert subaxis
                subtree = self.make_ragged_tree(
                    path | (axis.label, cidx),
                    mylayout_path,
                    subaxis,
                    mysizes,
                )
                nodedata.append(((axis.id, cidx), None))
                subtrees.append(subtree)

        parent_to_children = {}
        for subtree in subtrees:
            parent_to_children |= {root_id: subtree.root} | dict(
                subtree.parent_to_children
            )
        root = MyNode(data=tuple(nodedata), degree=len(axis.components), id=root_id)
        return FixedAryTree(root, parent_to_children)

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

    def _set_up(self):
        offset = IntRef(0)
        indices = PrettyTuple()

        # # return a nested list structure or nothing. if the former then haven't set all
        # # the layouts (needs) to be end of the loop
        #
        # # loop over all points in all parts of the multi-axis
        # # initialise layout array per axis part
        # # import pdb; pdb.set_trace()
        # layouts = {}
        # for i, cpt in enumerate(self.root.components):
        #     layouts |= prepare_layouts(self, self.root, cpt, i)
        #
        # set_null_layouts(layouts, self)
        # handle_const_starts(self, layouts)
        #
        # assert offset.value == 0
        layouts = self.finish_off_layouts(offset)

        # self.turn_lists_into_layout_functions(layouts)
        #
        # for layout in layouts.values():
        #     if isinstance(layout, list):
        #         if any(item is None for item in layout):
        #             raise AssertionError

        return layouts

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
    fields = LabelledNode.fields - {"degree"} | {"components", "permutation"}

    def __init__(
        self,
        components: Sequence[AxisComponent] | AxisComponent | int,
        label: Hashable | None = None,
        *,
        permutation: Sequence[int] | None = None,
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}([{', '.join(str(cpt) for cpt in self.components)}], label={self.label})"


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


def requires_external_index(axtree, axis, component, component_index):
    """Return ``True`` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index(
        axtree, axis, component, component_index
    ) or numbering_requires_external_index(axtree, axis, component, component_index)


def size_requires_external_index(axtree, node, part, component_index, depth=0):
    if not part.has_integer_count and part.count.dim.depth > depth:
        return True
    else:
        if child := axtree.find_node((node.id, component_index)):
            for i, cpt in enumerate(child.components):
                if size_requires_external_index(axtree, child, cpt, i, depth + 1):
                    return True
    return False


def numbering_requires_external_index(
    axtree, axis, component, component_index, depth=1
):
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


def has_constant_step(axtree, node, part, component_index, depth=0):
    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if child := axtree.find_node((node.id, component_index)):
        return all(
            not size_requires_external_index(axtree, child, cpt, i, depth)
            for i, cpt in enumerate(child.components)
        )
    else:
        return True


class AxisComponent(pytools.ImmutableRecord):
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
        "name",
        "max_count",
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
        max_count=None,
        overlap=None,
        indexed=False,
        lgmap=None,
    ):
        from pyop3.distarray import MultiArray

        if isinstance(count, numbers.Integral):
            assert not max_count or max_count == count
            max_count = count
        elif not max_count:
            max_count = max(count.data)

        super().__init__()

        self.count = count
        self.name = name
        self.indices = indices
        self.max_count = max_count
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
        # TODO This should probably raise an exception if we do weird things with maps
        # (as in fix the layouts for reduced data movement)
        # if self.count == -1:  # scalar thing
        #     count = 1
        # elif not self.indexed:
        #     count = self.max_count
        # else:
        #     count = 1
        # i dont think i need the above any more
        count = self.max_count
        if subaxis := axtree.find_node((axis.id, component_index)):
            return count * axtree.alloc_size(subaxis)
        else:
            return count

    # TODO make a free function or something - this is horrible
    def calc_size(self, axtree, axis, component_index, indices=PrettyTuple()):
        extent = self.find_integer_count(indices)
        if subaxis := axtree.find_node((axis.id, component_index)):
            return sum(axtree.calc_size(subaxis, indices | i) for i in range(extent))
        else:
            return extent

    def find_integer_count(self, indices=PrettyTuple()):
        from pyop3.distarray import MultiArray

        if isinstance(self.count, MultiArray):
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
    fields = set()


# from indices to offsets
class LayoutFunction(IndexFunction, abc.ABC):
    pass


class AffineLayoutFunction(LayoutFunction):
    fields = LayoutFunction.fields | {"step", "start"}

    def __init__(self, step, start=0):
        self.step = step
        self.start = start


class IndirectLayoutFunction(LayoutFunction):
    fields = LayoutFunction.fields | {"data"}

    def __init__(self, data):
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
def fill_shape(axes, axis_path=None, prev_indices=PrettyTuple()):
    from pyop3.distarray import MultiArray

    axis_path = axis_path or {}

    axis = axes.find_node(dict(axis_path))
    indices = []
    subindex_trees = []
    for i, component in enumerate(axis.components):
        if isinstance(component.count, MultiArray):
            # turn prev_indices into a tree
            assert len(prev_indices) > 0
            mynewtree = IndexTree()
            parent = (None, 0)
            for prev_idx in prev_indices:
                mynewtree = mynewtree.add_node(prev_idx, parent=parent)
                parent = (prev_idx.id, 0)
            count = component.count[mynewtree]
        else:
            count = component.count

        new_index = Range((axis.label, i), count)
        indices.append(new_index)

        new_axis_path = dict(axis_path) | {axis.label: i}
        if axes.find_node(new_axis_path):
            subindex_tree = fill_shape(axes, new_axis_path, prev_indices | new_index)
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


def expand_indices_to_fill_empty_shape(
    axis,
    itree,
    multi_index=None,
    path=PrettyTuple(),
    prev_indices=PrettyTuple(),
):
    multi_index = multi_index or itree.root

    subroots = []
    subnodes = {}
    for i, index in enumerate(multi_index.indices):
        # TODO: this bit is very similar to myinnerfunc in loopy.py
        if isinstance(index, Range):
            path = path | index.path
        else:
            assert isinstance(index, Map)
            path = path[: -len(index.from_labels)] + index.to_labels

        if submidx := itree.find_node((multi_index.id, i)):
            subitree = expand_indices_to_fill_empty_shape(
                axis, itree, submidx, path, prev_indices | index
            )
            subroots.append(subitree.root)
            subnodes |= subitree.parent_to_children
        elif axis.find_node(dict(path)):
            subitree = fill_shape(axis, path, prev_indices | index)
            subroots.append(subitree.root)
            subnodes |= subitree.parent_to_children
        else:
            subroots.append(None)

    root = MultiIndex(multi_index.indices)
    return IndexTree(root, {root.id: subroots} | subnodes)


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


def order_axes(layout):
    axes = MultiAxisTree()
    layout = list(layout)
    axis_to_constraint = {}
    history = set()
    while layout:
        if tuple(layout) in history:
            raise ValueError("Seen this before, cyclic")
        history.add(tuple(layout))

        constrained_axis = layout.pop(0)
        inserted = _insert_axis(axes, constrained_axis, axes.root, axis_to_constraint)
        if not inserted:
            layout.append(constrained_axis)
    return axes


def _insert_axis(
    axes: MultiAxisTree,
    new_caxis: ConstrainedMultiAxis,
    current_axis: MultiAxis,
    axis_to_caxis: dict[MultiAxis, ConstrainedMultiAxis],
    path: dict[Hashable] | None = None,
):
    path = path or {}

    within_labels = set(path.items())

    # alias - remove
    axis_to_constraint = axis_to_caxis

    if new_caxis.axis not in axis_to_constraint:
        axis_to_constraint[new_caxis.axis.label] = new_caxis

    if not axes.root:
        if not new_caxis.within_labels:
            axes.add_node(new_caxis.axis)
            return True
        else:
            return False

    # current_axis = current_axis or axes.root
    current_caxis = axis_to_constraint[current_axis.label]

    if new_caxis.priority < current_caxis.priority:
        if new_caxis.within_labels <= within_labels:
            # diagram or something?
            parent_axis = axes.parent(current_axis)
            subtree = axes.pop_subtree(current_axis)
            betterid = new_caxis.axis.copy(id=next(MultiAxis._id_generator))
            if not parent_axis:
                axes.add_node(betterid)
            else:
                axes.add_node(betterid, path)

            # must already obey the constraints - so stick  back in for all sub components
            for comp in betterid.components:
                stree = subtree.copy()
                # stree.replace_node(stree.root.copy(id=next(MultiAxis._id_generator)))
                mypath = (axes._node_to_path[betterid.id] or {}) | {
                    betterid.label: comp.label
                }
                axes.add_subtree(stree, mypath, uniquify=True)
                axes._parent_and_label_to_child[(betterid, comp.label)] = stree.root.id
                # need to register the right parent label
            return True
        else:
            # The priority is less so the axes should definitely
            # not be inserted below here - do not recurse
            return False
    elif axes.is_leaf(current_axis):
        assert new_caxis.priority >= current_caxis.priority
        for cpt in current_axis.components:
            if new_caxis.within_labels <= within_labels | {
                (current_axis.label, cpt.label)
            }:
                betterid = new_caxis.axis.copy(id=next(MultiAxis._id_generator))
                axes.add_node(betterid, path | {current_axis.label: cpt.label})
        return True
    else:
        inserted = False
        for cpt in current_axis.components:
            subaxis = axes.child_by_label(current_axis, cpt.label)
            # if not subaxis then we dont insert here
            if subaxis:
                inserted = inserted or _insert_axis(
                    axes,
                    new_caxis,
                    subaxis,
                    axis_to_constraint,
                    path | {current_axis.label: cpt.label},
                )
        return inserted


# aliases
MultiAxisTree = AxisTree
MultiAxis = Axis
MultiAxisComponent = AxisComponent


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
def _(arg: int) -> AxisComponent:
    return AxisComponent(arg)

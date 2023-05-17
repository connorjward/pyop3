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
    AffineMapNode,
    Index,
    IndexTree,
    MapNode,
    MultiIndex,
    RangeNode,
    TabulatedMapNode,
)
from pyop3.tree import (
    LabelledNode,
    LabelledNodeComponent,
    LabelledTree,
    Node,
    NullRootNode,
    NullRootTree,
    Tree,
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
    single_valued,
    strict_int,
    strictly_all,
    unique,
)

# old alias
NewNode = Node

DEFAULT_PRIORITY = 100


class InvalidConstraintsException(Exception):
    pass


class ConstrainedMultiAxis(pytools.ImmutableRecord):
    fields = {"axis", "priority", "within_labels"}
    # TODO We could use 'label' to set the priority
    # via commandline options

    def __init__(
        self,
        axis: "MultiAxis",
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
    for cpt in axis.components:
        if (
            cpt.is_distributed
            or (subaxis := axtree.child_by_label(axis, cpt.label))
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
    return all(part_is_set_up(axtree, axis, cpt) for cpt in axis.components)


# this would be an easy place to start with writing a tree visitor instead
def part_is_set_up(axtree, axis, component):
    if (subaxis := axtree.child_by_label(axis, component.label)) and not is_set_up(
        axtree, subaxis
    ):
        return False
    if component.layout_fn is None:
        return False
    return True


def has_independently_indexed_subaxis_parts(axtree, axis, component):
    """
    subaxis parts are independently indexed if they don't depend on the index from
    ``part``.

    if one sub-part needs this index to determine its extent then we need to create
    a layout function as the step sizes will differ.

    Note that we need to consider both ragged sizes and permutations here
    """
    if child := axtree.child_by_label(axis, component.label):
        return not any(
            requires_external_index(axtree, child, cpt) for cpt in child.components
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
        path = ((axis.name, component.label),)
    # path += ((axis.name, component.label),)

    # import pdb; pdb.set_trace()
    layouts = {path: None}

    if child := axtree.child_by_label(axis, component.label):
        for subcpt in child.components:
            layouts |= prepare_layouts(
                axtree, child, subcpt, path + ((child.name, subcpt.label),)
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
    for part in axis.components:
        new_path = path + ((axis.name, part.label),)
        if not has_constant_step(axtree, axis, part) or part.indexed:
            layouts[new_path] = "null layout"

        if child := axtree.child_by_label(axis, part.label):
            set_null_layouts(layouts, axtree, new_path, child)


def can_be_affine(axtree, axis, component):
    return (
        has_independently_indexed_subaxis_parts(axtree, axis, component)
        and component.numbering is None
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
    for part in axis.components:
        # catch already set null layouts
        if layouts[path | (axis.name, part.label)] is not None:
            continue
        # need to track if ragged permuted below
        # check for None here in case we have already set this to a null layout
        if can_be_affine(axtree, axis, part) and has_constant_start(
            axtree, axis, part, outer_axes_are_all_indexed
        ):
            step = step_size(axtree, axis, part)
            layouts[path | (axis.name, part.label)] = AffineLayoutFunction(step, offset)

        if not has_fixed_size(axtree, axis, part):
            # can't do any more as things to the right of this will also move around
            break
        else:
            offset += part.calc_size(axtree, axis)

    for part in axis.components:
        if child := axtree.child_by_label(axis, part.label):
            handle_const_starts(
                axtree,
                layouts,
                path | (axis.name, part.label),
                outer_axes_are_all_indexed and (part.indexed or part.count == 1),
                child,
            )


def has_constant_start(axtree, axis, component, outer_axes_are_all_indexed: bool):
    """
    We will have an affine layout with a constant start (usually zero) if either we are not
    ragged or if we are ragged but everything above is indexed (i.e. a temporary).
    """
    assert can_be_affine(axtree, axis, component)
    return isinstance(component.count, numbers.Integral) or outer_axes_are_all_indexed


def has_fixed_size(axtree, axis, component):
    return not size_requires_external_index(axtree, axis, component)


def step_size(axtree, axis, component, indices=PrettyTuple()):
    """Return the size of step required to stride over a multi-axis component.

    Non-constant strides will raise an exception.
    """
    if not has_constant_step(axtree, axis, component) and not indices:
        raise ValueError
    if subaxis := axtree.child_by_label(axis, component.label):
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

    cdim = axis.calc_size(part) if axis.children(part) else 1
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


class MultiAxisTree(LabelledTree):
    # fields = {"parts", "sf", "shared_sf"}

    def __init__(self, root=None, *, sf=None, shared_sf=None):
        super().__init__()
        self.sf = sf
        self.shared_sf = shared_sf
        if root:
            self.add_node(root)

    def add_nodes(self, axes, parent=None):
        # make sure all parts have labels, default to integers if necessary
        if strictly_all(pt.label is None for pt in axes):
            # set the label to the index
            axes = tuple(pt.copy(label=i) for i, pt in enumerate(axes))

        if utils.some_but_not_all(pt.is_distributed for pt in axes):
            raise ValueError(
                "Cannot have a multi-axis with some parallel parts and some not"
            )

        if not has_unique_entries(pt.label for pt in axes):
            raise ValueError(
                "Axis parts in the same multi-axis must have unique labels"
            )

        super().add_nodes(axes, parent)

    def __mul__(self, other):
        """Return the (dense) outer product."""
        return self.mul(other)

    @property
    def part(self):
        try:
            (pt,) = self.parts
            return pt
        except ValueError:
            raise RuntimeError

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

            layout = component.layout_fn
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
            axis = self.child_by_label(axis, component.label)

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
        return sum(cpt.calc_size(self, axis, indices) for cpt in axis.components)

    def alloc_size(self, axis=None):
        axis = axis or self.root
        return sum(cpt.alloc_size(self, axis) for cpt in axis.components)

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
        self.apply_layouts(layouts)
        assert is_set_up(self)

        # set the .sf and .owned_sf properties of new_axis
        if with_sf and is_distributed(self):
            attach_star_forest(self)
            attach_star_forest(self, with_halo_points=False)

            # attach a local to global map
            if len(self.children(self.root)) > 1:
                raise NotImplementedError(
                    "Currently only compute lgmaps for a single part, shouldn't "
                    "be hard to fix"
                )
            lgmap = create_lgmap(self)
            new_part = just_one(self.children(self.root))
            self.replace_node(new_part.copy(lgmap=lgmap))
            # self.copy(parts=[new_axis.part.copy(lgmap=lgmap)])
        # new_axis = attach_owned_star_forest(new_axis)
        self.frozen = True
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
        new_parts = []
        for part in axis.components:
            pth = path | (axis.name, part.label)

            layout = layouts[pth]
            if subaxis := self.child_by_label(axis, part.label):
                self.apply_layouts(layouts, pth, subaxis)
            new_part = part.copy(layout_fn=layout)
            new_parts.append(new_part)

        new_axis = axis.copy(components=new_parts)
        self.replace_node(new_axis)

    def finish_off_layouts(
        self, layouts, offset, path=PrettyTuple(), layout_path=PrettyTuple(), axis=None
    ):
        axis = axis or self.root
        # import pdb; pdb.set_trace()
        # since this search loops over all entries in the multi-axis
        # (not just the axis part) we need to execute the function if *any* axis part
        # requires tabulation.
        test1 = any(
            (subaxis := self.child_by_label(axis, cpt.label))
            and any(
                requires_external_index(self, subaxis, pt) for pt in subaxis.components
            )
            for cpt in axis.components
        )
        test2 = any(cpt.permutation for cpt in axis.components)

        # fixme very hard to read - the conditions above aren't quite right
        test3 = path not in layouts or layouts[path] != "null layout"
        if (test1 or test2) and test3:
            data = self.create_layout_lists(path, layout_path, offset, axis)

            # we shouldn't reproduce anything here
            # if any(layouts[k] is not None for k in data):
            #     raise AssertionError("shouldn't be doing things twice")
            # this is nasty hack - sometimes (ragged temporaries) we compute the offset
            # both as affine and also as a lookup table. We only want the former so drop
            # the latter here
            layouts |= {k: v for k, v in data.items() if layouts[k] is None}

        for part in axis.components:
            # zero offset if layout exists or if we are part of a temporary (and indexed)
            if layouts[path | (axis.name, part.label)] != "null layout" or part.indexed:
                subaxis_needs_part = False
                saved_offset = offset.value
                offset.value = 0
            else:
                subaxis_needs_part = True
                saved_offset = 0

            if subaxis := self.child_by_label(axis, part.label):
                if subaxis_needs_part:
                    new_layout_path = layout_path | (axis.name, part.label)
                else:
                    new_layout_path = PrettyTuple()
                self.finish_off_layouts(
                    layouts,
                    offset,
                    path | (axis.name, part.label),
                    new_layout_path,
                    subaxis,
                )

            offset.value += saved_offset

    _tmp_axis_namer = 0

    @classmethod
    def from_layout(cls, layout: Sequence[ConstrainedMultiAxis]) -> Any:  # TODO
        return order_axes(layout)

    def create_layout_lists(
        self, path, layout_path, offset, axis, indices=PrettyTuple()
    ):
        from pyop3.distarray import MultiArray

        npoints = 0
        # data = {}
        for part in axis.components:
            if part.has_integer_count:
                count = part.count
            else:
                count = part.count.get_value(indices)
                assert int(count) == count
                count = int(count)

            # data[npart] = [None] * count
            npoints += count

        new_layouts = collections.defaultdict(dict)

        # import pdb; pdb.set_trace()

        # taken from const function - handle starts
        myoff = offset.value
        found = set()
        for part in axis.components:
            # need to track if ragged permuted below
            if (
                has_independently_indexed_subaxis_parts(self, axis, part)
                and part.numbering is None
            ):
                new_layouts[path | (axis.name, part.label)] = (layout_path, myoff)
                found.add(part.label)

            if not has_fixed_size(self, axis, part):
                # can't do any more as things to the right of this will also move around
                break
            else:
                myoff += part.calc_size(self, axis)

        # if no numbering is provided create one
        # TODO should probably just set an affine one by default so we can mix these up
        if not strictly_all(cpt.numbering for cpt in axis.components):
            axis_numbering = []
            start = 0
            stop = start + axis.components[0].find_integer_count(indices)
            numb = np.arange(start, stop, dtype=PointerType)
            # don't set up to avoid recursion (should be able to use affine anyway)
            myaxes = MultiAxisTree()
            myaxes.register_node(MultiAxisNode([MultiAxisComponent(len(numb))]))
            myaxes.set_up()  # ???
            numb = MultiArray(
                dim=myaxes,
                name=f"ord{self._tmp_axis_namer}",
                data=numb,
                dtype=PointerType,
            )
            self._tmp_axis_namer += 1
            axis_numbering.append(numb)
            start = stop
            for i in range(1, len(axis.components)):
                stop = start + axis.components[i].find_integer_count(indices)
                numb = np.arange(start, stop, dtype=PointerType)
                myaxes = MultiAxisTree()
                myaxes.register_node(MultiAxis([MultiAxisComponent(len(numb))]))
                myaxes.set_up()  # ???
                numb = MultiArray(
                    dim=myaxes,
                    name=f"ord{self._tmp_axis_namer}_{i}",
                    data=numb,
                    dtype=PointerType,
                )
                axis_numbering.append(numb)
                start = stop
            self._tmp_axis_namer += 1
        else:
            axis_numbering = [cpt.numbering for cpt in axis.components]

        assert all(isinstance(num, MultiArray) for num in axis_numbering)

        for i in range(npoints):
            # TODO add a shortcut here to catch if i is inside the numbering of an affine
            # subpart.

            # find the right axis part and index thereof for the current 'global' numbering
            selected_part = None
            selected_part_label = None
            selected_index = None
            for part_num, axis_part in enumerate(axis.components):
                try:
                    # is the current global index found in the numbering of this axis part?
                    if axis_numbering[part_num].axes.depth > 1:
                        raise NotImplementedError("Need better indexing approach")
                    selected_index = list(axis_numbering[part_num].data).index(i)
                    selected_part = axis_part
                    selected_part_label = axis_part.label
                except ValueError:
                    continue
            if selected_part is None or selected_index is None:
                raise ValueError(f"{i} not found in any numberings")

            # skip those where we just set start and return an integer
            if selected_part_label in found:
                offset += step_size(self, axis, selected_part, indices | selected_index)
                continue

            if has_independently_indexed_subaxis_parts(self, axis, selected_part):
                # the layout is indirect and requires the current axis part if there only
                # if there is numbering
                if selected_part.numbering is not None:
                    new_layouts[path | (axis.name, selected_part_label)][
                        selected_index
                    ] = (
                        layout_path | (axis.name, selected_part_label),
                        offset.value,
                    )
                else:
                    new_layouts[path | (axis.name, selected_part_label)][
                        selected_index
                    ] = (
                        layout_path,
                        offset.value,
                    )
                # FIXME
                offset += step_size(self, axis, selected_part, indices | selected_index)
                # offset += step_size(self, selected_part, indices | selected_index)
            else:
                subaxis = self.child_by_label(axis, selected_part.label)
                assert subaxis
                subdata = self.create_layout_lists(
                    path | (axis.name, selected_part_label),
                    layout_path | (axis.name, selected_part_label),
                    offset,
                    subaxis,
                    indices | selected_index,
                )

                for subpath, subdata in subdata.items():
                    new_layouts[subpath][selected_index] = subdata

        # catch zero-sized sub-bits
        # for n in range(self.nparts):
        #     path_ = path | n
        #     if isinstance(new_layouts[path_], dict) and len(new_layouts[path_]) != npoints:
        #         assert len(new_layouts[path_]) < npoints
        #         for i in range(npoints):
        #             if i not in new_layouts[path_]:
        #                 new_layouts[path_][i] = []

        # now add the right labels for the layout (it might not need all of them)
        ret = {
            path_: self.unpack_index_dict(layout_)
            for path_, layout_ in new_layouts.items()
        }

        return ret

    @staticmethod
    def unpack_index_dict(idict):
        # catch just a number
        # if isinstance(idict, numbers.Integral):
        if isinstance(idict, tuple):
            return idict

        paths = set()
        ret = [None] * len(idict)
        for i, (new_path, j) in idict.items():
            ret[i] = j
            # these should all be the same value
            paths.add(new_path)

        assert not any(item is None for item in ret)
        return just_one(paths), ret

    def get_part_from_path(self, path, axis=None):
        axis = axis or self.root

        label, *sublabels = path

        (component,) = [
            cpt for cpt in axis.components if (axis.name, cpt.label) == label
        ]
        if sublabels:
            return self.get_part_from_path(
                sublabels, self.child_by_label(axis, component.label)
            )
        else:
            return axis, component

    _my_layout_namer = NameGenerator("layout")

    def turn_lists_into_layout_functions(self, layouts):
        from pyop3.distarray import MultiArray

        for path, layout in layouts.items():
            axis, part = self.get_part_from_path(path)
            if can_be_affine(self, axis, part):
                if isinstance(layout, tuple):
                    layout_path, layout = layout
                    name = self._my_layout_namer.next()
                    starts = MultiArray.from_list(
                        layout, layout_path, name, PointerType
                    )
                    step = step_size(self, axis, part)
                    layouts[path] = AffineLayoutFunction(step, starts)
            else:
                if layout == "null layout":
                    continue
                layout_path, layout = layout
                name = self._my_layout_namer.next()
                data = MultiArray.from_list(layout, layout_path, name, PointerType)
                layouts[path] = IndirectLayoutFunction(data)

    def _set_up(self, indices=PrettyTuple(), offset=None):
        # should probably set up constant layouts as a first step

        if not offset:
            offset = IntRef(0)

        # return a nested list structure or nothing. if the former then haven't set all
        # the layouts (needs) to be end of the loop

        # loop over all points in all parts of the multi-axis
        # initialise layout array per axis part
        # import pdb; pdb.set_trace()
        layouts = {}
        for cpt in self.root.components:
            layouts |= prepare_layouts(self, self.root, cpt)

        set_null_layouts(layouts, self)
        handle_const_starts(self, layouts)

        assert offset.value == 0
        self.finish_off_layouts(layouts, offset)

        self.turn_lists_into_layout_functions(layouts)

        for layout in layouts.values():
            if isinstance(layout, list):
                if any(item is None for item in layout):
                    raise AssertionError

        return layouts

    def drop_last(self):
        """Remove the last subaxis"""
        if not self.part.subaxis:
            return None
        else:
            return self.copy(
                parts=[self.part.copy(subaxis=self.part.subaxis.drop_last())]
            )

    def without_numbering(self):
        assert False, "dont touch?"
        return self.copy(parts=[pt.without_numbering() for pt in self.parts])

    @property
    def is_linear(self):
        """Return ``True`` if the multi-axis contains no branches at any level."""
        if self.nparts == 1:
            return self.part.subaxis.is_linear if self.part.subaxis else True
        else:
            return False

    def add_part(self, axis_id, *args):
        assert False, "dont touch?"
        if axis_id not in self._all_axis_ids:
            raise ValueError

        part = self._parse_part(*args)
        return self._add_part(axis_id, part)

    # old syntax - keep? add_subaxes?
    def add_subaxis(self, part_id, subaxes):
        self.add_nodes(subaxes, part_id)
        return self
        # return self._add_subaxis(part_id, subaxis)

    @property
    def _all_axis_ids(self):
        all_ids = [self.id]
        for part in self.parts:
            if part.subaxis:
                all_ids.extend(part.subaxis._all_axis_ids)

        if not has_unique_entries(all_ids):
            # TODO if duplicate entries exist
            raise NotImplementedError(
                """
Need to handle the case where axes have duplicated labels.

This can happen for example with matrices where the inner dimension can also be "cells",
"edges" etc. To get around this we should require that the label 'path' be specified
by the user as a tuple. E.g. ("cells", "edges"). We should only allow the syntactic sugar
of a single label value if that label is unique in the whole tree.
"""
            )
        return frozenset(all_ids)

    @property
    def _all_part_ids(self):
        all_ids = []
        for part in self.parts:
            if part.id is not None:
                all_ids.append(part.id)
            if part.subaxis:
                all_ids.extend(part.subaxis._all_part_ids)

        if len(unique(all_ids)) != len(all_ids):
            raise RuntimeError("ids must be unique")
        return frozenset(all_ids)

    def _add_part(self, axis_id, part):
        if axis_id == self.id:
            return self.copy(parts=self.parts + (part,))
        elif axis_id not in self._all_axis_ids:
            return self
        else:
            new_parts = []
            for pt in self.parts:
                if pt.subaxis:
                    new_subaxis = pt.subaxis._add_part(axis_id, part)
                    new_parts.append(pt.copy(subaxis=new_subaxis))
                else:
                    new_parts.append(pt)
            return self.copy(parts=new_parts)

    def _add_subaxis(self, part_id, subaxis):
        # TODO clean this up
        if part_id in self._all_part_ids:
            new_parts = []
            for part in self.parts:
                if part.id == part_id:
                    if part.subaxis:
                        raise RuntimeError("Already has a subaxis")
                    new_part = part.copy(subaxis=subaxis)
                else:
                    if part.subaxis:
                        new_part = part.copy(
                            subaxis=part.subaxis._add_subaxis(part_id, subaxis)
                        )
                    else:
                        new_part = part
                new_parts.append(new_part)
            return self.copy(parts=new_parts)
        else:
            return self

    @staticmethod
    def _parse_part(*args):
        if len(args) == 1 and isinstance(args[0], AxisPart):
            return args[0]
        else:
            return AxisPart(*args)

    @staticmethod
    def _parse_multiaxis(*args):
        if len(args) == 1 and isinstance(args[0], MultiAxis):
            return args[0]
        else:
            return MultiAxis(*args)


def get_slice_bounds(array, indices):
    from pyop3.distarray import MultiArray

    part = just_one(array.axes.children(array.axes.root))
    for _ in indices:
        part = just_one(array.axes.children(part))

    if isinstance(part.layout_fn, AffineLayoutFunction):
        if isinstance(part.layout_fn.start, MultiArray):
            start = part.layout_fn.start.get_value(indices)
        else:
            start = part.layout_fn.start
        size = part.calc_size(array.axes, indices)
    else:
        # I don't think that this ever happens. We only use IndirectLayoutFunctions when
        # we have numbering and that is not permitted with sparsity
        raise NotImplementedError

    return strict_int(start), strict_int(start + size)


def requires_external_index(axtree, axis, component):
    """Return ``True`` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index(
        axtree, axis, component
    ) or numbering_requires_external_index(axtree, axis, component)


def size_requires_external_index(axtree, node, part, depth=0):
    if not part.has_integer_count and part.count.dim.depth > depth:
        return True
    else:
        if child := axtree.child_by_label(node, part.label):
            for cpt in child.components:
                if size_requires_external_index(axtree, child, cpt, depth + 1):
                    return True
    return False


def numbering_requires_external_index(axtree, axis, component, depth=1):
    if component.numbering is not None and component.numbering.dim.depth > depth:
        return True
    else:
        if subaxis := axtree.child_by_label(axis, component.label):
            for cpt in subaxis.components:
                if numbering_requires_external_index(axtree, subaxis, cpt, depth + 1):
                    return True
    return False


def has_constant_step(axtree, node, part):
    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if child := axtree.child_by_label(node, part.label):
        return all(
            not size_requires_external_index(axtree, child, cpt)
            for cpt in child.components
        )
    else:
        return True


class MultiAxis(LabelledNode):
    fields = {"components", "label", "id"}

    _label_generator = UniqueNameGenerator("_MultiAxis_label")

    def __init__(self, components, label: Hashable | None = None, **kwargs):
        super().__init__(components, label, **kwargs)

    def __str__(self) -> str:
        return f"{self.name} : [{', '.join(str(cpt.label) for cpt in self.components)}]"

    # old alias
    @property
    def name(self):
        return self.label


# old alias
MultiAxisNode = MultiAxis


class MultiAxisComponent(LabelledNodeComponent):
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
        "numbering",
        "max_count",
        "layout_fn",
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
        numbering=None,
        max_count=None,
        layout_fn=None,
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

        if numbering is not None and not isinstance(numbering, MultiArray):
            # FIXME: This is an excellent argument for why the numbering should
            # be a property of the multi-axis, not the particular component
            # this name attribute, needed for indexing, is very nasty to determine here
            assert isinstance(numbering, tuple) and len(numbering) == 2
            axis_name, numbering = numbering

            if isinstance(numbering, np.ndarray):
                numbering = list(numbering)

            if isinstance(numbering, collections.abc.Sequence):
                numbering = MultiArray.from_list(
                    numbering,
                    [(axis_name, label)],
                    name=f"{axis_name}_{label}_num",
                    dtype=PointerType,
                )

        super().__init__(label)

        self.count = count
        self.indices = indices
        self.numbering = numbering
        self.max_count = max_count
        self.layout_fn = layout_fn
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
        return f"{self.__class__.__name__}(id={self.id}, label={self.label})"

    # old alias
    @property
    def name(self) -> str:
        return self.label

    @property
    def is_distributed(self):
        return self.overlap is not None

    @property
    def has_integer_count(self):
        return isinstance(self.count, numbers.Integral)

    def without_numbering(self):
        if self.subaxis:
            return self.copy(numbering=None, subaxis=self.subaxis.without_numbering())
        else:
            return self.copy(numbering=None)

    @property
    def is_ragged(self):
        from pyop3.distarray import MultiArray

        return isinstance(self.count, MultiArray)

    # deprecated alias
    @property
    def permutation(self):
        return self.numbering

    def alloc_size(self, axtree, axis):
        # TODO This should probably raise an exception if we do weird things with maps
        # (as in fix the layouts for reduced data movement)
        if self.count == -1:  # scalar thing
            count = 1
        elif not self.indexed:
            count = self.max_count
        else:
            count = 1
        if subaxis := axtree.child_by_label(axis, self.label):
            return count * axtree.alloc_size(subaxis)
        else:
            return count

    # TODO make a free function or something - this is horrible
    def calc_size(self, axtree, axis, indices=PrettyTuple()):
        extent = self.find_integer_count(indices)
        if subaxis := axtree.child_by_label(axis, self.label):
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

    def add_subaxis(self, part_id, subaxis):
        if part_id == self.id and self.subaxis:
            raise RuntimeError

        if part_id == self.id:
            return self.copy(subaxis=subaxis)
        else:
            return self.copy(subaxis=self.subaxis.add_subaxis(part_id, subaxis))


# what's the best name?
AxisPart = MultiAxisComponent


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


@dataclasses.dataclass
class SyncStatus:
    pending_write_op: Optional[Any] = None
    halo_valid: bool = True
    halo_modified: bool = False


"""
TODO I don't know if I need axis parts here. Perhaps just a multi-axis is a sufficient
abstraction.

Things might get simpler if I just defined some sort of null "root" node
"""


def fill_shape(axtree, axis, component, itree, iparent, prev_indices):
    from pyop3.distarray import MultiArray

    if isinstance(component.count, MultiArray):
        # turn prev_indices into a tree
        assert len(prev_indices) > 0
        mynewtree = IndexTree()
        prev = NullRootNode.ID
        for prev_idx in prev_indices:
            mynewtree.add_node(prev_idx, parent=prev)
            prev = prev_idx
        count = component.count[mynewtree]
    else:
        count = component.count
    new_index = RangeNode((axis.name, component.label), count)
    itree.add_node(new_index, parent=iparent)
    if subaxis := axtree.child_by_label(axis, component.label):
        for cpt in subaxis.components:
            fill_shape(axtree, subaxis, cpt, itree, new_index, prev_indices | new_index)


def expand_indices_to_fill_empty_shape(
    axis,
    itree,
    index,
    labels=PrettyTuple(),
    prev_indices=PrettyTuple(),
):
    if isinstance(index, RangeNode):
        new_labels = labels | index.label
    else:
        assert isinstance(index, MapNode)
        new_labels = labels[: -len(index.from_labels)] + index.to_labels

    if children := itree.children(index):
        for child in children:
            expand_indices_to_fill_empty_shape(
                axis, itree, child, new_labels, prev_indices | index
            )
    else:
        # traverse where we have got to to get the bottom unindexed bit
        myaxis = axis.root
        for label in new_labels:
            selected_part = just_one(
                [pt for pt in myaxis.components if (myaxis.name, pt.label) == label]
            )
            myaxis = axis.child_by_label(myaxis, selected_part.label)

        if myaxis:
            # this is controversial if we allow arbitrary ordering of axes
            raise AssertionError("dont want to do this any more")
            for part in myaxis.components:
                fill_shape(axis, myaxis, part, itree, index, prev_indices | index)


def create_lgmap(axes):
    # TODO: Fix imports
    from pyop3.multiaxis import Owned, Shared

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
    # TODO add missing bits to the mesh axes (zero-sized) here or,
    # more likely, in the dat constructor
    # axes = Tree()
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

    # should be able to delete
    # _check_constraints(tree)

    # now turn the tree into a proper multi-axis
    return axes
    return _create_multiaxis(axes)


"""
So constrained multiaxes should have a name associated with them so they can
be identified by PETSc options etc (e.g. 'field'). This is distinct from 'id' in the
tree because we can have multiple 'field' multiaxes if they are nested.

The problem that we have is that we sometimes have duplicated labels in our code
(e.g. if we have 2 meshes then there can be 2 "cell" labels) To resolve this I
think we should, in this function, store labels and a 2-tuple of (mesh name, label)
e.g. ("mesh0", "cells"). This is sufficiently unique.
"""


# class ConstrainedMultiAxisNode(Node):
#     fields = Node.fields | {"axis", "parent_label"}
#
#     _id_generator = UniqueNameGenerator("_id_ConstrainedMultiAxisNode")
#
#     def __init__(
#         self, axis, parent_label: Hashable | None = None, *, id: Hashable | None = None
#     ):
#         self.axis = axis
#         self.parent_label = parent_label
#         super().__init__(id or next(self._id_generator))
#
#     def __str__(self) -> str:
#         return f"{self.__class__.__name__}(axis={self.axis}, parent_label={self.parent_label})"


def _insert_axis(
    axes: MultiAxisTree,
    new_caxis: ConstrainedMultiAxis,
    current_axis: MultiAxis,
    axis_to_caxis: dict[MultiAxis, ConstrainedMultiAxis],
    path: dict[Hashable] | None = None,
):
    # breakpoint()
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
        # breakpoint()
        if new_caxis.within_labels <= within_labels:
            # diagram or something?
            parent_axis = axes.parent(current_axis)
            #
            # if parent_axis:
            #     mylabel = None
            #     for key, val in axes._parent_and_label_to_child.items():
            #         if val == current_axis.id:
            #             assert parent_axis.id == key[0]
            #             mylabel = key[1]
            #     assert mylabel
            # else:
            #     mylabel = None

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


def _check_constraints(ctree):
    def check(node, within_labels):
        if (parent := ctree.parent(node)) and parent.priority > node.priority:
            raise ConstraintsNotMetException
        if node.within_labels > within_labels:
            raise ConstraintsNotMetException

        return within_labels | {node.label}

    previsit(ctree, check, ctree.root, frozenset())


def _create_multiaxis(tree: Tree) -> MultiAxisTree:
    axes = MultiAxisTree()

    # note: constrained things contain multiple nodes
    def build(node, *_):
        if node == tree.root:
            target_parent = None
            parent_axis = None
        else:
            parent = tree.parent(node)
            target_parent = just_one(
                cpt
                for cpt in parent.data[0].axis.components
                if cpt.label == node.data[1]
            )
            parent_axis = parent.data[0].axis
        parent_component = node.data[1]
        axes.add_node(node.data[0].axis, (parent_axis, parent_component))

    previsit(tree, build)
    return axes

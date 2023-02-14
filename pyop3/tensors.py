import copy
import dataclasses
import enum
import functools
import numpy as np
import operator
import abc
import itertools
import collections
import dataclasses
from typing import Tuple, Union, Any, Optional, Sequence
import numbers
import threading

from mpi4py import MPI
from petsc4py import PETSc
import pymbolic as pym

import pytools
import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple, checked_zip, NameGenerator, unique, PrettyTuple, strictly_all, has_unique_entries
from pyop3 import utils


myprint = utils.print_with_rank


class IntRef:
    """Pass-by-reference integer."""
    def __init__(self, value):
        self.value = value

    def __iadd__(self, other):
        self.value += other
        return self

def get_bottom_part(axis):
    # must be linear
    if axis.part.subaxis:
        return get_bottom_part(axis.part.subaxis)
    else:
        return axis.part


def as_prepared_multiaxis(axis):
    if isinstance(axis, PreparedMultiAxis):
        return axis
    elif isinstance(axis, PreparedAxisPart):
        return PreparedMultiAxis(axis)
    else:
        raise TypeError


def as_multiaxis(axis):
    if isinstance(axis, MultiAxis):
        return axis
    elif isinstance(axis, AxisPart):
        return MultiAxis(axis)
    else:
        raise TypeError


def compute_offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)[:-1]], dtype=np.int32)


def is_set_up(axis):
    """Return ``True`` if all parts (recursively) of the multi-axis have an associated
    layout function.
    """
    return all(part_is_set_up(pt) for pt in axis.parts)


def part_is_set_up(part):
    if part.subaxis and not is_set_up(part.subaxis):
        return False
    if part.layout_fn is None:
        return False
    return True


def has_independently_indexed_subaxis_parts(part):
    """
    subaxis parts are independently indexed if they don't depend on the index from
    ``part``.

    if one sub-part needs this index to determine its extent then we need to create
    a layout function as the step sizes will differ.

    Note that we need to consider both ragged sizes and permutations here
    """
    if part.subaxis:
        return not any(requires_external_index(pt) for pt in part.subaxis.parts)
    else:
        return True


def only_linear(func):
    def wrapper(self, *args, **kwargs):
        if not self.is_linear:
            raise RuntimeError(f"{func.__name__} only admits linear multi-axes")
        return func(self, *args, **kwargs)
    return wrapper


def prepare_layouts(part, npart, path=None):
    """Make a magic nest of dictionaries for storing intermediate results."""
    # path is a tuple key for holding the different axis parts
    if not path:
        path = (npart,)

    # import pdb; pdb.set_trace()
    layouts = {path: None}

    if part.subaxis:
        for npt, subpart in enumerate(part.subaxis.parts):
            layouts |= prepare_layouts(subpart, npt, path+(npt,))

    # if not part.subaxis:
    #     layouts[path] = data.copy()
    # else:
        # FIXME do in _set_up instead of here (make this dict exhaustive)
        # # if any of the inner axes have a constant size then we need to tabulate here
        # # as the internal layout doesn't need this index
        # if any(not size_requires_external_index(pt) for pt in part.subaxis.parts):
        #     layouts[path] = data.copy()
        #
        # # recurse for any subparts for which the layouts requires this extra index to be
        # # fully specified
        # for npt, subpart in enumerate(part.subaxis.parts):
        #     if requires_external_index(subpart):
        #         layouts |= prepare_layouts(subpart, npt, data, path+(npt,))
    return layouts


def set_null_layouts(layouts, axis, path=()):
    """
    we have null layouts whenever the step in non-const (the variability is captured by
    the start property of the affine layout below it).
    """
    for npart, part in enumerate(axis.parts):
        new_path = path + (npart,)
        if not has_constant_step(part):
            layouts[new_path] = "null layout"

        if part.subaxis:
            set_null_layouts(layouts, part.subaxis, new_path)


def can_be_affine(part):
    return has_independently_indexed_subaxis_parts(part) and part.numbering is None


def handle_const_starts(axis, layouts, path=PrettyTuple()):
    offset = 0
    for i, part in enumerate(axis.parts):
        # need to track if ragged permuted below
        if can_be_affine(part):
            step = step_size(part)
            layouts[path|i] = AffineLayoutFunction(step, offset)

        if not has_fixed_size(part):
            # can't do any more as things to the right of this will also move around
            break
        else:
            offset += part.calc_size()

    for i, part in enumerate(axis.parts):
        if part.subaxis:
            handle_const_starts(part.subaxis, layouts, path|i)


def has_fixed_size(part):
    return not size_requires_external_index(part)


def step_size(part, indices=PrettyTuple()):
    if not has_constant_step(part) and not indices:
        raise ValueError
    return part.subaxis.calc_size(indices) if part.subaxis else 1


def attach_affine_layouts(layouts, part, npart, offset, path=None):
    """only for const start values"""
    if not path:
        path = (npart,)

    # we are affine if the step is constant and there is no numbering
    if has_constant_step(part) and part.numbering is None:
        step = step_size(part)
        start = offset.value
        assert layouts[path] is None
        layouts[path] = AffineLayoutFunction(step, start)

    # if layout is not null, zero the offset before recursing
    if layouts[path] != "null layout":
        old_offset = offset  # track a reference
        offset = IntRef(0)
    else:
        old_offset = None

    if part.subaxis:
        for npt, subpart in enumerate(part.subaxis.parts):
            attach_affine_layouts(layouts, subpart, npt, offset, path+(npt,))

            if isinstance(part.count, MultiArray):
                # note: this will fail if part.count is not an integer at the moment
                # but this is actually the behaviour that we want. If we have an array that
                # looks like
                # MultiAxis(AxisPart(5)).add_subaxis([AxisPart(nnz), AxisPart(3)])
                # then the outer axis part will have a lookup table striding by nnz[i]+3 but
                # the second inner axis will have regular steps of 1 (affine) but each loop
                # will start at nnz[i].
                # Remember: we need indirection maps for irregular strides, and affine functions
                # for regular ones. The actual start offset can be a lookup.
                # N.B. This could get even more complicated if we needed to add multiple
                # multi-arrays.

                # in fact, we can have a null layout for the outer axis and just use
                # a lookup for the nnz affine start part too.
                raise NotImplementedError("see comment")

            offset += part.calc_size()

    # restore the value
    if old_offset is not None:
        old_offset += offset.value


def attach_blank_data_to_layouts(layouts, part, npart, data, path=None):
    # path is a tuple key for holding the different axis parts
    if not path:
        path = (npart,)

    if layouts[path] is None:
        layouts[path] = data.copy()

    if part.subaxis:
        for npt, subpart in enumerate(part.subaxis.parts):
            attach_blank_data_to_layouts(layouts, subpart, npt, data, path+(npt,))


def attach_star_forest(axis, with_halo_points=True):
    comm = MPI.COMM_WORLD

    # 1. construct the point-to-point SF per axis part
    if axis.nparts != 1:
        raise NotImplementedError
    for part in axis.parts:
        part_sf = make_star_forest_per_axis_part(part, comm)

    # for now, will want to concat or something
    point_sf = part_sf

    # 2. broadcast the root offset to all leaves
    # TODO use a single buffer
    from_buffer = np.zeros(axis.part.count, dtype=np.uintp)
    to_buffer = np.zeros(axis.part.count, dtype=np.uintp)

    for pt, label in enumerate(axis.part.overlap):
        # only need to broadcast offsets for roots
        if isinstance(label, Shared) and not label.root:
            from_buffer[pt] = axis.get_offset((pt,))

    # TODO: It's quite bad to allocate a massive buffer when not much of it gets
    # moved. Perhaps good to use some sort of map and create a minimal SF.

    cdim = axis.part.subaxis.calc_size() if axis.part.subaxis else 1
    dtype, _ = get_mpi_dtype(np.dtype(np.uintp), cdim)
    bcast_args = dtype, from_buffer, to_buffer, MPI.REPLACE
    point_sf.bcastBegin(*bcast_args)
    point_sf.bcastEnd(*bcast_args)

    # 3. construct a new SF with these offsets
    nroots, _local, _remote = part_sf.getGraph()

    local_offsets = []
    remote_offsets = []
    i = 0
    for pt, label in enumerate(axis.part.overlap):
        # TODO not a nice check (is_leaf?)
        cond1 = not isinstance(label, Owned) and label.root
        if cond1:
            if with_halo_points or (not with_halo_points and isinstance(label, Shared)):
                local_offsets.append(axis.get_offset((pt,)))
                remote_offsets.append((_remote[i, 0], to_buffer[pt]))
            i += 1

    local_offsets = np.array(local_offsets, dtype=np.int32)
    remote_offsets = np.array(remote_offsets, dtype=np.int32)

    sf = PETSc.SF().create(comm)
    if not with_halo_points:
        utils.print_with_rank(f"local_offsets: {local_offsets}")
        utils.print_with_rank(f"remote_offsets: {remote_offsets}")
    sf.setGraph(nroots, local_offsets, remote_offsets)
    if with_halo_points:
        return axis.copy(sf=sf)
    else:
        return axis.copy(shared_sf=sf)


def make_star_forest_per_axis_part(part, comm):
    if part.is_distributed:
        # we have a root if a point is shared but doesn't point to another rank
        nroots = len([pt for pt in part.overlap if isinstance(pt, Shared) and not pt.root])

        # which local points are leaves?
        local_points = [i for i, pt in enumerate(part.overlap) if not isinstance(pt, Owned) and pt.root]

        # roots of other processes (rank, index)
        remote_points = utils.flatten([
            pt.root.as_tuple()
            for pt in part.overlap
            if not isinstance(pt, Owned) and pt.root
        ])

        # import pdb; pdb.set_trace()

        sf = PETSc.SF().create(comm)
        sf.setGraph(nroots, local_points, remote_points)
        return sf
    else:
        raise NotImplementedError(
            "Need to think about concatenating star forests. This will happen if mixed.")


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

# --------------------- \/ lifted from halo.py \/ -------------------------

from mpi4py import MPI
_MPI_types = {}


def get_mpi_dtype(numpy_dtype, cdim=1):
    """Get an MPI datatype corresponding to a Dat.

    This builds (if necessary a contiguous derived datatype of the
    correct size).

    Also returns if it is a builtin type.
    """
    key = (numpy_dtype, cdim)
    try:
        return _MPI_types[key]
    except KeyError:
        tdict = MPI._typedict
        try:
            btype = tdict[numpy_dtype.char]
        except KeyError:
            raise RuntimeError("Unknown base type %r", numpy_dtype)
        if cdim == 1:
            typ = btype
            builtin = True
        else:
            typ = btype.Create_contiguous(cdim)
            typ.Commit()
            builtin = False
        return _MPI_types.setdefault(key, (typ, builtin))


_numpy_types = {}


def get_numpy_dtype(datatype):
    """Get a numpy datatype corresponding to an MPI datatype.

    Only works for contiguous datatypes."""
    try:
        # possibly unsafe if handles are recycled, but OK, because we
        # hold on to the contig types
        return _numpy_types[datatype.py2f()]
    except KeyError:
        base, combiner, _ = datatype.decode()
        while combiner == "DUP":
            base, combiner, _ = base.decode()
        if combiner != "CONTIGUOUS":
            raise RuntimeError("Can only handle contiguous types")
        try:
            tdict = MPI.__TypeDict__
        except AttributeError:
            tdict = MPI._typedict

        tdict = dict((v.py2f(), k) for k, v in tdict.items())
        try:
            base = tdict[base.py2f()]
        except KeyError:
            raise RuntimeError("Unhandled base datatype %r", base)
        return _numpy_types.setdefault(datatype.py2f(), base)


def reduction_op(op, invec, inoutvec, datatype):
    dtype = get_numpy_dtype(datatype)
    invec = np.frombuffer(invec, dtype=dtype)
    inoutvec = np.frombuffer(inoutvec, dtype=dtype)
    inoutvec[:] = op(invec, inoutvec)


_contig_min_op = MPI.Op.Create(functools.partial(reduction_op, np.minimum), commute=True)
_contig_max_op = MPI.Op.Create(functools.partial(reduction_op, np.maximum), commute=True)

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


class Sparsity:
    def __init__(self, maps):
        if isinstance(maps, collections.abc.Sequence):
            rmap, cmap = maps
        else:
            rmap, cmap = maps, maps

        ...

        raise NotImplementedError


class MultiAxis(pytools.ImmutableRecord):
    fields = {"parts", "id", "parent", "is_set_up", "sf", "shared_sf"}

    id_generator = NameGenerator("ax")

    def __init__(self, parts, *, id=None, parent=None, is_set_up=False, sf=None, shared_sf=None):
        # make sure all parts have labels, default to integers if necessary
        if strictly_all(pt.label is None for pt in parts):
            # set the label to the index
            parts = tuple(pt.copy(label=i) for i, pt in enumerate(parts))

        if utils.some_but_not_all(pt.is_distributed for pt in parts):
            raise ValueError("Cannot have a multi-axis with some parallel parts and some not")

        if not has_unique_entries(pt.label for pt in parts):
            raise ValueError("Axis parts in the same multi-axis must have unique labels")

        self.parts = tuple(parts)
        self.id = id or self.id_generator.next()
        self.parent = parent
        self.is_set_up = is_set_up
        self.sf = sf
        self.shared_sf = shared_sf
        super().__init__()

    def __mul__(self, other):
        """Return the (dense) outer product."""
        return self.mul(other)

    @property
    def part(self):
        try:
            pt, = self.parts
            return pt
        except ValueError:
            raise RuntimeError

    def find_part(self, label):
        return self._parts_by_label[label]

    def get_offset(self, indices):
        # use layout functions to access the right thing
        # indices here are integers, so this will only work for multi-arrays that
        # are not multi-part
        # if self.is_multi_part:
        #   raise Exception("cannot index with integers here")

        # accumulate offsets from the layout functions
        offset = 0
        depth = 0
        axis = self

        # effectively loop over depth
        while axis:
            assert axis.nparts == 1

            # import pdb; pdb.set_trace()
            if axis.part.is_layout:
                if indices[depth] > 0:
                    if axis.part.subaxis:
                        # add the size of the one before
                        newidxs = indices[:-1] + (indices[-1]-1,)
                        offset += axis.part.subaxis.calc_size(newidxs)
                    else:
                        offset += 1
            else:
                layout = axis.part.layout_fn
                if isinstance(layout, IndirectLayoutFunction):
                    offset += layout.data.get_value(indices[:depth+1])
                elif layout == "null layout":
                    pass
                else:
                    assert isinstance(layout, AffineLayoutFunction)
                    if isinstance(layout.start, MultiArray):
                        start = layout.start.get_value(indices)
                    else:
                        start = layout.start
                    offset += indices[depth] * layout.step + start

            depth += 1
            axis = axis.part.subaxis

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

    @functools.cached_property
    def _parts_by_label(self):
        return {part.label: part for part in self.parts}

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

    def calc_size(self, indices=PrettyTuple()):
        # NOTE: this works because the size array cannot be multi-part, therefore integer
        # indices (as opposed to typed ones) are valid.
        return sum(pt.calc_size(indices) for pt in self.parts)

    @property
    def alloc_size(self):
        return sum(pt.alloc_size for pt in self.parts)

    @property
    def count(self):
        """Return the total number of entries in the axis across all axis parts.
        Will fail if axis parts do not have integer counts.
        """
        if not all(isinstance(pt.count, numbers.Integral) for pt in self.parts):
            raise RuntimeError()
        return sum(pt.count for pt in self.parts)

    # TODO ultimately remove this as it leads to duplication of data
    layout_namer = NameGenerator("layout")

    def set_up(self):
        """Initialise the multi-axis by computing the layout functions."""
        layouts = self._set_up()
        # import pdb; pdb.set_trace()
        new_axis = self.apply_layouts(layouts)
        assert is_set_up(new_axis)

        # set the .sf and .owned_sf properties of new_axis
        new_axis = attach_star_forest(new_axis)
        new_axis = attach_star_forest(new_axis, with_halo_points=False)
        # new_axis = attach_owned_star_forest(new_axis)
        return new_axis

    def apply_layouts(self, layouts, path=PrettyTuple()):
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
        new_parts = []
        for npart, part in enumerate(self.parts):
            pth = path | npart

            layout = layouts[pth]
            if part.subaxis:
                new_subaxis = part.subaxis.apply_layouts(layouts, pth)
            else:
                new_subaxis = None
            new_part = part.copy(layout_fn=layout, subaxis=new_subaxis)
            new_parts.append(new_part)
        return self.copy(parts=new_parts)

    def finish_off_layouts(self, layouts, offset, path=PrettyTuple()):

        # since this search loops over all entries in the multi-axis
        # (not just the axis part) we need to execute the function if *any* axis part
        # requires tabulation.
        test1 = any(
            part.subaxis and any(requires_external_index(pt) for pt in part.subaxis.parts)
            for part in self.parts
        )
        test2 = any(pt.permutation for pt in self.parts)

        # fixme very hard to read - the conditions above aren't quite right
        test3 = path not in layouts or layouts[path] != "null layout"
        # import pdb; pdb.set_trace()
        if (test1 or test2) and test3:
            data = self.create_layout_lists(path, offset)
            layouts |= data

        for npart, part in enumerate(self.parts):
            # zero offset if layout exists
            if layouts[path|npart] != "null layout":
                saved_offset = offset.value
                offset.value = 0
            else:
                saved_offset = 0

            if part.subaxis:
                part.subaxis.finish_off_layouts(layouts, offset, path|npart)

            offset.value += saved_offset

    def create_layout_lists(self, path, offset, indices=PrettyTuple()):
        npoints = 0
        # data = {}
        for npart, part in enumerate(self.parts):
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
        for i, part in enumerate(self.parts):
            # need to track if ragged permuted below
            if has_independently_indexed_subaxis_parts(part) and part.numbering is None:
                new_layouts[path|i] = myoff
                found.add(i)

            if not has_fixed_size(part):
                # can't do any more as things to the right of this will also move around
                break
            else:
                myoff += part.calc_size()


        # if no numbering is provided create one
        # TODO should probably just set an affine one by default so we can mix these up
        # import pdb; pdb.set_trace()
        if not strictly_all(pt.numbering is not None for pt in self.parts):
            axis_numbering = []
            start = 0
            stop = start + self.parts[0].find_integer_count(indices)
            numb = np.arange(start, stop, dtype=np.uintp)
            # don't set up to avoid recursion (should be able to use affine anyway)
            axis = MultiAxis([AxisPart(len(numb))])
            numb = MultiArray(dim=axis,name=f"{self.id}_ord0", data=numb, dtype=np.uintp)
            axis_numbering.append(numb)
            start = stop
            for i in range(1, self.nparts):
                stop = start + self.parts[i].find_integer_count(indices)
                numb = np.arange(start, stop, dtype=np.uintp)
                axis = MultiAxis([AxisPart(len(numb))])
                numb = MultiArray(dim=axis,name=f"{self.id}_ord{i}", data=numb, dtype=np.uintp)
                axis_numbering.append(numb)
                start = stop
        else:
            axis_numbering = [pt.numbering for pt in self.parts]

        assert all(isinstance(num, MultiArray) for num in axis_numbering)

        for i in range(npoints):
            # TODO add a shortcut here to catch if i is inside the numbering of an affine
            # subpart.

            # find the right axis part and index thereof for the current 'global' numbering
            selected_part = None
            selected_part_num = None
            selected_index = None
            for part_num, axis_part in enumerate(self.parts):
                try:
                    # is the current global index found in the numbering of this axis part?
                    if axis_numbering[part_num].depth > 1:
                        raise NotImplementedError("Need better indexing approach")
                    selected_index = list(axis_numbering[part_num].data).index(i)
                    selected_part = axis_part
                    selected_part_num = part_num
                except ValueError:
                    continue
            if selected_part is None or selected_index is None:
                raise ValueError(f"{i} not found in any numberings")

            # skip those where we just set start and return an integer
            if selected_part_num in found:
                offset += step_size(selected_part, indices|selected_index)
                continue

            if has_independently_indexed_subaxis_parts(selected_part):
                new_layouts[path|selected_part_num][selected_index] = offset.value
                offset += step_size(selected_part, indices|selected_index)
            else:
                assert selected_part.subaxis
                subdata = selected_part.subaxis.create_layout_lists(
                    path|selected_part_num, offset, indices|selected_index
                )

                for subpath, subdata in subdata.items():
                    new_layouts[subpath][selected_index] = subdata

        # catch zero-sized sub-bits
        # import pdb; pdb.set_trace()
        # for n in range(self.nparts):
        #     path_ = path | n
        #     if isinstance(new_layouts[path_], dict) and len(new_layouts[path_]) != npoints:
        #         assert len(new_layouts[path_]) < npoints
        #         for i in range(npoints):
        #             if i not in new_layouts[path_]:
        #                 new_layouts[path_][i] = []

        # import pdb; pdb.set_trace()
        ret = {path_: self.unpack_index_dict(layout_) for path_, layout_ in new_layouts.items()}
        return ret

    @staticmethod
    def unpack_index_dict(idict):
        # catch just a number
        if isinstance(idict, numbers.Integral):
            return idict

        ret = [None] * len(idict)
        for i, j in idict.items():
            ret[i] = j

        assert not any(item is None for item in ret)
        return ret

    def get_part_from_path(self, path):
        pidx, *subpath = path

        part = self.parts[pidx]
        if subpath:
            return part.subaxis.get_part_from_path(subpath)
        else:
            return part

    def turn_lists_into_layout_functions(self, layouts):
        for path, layout in layouts.items():
            part = self.get_part_from_path(path)
            if can_be_affine(part):
                if isinstance(layout, list):
                    starts = MultiArray.from_list(layout, f"layout_{'_'.join(str(p) for p in path)}", np.uintp)
                    step = step_size(part)
                    layouts[path] = AffineLayoutFunction(step, starts)
            else:
                if layout == "null layout":
                    continue
                assert isinstance(layout, list)
                data = MultiArray.from_list(layout, f"layout_{'_'.join(str(p) for p in path)}", np.uintp)
                layouts[path] = IndirectLayoutFunction(data)

    def _set_up(self, indices=PrettyTuple(), offset=None):
        # should probably set up constant layouts as a first step

        if not offset:
            offset = IntRef(0)

        # return a nested list structure or nothing. if the former then haven't set all
        # the layouts (needs) to be end of the loop

        # import pdb; pdb.set_trace()

        # loop over all points in all parts of the multi-axis
        # initialise layout array per axis part
        layouts = {}
        for npart, part in enumerate(self.parts):
            layouts |= prepare_layouts(part, npart)

        set_null_layouts(layouts, self)
        handle_const_starts(self, layouts)

        # myoff = IntRef(offset.value)
        # npoints = 0
        # for npart, part in enumerate(self.parts):
        #     if part.has_integer_count:
        #         count = part.count
        #     else:
        #         count = part.count.get_value(indices)
        #
        #     attach_affine_layouts(layouts, part, npart, myoff)
        #
        #     data = [None] * count
        #
        #     attach_blank_data_to_layouts(layouts, part, npart, data)
        #     npoints += count

            # I think I can avoid this as modified in the function - should pass explicitly
            # myoff += 


        # import pdb; pdb.set_trace()
        assert offset.value == 0
        self.finish_off_layouts(layouts, offset)

        # import pdb; pdb.set_trace()
        self.turn_lists_into_layout_functions(layouts)

        # import pdb; pdb.set_trace()

        for layout in layouts.values():
            if isinstance(layout, list):
                if any(item is None for item in layout):
                    raise AssertionError

        return layouts

        # now convert valid layouts to actual layout functions. skip those who still
        # need extra indices (handled up the stack)
        for npart, part in enumerate(self.parts):
            if not requires_external_index(part):
                # TODO complicated traversal to build layout functions
                path = (npart,)

                # sometimes we can have zero-sized arrays (ragged)
                if len(layouts[path]) == 0:
                    step = 1  # this kinda ick
                    start = 0
                    new_layout = AffineLayoutFunction(step, start=start)
                elif len(layouts[path]) == 1:
                    step = 1  # this kinda ick
                    start = layouts[path][0]
                    new_layout = AffineLayoutFunction(step, start=start)
                else:
                    steps = [
                        layouts[path][i+1] - layouts[path][i] for i in range(len(layouts[path])-1)
                    ]
                    if pytools.is_single_valued(steps):
                        step = pytools.single_valued(steps)
                        start = layouts[path][0]
                        new_layout = AffineLayoutFunction(step, start=start)
                    else:
                        new_layout = IndirectLayoutFunction(
                            MultiArray.from_list(layouts[path], f"{self.id}_ord{part_num}", np.uintp)
                        )

                layouts[path] = new_layout

        return layouts

    def drop_last(self):
        """Remove the last subaxis"""
        if not self.part.subaxis:
            return None
        else:
            return self.copy(parts=[self.part.copy(subaxis=self.part.subaxis.drop_last())])

    def without_numbering(self):
        return self.copy(parts=[pt.without_numbering() for pt in self.parts])

    @functools.cached_property
    def is_linear(self):
        """Return ``True`` if the multi-axis contains no branches at any level."""
        if self.nparts == 1:
            return self.part.subaxis.is_linear if self.part.subaxis else True
        else:
            return False

    @only_linear
    @functools.cached_property
    def depth(self):
        if subaxis := self.part.subaxis:
            return subaxis.depth + 1
        else:
            return 1

    def as_layout(self):
        new_parts = tuple(pt.as_layout() for pt in self.parts)
        return self.copy(parts=new_parts)

    def set_up_terminal(self, subaxes, indices, depth):
        if any(pt.permutation is not None for pt in self.parts):
            assert (
                sorted(sum([list(pt.permutation) for pt in self.parts], []))
                == np.arange(sum(pt.find_integer_count(indices) for pt in self.parts), dtype=int),
                "permutations must be exhaustive"
            )

        layout_fn_per_part = []

        # layout functions are not needed if no numbering is specified (i.e. they are just
        # contiguous) and if they are not ragged 'below'
        # import pdb; pdb.set_trace()
        if strictly_all(pt.numbering is None and pt.has_constant_step for pt in self.parts):
            start = 0
            for part, subaxis in checked_zip(self.parts, subaxes):
                # TODO This will fail is subaxis is ragged
                if subaxis:
                    step = subaxis.calc_size(indices)
                else:
                    step = 1
                layout_fn = AffineLayoutFunction(step, start)
                layout_fn_per_part.append(layout_fn)

                # TODO this will fail if things are ragged - need to store starts as
                # expressions somehow
                start += part.calc_size(indices)
            return layout_fn_per_part

        # initialise layout array per axis part
        # note that this just needs to be a numpy array. the clever indexing into trees
        # happens outside this function
        layouts = tuple(
            np.zeros(pt.find_integer_count(indices), dtype=np.uintp)
            for pt in self.parts
        )

        # if no numbering is provided create one
        if not strictly_all(pt.numbering is not None for pt in self.parts):
            axis_numbering = []
            start = 0
            stop = self.parts[0].find_integer_count(indices)
            numb = np.arange(start, stop, dtype=np.uintp)
            axis_numbering.append(numb)
            for i in range(1, self.nparts):
                numb = np.arange(start, stop, dtype=np.uintp)
                axis_numbering.append(numb)
                start = stop
                stop += self.parts[i].find_integer_count(indices)
        else:
            axis_numbering = [pt.numbering for pt in self.parts]


        axis_length = sum(pt.find_integer_count(indices) for pt in self.parts)
        offset = 0
        for current_idx in range(axis_length):
            # find the right axis part and index thereof for the current 'global' numbering
            selected_part_num = None
            selected_index = None
            for part_num, axis_part in enumerate(self.parts):
                try:
                    # is the current global index found in the numbering of this axis part?
                    # FIXME this will likely break as numbering is not an array - need to implement this fn
                    selected_index = list(axis_numbering[part_num]).index(current_idx)
                    selected_part_num = part_num
                except ValueError:
                    continue

            if selected_part_num is None or selected_index is None:
                assert selected_part_num is None and selected_index is None, "must be both"
                raise ValueError(f"{current_idx} not found in any numberings")

            # now store the offset in the right place
            layouts[selected_part_num][selected_index] = offset

            # lastly increment the pointer
            if subaxes[selected_part_num]:
                # FIXME but if nested this won't work
                offset += subaxes[selected_part_num].calc_size(indices|selected_index)
            else:
                offset += 1

        return layouts

    def add_part(self, axis_id, *args):
        if axis_id not in self._all_axis_ids:
            raise ValueError

        part = self._parse_part(*args)
        return self._add_part(axis_id, part)

    def add_subaxis(self, part_id, *args):
        if part_id not in self._all_part_ids:
            raise ValueError

        subaxis = self._parse_multiaxis(*args)
        return self._add_subaxis(part_id, subaxis)

    @functools.cached_property
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

    @functools.cached_property
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
            return self.copy(parts=self.parts+(part,))
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
                        new_part = part.copy(subaxis=part.subaxis._add_subaxis(part_id, subaxis))
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


    @staticmethod
    def _parse_part(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], PreparedAxisPart):
            return args[0]
        else:
            return PreparedAxisPart(*args, **kwargs)

    @staticmethod
    def _parse_multiaxis(*args):
        if len(args) == 1 and isinstance(args[0], PreparedMultiAxis):
            return args[0]
        else:
            return PreparedMultiAxis(*args)

def requires_external_index(part):
    """Return ``True`` if more indices are required to index the multi-axis layouts
    than exist in the given subaxis.
    """
    return size_requires_external_index(part) or numbering_requires_external_index(part)

def size_requires_external_index(part, depth=0):
    if not part.has_integer_count and part.count.depth > depth:
        return True
    else:
        if part.subaxis:
            for pt in part.subaxis.parts:
                if size_requires_external_index(pt, depth+1):
                    return True
    return False

def numbering_requires_external_index(part, depth=1):
    if part.numbering is not None and part.numbering.depth > depth:
        return True
    else:
        if part.subaxis:
            for pt in part.subaxis.parts:
                if numbering_requires_external_index(pt, depth+1):
                    return True
    return False

def has_constant_step(part):
    # we have a constant step if none of the internal dimensions need to index themselves
    # with the current index (numbering doesn't matter here)
    if part.subaxis:
        return all(not size_requires_external_index(pt) for pt in part.subaxis.parts)
    else:
        return True

PreparedMultiAxis = MultiAxis




class AbstractAxisPart(pytools.ImmutableRecord, abc.ABC):
    fields = {"count", "subaxis", "numbering", "label", "id", "max_count", "is_layout", "layout_fn", "overlap", "overlap_sf"}

    id_generator = NameGenerator("_p")

    def __init__(self, count, subaxis=None, *, numbering=None, label=None, id=None, max_count=None, is_layout=False, layout_fn=None, overlap=None):
        if isinstance(count, numbers.Integral):
            assert not max_count or max_count == count
            max_count = count
        else:
            if max_count is not None:
                max_count = max(count.data)

        if isinstance(numbering, np.ndarray):
            numbering = list(numbering)

        if isinstance(numbering, collections.abc.Sequence):
            numbering = MultiArray.from_list(numbering, name=f"{id}_ord", dtype=np.uintp)

        if not isinstance(label, collections.abc.Hashable):
            raise ValueError("Provided label must be hashable")

        if not id:
            id = self.id_generator.next()

        self.count = count
        self.subaxis = subaxis
        self.numbering = numbering
        self.label = label
        self.id = id
        self.max_count = max_count
        self.is_layout = is_layout
        self.layout_fn = layout_fn
        self.overlap = overlap
        super().__init__()

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

    def as_layout(self):
        new_subaxis = self.subaxis.as_layout() if self.subaxis else None
        return self.copy(is_layout=True, layout_fn=None, subaxis=new_subaxis)

    @property
    def is_ragged(self):
        return isinstance(self.count, MultiArray)

    # deprecated alias
    @property
    def permutation(self):
        return self.numbering

    @property
    def alloc_size(self):
        # TODO This should probably raise an exception if we do weird things with maps
        if self.subaxis:
            return self.max_count * self.subaxis.alloc_size
        else:
            return self.max_count

    def calc_size(self, indices=PrettyTuple()):
        extent = self.find_integer_count(indices)
        if self.subaxis:
            return sum(self.subaxis.calc_size(indices|i) for i in range(extent))
        else:
            return extent

    def find_integer_count(self, indices=PrettyTuple()):
        if isinstance(self.count, MultiArray):
            return self.count.get_value(indices)
        else:
            assert isinstance(self.count, numbers.Integral)
            return self.count


class AxisPart(AbstractAxisPart):
    fields = AbstractAxisPart.fields
    def __init__(self, count, subaxis=None, **kwargs):
        if subaxis:
            subaxis = as_multiaxis(subaxis)
        super().__init__(count, subaxis, **kwargs)

    def add_subaxis(self, part_id, subaxis):
        if part_id == self.id and self.subaxis:
            raise RuntimeError

        if part_id == self.id:
            return self.copy(subaxis=subaxis)
        else:
            return self.copy(subaxis=self.subaxis.add_subaxis(part_id, subaxis))


PreparedAxisPart = AxisPart




# not used
class ExpressionTemplate:
    """A thing that evaluates to some collection of loopy instructions when
    provided with the right inames.

    Useful for (e.g.) map0_getSize() since function calls are not allowed for GPUs.
    """

    def __init__(self, fn):
        self._fn = fn
        """Callable taking indices that evaluates to a pymbolic expression"""

    def generate(self, _):
        pass


class Index:
    def __init__(self, *args):
        raise NotImplementedError("deprecated")


class IndexSet(pytools.ImmutableRecord):
    """A set of entries to iterate over."""
    fields = {"size", "subset_indices"}

    def __init__(self, size, subset_indices=None):
        self.size = size
        self.subset_indices = subset_indices
        """indices is not None if we are dealing with a subset (e.g. mesh.interior_facets)"""


class TypedIndex(pytools.ImmutableRecord):
    fields = {"part_label", "iset", "id"}

    _id_generator = NameGenerator(prefix="typed_idx")

    def __init__(self, part_label: collections.abc.Hashable, iset: IndexSet, id=None):
        self.part_label = part_label
        self.iset = iset
        self.id = id or self._id_generator.next()

    @property
    def part(self):
        import warnings
        warnings.warn("use part_label now", DeprecationWarning)
        return self.part_label


class MultiIndex(pytools.ImmutableRecord):
    fields = {"typed_indices"}

    def __init__(self, typed_indices):
        if any(not isinstance(idx, TypedIndex) for idx in typed_indices):
            raise TypeError
        self.typed_indices = tuple(typed_indices)

    def __iter__(self):
        return iter(self.typed_indices)

    @property
    def depth(self):
        return len(self.indices)


class MultiIndexCollection(pytools.ImmutableRecord, abc.ABC):
    fields = {"multi_indices"}

    def __init__(self, multi_indices):
        if not all(isinstance(idx, MultiIndex) for idx in multi_indices):
            raise ValueError

        self.multi_indices = tuple(multi_indices)

    def __iter__(self):
        return iter(self.multi_indices)


# class Index(pytools.ImmutableRecord, abc.ABC):
#     fields = {"nparts", "sizes"}
#
#     def __init__(self, parts, sizes, depth: int=1):
#         self.parts = parts
#         """List of integers selecting the parts produced by this index."""
#         self.sizes = sizes
#         """Function returning an integer given a part number describing the size of the loop."""
#
#         self.depth = depth
#         """The multi-index size"""
#         super().__init__()


class Map(MultiIndexCollection):
    fields = MultiIndexCollection.fields | {"from_multi_indices"}

    def __init__(self, multi_indices, from_multi_indices):
        super().__init__(multi_indices=multi_indices)
        self.from_multi_indices = from_multi_indices

    def __mul__(self, other):
        """The product of two maps produces a sparsity."""
        if isinstance(other, Map):
            return self.mul(other)
        else:
            return NotImplemented

    def mul(self, other, is_nonzero=None):
        """is_nonzero is a function letting us exploit additional sparsity.

        Something like:

            for cell in cells:
                # rdof and cdof are multi-indices
                for rdof in self.dofs[cell]:
                    for cdof in other.dofs[cell]:
                        # maybe have extra loops here if we have multiple DoFs per entity
                        if is_nonzero(rdof, cdof):
                            # store the non-zero
        """
        ...
        raise NotImplementedError
        return Sparsity(...)

class Slice(Map):
    fields = Map.fields | {"start", "step"}

    def __init__(self, indices, from_indices, start=None, step=None):
        # FIXME need to think about how slices with starts and steps work
        # with multi-part axes
        if start or step:
            raise NotImplementedError

        super().__init__(indices, from_indices)
        self.start = start
        self.step = step


class IndirectMap(Map):
    fields = Map.fields | {"data"}

    def __init__(self, indices, from_indices, data):
        super().__init__(indices, from_indices)
        self.data = data


# TODO need to specify the output types I reckon - parents can vary but base outputs
# are absolutely needed.
# class Map(Index):
#     fields = Index.fields | {"from_index", "to"}
#
#     def __init__(self, from_, depth: int, parts, sizes, to):
#         if depth != from_.depth:
#             raise ValueError("Can only map between multi-indices of the same size")
#
#         super().__init__(parts=parts,sizes=sizes, depth=depth)
#         """The number of indices 'consumed' by this map"""
#
#         self.from_index = from_
#         """The input multi-index mapped from"""
#
#         self.to = to
#         """A function mapping between multi-indices"""


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


# class IndexFunction(Map):
#     """The idea here is that we provide an expression, say, "2*x0 + x1 - 3"
#     and then use pymbolic maps to replace the xN with the correct inames for the
#     outer domains. We could also possibly use pN (or pym.var subclass called Parameter)
#     to describe parameters."""
#     fields = Map.fields | {"expr", "vars"}
#     def __init__(self, expr, arity, vars, **kwargs):
#         """
#         vardims:
#             iterable of 2-tuples of the form (var, label) where var is the
#             pymbolic Variable in expr and label is the dim label associated with
#             it (needed to select the right iname) - note, this is ordered
#         """
#         self.expr = expr
#         self.vars = as_tuple(vars)
#         super().__init__(arity=arity, **kwargs)
#
#     @property
#     def size(self):
#         return self.arity


class NonAffineMap(Map):
    fields = Map.fields | {"tensor"}

    # TODO is this ever not valid?
    offset = 0

    def __init__(self, tensor, **kwargs):
        self.tensor = tensor

        # TODO this is AWFUL
        arity_ = self.tensor.indices[-1].size
        if "arity" in kwargs:
            assert arity_ == kwargs["arity"] 
            super().__init__(**kwargs)
        else:
            super().__init__(arity=arity_, **kwargs)

    @property
    def input_indices(self):
        return self.tensor.indices[:-1]

    @property
    def map(self):
        return self.tensor


@dataclasses.dataclass
class SyncStatus:
    pending_write_op: Optional[Any] = None
    halo_valid: bool = True
    halo_modified: bool = False


# TODO this shouldn't be an immutable record
class MultiArray(pym.primitives.Variable, pytools.ImmutableRecordWithoutPickling):
    """Multi-dimensional, hierarchical array.

    Parameters
    ----------
    sf : ???
        PETSc star forest connecting values (offsets) in the local array with
        remote equivalents.

    """

    fields = {"dim", "indices", "dtype", "mesh", "name", "data", "max_value", "sf"}

    name_generator = pyop3.utils.MultiNameGenerator()
    prefix = "ten"

    def __init__(self, dim, indices=None, dtype=None, *, mesh = None, name: str = None, prefix: str=None, data=None, max_value=32, sf=None):
        dim = as_prepared_multiaxis(dim)

        # TODO raise NotImplementedError if the multi-axis contains multiple parallel axes

        if not isinstance(dim, PreparedMultiAxis):
            raise ValueError("dim needs to be prepared. call .set_up()")

        self.data = data
        self.params = {}
        self._param_namer = NameGenerator(f"{name}_p")
        assert dtype is not None

        self.dim = dim
        # if not self._is_valid_indices(indices, dim.root):
        # assert all(self._is_valid_indices(idxs, dim) for idxs in indicess)
        self.indices = indices or MultiIndexCollection(self._extend_multi_index(None))

        self.mesh = mesh
        self.dtype = np.dtype(dtype)  # since np.float64 is not actually the right thing
        self.max_value = max_value
        self.sf = sf
        super().__init__(name)

        self._pending_write_op = None
        self._halo_modified = False
        self._halo_valid = True

        self._sync_thread = None

    # TODO delete this and just use constructor
    @classmethod
    def new(cls, dim, indices=None, *args, prefix=None, name=None, **kwargs):
        name = name or cls.name_generator.next(prefix or cls.prefix)

        dim = as_prepared_multiaxis(dim)

        # if not indicess:
        #     indicess = cls._fill_with_slices(dim)
        # else:
        #     if not isinstance(indicess[0], collections.abc.Sequence):
        #         indicess = (indicess,)
        #     indicess = [cls._parse_indices(dim, idxs) for idxs in indicess]

        # dim = cls.compute_layouts(dim)

        return cls(dim, indices, *args, name=name, **kwargs)

    @classmethod
    def compute_part_size(cls, part):
        size = 0
        if isinstance(part.size, numbers.Integral):
            return part.size
        return size

    @classmethod
    def from_list(cls, data, name, dtype, inc=0):
        """Return a (linear) multi-array formed from a list of lists."""
        flat, count = cls._get_count_data(data)

        if isinstance(count, list):
            count = cls.from_list(count, name, dtype, inc+1)

        flat = np.array(flat, dtype=dtype)

        axis = MultiAxis([AxisPart(count)])
        if isinstance(count, MultiArray):
            base_part = get_bottom_part(count.root)
            axis = count.root.add_subaxis(base_part.id, axis)
        return cls(axis.set_up(), name=f"{name}_{inc}", data=flat, dtype=dtype)

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

    reduction_ops = {
        pyop3.exprs.INC: MPI.SUM,
    }

    def reduce_leaves_to_roots(self, sf, pending_write_op):
        mpi_dtype, _ = get_mpi_dtype(self.data.dtype)
        mpi_op = self.reduction_ops[pending_write_op]
        args = (mpi_dtype, self.data, self.data, mpi_op)
        sf.reduceBegin(*args)
        sf.reduceEnd(*args)

    def broadcast_roots_to_leaves(self, sf):
        mpi_dtype, _ = get_mpi_dtype(self.data.dtype)
        mpi_op = MPI.REPLACE
        args = (mpi_dtype, self.data, self.data, mpi_op)
        sf.bcastBegin(*args)
        sf.bcastEnd(*args)

    def sync_begin(self, need_halo_values=False):
        """Begin synchronizing shared data."""
        self._sync_thread = threading.Thread(
            target=self.__class__.sync,
            args=(self,),
            kwargs={"need_halo_values": need_halo_values},
        )
        self._sync_thread.start()

    def sync_end(self):
        """Finish synchronizing shared data."""
        if not self._sync_thread:
            raise RuntimeError(
                "Cannot call sync_end without a prior call to sync_begin")
        self._sync_thread.join()
        self._sync_thread = None

    # TODO create Synchronizer object for encapsulation?
    def sync(self, need_halo_values=False):
        """Perform halo exchanges to ensure that all ranks store up-to-date values.

        Parameters
        ----------
        need_halo_values : bool
            Whether or not halo values also need to be synchronized.

        Notes
        -----
        This is a blocking operation. For the non-blocking alternative use
        :meth:`sync_begin` and :meth:`sync_end` (FIXME)

        Note that this method should only be called when one needs to read from
        the array.
        """
        # 1. Reduce leaf values to roots if they have been written to.
        # (this is basically local-to-global)
        if self._pending_write_op:
            assert not self._halo_valid, "If a write is pending the halo cannot be valid"
            # If halo entries have also been written to then we need to use the
            # full SF containing both shared and halo points. If the halo has not
            # been modified then we only need to reduce with shared points.
            if self._halo_modified:
                self.reduce_leaves_to_roots(self.root.sf, self._pending_write_op)
            else:
                # only reduce with values from owned points
                self.reduce_leaves_to_roots(self.root.shared_sf, self._pending_write_op)

        # implicit barrier? can only broadcast reduced values

        # 3. at this point only one of the owned points knows the correct result which
        # now needs to be scattered back to some (but not necessarily all) of the other ranks.
        # (this is basically global-to-local)

        # only send to halos if we want to read them and they are out-of-date
        if need_halo_values and not self._halo_valid:
            # send the root value back to all the points
            self.broadcast_roots_to_leaves(self.root.sf)
            self._halo_valid = True  # all the halo points are now up-to-date
        else:
            # only need to update owned points if we did a reduction earlier
            if self._pending_write_op:
                # send the root value back to just the owned points
                self.broadcast_roots_to_leaves(self.root.shared_sf)
                # (the halo is still dirty)

        # set self.last_op to None here? what if halo is still off?
        # what if we read owned values and then owned+halo values?
        # just perform second step
        self._pending_write_op = None
        self._halo_modified = False


    # @classmethod
    # def compute_layouts(cls, axis):
    #     if axis.permutation:
    #         layouts = cls.make_offset_map(axis)
    #     else:
    #         layouts = [None] * len(axis.parts)
    #
    #     new_parts = []
    #     offset = 0  # for mixed
    #     for part, mylayout in zip(axis.parts, layouts):
    #         if isinstance(part, ScalarAxisPart):
    #             # FIXME may not work with mixed
    #             new_part = part
    #             offset += 1
    #         else:
    #             subaxis = cls.compute_layouts(part.subaxis) if part.subaxis else None
    #
    #             if axis.permutation:
    #                 layout = mylayout, 0  # offset here is always 0 as accounted for in map
    #                 # import pdb; pdb.set_trace()
    #             else:
    #                 if isinstance(part.size, pym.primitives.Expression):
    #                     offsets = compute_offsets(part.size.data)
    #                     layout = part.size.copy(name=part.size.name+"c", data=offsets), offset
    #                 else:
    #                     layout = part.size, offset
    #             new_part = part.copy(layout=layout, subaxis=subaxis)
    #             # import pdb; pdb.set_trace()
    #             offset += cls._compute_full_part_size(part)
    #
    #         new_parts.append(new_part)
    #     return axis.copy(parts=new_parts)

    @classmethod
    def _get_part_size(cls, part, parent_indices):
        size = part.size
        if isinstance(size, numbers.Integral):
            return size
        elif isinstance(size, MultiArray):
            return cls._read_tensor(size, parent_indices)
        else:
            raise TypeError

    @classmethod
    def make_offset_map(cls, axis):
        offsets = collections.defaultdict(dict)
        offset = 0
        npoints = sum(p.size for p in axis.parts)
        for pt in range(npoints):
            pt = axis.permutation[pt]

            npart, part = cls._get_subdim(axis, pt)
            offsets[npart][pt] = offset

            # increment the pointer by the size of the step for this subdim
            # FIXME This does not work for ragged as the wrong result is returned here...
            if part.subaxis:
                offset += cls._compute_full_axis_size(part.subaxis, [pt])
            else:
                offset += 1

        layouts = []
        for npart in sorted(offsets):
            idxs = np.array([offsets[npart][i] for i in sorted(offsets[npart])], dtype=np.int32)
            new_section = MultiArray.new(MultiAxis(len(idxs)), data=idxs, prefix="sec", dtype=np.int32)
            layouts.append(new_section)

        return layouts

    @classmethod
    def _generate_looping_indices(cls, part):
        if isinstance(part.size, numbers.Integral):
            return [([], range(part.size))]
        else:
            result = []
            for parent_indices in part.size.mygenerateindices():
                result.append([parent_indices, range(cls._read_tensor(part.size, parent_indices))])
            return result

    @classmethod
    def _generate_indices(cls, part, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        if isinstance(part.size, MultiArray):
            # there must already be an outer dim or this makes no sense
            assert parent_indices
            idxs = [i for i in range(cls._read_tensor(part.size, parent_indices))]
        else:
            idxs = [i for i in range(part.size)]

        if part.subaxis:
            idxs = [
                [i, *subidxs]
                for i in idxs
                for subidxs in cls._generate_indices(part.subaxis, parent_indices=parent_indices+[i])
            ]
        else:
            idxs = [[i] for i in idxs]

        return idxs

    def mygenerateindices(self):
        return self._mygenindices(self.dim)

    @classmethod
    def _mygenindices(cls, axis, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        idxs = []
        for i in range(cls._get_size(axis, parent_indices)):
            if axis.part.subaxis:
                for subidxs in cls._mygenindices(axis.part.subaxis, parent_indices+[i]): 
                    idxs.append([i] + subidxs)
            else:
                idxs.append([i])
        return idxs

    @classmethod
    def _compute_full_part_size(cls, part, parent_indices=None, current_size=1):
        if not parent_indices:
            parent_indices = []

        if isinstance(part, ScalarAxisPart):
            return 1

        # if we encounter an array then discard everything before and make this the new size
        # e.g. if we have 2 * 2 * [1, 2, 3, 4] then the actual size is 1+2+3+4 = 10
        if isinstance(part.size, MultiArray):
            d = cls._slice_marray(part.size, parent_indices)
            current_size = sum(d)
        else:
            current_size *= part.size

        if part.subaxis:
            return sum(cls._compute_full_part_size(pt, parent_indices, current_size) for pt in part.subaxis.parts)
        else:
            return current_size

    @classmethod
    def _slice_marray(cls, marray, parent_indices):
        def compute_subaxis_size(subaxis, idxs):
            if subaxis:
                return cls._compute_full_axis_size(subaxis, idxs)
            else:
                return 1

        if not parent_indices:
            return marray.data
        elif len(parent_indices) == 1:
            if marray.dim.part.subaxis:
                ptr = 0
                parent_idx, = parent_indices
                for i in range(parent_idx):
                    ptr += compute_subaxis_size(marray.dim.part.subaxis, parent_indices+[i])
                return marray.data[ptr:ptr+compute_subaxis_size(marray.dim.part.subaxis, parent_indices+[parent_idx])]
            else:
                idx, = parent_indices
                return marray.data[idx],
        else:
            raise NotImplementedError

    @classmethod
    def _compute_full_axis_size(cls, axis, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        return sum(cls._compute_full_part_size(pt, parent_indices) for pt in axis.parts)

    @classmethod
    def _get_subdim(cls, dim, point):
        bounds = list(np.cumsum([p.size for p in dim.parts]))
        for i, (start, stop) in enumerate(zip([0]+bounds, bounds)):
            if start <= point < stop:
                npart = i
                break
        return npart, dim.parts[npart]

    def get_value(self, indices):
        offset = self.root.get_offset(indices)
        return self.data[int(offset)]

    # aliases
    @property
    def axes(self):
        return self.dim

    @property
    def root(self):
        return self.dim

    def __getitem__(self, multi_indicesss):
        """
        pass in an iterable of an iterable of multi-indices (e.g. returned by closure)
        """
        """The (outdated) plan of action here is as follows:

        - if a tensor is indexed by a set of stencils then that's great.
        - if it is indexed by a set of slices and integers then we convert
          that to a set of stencils.
        - if passed a combination of stencil groups and integers/slices then
          the integers/slices are first converted to stencil groups and then
          the groups are concatenated, multiplying where required.

        N.B. for matrices, we want to take a tensor product of the stencil groups
        rather than concatenate them. For example:

        mat[f(p), g(q)]

        where f(p) is (0, map[i]) and g(q) is (0, map[j]) would produce the wrong
        thing because we would get mat[0, map[i], 0, map[j]] where what we really
        want is mat[0, 0, map[i], map[j]]. Therefore we instead write:

        mat[f(p)*g(q)]

        to get the correct behaviour.
        """

        # convert an iterable of multi-index collections into a single set of multi-index
        # collections- - needed for [closure(p), closure(p)] each of which returns
        # a list of multi-indices that need to be multiplicatively combined.
        # multi_indicess = self.merge_multiindicesss(multi_indicesss)
        multi_indicess, = multi_indicesss  # for now assert length one

        # TODO Add support for already indexed items
        # This is complicated because additional indices should theoretically index
        # pre-existing slices, rather than get appended/prepended as is currently
        # assumed.
        # if self.is_indexed:
        #     raise NotImplementedError("Needs more thought")

        # if not isinstance(indicess[0], collections.abc.Sequence):
        #     indicess = (indicess,)
        # import pdb; pdb.set_trace()
        multi_indicess = MultiIndexCollection(tuple(
            multi_idx_
            for multi_idx in multi_indicess
            for multi_idx_ in self._extend_multi_index(multi_idx)
        ))
        # import pdb; pdb.set_trace()

        # NOTE: indices should not be a property of this data structure. The indices
        # are only relevant for codegen so I think an IndexedMultiArray or similar would
        # be better. It would also help if we wanted to swap out the data structure later
        # on (but naturally retain the same indices).
        return self.copy(indices=multi_indicess)

    def _extend_multi_index(self, multi_index, axis=None):
        """Apply a multi-index to own axes and return a tuple of 'full' multi-indices.

        This is required in case an inner dimension is multi-part which would require two
        multi-indices to correctly index both.
        """
        # import pdb; pdb.set_trace()
        if not axis:
            axis = self.root

        if multi_index:
            idx, *subidxs = multi_index

            if subaxis := axis.find_part(idx.part_label).subaxis:
                return tuple(
                    MultiIndex((idx,) + subidxs_.typed_indices)
                    for subidxs_ in self._extend_multi_index(subidxs, subaxis)
                )
            else:
                if subidxs:
                    raise ValueError
                return (MultiIndex((idx,)),)
        else:
            new_idxs = []
            for pt in axis.parts:
                idx, subidxs = TypedIndex(pt.label, IndexSet(pt.count)), []

                if subaxis := axis.find_part(idx.part_label).subaxis:
                    new_idxs.extend(
                        MultiIndex((idx,) + subidxs_.typed_indices)
                        for subidxs_ in self._extend_multi_index(subidxs, subaxis)
                    )
                else:
                    new_idxs.append(MultiIndex((idx,)))
            return tuple(new_idxs)


    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)

    @classmethod
    def _fill_with_slices(cls, axis, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        idxs = []
        for i, part in enumerate(axis.parts):
            if isinstance(part, ScalarAxisPart):
                idxs.append([])
                continue
            idx = Slice(part.size, npart=i)
            if part.subaxis:
                idxs += [[idx, *subidxs]
                    for subidxs in cls._fill_with_slices(part.subaxis, parent_indices+[idx])]
            else:
                idxs += [[idx]]
        return idxs

    @classmethod
    def _is_valid_indices(cls, indices, dim):
        # deal with all of this later - need a good scalar solution before this will make sense I think.
        return True

        # not sure what I'm trying to do here
        if not indices and dim.sizes and None in dim.sizes:
            return True

        if dim.sizes and not indices:
            return False

        # scalar case
        if not dim.sizes and not indices:
            return True

        idx, *subidxs = indices
        assert idx.label in dim.labels

        if isinstance(idx, NonAffineMap):
            mapvalid = cls._is_valid_indices(idx.tensor.indices, idx.tensor.dim)
            if not mapvalid:
                return False
            # import pdb; pdb.set_trace()
            npart = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[npart]
                if not cls._is_valid_indices(subidxs, subdim):
                    return False
            return True
        elif isinstance(idx, (Slice, IndexFunction)):
            npart = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[npart]
                if not cls._is_valid_indices(subidxs, subdim):
                    return False
            return True
        else:
            raise TypeError

    def __str__(self):
        return self.name

    @property
    def is_indexed(self):
        return all(self._check_indexed(self.dim, idxs) for idxs in self.indicess)

    def _check_indexed(self, dim, indices):
        for label, size in zip(dim.labels, dim.sizes):
            try:
                (index, subindices), = [(idx, subidxs) for idx, subidxs in indices if idx.label == label]

                npart = dim.labels.index(index.label)

                if subdims := self.dim.get_children(dim):
                    subdim = subdims[npart]
                    return self._check_indexed(subdim, subindices)
                else:
                    return index.size != size
            except:
                return True

    @classmethod
    def _parse_indices(cls, dim, indices, parent_indices=None):
        # import pdb; pdb.set_trace()
        if not parent_indices:
            parent_indices = []

        # cannot assume a slice if we are mixed - is this right?
        # could duplicate parent indices I suppose
        if not indices:
            if len(dim.parts) > 1:
                raise ValueError
            else:
                indices = [Slice(dim.part.size)]

        # import pdb; pdb.set_trace()
        idx, *subidxs = indices

        if isinstance(idx, Map):
            npart = idx.npart
            part = dim.get_part(npart)
            if part.subaxis:
                return [idx] + cls._parse_indices(part.subaxis, subidxs, parent_indices+[idx])
            else:
                return [idx]
        elif isinstance(idx, Slice):
            if isinstance(idx.size, pym.primitives.Expression):
                if not isinstance(idx.size, MultiArray):
                    raise NotImplementedError

            part = dim.get_part(idx.npart)
            if part.subaxis:
                return [idx] + cls._parse_indices(part.subaxis, subidxs, parent_indices+[idx])
            else:
                return [idx]
        else:
            raise TypeError

    # @property
    # def indices(self):
    #     try:
    #         idxs, = self.indicess
    #         return idxs
    #     except ValueError:
    #         raise RuntimeError

    # @property
    # def linear_indicess(self):
    #     # import pdb; pdb.set_trace()
    #     if not self.indices:
    #         return [[]]
    #     return [val for item in self.indices for val in self._linearize(item)]

    def _linearize(self, item):
        # import pdb; pdb.set_trace()
        value, children = item

        if children:
            return [[value] + result for child in children for result in self._linearize(child)]
        else:
            return [[value]]

    @property
    def indexed_shape(self):
        try:
            sh, = self.indexed_shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def indexed_shapes(self):
        return indexed_shapes(self)

    @property
    def indexed_size(self):
        return functools.reduce(operator.mul, self.indexed_shape, 1)

    @property
    def shape(self):
        try:
            sh, = self.shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def shapes(self):
        return self._compute_shapes(self.dim)

    @property
    def size(self):
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def depth(self):
        """Will only work if thing is not multi-part"""
        # TODO should probably still be considered fine if the diverging parts have the same depth
        # would likely require a tree traversal
        depth_ = 0
        axis = self.axes
        while axis:
            depth_ += 1
            if axis.nparts > 1:
                raise RuntimeError("depth not possible for multi-part layouts")
            axis = axis.part.subaxis
        return depth_

    @property
    def order(self):
        return self._compute_order(self.dim)

    def _parametrise_if_needed(self, value):
        if isinstance(value, MultiArray):
            if (param := pym.var(value.name)) in self.params:
                assert self.params[param] == value
            else:
                self.params[param] = value
            return param
        else:
            return value

    def _compute_order(self, dim):
        subdims = dim.subdims
        ords = {self._compute_order(subdim) for subdim in subdims}

        if len(ords) == 0:
            return 1
        elif len(ords) == 1:
            return 1 + ords.pop()
        if len(ords) > 1:
            raise Exception("tensor order cannot be established (subdims are different depths)")

    def _merge_stencils(self, stencils1, stencils2):
        return _merge_stencils(stencils1, stencils2, self.dim)

    def _compute_shapes(self, axis):
        shapes = []
        for part in axis.parts:
            if isinstance(part, ScalarAxisPart):
                shapes.append(())
            elif part.subaxis:
                for shape in self._compute_shapes(part.subaxis):
                    shapes.append((part.size, *shape))
            else:
                shapes.append((part.size,))
        return tuple(shapes)


def indexed_shapes(tensor):
    return tuple(_compute_indexed_shape(idxs) for idxs in tensor.indicess)


def _compute_indexed_shape(indices):
    if not indices:
        return ()

    index, *subindices = indices

    return index_shape(index) + _compute_indexed_shape(subindices)


def _compute_indexed_shape2(flat_indices):
    import warnings
    warnings.warn("need to remove", DeprecationWarning)
    shape = ()
    for index in flat_indices:
        shape += index_shape(index)
    return shape


@functools.singledispatch
def index_shape(index):
    raise TypeError

@index_shape.register(Slice)
@index_shape.register(IndexFunction)
def _(index):
    # import pdb; pdb.set_trace()
    if index.is_loop_index:
        return ()
    return (index.size,)

@index_shape.register(NonAffineMap)
def _(index):
    if index.is_loop_index:
        return ()
    else:
        return index.tensor.indexed_shape


def _merge_stencils(stencils1, stencils2, dims):
    stencils1 = as_stencil_group(stencils1, dims)
    stencils2 = as_stencil_group(stencils2, dims)

    return StencilGroup(
        Stencil(
            idxs1+idxs2
            for idxs1, idxs2 in itertools.product(stc1, stc2)
        )
        for stc1, stc2 in itertools.product(stencils1, stencils2)
    )

def as_stencil_group(stencils, dims):
    if isinstance(stencils, StencilGroup):
        return stencils

    is_sequence = lambda seq: isinstance(seq, collections.abc.Sequence)
    # case 1: dat[x]
    if not is_sequence(stencils):
        return StencilGroup([
            Stencil([
                _construct_indices([stencils], dims, dims.root)
            ])
        ])
    # case 2: dat[x, y]
    elif not is_sequence(stencils[0]):
        return StencilGroup([
            Stencil([
                _construct_indices(stencils, dims, dims.root)
            ])
        ])
    # case 3: dat[(a, b), (c, d)]
    elif not is_sequence(stencils[0][0]):
        return StencilGroup([
            Stencil([
                _construct_indices(idxs, dims, dims.root)
                for idxs in stencils
            ])
        ])
    # case 4: dat[((a, b), (c, d)), ((e, f), (g, h))]
    elif not is_sequence(stencils[0][0][0]):
        return StencilGroup([
            Stencil([
                _construct_indices(idxs, dims, dims.root)
                for idxs in stencil
            ])
            for stencil in stencils
        ])
    # default
    else:
        raise ValueError


def _construct_indices(input_indices, dims, current_dim, parent_indices=None):
    if not parent_indices:
        parent_indices = []
    # import pdb; pdb.set_trace()
    if not current_dim:
        return ()

    if not input_indices:
        if len(dims.get_children(current_dim)) > 1:
            raise RuntimeError("Ambiguous npart")
        input_indices = [Slice.from_dim(current_dim, 0)]

    index, *subindices = input_indices

    npart = current_dim.labels.index(index.label)

    if subdims := dims.get_children(current_dim):
        subdim = subdims[npart]
    else:
        subdim = None

    return (index,) + _construct_indices(subindices, dims, subdim, parent_indices + [index.copy(is_loop_index=True)])



def index(indices):
    """wrap all slices and maps in loop index objs."""
    # cannot be multiple sets of indices if we are shoving this into a loop
    if isinstance(indices[0], collections.abc.Sequence):
        (indices,) = indices
    return tuple(_index(idx) for idx in indices)


def _index(idx):
    if isinstance(idx, NonAffineMap):
        return idx.copy(is_loop_index=True, tensor=idx.tensor.copy(indicess=(index(idx.tensor.indices),)))
    else:
        return idx.copy(is_loop_index=True)


def _break_mixed_slices(stencils, dtree):
    return tuple(
        tuple(idxs
            for indices in stencil
            for idxs in _break_mixed_slices_per_indices(indices, dtree)
        )
        for stencil in stencils
    )


def _break_mixed_slices_per_indices(indices, dtree):
    """Every slice over a mixed dim should branch the indices."""
    if not indices:
        yield ()
    else:
        index, *subindices = indices
        for i, idx in _partition_slice(index, dtree):
            subtree = dtree.children[i]
            for subidxs in _break_mixed_slices_per_indices(subindices, subtree):
                yield (idx, *subidxs)


"""
so I like it if we could go dat[:mesh.ncells, 2] to access the right part of the mixed dim. How to do multiple stencils?

dat[(), ()] for a single stencil
or dat[((), ()), ((), ())] for stencils

also dat[:] should return multiple indicess (for each part of the mixed dim) (but same stencil/temp)

how to handle dat[2:mesh.ncells] with raggedness? Global offset required.

N.B. dim tree no longer used for codegen - can probably get removed. Though a tree still needed since dims should
be independent of their children.

What about LoopIndex?

N.B. it is really difficult to support partial indexing (inc. integers) because, for ragged tensors, the initial offset
is non-trivial and needs to be computed by summing all preceding entries.
"""


def _partition_slice(slice_, dtree):
    if isinstance(slice_, slice):
        ptr = 0
        for i, child in enumerate(dtree.children):
            dsize = child.value.size
            dstart = ptr
            dstop = ptr + dsize

            # check for overlap
            if ((slice_.stop is None or dstart < slice_.stop)
                    and (slice_.start is None or dstop > slice_.start)):
                start = max(dstart, slice_.start) if slice_.start is not None else dstart
                stop = min(dstop, slice_.stop) if slice_.stop is not None else dstop
                yield i, slice(start, stop)
            ptr += dsize
    else:
        yield 0, slice_


def Global(*, name: str = None):
    raise NotImplementedError
    # return MultiArray(name=name, prefix="glob")


def Dat(mesh, dofs, **kwargs):
    """
    dofs:
        A dict mapping part IDs (usually topological dims) to a new
        subaxis.
    """
    axes = mesh.axis
    for id, subaxis in dofs.items():
        axes = axes.add_subaxis(id, subaxis)
    return MultiArray.new(axes, prefix="dat", **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    raise NotImplementedError

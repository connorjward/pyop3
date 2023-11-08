import threading
from operator import attrgetter

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import pyop3 as op3
from pyop3.extras.debug import print_with_rank


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def paxes(comm):
    """Return a parallel `pyop3.AxisTree`.

    The point SF for the distributed axis is given by

                        * g   g
    [rank 0]  0---1---2-*-3---4
                  |   | * |   |
    [rank 1]      0---1-*-2---3---4---5
                  g   g *

    where 'g' means a ghost point. These are the leaves of the SF.

    """
    # abort in serial
    if comm.size == 1:
        return

    if comm.rank == 0:
        npoints = 5
        nroots = 2
        ilocal = (3, 4)
        iremote = tuple((1, i) for i in (2, 3))
    else:
        assert comm.rank == 1
        npoints = 6
        nroots = 2
        ilocal = (0, 1)
        iremote = tuple((0, i) for i in (1, 2))
    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)

    serial = op3.Axis(npoints)
    axis = op3.Axis.from_serial(serial, sf)
    print_with_rank(repr(axis))
    # return op3.AxisTree(axis, {axis.id: op3.Axis(3)})
    axes = op3.AxisTree(axis, {axis.id: op3.Axis(3)})
    print_with_rank(axes)
    print_with_rank(repr(axes))
    return axes


@pytest.mark.parallel(nprocs=2)
def test_new_array_has_valid_roots_and_leaves(paxes):
    array = op3.MultiArray(paxes)
    assert array._roots_valid and array._leaves_valid


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize(
    ("accessor", "leaves_valid"),
    [
        ("data_rw", False),
        ("data_ro", True),
        ("data_wo", False),
        ("data_rw_with_ghosts", True),
        ("data_ro_with_ghosts", True),
        ("data_wo_with_ghosts", True),
    ],
)
def test_leaf_invalidation(paxes, accessor, leaves_valid):
    array = op3.MultiArray(paxes)
    assert array._leaves_valid

    attrgetter(accessor)(array)
    assert array._leaves_valid == leaves_valid


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize(
    ("accessor", "leaves_valid"),
    [
        ("data_rw", False),
        ("data_ro", False),
        ("data_wo", False),
        ("data_rw_with_ghosts", True),
        ("data_ro_with_ghosts", True),
        ("data_wo_with_ghosts", True),
    ],
)
def test_accessors_update_roots_and_leaves(comm, paxes, accessor, leaves_valid):
    array = op3.MultiArray(paxes, dtype=int)
    sf = array.axes.sf

    assert comm.size == 2
    rank = comm.rank
    other_rank = (comm.rank + 1) % 2

    # invalidate root and leaf data
    array.data_wo_with_ghosts[...] = rank
    array._roots_valid = False
    array._last_write_op = op3.INC
    assert not array._leaves_valid

    attrgetter(accessor)(array)

    # roots should be always be updated
    assert array._roots_valid
    assert array._data[sf.root_indices] == rank + other_rank
    assert array._last_write_op is None

    assert array._leaves_valid == leaves_valid
    if array._leaves_valid:
        assert array._data[sf.leaf_indices] == other_rank
    else:
        assert array._data[sf.leaf_indices] == rank


# old code, for reference only

# this runs in both serial and parallel if a fixture
# @pytest.fixture
# def make_overlapped_array(comm):
#     # map end of rank 0 to start of rank 1
#
#     # construct the overlap SF
#     # it should be possible to create this using the point ownerships!
#     # sf = PETSc.SF().create(comm)
#     if comm.rank == 0:
#         owned_points = [0, 1, 2, 3]
#         shared_points = [4, 5, 6]
#         halo_points = [7, 8, 9]
#         remote_halo_points = [0, 1, 2]
#         remote_shared_points = [3, 4, 5]
#
#         overlap = [None] * 10
#         for pt in owned_points:
#             overlap[pt] = Owned()
#         for lpt, rpt in zip(shared_points, remote_halo_points):
#             overlap[lpt] = Shared()
#         for lpt, rpt in zip(halo_points, remote_shared_points):
#             overlap[lpt] = Halo(RemotePoint(1, rpt))
#     else:
#         assert comm.rank == 1
#         owned_points = [6, 7, 8, 9]
#         shared_points = [3, 4, 5]
#         halo_points = [0, 1, 2]
#         remote_halo_points = [7, 8, 9]
#         remote_shared_points = [4, 5, 6]
#
#         overlap = [None] * 10
#         for pt in owned_points:
#             overlap[pt] = Owned()
#         for lpt, rpt in zip(shared_points, remote_halo_points):
#             overlap[lpt] = Shared()
#         for lpt, rpt in zip(halo_points, remote_shared_points):
#             overlap[lpt] = Halo(RemotePoint(0, rpt))
#
#     # sf.setGraph(3, halo_points, remote_offsets)
#
#     root = MultiAxis([AxisPart(10, overlap=overlap)]).set_up()
#     array = MultiArray(root, data=np.ones(10), dtype=np.float64, name="myarray")
#     assert not array._pending_write_op
#     assert array._halo_valid
#     return array
#
#
# def make_global(comm):
#     assert comm.size == 2
#
#     if comm.rank == 0:
#         overlap = [Shared(), Shared(), Shared(RemotePoint(1, 1))]
#     else:
#         assert comm.rank == 1
#         overlap = [Shared(RemotePoint(0, 1)), Shared(), Shared(RemotePoint(0, 0))]
#     root = MultiAxis([AxisPart(10, overlap=overlap)]).set_up()
#     array = MultiArray(root, data=np.ones(3), dtype=np.float64, name="myarray")
#     assert array._halo_valid
#     assert not array._pending_write_op
#     return array
#
#
# @pytest.mark.parallel(nprocs=2)
# def test_sf_exchanges_data(comm):
#     # pretend like the last thing we did was INC into the arrays
#     # array.last_op = INC  # TODO distinguish local and stencil ops
#     overlapped_array = make_overlapped_array(comm)
#
#     mpi_dtype, _ = get_mpi_dtype(overlapped_array.dtype)
#     buffer = overlapped_array.data
#
#     mpi_op = MPI.SUM
#     overlapped_array.root.sf.reduceBegin(mpi_dtype, buffer, buffer, mpi_op)
#     overlapped_array.root.sf.reduceEnd(mpi_dtype, buffer, buffer, mpi_op)
#
#     mpi_op = MPI.REPLACE
#     overlapped_array.root.sf.bcastBegin(mpi_dtype, buffer, buffer, mpi_op)
#     overlapped_array.root.sf.bcastEnd(mpi_dtype, buffer, buffer, mpi_op)
#
#     if comm.rank == 0:
#         assert np.allclose(overlapped_array.data, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
#     else:
#         assert comm.rank == 1
#         assert np.allclose(overlapped_array.data, [2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
#
#     # assert not array.halo_valid  # or similar
#
#     # assert data == ???
#
#
# @pytest.mark.parallel(nprocs=2)
# def test_sync(comm):
#     array = make_overlapped_array(comm)
#
#     # pretend like the last thing we did was INC into the arrays
#     array._pending_write_op = INC
#     array._halo_modified = True
#     array._halo_valid = False
#
#     array.sync(need_halo_values=True)
#
#     if comm.rank == 0:
#         assert np.allclose(array.data, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
#     else:
#         assert comm.rank == 1
#         assert np.allclose(array.data, [2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
#
#     # assert not array.halo_valid  # or similar
#
#     # assert data == ???
#
#
# @pytest.mark.parallel(nprocs=2)
# def test_sync_with_threads(comm):
#     array = make_overlapped_array(comm)
#
#     # pretend like the last thing we did was INC into the arrays
#     array._pending_write_op = INC
#     array._halo_modified = True
#     array._halo_valid = False
#
#     array.sync_begin(need_halo_values=True)
#     array.sync_end()
#
#     if comm.rank == 0:
#         assert np.allclose(array.data, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
#     else:
#         assert comm.rank == 1
#         assert np.allclose(array.data, [2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
#
#     # assert not array.halo_valid  # or similar
#
#     # assert data == ???
#
#
# @pytest.mark.parallel(nprocs=2)
# def test_global_sync(comm):
#     global_ = make_global(comm)
#     assert np.allclose(global_.data, 1)
#
#     # pretend like the last thing we did was INC into the array
#     global_._pending_write_op = INC
#     global_._halo_modified = True
#     global_._halo_valid = False
#     global_.sync()
#     assert np.allclose(global_.data, 2)
#
#
# @pytest.mark.parallel(nprocs=2)
# def test_contrived_distribution(comm):
#     # Construct an array with some shared points and some halo points
#     if comm.rank == 0:
#         overlap = [
#             Shared(RemotePoint(1, 2)),
#             Shared(),
#             Shared(),
#             Halo(RemotePoint(1, 0)),
#         ]
#         npoints = 4
#     else:
#         assert comm.rank == 1
#         overlap = [
#             Shared(),
#             Owned(),
#             Shared(),
#             Halo(RemotePoint(0, 2)),
#             Shared(RemotePoint(0, 1)),
#         ]
#         npoints = 5
#
#     root = MultiAxis([AxisPart(npoints, overlap=overlap)]).set_up()
#     array = MultiArray(root, data=np.ones(npoints), dtype=np.float64, name="myarray")
#     assert array._halo_valid
#     assert not array._pending_write_op
#
#     # halo_valid and pending_write_op should not be allowed
#     array._pending_write_op = INC
#     array._halo_modified = False
#     array._halo_valid = True
#     with pytest.raises(AssertionError):
#         array.sync()
#
#     # halo has not been modified and we don't want to read it
#     # this means that all shared points that are pointed to by other shared points should
#     # be set to 2 and all owned or halo stay
#     # at 1
#     array.data[...] = 1
#     array._pending_write_op = INC
#     array._halo_modified = False
#     array._halo_valid = False
#     array.sync(need_halo_values=False)
#
#     if comm.rank == 0:
#         assert np.allclose(array.data, [2, 2, 1, 1])
#     else:
#         assert np.allclose(array.data, [1, 1, 2, 1, 2])
#
#     # halo has not been modified but we do want to read it
#     array.data[...] = 1
#     array._pending_write_op = INC
#     array._halo_modified = False
#     array._halo_valid = False
#     array.sync(need_halo_values=True)
#
#     if comm.rank == 0:
#         assert np.allclose(array.data, [2, 2, 1, 1])
#     else:
#         assert np.allclose(array.data, [1, 1, 2, 1, 2])
#
#     # halo has been modified but we do not want to read it
#     array.data[...] = 1
#     array._pending_write_op = INC
#     array._halo_modified = True
#     array._halo_valid = False
#     array.sync(need_halo_values=False)
#
#     if comm.rank == 0:
#         assert np.allclose(array.data, [2, 2, 2, 1])
#     else:
#         assert np.allclose(array.data, [2, 1, 2, 1, 2])
#
#     # halo has been modified and we do want to read it
#     array.data[...] = 1
#     array._pending_write_op = INC
#     array._halo_modified = True
#     array._halo_valid = False
#     array.sync(need_halo_values=True)
#
#     if comm.rank == 0:
#         assert np.allclose(array.data, [2, 2, 2, 2])
#     else:
#         assert np.allclose(array.data, [2, 1, 2, 2, 2])

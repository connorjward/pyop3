import threading

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

# from pyop3 import (
#     INC,
#     AxisPart,
#     Halo,
#     MultiArray,
#     MultiAxis,
#     Owned,
#     PointOwnershipLabel,
#     RemotePoint,
#     Shared,
#     get_mpi_dtype,
#     utils,
# )


# This file is pretty outdated
pytest.skip(allow_module_level=True)


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


# this runs in both serial and parallel if a fixture
# @pytest.fixture
def make_overlapped_array(comm):
    # map end of rank 0 to start of rank 1

    # construct the overlap SF
    # it should be possible to create this using the point ownerships!
    # sf = PETSc.SF().create(comm)
    if comm.rank == 0:
        owned_points = [0, 1, 2, 3]
        shared_points = [4, 5, 6]
        halo_points = [7, 8, 9]
        remote_halo_points = [0, 1, 2]
        remote_shared_points = [3, 4, 5]

        overlap = [None] * 10
        for pt in owned_points:
            overlap[pt] = Owned()
        for lpt, rpt in zip(shared_points, remote_halo_points):
            overlap[lpt] = Shared()
        for lpt, rpt in zip(halo_points, remote_shared_points):
            overlap[lpt] = Halo(RemotePoint(1, rpt))
    else:
        assert comm.rank == 1
        owned_points = [6, 7, 8, 9]
        shared_points = [3, 4, 5]
        halo_points = [0, 1, 2]
        remote_halo_points = [7, 8, 9]
        remote_shared_points = [4, 5, 6]

        overlap = [None] * 10
        for pt in owned_points:
            overlap[pt] = Owned()
        for lpt, rpt in zip(shared_points, remote_halo_points):
            overlap[lpt] = Shared()
        for lpt, rpt in zip(halo_points, remote_shared_points):
            overlap[lpt] = Halo(RemotePoint(0, rpt))

    # sf.setGraph(3, halo_points, remote_offsets)

    root = MultiAxis([AxisPart(10, overlap=overlap)]).set_up()
    array = MultiArray(root, data=np.ones(10), dtype=np.float64, name="myarray")
    assert not array._pending_write_op
    assert array._halo_valid
    return array


def make_global(comm):
    assert comm.size == 2

    if comm.rank == 0:
        overlap = [Shared(), Shared(), Shared(RemotePoint(1, 1))]
    else:
        assert comm.rank == 1
        overlap = [Shared(RemotePoint(0, 1)), Shared(), Shared(RemotePoint(0, 0))]
    root = MultiAxis([AxisPart(10, overlap=overlap)]).set_up()
    array = MultiArray(root, data=np.ones(3), dtype=np.float64, name="myarray")
    assert array._halo_valid
    assert not array._pending_write_op
    return array


@pytest.mark.parallel(nprocs=2)
def test_sf_exchanges_data(comm):
    # pretend like the last thing we did was INC into the arrays
    # array.last_op = INC  # TODO distinguish local and stencil ops
    overlapped_array = make_overlapped_array(comm)

    mpi_dtype, _ = get_mpi_dtype(overlapped_array.dtype)
    buffer = overlapped_array.data

    mpi_op = MPI.SUM
    overlapped_array.root.sf.reduceBegin(mpi_dtype, buffer, buffer, mpi_op)
    overlapped_array.root.sf.reduceEnd(mpi_dtype, buffer, buffer, mpi_op)

    mpi_op = MPI.REPLACE
    overlapped_array.root.sf.bcastBegin(mpi_dtype, buffer, buffer, mpi_op)
    overlapped_array.root.sf.bcastEnd(mpi_dtype, buffer, buffer, mpi_op)

    if comm.rank == 0:
        assert np.allclose(overlapped_array.data, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    else:
        assert comm.rank == 1
        assert np.allclose(overlapped_array.data, [2, 2, 2, 2, 2, 2, 1, 1, 1, 1])

    # assert not array.halo_valid  # or similar

    # assert data == ???


@pytest.mark.parallel(nprocs=2)
def test_sync(comm):
    array = make_overlapped_array(comm)

    # pretend like the last thing we did was INC into the arrays
    array._pending_write_op = INC
    array._halo_modified = True
    array._halo_valid = False

    array.sync(need_halo_values=True)

    if comm.rank == 0:
        assert np.allclose(array.data, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    else:
        assert comm.rank == 1
        assert np.allclose(array.data, [2, 2, 2, 2, 2, 2, 1, 1, 1, 1])

    # assert not array.halo_valid  # or similar

    # assert data == ???


@pytest.mark.parallel(nprocs=2)
def test_sync_with_threads(comm):
    array = make_overlapped_array(comm)

    # pretend like the last thing we did was INC into the arrays
    array._pending_write_op = INC
    array._halo_modified = True
    array._halo_valid = False

    array.sync_begin(need_halo_values=True)
    array.sync_end()

    if comm.rank == 0:
        assert np.allclose(array.data, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    else:
        assert comm.rank == 1
        assert np.allclose(array.data, [2, 2, 2, 2, 2, 2, 1, 1, 1, 1])

    # assert not array.halo_valid  # or similar

    # assert data == ???


@pytest.mark.parallel(nprocs=2)
def test_global_sync(comm):
    global_ = make_global(comm)
    assert np.allclose(global_.data, 1)

    # pretend like the last thing we did was INC into the array
    global_._pending_write_op = INC
    global_._halo_modified = True
    global_._halo_valid = False
    global_.sync()
    assert np.allclose(global_.data, 2)


@pytest.mark.parallel(nprocs=2)
def test_contrived_distribution(comm):
    # Construct an array with some shared points and some halo points
    if comm.rank == 0:
        overlap = [
            Shared(RemotePoint(1, 2)),
            Shared(),
            Shared(),
            Halo(RemotePoint(1, 0)),
        ]
        npoints = 4
    else:
        assert comm.rank == 1
        overlap = [
            Shared(),
            Owned(),
            Shared(),
            Halo(RemotePoint(0, 2)),
            Shared(RemotePoint(0, 1)),
        ]
        npoints = 5

    root = MultiAxis([AxisPart(npoints, overlap=overlap)]).set_up()
    array = MultiArray(root, data=np.ones(npoints), dtype=np.float64, name="myarray")
    assert array._halo_valid
    assert not array._pending_write_op

    # halo_valid and pending_write_op should not be allowed
    array._pending_write_op = INC
    array._halo_modified = False
    array._halo_valid = True
    with pytest.raises(AssertionError):
        array.sync()

    # halo has not been modified and we don't want to read it
    # this means that all shared points that are pointed to by other shared points should
    # be set to 2 and all owned or halo stay
    # at 1
    array.data[...] = 1
    array._pending_write_op = INC
    array._halo_modified = False
    array._halo_valid = False
    array.sync(need_halo_values=False)

    if comm.rank == 0:
        assert np.allclose(array.data, [2, 2, 1, 1])
    else:
        assert np.allclose(array.data, [1, 1, 2, 1, 2])

    # halo has not been modified but we do want to read it
    array.data[...] = 1
    array._pending_write_op = INC
    array._halo_modified = False
    array._halo_valid = False
    array.sync(need_halo_values=True)

    if comm.rank == 0:
        assert np.allclose(array.data, [2, 2, 1, 1])
    else:
        assert np.allclose(array.data, [1, 1, 2, 1, 2])

    # halo has been modified but we do not want to read it
    array.data[...] = 1
    array._pending_write_op = INC
    array._halo_modified = True
    array._halo_valid = False
    array.sync(need_halo_values=False)

    if comm.rank == 0:
        assert np.allclose(array.data, [2, 2, 2, 1])
    else:
        assert np.allclose(array.data, [2, 1, 2, 1, 2])

    # halo has been modified and we do want to read it
    array.data[...] = 1
    array._pending_write_op = INC
    array._halo_modified = True
    array._halo_valid = False
    array.sync(need_halo_values=True)

    if comm.rank == 0:
        assert np.allclose(array.data, [2, 2, 2, 2])
    else:
        assert np.allclose(array.data, [2, 1, 2, 2, 2])

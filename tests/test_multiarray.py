from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import pytest

from pyop3 import utils
from pyop3 import MultiArray, MultiAxis, AxisPart, PointOwnershipLabel, get_mpi_dtype, Owned, Halo, RemotePoint, Shared, INC

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
    # assert array.halo_valid  # or similar
    # assert not array.last_op
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


"""
# Notes

I need a number of different star forests:

- all owned and halo points -> single owned point
    - gathering if halo is modified
    - scattering if the above holds and the halo is read
- all owned points -> single owned point
    - gathering if halo is not modified
    - scattering if halo is not read

i.e.

Ahhhh, halo_valid is a different thing to "halo has been modified"!
instead it means "the halo points store the same data as the owned points"

# maybe add "owned_modified" flag? might be clearer

def sync(reading_halo=False):
    # 1. if modified, reduce leaves to their roots
    if self.pending_write_op:
        # the halo should definitely not be considered correct
        assert not self.halo_valid, msg
        if self.halo_written_to:
            self.all_points_sf.reduce()
        else:
            self.owned_points_sf.reduce()  # only reduce with values from owned points

    # 3. at this point only one of the owned points knows the correct result which
    # now needs to be scattered back to some (but not necessarily all) of the other ranks.

    # only send to halos if we want to read them and they are out-of-date
    if reading_halo and not self.halo_valid:
        # send the root value back to all the points
        all_points_sf.broadcast()
        self.halo_valid = True  # all the halo points are now up-to-date
    else:
        # only need to update owned points if we did a reduction earlier
        if self.pending_write_op:
            # send the root value back to just the owned points
            owned_points_sf.broadcast()
            # (the halo is still dirty)

    # set self.last_op to None here? what if halo is still off?
    # what if we read owned values and then owned+halo values?
    # just perform second step
    self.last_op = None
    self.halo_modified = False

Also need to store `last_write_op` property for `DistributedArray` (default `None`). Possible
values include: INC, MIN, MAX, WRITE. This indicates what reduction operations are
required.
"""

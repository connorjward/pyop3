import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import pyop3 as op3
from pyop3.axes.parallel import grow_dof_sf
from pyop3.extras.debug import print_with_rank


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def sf(comm):
    """Create a star forest for a distributed array.

    The created star forest will be distributed as follows:

                           g  g
    rank 0: [0, 4, 1, 2, * 5, 3]
                   |  |  * |  |
    rank 1:       [0, 1, * 3, 2, 4, 5]
                   g  g

    "g" denotes ghost points and "*" is the location of the partition.

    Note that the numberings [0, 4, 1, 2, 5, 3] and [0, 1, 3, 2, 4, 5]
    will be used when creating the axis tree on each rank.

    """
    # abort in serial
    if comm.size == 1:
        return

    if comm.rank == 0:
        nroots = 2
        ilocal = (5, 3)
        iremote = tuple((1, i) for i in (3, 2))
    else:
        assert comm.rank == 1
        nroots = 2
        ilocal = (0, 1)
        iremote = tuple((0, i) for i in (1, 2))

    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)
    return sf


@pytest.fixture
def axis(comm, sf):
    # abort in serial
    if comm.size == 1:
        return

    if sf.comm.rank == 0:
        numbering = [0, 4, 1, 2, 5, 3]
    else:
        assert sf.comm.rank == 1
        numbering = [0, 1, 3, 2, 4, 5]
    serial = op3.Axis(6, numbering=numbering)
    return op3.Axis.from_serial(serial, sf)


@pytest.fixture
def msf(comm):
    # abort in serial
    if comm.size == 1:
        return

    """
                               g   g
    rank 0: [a0, b2, a1, b1, * b0, a2]
            [0,  5,  1,  4,  * 3,  2]
                     |   |   * |   |
                    [0,  6,  * 4,  2,  1,  5,  3]
    rank 1:         [a0, b2, * b0, a2, a1, b1, a3]
                     g   g
    """
    if comm.rank == 0:
        nroots = 2
        ilocal = (3, 2)
        iremote = tuple((1, i) for i in (4, 2))
    else:
        assert comm.rank == 1
        nroots = 2
        ilocal = (0, 6)
        iremote = tuple((0, i) for i in (1, 4))
    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)
    return sf


@pytest.fixture
def maxis(comm, msf):
    # abort in serial
    if comm.size == 1:
        return

    if comm.rank == 0:
        numbering = [0, 5, 1, 4, 3, 2]
        serial = op3.Axis([3, 3], numbering=numbering)
    else:
        assert comm.rank == 1
        numbering = [0, 6, 4, 2, 1, 5, 3]
        serial = op3.Axis([4, 3], numbering=numbering)
    return op3.Axis.from_serial(serial, msf)


@pytest.mark.parallel(nprocs=2)
def test_halo_data_stored_at_end_of_array(axis):
    if axis.sf.comm.rank == 0:
        # unchanged as halo data already at the end
        reordered = [0, 4, 1, 2, 5, 3]
    else:
        assert axis.sf.comm.rank == 1
        reordered = [3, 2, 4, 5, 0, 1]
    assert np.equal(axis.numbering, reordered).all()


@pytest.mark.parallel(nprocs=2)
def test_multi_component_halo_data_stored_at_end(maxis):
    if maxis.sf.comm.rank == 0:
        # unchanged as halo data already at the end
        reordered = [0, 5, 1, 4, 3, 2]
    else:
        assert maxis.sf.comm.rank == 1
        reordered = [4, 2, 1, 5, 3, 0, 6]
    assert np.equal(maxis.numbering, reordered).all()


@pytest.mark.parallel(nprocs=2)
def test_distributed_subaxes_partition_halo_data(comm):
    # Check that
    #
    #        +--+--+
    #        |  |  |
    #        +--+--+
    #       /       \
    #   +-----+   +-----+
    #   |   xx|   |   xx|
    #   +-----+   +-----+
    #
    # transforms to move all of the halo data to the end. Inspect the layouts.
    pass


@pytest.mark.parallel(nprocs=2)
def test_stuff(comm):
    raise NotImplementedError
    sf = create_sf(comm)

    if comm.rank == 0:
        axes = op3.AxisTree(
            DistributedAxis(
                [
                    op3.AxisComponent(4, "a"),
                    op3.AxisComponent(2, "b"),
                ],
                permutation=[0, 2, 3, 5, 1, 4],
                sf=sf,
            )
        )
    else:
        assert comm.rank == 1
        axes = op3.AxisTree(
            DistributedAxis(
                [
                    op3.AxisComponent(3, "a"),
                    op3.AxisComponent(3, "b"),
                ],
                permutation=[0, 1, 3, 2, 4, 5],
                sf=sf,
            )
        )

    dof_sf = grow_dof_sf(axes.freeze())

    dof_sf.view()

    raise NotImplementedError


if __name__ == "__main__":
    test_stuff(MPI.COMM_WORLD)

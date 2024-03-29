import pytest
from mpi4py import MPI
from petsc4py import PETSc

import pyop3 as op3


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def sf(comm):
    """Create a star forest for a distributed array.

    The created star forest will be distributed as follows:

                   g  g
    rank 0:       [0, 1, * 2, 3, 4, 5]
                   |  |  * |  |
    rank 1: [0, 1, 2, 3, * 4, 5]
                           g  g

    "g" denotes ghost points and "*" is the location of the partition.

    Note that we use a "naive" point numbering here because this needs to be
    composed with a serial numbering provided by the distributed axis. The tests
    get very hard to parse if we also have a tricky numbering here.

    """
    # abort in serial
    if comm.size == 1:
        return

    # the sf is created independently of the renumbering
    if comm.rank == 0:
        nroots = 2
        ilocal = (0, 1)
        iremote = tuple((1, i) for i in (2, 3))
    else:
        assert comm.rank == 1
        nroots = 2
        ilocal = (4, 5)
        iremote = tuple((0, i) for i in (2, 3))

    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)
    return sf


@pytest.fixture
def paxis(comm, sf):
    # abort in serial
    if comm.size == 1:
        return

    if sf.comm.rank == 0:
        numbering = [0, 1, 3, 2, 4, 5]
    else:
        assert sf.comm.rank == 1
        numbering = [0, 4, 1, 2, 5, 3]
    serial = op3.Axis(6, numbering=numbering)
    return op3.Axis.from_serial(serial, sf)

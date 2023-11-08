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

                  g   g *
    [rank 0]      0---1-*-2---3---4---5
                  |   | * |   |
    [rank 1]  0---1---2-*-3---4
                        * g   g

    where 'g' means a ghost point. These are the leaves of the SF.

    """
    # abort in serial
    if comm.size == 1:
        return

    if comm.rank == 0:
        npoints = 6
        nroots = 2
        ilocal = (0, 1)
        iremote = tuple((1, i) for i in (1, 2))
    else:
        assert comm.rank == 1
        npoints = 5
        nroots = 2
        ilocal = (3, 4)
        iremote = tuple((0, i) for i in (2, 3))
    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)

    serial = op3.Axis(npoints)
    axis = op3.Axis.from_serial(serial, sf)
    return op3.AxisTree(axis, {axis.id: op3.Axis(3)})


@pytest.mark.parallel(nprocs=2)
def test_new_array_has_valid_roots_and_leaves(paxes):
    array = op3.MultiArray(paxes)
    assert array._roots_valid and array._leaves_valid


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("accessor", ["data_rw", "data_ro", "data_wo"])
def test_accessors_update_roots_and_leaves(comm, paxes, accessor):
    array = op3.MultiArray(paxes, dtype=int)
    sf = array.axes.sf

    if comm.rank == 0:
        self_num = 1
        other_num = 2
    else:
        assert comm.rank == 1
        self_num = 2
        other_num = 1

    # invalidate root and leaf data
    array._data[...] = self_num
    array._roots_valid = False
    array._leaves_valid = False
    array._last_write_op = op3.INC

    attrgetter(accessor)(array)

    # core points (not in SF) should be unchanged
    assert (array._data[sf.icore] == self_num).all()

    if accessor in {"data_rw", "data_ro"}:
        # roots should be always be updated
        assert array._roots_valid
        assert (array._data[sf.iroot] == self_num + other_num).all()
        assert array._last_write_op is None

        # ghost values are not yet updated
        assert not array._leaves_valid
        assert (array._data[sf.ileaf] == self_num).all()
        array.broadcast_roots_to_leaves()
        assert (array._data[sf.ileaf] == self_num + other_num).all()
    else:
        assert accessor == "data_wo"
        # roots should be considered up-to-date but the pending write
        # will have been dropped
        assert array._roots_valid
        assert (array._data[sf.iroot] == self_num).all()
        assert array._last_write_op is None

        # ghost values are not yet updated
        assert not array._leaves_valid
        assert (array._data[sf.ileaf] == self_num).all()
        array.broadcast_roots_to_leaves()
        assert (array._data[sf.ileaf] == other_num).all()

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze

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
def test_distributed_subaxes_partition_halo_data(axis):
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
    root = op3.Axis([1, 1])
    subaxis0 = axis
    subaxis1 = axis.copy(id=op3.Axis.unique_id())
    axes = op3.AxisTree(root, {root.id: [subaxis0, subaxis1]}).freeze()

    path0 = freeze(
        {
            root.label: root.components[0].label,
            subaxis0.label: subaxis0.components[0].label,
        }
    )
    path1 = freeze(
        {
            root.label: root.components[1].label,
            subaxis1.label: subaxis1.components[0].label,
        }
    )

    _, ilocal, _ = axis.sf.getGraph()
    npoints = axis.components[0].count
    nghost = len(ilocal)
    nowned = npoints - nghost

    layout0 = axes.layouts[path0].array
    layout1 = axes.layouts[path1].array

    # print_with_rank(layout0.data)
    # print_with_rank(layout0.data)

    # check that we have tabulated offsets like:
    # ["owned pt0", "owned pt1", "halo pt0", "halo pt1"]
    assert (
        layout0.get_value([0, 0])
        < layout0.get_value([0, nowned - 1])
        < layout1.get_value([0, 0])
        < layout1.get_value([0, nowned - 1])
        < layout0.get_value([0, nowned])
        < layout0.get_value([0, npoints - 1])
        < layout1.get_value([0, nowned])
        < layout1.get_value([0, npoints - 1])
    )


@pytest.mark.parallel(nprocs=2)
def test_nested_parallel_axes_produce_correct_sf(axis):
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
    # builds the right star forest.
    root = op3.Axis([1, 1])
    subaxis0 = axis
    subaxis1 = axis.copy(id=op3.Axis.unique_id())
    axes = op3.AxisTree(root, {root.id: [subaxis0, subaxis1]}).freeze()

    rank = axis.sf.comm.rank
    other_rank = (axis.sf.comm.rank + 1) % 2

    array = op3.MultiArray(axes, dtype=op3.ScalarType)
    array.data[...] = axis.sf.comm.rank
    array.broadcast_roots_to_leaves()

    print_with_rank(array.data)
    array.axes.sf.view()

    _, ilocal, _ = axes.sf.getGraph()
    nghost = len(ilocal)
    assert nghost == 4
    # TODO ultimately will be _with_halos
    assert np.equal(array.data[:-nghost], rank).all()
    assert np.equal(array.data[-nghost:], other_rank).all()

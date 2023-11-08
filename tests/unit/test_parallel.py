# TODO move these tests into something matching an appropriate module
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze

import pyop3 as op3
from pyop3.axes.parallel import grow_dof_sf
from pyop3.extras.debug import print_with_rank
from pyop3.indices.tree import IterationType, partition_iterset
from pyop3.utils import just_one


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
def axis(comm, sf):
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
def test_halo_data_stored_at_end_of_array(comm, axis):
    if comm.rank == 0:
        reordered = [3, 2, 4, 5, 0, 1]
    else:
        assert comm.rank == 1
        # unchanged as halo data already at the end
        reordered = [0, 1, 2, 3, 4, 5]
    assert np.equal(axis.numbering, reordered).all()


@pytest.mark.parallel(nprocs=2)
def test_multi_component_halo_data_stored_at_end(comm, maxis):
    if comm.rank == 0:
        # unchanged as halo data already at the end
        reordered = [0, 5, 1, 4, 3, 2]
    else:
        assert comm.rank == 1
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

    npoints = axis.sf.size
    nowned = npoints - axis.sf.nleaves

    layout0 = axes.layouts[path0].array
    layout1 = axes.layouts[path1].array

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
def test_nested_parallel_axes_produce_correct_sf(comm, axis):
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

    rank = comm.rank
    other_rank = (rank + 1) % 2

    array = op3.MultiArray(axes, dtype=op3.ScalarType)
    array.data[...] = rank
    array.broadcast_roots_to_leaves()

    nghost = axes.sf.nleaves
    assert nghost == 4
    assert np.equal(array.data_ro_with_ghosts[:-nghost], rank).all()
    assert np.equal(array.data_ro_with_ghosts[-nghost:], other_rank).all()


@pytest.mark.parallel(nprocs=2)
def test_partition_iterset_scalar(comm, axis):
    array = op3.MultiArray(axis, dtype=op3.ScalarType)

    p = axis.index()
    tmp = array[p]
    core, root, leaf = partition_iterset(p, [tmp])

    print_with_rank(core, root, leaf)

    if comm.rank == 0:
        # from [0, 1, 3, 2, 4, 5] and knowing that ...
        # this is so confusing
        # basically for this case the numbering is such that the root entities
        # come before the core ones. Ghost will always be the final entries because that
        # is how we do the numbering in the first place.
        assert np.equal(core, [2, 3]).all()
        assert np.equal(root, [0, 1]).all()
        assert np.equal(leaf, [4, 5]).all()
    else:
        assert comm.rank == 1
        # numbering = [0, 4, 1, 2, 5, 3]
        assert np.equal(core, [0, 1]).all()
        assert np.equal(root, [2, 3]).all()
        assert np.equal(leaf, [4, 5]).all()


@pytest.mark.parallel(nprocs=2)
def test_partition_iterset_with_map(comm, axis):
    axis_label = axis.label
    component_label = just_one(axis.components).label

    # connect nearest neighbours (and self at ends)
    # note that this is with the renumbered axis numbering
    if comm.rank == 0:
        # slightly different because the "end" point is actually 3 and the start is 4
        map_data = np.asarray(
            [[5, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 0]], dtype=op3.IntType
        )
    else:
        assert comm.rank == 1
        map_data = np.asarray(
            [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 5]], dtype=op3.IntType
        )
    map_axes = op3.AxisTree(op3.Axis(6, axis.label, id="root"), {"root": op3.Axis(2)})
    map_array = op3.MultiArray(map_axes, data=map_data.flatten())
    map0 = op3.Map(
        {
            freeze({axis_label: component_label}): [
                op3.TabulatedMapComponent(
                    axis_label, component_label, map_array, label=component_label
                )
            ]
        },
        "map0",
        label=axis_label,
    )

    array = op3.MultiArray(axis, dtype=op3.ScalarType)
    p = axis.index()
    tmp = array[map0(p)]
    core, root, leaf = partition_iterset(p, [tmp])

    if comm.rank == 0:
        assert np.equal(core, [3]).all()
        assert np.equal(root, [1, 2]).all()
        assert np.equal(leaf, [0, 4, 5]).all()
    else:
        assert comm.rank == 1
        assert np.equal(core, [0]).all()
        assert np.equal(root, [1, 2]).all()
        assert np.equal(leaf, [3, 4, 5]).all()

import loopy as lp
import numpy as np
import pytest
from petsc4py import PETSc
from pyrsistent import freeze

import pyop3 as op3
from pyop3.extras.debug import print_with_rank
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


def set_kernel(size, intent):
    return op3.Function(
        lp.make_kernel(
            f"{{ [i]: 0 <= i < {size} }}",
            "y[i] = x[0]",
            [
                lp.GlobalArg("x", int, (1,), is_input=True, is_output=False),
                lp.GlobalArg("y", int, (size,), is_input=False, is_output=True),
            ],
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        ),
        [op3.READ, intent],
    )


@pytest.fixture
def mesh_axis(comm):
    """Return an axis corresponding to an interval mesh distributed between two ranks.

    The mesh looks like the following:

                         r      g  g
             6  2  5  1  4  *   0  3
    [rank 0] x-----x-----x  * -----x
                            *
    [rank 1]             x  * -----x-----x-----x-----x
                         4      0  5  1  6  2  7  3  8




    [rank 1]             x  * -----x-----x-----x
                         3  *   0  4  1  5  2  6
                         g      r  r


    Ghost points (leaves) are marked with "g" and roots with "r".

    The axes are also given an arbitrary numbering.

    """
    # abort in serial
    if comm.size == 1:
        return

    # the sf is created independently of the renumbering
    if comm.rank == 0:
        nroots = 1
        ilocal = [0, 3]
        iremote = [(1, 0), (1, 5)]
    else:
        assert comm.rank == 1
        nroots = 2
        ilocal = [4]
        iremote = [(0, 4)]
    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)

    # numberings chosen to stress ghost partitioning algorithms
    if comm.rank == 0:
        ncells = 3
        nverts = 4
        numbering = [1, 5, 4, 0, 6, 3, 2]
    else:
        ncells = 4
        nverts = 5
        numbering = [3, 4, 7, 0, 2, 1, 6, 8, 5]
        # numbering = [3, 4, 0, 2, 1, 6, 5]
    serial = op3.Axis(
        [op3.AxisComponent(ncells, "cells"), op3.AxisComponent(nverts, "verts")],
        "mesh",
        numbering=numbering,
    )
    return op3.Axis.from_serial(serial, sf)


@pytest.fixture
def cone_map(comm, mesh_axis):
    """Return a map from cells to incident vertices."""
    # abort in serial
    if comm.size == 1:
        return

    ncells = mesh_axis.components[0].count
    nverts = mesh_axis.components[1].count
    arity = 2
    maxes = op3.AxisTree(
        op3.Axis([op3.AxisComponent(ncells, "cells")], "mesh", id="root"),
        {"root": op3.Axis(arity)},
    )

    if comm.rank == 0:
        mdata = np.asarray([[4, 3], [5, 4], [6, 5]])
    else:
        assert comm.rank == 1
        # mdata = np.asarray([[3, 4], [4, 5], [5, 6]])
        mdata = np.asarray([[4, 5], [5, 6], [6, 7], [7, 8]])

    # NOTES
    # Question:
    # How does one map from the default component-wise numbering to the
    # correct component-wise numbering of the renumbered axis?
    #
    # Example:
    # Given the renumbering [c1, v2, v0, c0, v1], generate the maps from default to
    # renumbered (component-wise) points:
    #
    #   {c0: c1, c1: c0}, {v0: v1, v1: v2, v2: v0}
    #
    # Solution:
    #
    # The specified numbering is a map from the new numbering to the old. Therefore
    # the inverse of this maps from the old numbering to the new. To give an example,
    # consider the interval mesh numbering [c1, v2, v0, c0, v1]. With plex numbering
    # this becomes [1, 4, 2, 0, 3]. This tells us that point 0 in the new numbering
    # corresponds to point 1 in the default numbering, point 1 maps to point 4 and
    # so on. For this example, the inverse numbering is [3, 0, 2, 4, 1]. This tells
    # us that point 0 in the default numbering maps to point 3 in the new numbering
    # and so on.
    # Given this map, the final thing to do is map from plex-style numbering to
    # the component-wise numbering used in pyop3. We should be able to do this by
    # looping over the renumbering (NOT the inverse) and have a counter for each
    # component.

    # map default cell numbers to their renumbered equivalents
    cell_renumbering = np.empty(ncells, dtype=int)
    min_cell, max_cell = mesh_axis._component_numbering_offsets[:2]
    counter = 0
    for new_pt, old_pt in enumerate(mesh_axis.numbering):
        # is it a cell?
        if min_cell <= old_pt < max_cell:
            old_cell = old_pt - min_cell
            cell_renumbering[old_cell] = counter
            counter += 1
    assert counter == ncells

    # map default vertex numbers to their renumbered equivalents
    vert_renumbering = np.empty(nverts, dtype=int)
    min_vert, max_vert = mesh_axis._component_numbering_offsets[1:]
    counter = 0
    for new_pt, old_pt in enumerate(mesh_axis.numbering):
        # is it a vertex?
        if min_vert <= old_pt < max_vert:
            old_vert = old_pt - min_vert
            vert_renumbering[old_vert] = counter
            counter += 1
    assert counter == nverts

    # renumber the map
    mdata_renum = np.empty_like(mdata)
    for old_cell in range(ncells):
        new_cell = cell_renumbering[old_cell]
        for i, old_pt in enumerate(mdata[old_cell]):
            old_vert = old_pt - min_vert
            mdata_renum[new_cell, i] = vert_renumbering[old_vert]

    # print_with_rank("vertnum", vert_renumbering)
    print_with_rank("mdata", mdata)
    print_with_rank("mdata new", mdata_renum)

    mdat = op3.Dat(maxes, name="cone", data=mdata_renum.flatten())
    return op3.Map(
        {
            freeze({"mesh": "cells"}): [
                op3.TabulatedMapComponent("mesh", "verts", mdat),
            ]
        },
        "cone",
    )


@pytest.mark.parallel(nprocs=2)
# @pytest.mark.parametrize("intent", [op3.INC, op3.MIN, op3.MAX])
@pytest.mark.parametrize(["intent", "fill_value"], [(op3.WRITE, 0), (op3.INC, 0)])
def test_parallel_loop(comm, paxis, intent, fill_value):
    assert comm.size == 2

    dat = op3.Dat(paxis, data=np.full(paxis.size, fill_value, dtype=int))
    knl = rank_plus_one_kernel(comm, intent)

    op3.do_loop(
        p := paxis.index(),
        knl(dat[p]),
    )

    assert np.equal(dat.array._data[: paxis.owned_count], comm.rank + 1).all()
    assert np.equal(dat.array._data[paxis.owned_count :], fill_value).all()

    # since we do not modify ghost points no reduction is needed
    assert dat.array._pending_reduction is None


# can try with P1 and P2
@pytest.mark.parallel(nprocs=2)
def test_parallel_loop_with_map(comm, mesh_axis, cone_map, scalar_copy_kernel):
    assert comm.size == 2
    rank = comm.rank
    other_rank = (comm.rank + 1) % 2

    # could parametrise these
    intent = op3.INC
    fill_value = 0

    rank_dat = op3.Dat(
        op3.Axis(1), name="rank", data=np.asarray([comm.rank + 1]), dtype=int
    )
    dat = op3.Dat(mesh_axis, data=np.full(mesh_axis.size, fill_value), dtype=int)

    knl = set_kernel(2, intent)

    # op3.do_loop(
    loop = op3.loop(
        c := mesh_axis["cells"].index(),
        knl(rank_dat, dat[cone_map(c)]),
    )

    print_with_rank(loop.loopy_code.ir)

    loop()

    print_with_rank(dat.layouts[freeze({"mesh": "verts"})].array.data)

    print_with_rank(dat.array._data)
    print_with_rank(dat.array._pending_reduction)
    print_with_rank(dat.array._leaves_valid)


@pytest.mark.parallel(nprocs=2)
def test_same_reductions_commute():
    ...


@pytest.mark.parallel(nprocs=2)
def test_different_reductions_do_not_commute():
    ...

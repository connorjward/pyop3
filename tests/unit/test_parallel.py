import pytest
from mpi4py import MPI
from petsc4py import PETSc

import pyop3 as op3
from pyop3.axes.parallel import DistributedAxis, grow_dof_sf


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


# this runs in both serial and parallel if a fixture
# @pytest.fixture
def create_sf(comm):
    """
    type:    a  b  a  a    b  a
    rank 0: [0, 4, 1, 2, * 5, 3]
                   |  |  * |  |
    rank 1:       [0, 1, * 3, 2, 4, 5]
    type:          a  a    b  a  b  b

    any permutation should be independent of this
    """
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
    # sf.view()
    return sf


@pytest.mark.parallel(nprocs=2)
def test_stuff(comm):
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

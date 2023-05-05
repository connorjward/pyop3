import pytest
from pyop3 import *
from pyop3.distarray.petsc import *
from pyop3.extras.debug import print_with_rank


def test_read_sparse_matrix():
    """Read values from a matrix that looks like:

     0 10 20
     0 30 40
    50  0  0

    """
    nnzaxes = MultiAxis([AxisPart(3, id="p1")]).set_up()
    nnz = MultiArray(nnzaxes, name="nnz", data=np.array([2, 2, 1], dtype=np.uint64))

    indices = MultiArray.from_list(
        [[1, 2], [1, 2], [0]], labels=["p1", "any"], name="indices", dtype=np.uint64)

    mataxes = nnzaxes.copy().add_subaxis("p1", [AxisPart(nnz, indices=indices)]).set_up()
    mat = MultiArray(mataxes, name="mat", data=np.arange(10, 51, 10))

    assert mat.get_value([0, 1]) == 10
    assert mat.get_value([0, 2]) == 20
    assert mat.get_value([1, 1]) == 30
    assert mat.get_value([1, 2]) == 40
    assert mat.get_value([2, 0]) == 50


def test_read_sparse_rank_3_tensor():
    """Read values from a matrix that looks like:

    0 A
    B C

    with:
        A : [10  0 20]
        B : [30 40  0]
        C : [50  0 60]


    """
    ax1 = MultiAxis([AxisPart(2, id="p1")]).set_up()
    nnz = MultiArray(ax1, name="nnz", data=np.array([1, 2], dtype=np.uint64))

    indices1 = MultiArray.from_list(
        [[1], [0, 1]], labels=["p1", "any"], name="indices", dtype=np.uint64)

    indices2 = MultiArray.from_list(
        [[[0, 2]], [[0, 1], [0, 2]]], labels=["p1", "any1", "any2"], name="indices", dtype=np.uint64)

    ax2 = (
        ax1.copy()
        .add_subaxis("p1", [AxisPart(nnz, indices=indices1, id="p2")])
        .add_subaxis("p2", [AxisPart(2, indices=indices2)])).set_up()
    tensor = MultiArray(ax2, name="tensor", data=np.arange(10, 61, 10))

    assert tensor.get_value([0, 1, 0]) == 10
    assert tensor.get_value([0, 1, 2]) == 20
    assert tensor.get_value([1, 0, 0]) == 30
    assert tensor.get_value([1, 0, 1]) == 40
    assert tensor.get_value([1, 1, 0]) == 50
    assert tensor.get_value([1, 1, 2]) == 60


@pytest.fixture
def sparsity1dp1():
    """

    The cone sparsity of the following mesh:

    v0  v1  v2  v3
    x---x---x---x
      c0  c1  c2

    should look like:

       v0 v1 v2 v3
    v0 x  x
    v1 x  x  x
    v2    x  x  x
    v3       x  x

    """
    mapaxes = (
        MultiAxis([AxisPart(3, label="cells", id="cells")])
        .add_subaxis("cells", [AxisPart(2, label="any")])).set_up()
    mapdata = MultiArray(
        mapaxes, name="map0", data=np.array([0, 1, 1, 2, 2, 3], dtype=IntType))

    iterindex = RangeNode("cells", 3, id="i0")
    lmap = rmap = iterindex.add_child(
        "i0",
        TabulatedMapNode(["cells"], ["nodes"], arity=2,
                         data=mapdata[[iterindex]]))

    return make_sparsity(iterindex, lmap, rmap)


def test_make_sparsity(sparsity1dp1):
    expected = {
        (("nodes",), ("nodes",)): {
            ((0,), (0,)),
            ((0,), (1,)),
            ((1,), (0,)),
            ((1,), (1,)),
            ((1,), (2,)),
            ((2,), (1,)),
            ((2,), (2,)),
            ((2,), (3,)),
            ((3,), (2,)),
            ((3,), (3,)),
        },
    }

    assert sparsity1dp1 == expected


def test_make_matrix(sparsity1dp1):
    raxes = MultiAxis([AxisPart(4, label="nodes")])
    caxes = raxes.copy()

    mat = PetscMatAIJ(raxes, caxes, sparsity1dp1)

    import pdb; pdb.set_trace()

    assert False


@pytest.mark.parallel(nprocs=2)
def test_make_parallel_matrix():
    """TODO

    Construct a P1 matrix for the following 1D mesh:

    v0    v1    v2    v3
    x-----x-----x-----x
       c0    c1    c2

    The mesh is distributed between 2 processes so the local meshes are:

             v0    v1    v2
    proc 1:  x-----x-----o
                c0    c1

             v1    v2    v3
    proc 2:  o~~~~~x-----x
                c1    c2

    Where o and ~ (instead of x and -) denote that points are halo, not owned.

    It is essential that all owned points fully store all points in their
    adjacency. For FEM the adjacency is given by cl(support(pt)). For process 1
    this means that v2 must be stored, but for process 2, owning v2 requires
    that c1 and v1 both exist in the halo.

    Given the adjacency relation as described. The matrix sparsity should be:

       v0 v1 v2 v3
    v0 x  x
    v1 x  x  x
    v2    x  x  x
    v3       x  x

    The sparsities for each process are given by:

    proc 1:

       v0 v1 v2
    v0 x  x
    v1 x  s  s
    v2    h  h

    proc 2:

       v1 v2 v3
    v1 h  h
    v2 s  s  x
    v3    x  x

    Here "s" denotes shared and "h" halo.

    Since PETSc divides ownership across rows, the DoFs in (v3, :) are dropped for
    process 1 and the DoFs in (v1, :) are dropped for process 2.

    """
    comm = PETSc.Sys.getDefaultComm()
    assert comm.size == 2

    if comm.rank == 0:
        # v0, v1 and v2
        nnodes = 3
        overlap = [Owned(), Shared(), Halo(RemotePoint(1, 1))]

        # now make the sparsity
        mapaxes = (
            MultiAxis([AxisPart(2, label="cells", id="cells")])
            .add_subaxis("cells", [AxisPart(2, label="any")])).set_up()
        mapdata = MultiArray(
            mapaxes, name="map0", data=np.array([0, 1, 1, 2], dtype=IntType))

        iterindex = RangeNode("cells", 2, id="i0")
        lmap = rmap = iterindex.add_child(
            "i0",
            TabulatedMapNode(["cells"], ["nodes"], arity=2,
                             data=mapdata[[iterindex]]))
    else:
        # v1, v2 and v3
        nnodes = 3
        # FIXME: Unclear on the ordering of Shared and Owned
        # should they be handled by some numbering?
        overlap = [Owned(), Shared(), Halo(RemotePoint(0, 1))]

        # now make the sparsity
        mapaxes = (
            MultiAxis([AxisPart(2, label="cells", id="cells")])
            .add_subaxis("cells", [AxisPart(2, label="any")])).set_up()
        mapdata = MultiArray(
            mapaxes, name="map0", data=np.array([2, 1, 1, 0], dtype=IntType))

        iterindex = RangeNode("cells", 2, id="i0")
        lmap = rmap = iterindex.add_child(
            "i0",
            TabulatedMapNode(["cells"], ["nodes"], arity=2,
                             data=mapdata[[iterindex]]))

    axes = MultiAxis([AxisPart(nnodes, label="nodes", overlap=overlap)]).set_up()
    sparsity = make_sparsity(iterindex, lmap, rmap)

    print_with_rank(sparsity)

    # new_sparsity = distribute_sparsity(sparsity, axes, axes)

    # print_with_rank(new_sparsity)

    mat = PetscMatAIJ(axes, axes, sparsity)

    # import pdb; pdb.set_trace()
    #
    mat.petscmat.getLGMap()[0].view()
    mat.petscmat.getLGMap()[1].view()

    mat.petscmat.view()


if __name__ == "__main__":
    test_make_parallel_matrix()

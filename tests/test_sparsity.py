from pyop3 import *


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

    mataxes = nnzaxes.add_subaxis("p1", MultiAxis([AxisPart(nnz, indices=indices)])).set_up()
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

    ax2 = ax1.add_subaxis(
        "p1", MultiAxis([
            AxisPart(
                nnz,
                indices=indices1,
                subaxis=MultiAxis([AxisPart(2, indices=indices2)])),
        ])).set_up()
    tensor = MultiArray(ax2, name="tensor", data=np.arange(10, 61, 10))

    assert tensor.get_value([0, 1, 0]) == 10
    assert tensor.get_value([0, 1, 2]) == 20
    assert tensor.get_value([1, 0, 0]) == 30
    assert tensor.get_value([1, 0, 1]) == 40
    assert tensor.get_value([1, 1, 0]) == 50
    assert tensor.get_value([1, 1, 2]) == 60


def test_make_sparsity():
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
        MultiAxis([
            AxisPart(
                3, label="cells",
                subaxis=MultiAxis([
                    AxisPart(2, label="any"),
                ]),
            ),
        ]).set_up())
    mapdata = MultiArray(
        mapaxes, name="map0", data=np.array([0, 1, 1, 2, 2, 3], dtype=np.uint64))

    iterindex = RangeNode("cells", 3, id="i0")
    lmap = rmap = iterindex.add_child(
        "i0",
        TabulatedMapNode(["cells"], ["verts"], arity=2,
                         data=mapdata[[iterindex]]))

    sparsity = make_sparsity(iterindex, lmap, rmap)
    expected = {
        (("verts",), ("verts",)): {
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

    assert sparsity == expected

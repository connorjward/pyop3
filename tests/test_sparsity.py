from pyop3 import *


def test_read_sparse_matrix():
    """Read values from a matrix that looks like:

     0 10 20
     0 30 40
    50  0  0

    """
    nnzaxes = MultiAxis([AxisPart(3, id="p1")]).set_up()
    nnz = MultiArray(nnzaxes, name="nnz", data=np.array([2, 2, 1], dtype=np.uint64))

    mataxes = nnzaxes.add_subaxis(
        "p1", MultiAxis([AxisPart(nnz, indices=[1, 2, 1, 2, 0])])).set_up()
    mat = MultiArray(mataxes, name="mat", data=np.arange(10, 51, 10))

    assert mat.get_value([0, 1]) == 10
    assert mat.get_value([0, 2]) == 20
    assert mat.get_value([1, 1]) == 30
    assert mat.get_value([1, 2]) == 40
    assert mat.get_value([2, 0]) == 50

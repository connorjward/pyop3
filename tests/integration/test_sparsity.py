import numpy as np
import pytest

from pyop3 import Axis, AxisTree, IntType, MultiArray, ScalarType, do_loop
from pyop3.utils import flatten


def test_loop_over_ragged_subset(scalar_copy_kernel):
    # Simulate looping over a (3, 3) sparse matrix with non-zero layout:
    # [x x 0]
    # [x x x]
    # [0 x x]
    nnz_axes = AxisTree(Axis(3))
    nnz_data = np.asarray([2, 3, 2], dtype=IntType)
    nnz = MultiArray(nnz_axes, name="nnz", data=nnz_data)

    subset_axes = nnz_axes.add_subaxis(Axis(nnz), *nnz_axes.leaf)
    subset_data = np.asarray(flatten([[0, 1], [0, 1, 2], [1, 2]]), dtype=IntType)
    subset = MultiArray(
        subset_axes,
        name="subset",
        data=subset_data,
    )

    axes = nnz_axes.add_subaxis(Axis(3), *nnz_axes.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes[:, subset].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert False
    assert np.allclose(dat1.data[touched], dat0.data[touched])
    assert np.allclose(dat1.data[untouched], 0)

import numpy as np
import pytest

from pyop3 import (
    Axis,
    AxisComponent,
    AxisTree,
    IntType,
    MultiArray,
    ScalarType,
    do_loop,
)
from pyop3.utils import flatten


def test_loop_over_ragged_subset(scalar_copy_kernel):
    # Simulate looping over a (3, 3) sparse matrix with non-zero layout:
    # [x x 0]
    # [x x x]
    # [0 x x]
    nnz_axes = AxisTree(Axis([AxisComponent(3, "pt0")], "ax0"))
    nnz_data = np.asarray([2, 3, 2], dtype=IntType)
    nnz = MultiArray(nnz_axes, name="nnz", data=nnz_data)

    # this is unpleasant
    # subset_axes = nnz_axes.add_subaxis(Axis([AxisComponent(nnz, "pt0")], "ax1"), *nnz_axes.leaf)
    subset_axes = nnz_axes.add_subaxis(
        Axis([AxisComponent(nnz, "pt0")], "ax2"), *nnz_axes.leaf
    )
    subset_data = np.asarray(flatten([[0, 1], [0, 1, 2], [1, 2]]), dtype=IntType)
    subset = MultiArray(
        subset_axes,
        name="subset",
        data=subset_data,
    )

    axes = nnz_axes.add_subaxis(Axis([AxisComponent(3, "pt0")], "ax2"), *nnz_axes.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes[:, subset].index(), scalar_copy_kernel(dat0[p], dat1[p]))

    expected = np.zeros_like(dat0.data)
    subset_offset = 0
    for i in range(3):
        for j in range(nnz_data[i]):
            offset = i * 3 + subset_data[subset_offset]
            expected[offset] = dat0.data[offset]
            subset_offset += 1
    assert np.allclose(dat1.data, expected)

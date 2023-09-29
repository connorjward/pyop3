import numpy as np
import pytest

from pyop3 import (
    AffineSliceComponent,
    Axis,
    AxisComponent,
    AxisTree,
    IntType,
    MultiArray,
    ScalarType,
    Slice,
    Subset,
    do_loop,
    loop,
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

    expected = np.zeros_like(dat0.data)
    subset_offset = 0
    for i in range(3):
        for j in range(nnz_data[i]):
            offset = i * 3 + subset_data[subset_offset]
            expected[offset] = dat0.data[offset]
            subset_offset += 1
    assert np.allclose(dat1.data, expected)


def test_sparse_copy(scalar_copy_kernel):
    # Simulate accessing values from a (3, 3) sparse matrix with non-zero layout:
    # [x x 0]
    # [x x x]
    # [0 x x]
    nnz_axes = AxisTree(Axis([AxisComponent(3, "pt0")], "ax0"))
    nnz_data = np.asarray([2, 3, 2], dtype=IntType)
    nnz = MultiArray(nnz_axes, name="nnz", data=nnz_data)

    # dense
    axes0 = nnz_axes.add_subaxis(Axis([AxisComponent(3, "pt0")], "ax1"), *nnz_axes.leaf)
    # sparse
    axes1 = nnz_axes.add_subaxis(
        Axis([AxisComponent(nnz, "pt0")], "ax2"), *nnz_axes.leaf
    )

    dat0 = MultiArray(axes0, name="dat0", data=np.arange(axes0.size, dtype=ScalarType))
    dat1 = MultiArray(axes1, name="dat1", dtype=dat0.dtype)

    slice0 = Slice("ax0", [AffineSliceComponent("pt0", label="pt0")], label="ax0")

    subset_data = np.asarray(flatten([[0, 1], [0, 1, 2], [1, 2]]), dtype=IntType)
    subset = MultiArray(
        axes1,
        name="subset",
        data=subset_data,
    )
    slice1 = Slice("ax1", [Subset("pt0", subset, label="pt0")], label="ax2")

    # The following is equivalent to
    # for (i, j), (p, q) in axes[:, subset]:
    #   dat1[i, j] = dat0[p, q]
    # do_loop(p := axes0[:, subset].enumerate(), scalar_copy_kernel(dat0[p.value], dat1[p.index]))
    do_loop(
        p := axes0[slice0, slice1].index(),
        scalar_copy_kernel(dat0[p], dat1[p.i]),
    )

    # Since we are looping over the matrix [[0, 1, 2], [3, 4, 5], [6, 7, 8]] and
    # accessing [[0, 1], [0, 1, 2], [1, 2]] we expect
    # to have [[0, 1], [3, 4, 5], [7, 8]]
    expected = np.asarray([0, 1, 3, 4, 5, 7, 8])
    assert np.allclose(dat1.data, expected)


def test_sliced_array(scalar_copy_kernel):
    n = 10
    axes = Axis([AxisComponent(n, "pt0")], "ax0")

    array0 = MultiArray(
        axes, name="array0", data=np.arange(axes.size, dtype=ScalarType)
    )
    array1 = MultiArray(axes[::2][1:], name="array1", dtype=array0.dtype)

    # do_loop(p := axes[::2][1:].index(), scalar_copy_kernel(array0[p], array1[p]))
    l = loop(p := axes[::2][1:].index(), scalar_copy_kernel(array0[p], array1[p]))
    l()
    assert np.allclose(array1.data_ro, array0.data_ro[::2][1:])
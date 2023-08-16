import numpy as np
import pytest

from pyop3 import Axis, AxisComponent, AxisTree, MultiArray, ScalarType, do_loop


def test_copy_with_local_indices(scalar_copy_kernel):
    size = 10
    axes = AxisTree(Axis(size))
    dat0 = MultiArray(axes, data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, dtype=dat0.dtype)

    do_loop(
        p := axes.enumerate(),
        scalar_copy_kernel(dat0[p.global_index], dat1[p.local_index]),
    )
    assert np.allclose(dat1.data, dat0.data)


def test_copy_slice(scalar_copy_kernel):
    big_axes = Axis([AxisComponent(10, "pt0")], "ax0")
    small_axes = Axis([AxisComponent(5, "pt0")], "ax0")

    array0 = MultiArray(
        big_axes, name="array0", data=np.arange(big_axes.size, dtype=ScalarType)
    )
    array1 = MultiArray(small_axes, name="array1", dtype=array0.dtype)

    do_loop(
        p := big_axes[::2].enumerate(),
        scalar_copy_kernel(array0[p.global_index], array1[p.local_index]),
    )
    assert np.allclose(array1.data, array0.data[::2])


def test_inc_into_small_array(scalar_copy_kernel):
    size = 10
    dim = 3

    big = MultiArray(big_axes)
    small = MultiArray(small_axes)

    # The following is equivalent to
    # for p in big_axes.root:
    #   for i, q in enumerate(big_axes[p]):
    #     small[i] = big[p, q]
    do_loop(
        p := big_axes.root.index(),
        loop(
            q := big_axes[p, :].enumerate(),
            scalar_copy_kernel(big[p, q[1]], small[q[0]]),
        ),
    )

    assert False, "TODO"

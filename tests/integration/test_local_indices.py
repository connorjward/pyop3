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
    do_loop,
    loop,
)


def test_copy_with_local_indices(scalar_copy_kernel):
    size = 10
    axes = AxisTree(Axis(size))
    dat0 = MultiArray(axes, data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, dtype=dat0.dtype)

    do_loop(
        p := axes.index(),
        scalar_copy_kernel(dat0[p.j], dat1[p.i]),
    )
    assert np.allclose(dat1.data, dat0.data)


def test_copy_slice(scalar_copy_kernel):
    big_axes = Axis([AxisComponent(10, "pt0")], "ax0")
    small_axes = Axis([AxisComponent(5, "pt0")], "ax1")

    array0 = MultiArray(
        big_axes, name="array0", data=np.arange(big_axes.size, dtype=ScalarType)
    )
    array1 = MultiArray(small_axes, name="array1", dtype=array0.dtype)

    slice0 = Slice(
        "ax0", [AffineSliceComponent("pt0", step=2, label="pt0")], label="ax1"
    )

    do_loop(
        p := big_axes[slice0].index(),
        scalar_copy_kernel(array0[p], array1[p.i]),
    )
    assert np.allclose(array1.data, array0.data[::2])


# this isn't a very meaningful test since the local and global loop indices are identical
@pytest.mark.skip(reason="loop composition not currently supported")
def test_inc_into_small_array(scalar_inc_kernel):
    m, n = 10, 3

    small_axes = Axis(n, "ax1")
    big_axes = AxisTree(
        Axis([AxisComponent(m, "pt0")], "ax0", id="root"), {"root": small_axes}
    )

    big = MultiArray(
        big_axes, name="big", data=np.arange(big_axes.size, dtype=ScalarType)
    )
    small = MultiArray(small_axes, name="small", dtype=big.dtype)

    # The following is equivalent to
    # for p in big_axes.root:
    #   for i, q in enumerate(big_axes[p]):
    #     small[i] = big[p, q]
    do_loop(
        p := big_axes.root.index(),
        loop(
            # I think that q.value and q.index are actually exactly the same
            q := small_axes.enumerate(),
            scalar_inc_kernel(big[p, q.value], small[q.index]),
        ),
    )

    expected = np.zeros(n)
    for i in range(m):
        for j in range(n):
            expected[j] += big.data.reshape((m, n))[i, j]
    assert np.allclose(small.data, expected)

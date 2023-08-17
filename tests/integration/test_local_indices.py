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
    loop,
)


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


# this isn't a very meaningful test since the local and global loop indices are identical
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
            q := small_axes.enumerate(),
            scalar_inc_kernel(big[p, q.global_index], small[q.local_index]),
        ),
    )

    expected = np.zeros(n)
    for i in range(m):
        for j in range(n):
            expected[j] += big.data.reshape((m, n))[i, j]
    assert np.allclose(small.data, expected)


# TODO this does not belong in this test file
def test_copy_offset(scalar_copy_kernel_int):
    m = 10
    axes = Axis(m)
    out = MultiArray(axes, name="out", dtype=IntType)

    # do_loop(p := axes.index(), scalar_copy_kernel_int(axes(p), out[p]))
    # debug
    from pyrsistent import pmap

    from pyop3.index import IndexTree, SplitIndexTree, SplitLoopIndex

    p = axes.index()
    path = pmap({axes.label: axes.components[0].label})
    itree = SplitIndexTree({pmap({p: path}): IndexTree(SplitLoopIndex(p, path))})
    l = loop(p, scalar_copy_kernel_int(axes(itree), out[p]))
    l()
    assert np.allclose(out.data, np.arange(10))


# TODO this does not belong in this test file
def test_copy_vec_offset(scalar_copy_kernel_int):
    m, n = 10, 3
    # axes = AxisTree(Axis(m, id="root"), {"root": Axis(n)})
    axes = AxisTree(
        Axis([AxisComponent(m, "pt0")], "ax0", id="root"),
        {"root": Axis([AxisComponent(n, "pt0")], "ax1")},
    )

    out = MultiArray(axes.root, name="out", dtype=IntType)

    # do_loop(p := axes.root.index(), scalar_copy_kernel(axes(p, 0), out[p]))
    from pyrsistent import pmap

    from pyop3.index import (
        AffineSliceComponent,
        IndexTree,
        Slice,
        SplitIndexTree,
        SplitLoopIndex,
    )

    p = axes.root.index()
    path = pmap({"ax0": "pt0"})
    # i.e. [p, 0]
    itree = SplitIndexTree(
        {
            pmap({p: path}): IndexTree(
                root := SplitLoopIndex(p, path),
                {root.id: Slice("ax1", [AffineSliceComponent("pt0", 0, 1)])},
            )
        }
    )
    l = loop(p, scalar_copy_kernel_int(axes(itree), out[p]))

    l()
    assert np.allclose(out.data, np.arange(m * n, step=n))

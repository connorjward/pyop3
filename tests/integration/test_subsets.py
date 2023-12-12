import loopy as lp
import numpy as np
import pytest

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


@pytest.mark.parametrize(
    "touched,untouched",
    [
        (slice(2, None), slice(2)),
        (slice(6), slice(6, None)),
        (slice(None, None, 2), slice(1, None, 2)),
    ],
)
def test_loop_over_slices(scalar_copy_kernel, touched, untouched):
    npoints = 10
    axes = op3.Axis(npoints)
    dat0 = op3.HierarchicalArray(
        axes, name="dat0", data=np.arange(npoints), dtype=op3.ScalarType
    )
    dat1 = op3.HierarchicalArray(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes[touched].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro[untouched], 0)
    assert np.allclose(dat1.data_ro[touched], dat0.data_ro[touched])


@pytest.mark.parametrize("size,touched", [(6, [2, 3, 5, 0])])
def test_scalar_copy_of_subset(scalar_copy_kernel, size, touched):
    untouched = list(set(range(size)) - set(touched))
    subset_axes = op3.Axis({"pt0": len(touched)}, "ax0")
    subset = op3.HierarchicalArray(
        subset_axes, name="subset0", data=np.asarray(touched), dtype=op3.IntType
    )

    axes = op3.Axis({"pt0": size}, "ax0")
    dat0 = op3.HierarchicalArray(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.HierarchicalArray(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes[subset].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro[touched], dat0.data_ro[touched])
    assert np.allclose(dat1.data_ro[untouched], 0)


@pytest.mark.parametrize("size,indices", [(6, [2, 3, 5, 0])])
def test_write_to_subset(scalar_copy_kernel, size, indices):
    n = len(indices)

    subset_axes = op3.Axis({"pt0": n}, "ax0")
    subset = op3.HierarchicalArray(
        subset_axes, name="subset0", data=np.asarray(indices), dtype=op3.IntType
    )

    axes = op3.Axis({"pt0": size}, "ax0")
    dat0 = op3.HierarchicalArray(
        axes, name="dat0", data=np.arange(axes.size), dtype=op3.IntType
    )
    dat1 = op3.HierarchicalArray(subset_axes, name="dat1", dtype=dat0.dtype)

    kernel = op3.Function(
        lp.make_kernel(
            f"{{ [i]: 0 <= i < {n} }}",
            "y[i] = x[i]",
            [
                lp.GlobalArg("x", shape=(n,), dtype=dat0.dtype),
                lp.GlobalArg("y", shape=(n,), dtype=dat0.dtype),
            ],
            name="copy",
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        ),
        [op3.READ, op3.WRITE],
    )

    op3.do_loop(op3.Axis(1).index(), kernel(dat0[subset], dat1))
    assert (dat1.data_ro == indices).all()

import loopy as lp
import numpy as np
import pytest
from pyrsistent import pmap

from pyop3 import (
    INC,
    READ,
    WRITE,
    AffineSliceComponent,
    Axis,
    AxisComponent,
    AxisTree,
    Index,
    IndexTree,
    IntType,
    Map,
    MultiArray,
    ScalarType,
    Slice,
    SliceComponent,
    TabulatedMapComponent,
    do_loop,
    loop,
)
from pyop3.codegen.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


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
    axes = AxisTree(Axis(npoints))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes[touched].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data[untouched], 0)
    assert np.allclose(dat1.data[touched], dat0.data[touched])


@pytest.mark.parametrize("size,touched", [(6, [2, 3, 5, 0])])
def test_scalar_copy_of_subset(scalar_copy_kernel, size, touched):
    untouched = list(set(range(size)) - set(touched))
    subset_axes = Axis(len(touched))
    subset = MultiArray(
        subset_axes, name="subset0", data=np.asarray(touched, dtype=IntType)
    )

    axes = Axis(size)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes[subset].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data[touched], dat0.data[touched])
    assert np.allclose(dat1.data[untouched], 0)

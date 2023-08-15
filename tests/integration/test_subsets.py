import loopy as lp
import numpy as np
import pytest
from pyrsistent import pmap

from pyop3 import (
    INC,
    READ,
    WRITE,
    Axis,
    AxisComponent,
    AxisTree,
    Index,
    IndexTree,
    IntType,
    LoopyKernel,
    Map,
    MultiArray,
    ScalarType,
    Slice,
    SliceComponent,
    TabulatedMapComponent,
    do_loop,
    loop,
)
from pyop3.codegen import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.index import AffineSliceComponent, SplitIndexTree, SplitLoopIndex
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


def test_scalar_copy_of_subset(scalar_copy_kernel):
    m, n = 6, 4
    sdata = np.asarray([2, 3, 5, 0], dtype=IntType)
    untouched = [1, 4]

    axes = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    # a subset is really a map from a small set into a larger one
    saxes = AxisTree(
        Axis([AxisComponent(n, "scpt0")], "sax0", id="root"), {"root": Axis(1)}
    )
    subset = MultiArray(saxes, name="subset0", data=sdata)

    do_loop(p := axes[subset].index(), scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data[sdata], dat0.data[sdata])
    assert np.allclose(dat1.data[untouched], 0)

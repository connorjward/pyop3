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


# these could be separate tests
def test_loop_over_slices(scalar_copy_kernel):
    npoints = 10
    # axes = AxisTree(Axis(npoints))
    axes = AxisTree(Axis([AxisComponent(npoints, "pt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    # cleanup
    itreebag0 = SplitIndexTree(
        pmap({pmap(): IndexTree(Slice("ax0", [AffineSliceComponent("pt0", start=2)]))})
    )
    p = axes[itreebag0].index()
    unrolled_p = SplitLoopIndex(p, pmap({"ax0": "pt0"}))
    itree_bag = SplitIndexTree(
        pmap({pmap({p: pmap({"ax0": "pt0"})}): IndexTree(unrolled_p)})
    )

    # do_loop(p := axes[2:].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    # do_loop(p, scalar_copy_kernel(dat0[itree_bag], dat1[itree_bag]))
    l = loop(p, scalar_copy_kernel(dat0[itree_bag], dat1[itree_bag]))
    l()
    assert np.allclose(dat1.data[:2], 0)
    assert np.allclose(dat1.data[2:], dat0.data[2:])

    assert False, "fixme below"

    dat1.data[...] = 0
    do_loop(p := axes[:6].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data[:6], dat0.data[:6])
    assert np.allclose(dat1.data[6:], 0)

    dat1.data[...] = 0
    do_loop(p := axes[::2].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data[::2], dat0.data[::2])
    assert np.allclose(dat1.data[1::2], 0)


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

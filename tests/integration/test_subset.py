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
from pyop3.utils import flatten


@pytest.fixture
def scalar_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=False, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="scalar_copy",
        lang_version=(2018, 2),
    )
    return LoopyKernel(code, [READ, WRITE])


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
    subset_array = MultiArray(saxes, name="subset0", data=sdata)
    subset = Map(
        {
            pmap({"sax0": "scpt0"}): [
                TabulatedMapComponent("ax0", "cpt0", subset_array),
            ],
        },
        "subset",
    )

    do_loop(
        p := saxes.root.index(), scalar_copy_kernel(dat0[subset(p)], dat1[subset(p)])
    )

    assert np.allclose(dat1.data[sdata], dat0.data[sdata])
    assert np.allclose(dat1.data[untouched], 0)

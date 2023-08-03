import ctypes

import loopy as lp
import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import pmap

from pyop3.axis import Axis, AxisComponent, AxisTree
from pyop3.codegen import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.distarray import MultiArray
from pyop3.dtypes import IntType, ScalarType
from pyop3.index import (
    AffineMap,
    IdentityMap,
    Index,
    IndexTree,
    Map,
    Slice,
    TabulatedMap,
)
from pyop3.loopexpr import INC, READ, WRITE, LoopyKernel, do_loop, loop
from pyop3.utils import flatten


@pytest.fixture
def vec2_copy_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (2,), is_input=False, is_output=True),
        ],
        name="copy",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(lpy_kernel, [READ, WRITE])


def test_1d_slice_composition(vec2_copy_kernel):
    # equivalent to dat1[...] = dat0[::2][1:3] (== [2, 4])
    m, n = 10, 2
    dat0 = MultiArray(
        AxisTree(Axis([(m, "cpt0")], "ax0")),
        name="dat0",
        data=np.arange(m, dtype=ScalarType),
    )
    dat1 = MultiArray(Axis([(n, "cpt0")], "ax0"), name="dat1", dtype=dat0.dtype)

    p = Axis(1).index()
    itree0 = IndexTree(Slice([("ax0", "cpt0", 0, None, 2)]))
    itree1 = IndexTree(Slice([("ax0", "cpt0", 1, 3, 1)]))
    itree2 = IndexTree(Slice([("ax0", "cpt0", 0, None, 1)]))

    do_loop(p, vec2_copy_kernel(dat0[itree0][itree1], dat1[itree2]))

    assert np.allclose(dat1.data, dat0.data[::2][1:3])


def test_2d_slice_composition(vec2_copy_kernel):
    # equivalent to dat0.data[::2, 1:][2:4, 1]
    m0, m1, n = 10, 3, 2

    axes0 = AxisTree(
        Axis([(m0, "cpt0")], "ax0", id="root"), {"root": Axis([(m1, "cpt0")], "ax1")}
    )
    axes1 = AxisTree(Axis([(n, "cpt0")], "ax0"))

    dat0 = MultiArray(axes0, name="dat0", data=np.arange(axes0.size, dtype=ScalarType))
    dat1 = MultiArray(axes1, name="dat1", dtype=dat0.dtype)

    p = Axis(1).index()
    itree0 = IndexTree(
        Slice([("ax0", "cpt0", 0, None, 2)], id="slice0"),
        {"slice0": Slice([("ax1", "cpt0", 1, None, 1)])},
    )
    itree1 = IndexTree(
        Slice([("ax0", "cpt0", 2, 4, 1)], id="slice1"),
        {"slice1": Slice([("ax1", "cpt0", 1, 2, 1)])},
    )
    itree2 = IndexTree(Slice([("ax0", "cpt0", 0, None, 1)]))

    do_loop(
        p,
        vec2_copy_kernel(
            dat0[itree0][itree1],
            dat1[itree2],
        ),
    )

    assert np.allclose(dat1.data, dat0.data.reshape((m0, m1))[::2, 1:][2:4, 1])

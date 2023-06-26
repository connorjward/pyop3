import loopy as lp
import numpy as np
import pytest

from pyop3 import (
    READ,
    WRITE,
    Axis,
    AxisTree,
    Index,
    IndexTree,
    LoopyKernel,
    MultiArray,
    Range,
    ScalarType,
    Slice,
    do_loop,
    loop,
)
from pyop3.codegen import LOOPY_LANG_VERSION, LOOPY_TARGET


@pytest.fixture
def copy_kernel():
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


def test_1d_slice_composition(copy_kernel):
    m = 10
    dat0 = MultiArray(
        AxisTree(Axis(m, "ax0")), name="dat0", data=np.arange(m, dtype=ScalarType)
    )
    dat1 = MultiArray(Axis(2, "ax1"), name="dat1", dtype=ScalarType)

    # equivalent to dat1[...] = dat0[::2][1:3] (== [2, 4])

    # we have no outer iteration so just have a loop with extent 1
    do_loop(
        AxisTree(
            Axis(1, "any")
        ),  # like a range, could be called "p" and used to index sub-bits
        copy_kernel(
            dat0[Index(Slice(("ax0", 0), None, None, 2))][
                Index(Slice(("ax0", 0), 1, 3))
            ],
            dat1[Index(Slice(("ax1", 0), 2))],
        ),
    )

    assert np.allclose(dat1.data, dat0.data[::2][1:3])


def test_2d_slice_composition(copy_kernel):
    m, n = 10, 3

    axes0 = AxisTree(Axis(m, "ax0", id="root"), {"root": Axis(n, "ax1")})
    axes1 = AxisTree(Axis(2, "ax2"))

    dat0 = MultiArray(axes0, name="dat0", data=np.arange(axes0.size))
    dat1 = MultiArray(axes1, name="dat1", dtype=dat0.dtype)

    # equivalent to dat0.data[::2, 1:][2:4, 1]
    p0 = IndexTree(
        Index(Slice(("ax0", 0), None, None, 2), id="idx0"),
        {"idx0": Index(Slice(("ax1", 0), 1, None))},
    )
    p1 = IndexTree(
        Index(Slice(("ax0", 0), 2, 4), id="idx1"),
        {"idx1": Index(Slice(("ax1", 0), 1, 2))},
    )

    do_loop(
        Axis(1),  # for _ in range(1)
        copy_kernel(
            dat0[p0][p1],
            dat1[...],
        ),
    )

    assert np.allclose(dat1.data, dat0.data.reshape((m, n))[::2, 1:][2:4, 1])

import pytest
import loopy as lp
import numpy as np

from pyop3 import AxisTree, Axis, MultiArray, ScalarType, do_loop, Slice, Range, READ, WRITE, LoopyKernel, Index
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
    dat0 = MultiArray(AxisTree(Axis(m)), data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(Axis(2), dtype=ScalarType)

    # equivalent to dat1[...] = dat0[::2][1:3] (== [2, 4])

    # we have no outer iteration so just have a loop with extent 1
    do_loop(
        Range("any", 1),  # should be .index
        copy_kernel(dat0[Index(Slice(None, 10, 2, size=5))][Index(Slice(1, 3, size=2))], dat1[Index(Slice(2, size=2))]),
    )

    assert np.allclose(dat1.data, dat0.data[::2][1:3])

import pytest

import loopy as lp

from pyop3 import AxisTree, Axis, READ, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE, ScalarType, MultiArray, LoopyKernel, IndexTree, Index, Range, do_loop
from pyop3.codegen import LOOPY_TARGET, LOOPY_LANG_VERSION


# NOTE: It is only meaningful to test min/max in parallel as otherwise they behave the
# same as rw/write
@pytest.fixture
def min_rw_kernel():
    code = lp.make_kernel(
        "x[0] = min(x[0], y[0])",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=True, is_output=True),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=False),
        ],
        name="min_rw",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(code, [MIN_RW, READ])


@pytest.fixture
def min_write_kernel():
    code = lp.make_kernel(
        "x[0] = min(y[0], z[0])",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=False, is_output=True),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("z", ScalarType, (1,), is_input=True, is_output=False),
        ],
        name="min_write",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(code, [MIN_WRITE, READ, READ])


@pytest.fixture
def max_rw_kernel():
    code = lp.make_kernel(
        "x[0] = max(x[0], y[0])",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=True, is_output=True),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=False),
        ],
        name="max_rw",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(code, [MAX_RW, READ])


@pytest.fixture
def max_write_kernel():
    code = lp.make_kernel(
        "x[0] = max(y[0], z[0])",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=False, is_output=True),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("z", ScalarType, (1,), is_input=True, is_output=False),
        ],
        name="max_write",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(code, [MAX_WRITE, READ, READ])


@pytest.mark.parametrize("access", [MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE])
def test_pointwise_accesses_descriptors_fail_with_vector_shape(access):
    m = 3

    if access in {MIN_RW, MAX_RW}:
        kernel_data = [
            lp.GlobalArg("x", ScalarType, (m,), is_input=True, is_output=True),
            lp.GlobalArg("y", ScalarType, (m,), is_input=True, is_output=False),
        ]
    else:
        assert access in {MIN_WRITE, MAX_WRITE}
        kernel_data = [
            lp.GlobalArg("x", ScalarType, (m,), is_input=False, is_output=True),
            lp.GlobalArg("y", ScalarType, (m,), is_input=True, is_output=False),
            lp.GlobalArg("z", ScalarType, (m,), is_input=True, is_output=False),
        ]
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "",
        kernel_data,
        name="dummy",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )

    with pytest.raises(ValueError):
        LoopyKernel(lpy_kernel, [access] + [READ] * (len(kernel_data) - 1))

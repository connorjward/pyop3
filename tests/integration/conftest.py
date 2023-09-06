import loopy as lp
import pytest

from pyop3 import INC, READ, WRITE, IntType, LoopyKernel, ScalarType
from pyop3.codegen import loopy_lang_version, loopy_target


@pytest.fixture
def scalar_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=False, is_output=True),
        ],
        name="scalar_copy",
        target=loopy_target(),
        lang_version=loopy_lang_version(),
    )
    return LoopyKernel(code, [READ, WRITE])


@pytest.fixture
def scalar_copy_kernel_int():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", IntType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", IntType, (1,), is_input=False, is_output=True),
        ],
        name="scalar_copy_int",
        target=loopy_target(),
        lang_version=loopy_lang_version(),
    )
    return LoopyKernel(code, [READ, WRITE])


@pytest.fixture
def scalar_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = y[i] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="scalar_inc",
        target=loopy_target(),
        lang_version=loopy_lang_version(),
    )
    return LoopyKernel(lpy_kernel, [READ, INC])

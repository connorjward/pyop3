import loopy as lp
import pytest

from pyop3 import READ, WRITE, LoopyKernel, ScalarType
from pyop3.codegen import LOOPY_LANG_VERSION, LOOPY_TARGET


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
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(code, [READ, WRITE])

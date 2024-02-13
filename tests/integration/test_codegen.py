import loopy as lp

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


def test_dummy_arguments():
    kernel = op3.Function(
        lp.make_kernel(
            "{ [i]: 0 <= i < 1 }",
            [lp.CInstruction((), "y[0] = x[0];", read_variables=frozenset({"x", "y"}))],
            [
                lp.ValueArg("x", dtype=lp.types.OpaqueType("double*")),
                lp.ValueArg("y", dtype=lp.types.OpaqueType("double*")),
            ],
            name="subkernel",
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        ),
        [op3.NA, op3.NA],
    )
    # ccode = lp.generate_code_v2(kernel.code)
    # breakpoint()
    called_kernel = kernel(op3.DummyKernelArgument(), op3.DummyKernelArgument())

    # how do I know where to stop? at C code vs executable?
    code = op3.ir.lower.compile(called_kernel, name="dummy_kernel")

    ccode = lp.generate_code_v2(code.ir).device_code()

    # TODO validate that the generate code is correct, at the time of writing
    # it merely looks right

import loopy as lp
import numpy as np
import pytest

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


@pytest.fixture
def scalar_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=False, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="scalar_copy",
        lang_version=(2018, 2),
    )
    return op3.Function(code, [op3.READ, op3.WRITE])


@pytest.fixture
def vector_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (3,), is_input=False, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="vector_copy",
        lang_version=(2018, 2),
    )
    return op3.Function(code, [op3.READ, op3.WRITE])


def test_scalar_copy(scalar_copy_kernel):
    m = 10
    axis = op3.Axis(m)
    dat0 = op3.Dat(axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(
        axis,
        name="dat1",
        dtype=dat0.dtype,
    )

    op3.do_loop(p := axis.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data, dat0.data)


def test_vector_copy(vector_copy_kernel):
    m, n = 10, 3

    axes = op3.AxisTree.from_nest({op3.Axis(m): op3.Axis(n)})
    dat0 = op3.Dat(axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(
        axes,
        name="dat1",
        dtype=dat0.dtype,
    )

    op3.do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_multi_component_vector_copy(vector_copy_kernel):
    m, n, a, b = 4, 6, 2, 3

    axes = op3.AxisTree.from_nest(
        {op3.Axis({"pt0": m, "pt1": n}): [op3.Axis(a), op3.Axis(b)]}
    )
    dat0 = op3.Dat(
        axes,
        name="dat0",
        data=np.arange(m * a + n * b),
        dtype=op3.ScalarType,
    )
    dat1 = op3.Dat(
        axes,
        name="dat1",
        dtype=dat0.dtype,
    )

    op3.do_loop(
        p := axes.root["pt1"].index(),
        vector_copy_kernel(dat0[p, :], dat1[p, :]),
    )

    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])


def test_copy_multi_component_temporary(vector_copy_kernel):
    m = 4
    n0, n1 = 2, 1
    npoints = m * n0 + m * n1

    axes = op3.AxisTree.from_nest({op3.Axis(m): op3.Axis([n0, n1])})
    dat0 = op3.Dat(axes, name="dat0", data=np.arange(npoints), dtype=op3.ScalarType)
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_multi_component_scalar_copy_with_two_outer_loops(scalar_copy_kernel):
    m, n, a, b = 8, 6, 2, 3

    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": m, "pt1": n}): [
                op3.Axis(a),
                op3.Axis(b),
            ]
        },
    )
    dat0 = op3.Dat(
        axes, name="dat0", data=np.arange(m * a + n * b), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes["pt1", :].index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])

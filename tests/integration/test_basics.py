import ctypes

import loopy as lp
import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import pmap

from pyop3 import (
    INC,
    READ,
    WRITE,
    AffineSliceComponent,
    Axis,
    AxisComponent,
    AxisTree,
    Function,
    Index,
    IndexTree,
    IntType,
    MultiArray,
    ScalarType,
    Slice,
    do_loop,
    loop,
)
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
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
    return Function(code, [READ, WRITE])


@pytest.fixture
def vector_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (3,), is_input=False, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="vector_copy",
        lang_version=(2018, 2),
    )
    return Function(code, [READ, WRITE])


def test_scalar_copy(scalar_copy_kernel):
    m = 10

    axis = Axis([AxisComponent(m, "pt0")], "ax0")
    dat0 = MultiArray(
        axis,
        name="dat0",
        data=np.arange(m, dtype=ScalarType),
    )
    dat1 = MultiArray(
        axis,
        name="dat1",
        dtype=dat0.dtype,
    )

    do_loop(p := axis.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data, dat0.data)


def test_vector_copy(vector_copy_kernel):
    m, n = 10, 3

    axes = AxisTree(
        root := Axis(m),
        {
            root.id: Axis(n),
        },
    )
    dat0 = MultiArray(
        axes,
        name="dat0",
        data=np.arange(m * n, dtype=ScalarType),
    )
    dat1 = MultiArray(
        axes,
        name="dat1",
        dtype=dat0.dtype,
    )

    do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_multi_component_vector_copy(vector_copy_kernel):
    m, n, a, b = 4, 6, 2, 3

    axes = AxisTree(
        Axis([AxisComponent(m, "pt0"), AxisComponent(n, "pt1")], "ax0", id="root"),
        {
            "root": [
                Axis(a),
                Axis(b),
            ]
        },
    )
    dat0 = MultiArray(
        axes,
        name="dat0",
        data=np.arange(m * a + n * b, dtype=ScalarType),
    )
    dat1 = MultiArray(
        axes,
        name="dat1",
        dtype=dat0.dtype,
    )

    do_loop(
        p := axes.root[Slice("ax0", [AffineSliceComponent("pt1")])].index(),
        vector_copy_kernel(dat0[p, :], dat1[p, :]),
    )

    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])


def test_copy_multi_component_temporary(vector_copy_kernel):
    m = 4
    n0, n1 = 2, 1
    npoints = m * n0 + m * n1

    axes = AxisTree(Axis(m, "ax0", id="root"), {"root": Axis([n0, n1], "ax1")})
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_multi_component_scalar_copy_with_two_outer_loops(scalar_copy_kernel):
    m, n, a, b = 8, 6, 2, 3

    axes = AxisTree(
        Axis(
            [
                (m, 0),
                (n, 1),
            ],
            "ax0",
            id="root",
        ),
        {
            "root": [
                Axis(a),
                Axis([(b, 0)], "ax1"),
            ]
        },
    )
    dat0 = MultiArray(
        axes, name="dat0", data=np.arange(m * a + n * b, dtype=ScalarType)
    )
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    iterset = AxisTree(
        Axis([(n, 1)], "ax0", id="root"),
        {"root": Axis([(b, 0)], "ax1")},
    )
    # do_loop(p := iterset.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    l = loop(p := iterset.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    l()

    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])

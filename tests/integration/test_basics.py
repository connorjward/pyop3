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
from pyop3.index import Index, IndexTree, Slice
from pyop3.loopexpr import INC, READ, WRITE, LoopyKernel, do_loop, loop
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
    return LoopyKernel(code, [READ, WRITE])


def test_scalar_copy(scalar_copy_kernel):
    m = 10

    axis = Axis(m)
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
        root := Axis([(m, "cpt0"), (n, "cpt1")], "ax0"),
        {
            root.id: [
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

    # TODO It would be nice to express this as a loop over
    # p := axes.root[Slice(axis="ax0", cpt=1)].index
    # but this needs axis slicing to work first.
    iterset = Axis([(n, "cpt1")], "ax0")
    do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))

    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])


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
    do_loop(p := iterset.index(), scalar_copy_kernel(dat0[p], dat1[p]))

    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])


@pytest.mark.skip(reason="TODO")
def test_inc_with_shared_global_value():
    m0, m1 = 5, 1
    n = 2
    npoints = m0 * n + m1

    arity = 1
    mapdata = np.asarray([[3], [2], [1], [4], [1]], dtype=IntType)

    axes = AxisTree(root := Axis([m0, m1], "ax0"), {root.id: [Axis(n), None]})
    dat0 = MultiArray(axes, name="dat0", data=np.zeros(npoints, dtype=ScalarType))

    maxes = AxisTree(root := Axis(m0, "ax0"), {root.id: Axis(arity)})
    map0 = MultiArray(maxes, name="map0", data=mapdata.flatten())

    knl = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "x[i]  = x[i] + 1",
        [lp.GlobalArg("x", ScalarType, (3,), is_input=True, is_output=True)],
        target=LOOPY_TARGET,
        name="plus_one",
        lang_version=(2018, 2),
    )
    plus_one = LoopyKernel(knl, [INC])

    p = IndexTree(Index(Range("ax0", m0)))

    mapexpr = (tuple(pym.variables("x y")), 0)  # always yield 0
    q = p.put_node(
        Index(
            [
                TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=arity, data=map0[p]),
                AffineMap([("ax0", 0)], [("ax0", 1)], arity=1, expr=mapexpr),
            ]
        ),
        p.leaf,
    )

    do_loop(p, plus_one(dat0[q]))

    expected = np.zeros(npoints)
    for i0 in range(m0):
        for i1 in range(n):
            expected[mapdata[i0] * n + i1] += 1
        expected[m0 * n :] += 1
    assert np.allclose(dat0.data, expected)

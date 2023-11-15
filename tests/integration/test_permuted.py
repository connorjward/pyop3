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
    Axis,
    AxisComponent,
    AxisTree,
    Function,
    IntType,
    MultiArray,
    ScalarType,
    do_loop,
    loop,
)
from pyop3.ir.lower import LOOPY_LANG_VERSION, LOOPY_TARGET
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


def test_vector_copy_with_permuted_axis(vector_copy_kernel):
    m, n = 6, 3
    numbering = np.asarray([2, 5, 1, 0, 4, 3], dtype=IntType)

    axes = AxisTree(Axis(m, id="root"), {"root": Axis(n)})
    paxes = axes.with_modified_node(
        axes.root,
        numbering=numbering,
    )
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m * n, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_vector_copy_with_two_permuted_axes(vector_copy_kernel):
    a, b, c = 4, 2, 3
    numbering0 = [2, 1, 3, 0]
    numbering1 = [1, 0]

    axis0 = Axis(a, "ax0")
    axis1 = Axis(b, "ax1")
    axes = AxisTree(
        axis0,
        {
            axis0.id: axis1,
            axis1.id: Axis(c),
        },
    )
    paxes = axes.with_modified_node(axis0, numbering=numbering0).with_modified_node(
        axis1, numbering=numbering1
    )

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    iterset = AxisTree(axis0, {axis0.id: axis1})
    do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_vector_copy_with_permuted_inner_axis(vector_copy_kernel):
    a, b, c = 5, 4, 3
    numbering = [2, 1, 3, 0]

    root = Axis(a, "ax0")
    inner_axis = Axis(b)
    inner_paxis = inner_axis.copy(numbering=numbering)

    axes = AxisTree(root, {root.id: inner_axis, inner_axis.id: Axis(c)})
    paxes = AxisTree(root, {root.id: inner_paxis, inner_paxis.id: Axis(c)})

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    iterset = AxisTree(root, {root.id: inner_axis})
    do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_vector_copy_with_permuted_multi_component_axes(vector_copy_kernel):
    m, n = 3, 2
    a, b = 2, 3
    numbering = [4, 2, 0, 3, 1]

    root = Axis([AxisComponent(m, "pt0"), AxisComponent(n, "pt1")], "ax0")
    axes = AxisTree(root, {root.id: [Axis(a), Axis(b)]})
    paxes = axes.with_modified_node(axes.root, numbering=numbering)

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    iterset = AxisTree(Axis([root.components[1]], root.label))
    do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))

    # with the renumbering dat1 now looks like:
    #   [(pt1, 0), (pt0, 0), (pt0, 1), (pt1, 1), (pt0, 2)]
    # whereas dat0 looks like
    #   [(pt0, 0), (pt0, 1), (pt0, 2), (pt1, 0), (pt1, 1)]
    assert not np.allclose(dat1.data_ro, dat0.data_ro)

    izero = [
        [("pt0", 0), 0],
        [("pt0", 0), 1],
        [("pt0", 1), 0],
        [("pt0", 1), 1],
        [("pt0", 2), 0],
        [("pt0", 2), 1],
    ]
    icopied = [
        [("pt1", 0), 0],
        [("pt1", 0), 1],
        [("pt1", 0), 2],
        [("pt1", 1), 0],
        [("pt1", 1), 1],
        [("pt1", 1), 2],
    ]
    for ix in izero:
        assert np.allclose(dat1.get_value(ix), 0.0)
    for ix in icopied:
        assert np.allclose(dat1.get_value(ix), dat0.get_value(ix))


def test_scalar_copy_with_permuted_inner_axis(scalar_copy_kernel):
    m, n = 4, 3
    numbering = np.asarray([1, 2, 0], dtype=IntType)
    npoints = m * n

    axes = AxisTree(
        root := Axis(m, "ax0"), {root.id: Axis(n, "ax1", numbering=numbering)}
    )
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat0.data, dat1.data)

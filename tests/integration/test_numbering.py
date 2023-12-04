import ctypes

import loopy as lp
import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import pmap

import pyop3 as op3
from pyop3.ir.lower import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


@pytest.fixture
def vector_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (3,), is_input=False, is_output=True),
        ],
        name="vector_copy",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(code, [op3.READ, op3.WRITE])


def test_scalar_copy_with_permuted_inner_axis(scalar_copy_kernel):
    m, n = 4, 3
    numbering = [1, 2, 0]

    axis0 = op3.Axis(m)
    axis1 = op3.Axis(n)
    paxis1 = axis1.copy(numbering=numbering)
    axes = op3.AxisTree.from_nest({axis0: axis1})
    paxes = op3.AxisTree.from_nest({axis0: paxis1})

    dat0 = op3.Dat(axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(paxes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_vector_copy_with_permuted_axis(vector_copy_kernel):
    m, n = 6, 3
    numbering = [2, 5, 1, 0, 4, 3]

    axis0 = op3.Axis(m)
    axis1 = op3.Axis(n)
    axes = op3.AxisTree.from_nest({axis0: axis1})

    paxis0 = axis0.copy(numbering=numbering)
    paxes = op3.AxisTree.from_nest({paxis0: axis1})

    dat0 = op3.Dat(axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(paxes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data, dat0.data)


def test_vector_copy_with_two_permuted_axes(vector_copy_kernel):
    a, b, c = 4, 2, 3
    numbering0 = [2, 1, 3, 0]
    numbering1 = [1, 0]

    axis0 = op3.Axis(a)
    axis1 = op3.Axis(b)
    axis2 = op3.Axis(c)
    axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})

    paxis0 = axis0.copy(numbering=numbering0)
    paxis1 = axis1.copy(numbering=numbering1)
    paxes = op3.AxisTree.from_nest({paxis0: {paxis1: axis2}})

    dat0 = op3.Dat(axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(paxes, name="dat1", dtype=dat0.dtype)

    iterset = op3.AxisTree.from_nest({axis0: axis1})
    op3.do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_vector_copy_with_permuted_inner_axis(vector_copy_kernel):
    a, b, c = 5, 4, 3
    numbering = [2, 1, 3, 0]

    axis0 = op3.Axis(a)
    axis1 = op3.Axis(b)
    axis2 = op3.Axis(c)
    axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})

    paxis1 = axis1.copy(numbering=numbering)
    paxes = op3.AxisTree.from_nest({axis0: {paxis1: axis2}})

    dat0 = op3.Dat(axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(paxes, name="dat1", dtype=dat0.dtype)

    iterset = op3.AxisTree.from_nest({axis0: axis1})
    op3.do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_vector_copy_with_permuted_multi_component_axes(vector_copy_kernel):
    m, n = 3, 2
    a, b = 2, 3
    numbering = [4, 2, 0, 3, 1]

    root = op3.Axis({"a": m, "b": n})
    proot = root.copy(numbering=numbering)
    axes = op3.AxisTree.from_nest({root: [op3.Axis(a), op3.Axis(b)]})
    paxes = op3.AxisTree.from_nest({proot: [op3.Axis(a), op3.Axis(b)]})

    dat0 = op3.Dat(axes, name="dat0", data=np.arange(axes.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(paxes, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := root["b"].index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))

    # with the renumbering dat1 now looks like
    #   [b0, a0, a1, b1, a2]
    # whereas dat0 looks like
    #   [a0, a1, a2, b0, b1]
    assert not np.allclose(dat1.data_ro, dat0.data_ro)

    izero = [
        [("a", 0), 0],
        [("a", 0), 1],
        [("a", 1), 0],
        [("a", 1), 1],
        [("a", 2), 0],
        [("a", 2), 1],
    ]
    icopied = [
        [("b", 0), 0],
        [("b", 0), 1],
        [("b", 0), 2],
        [("b", 1), 0],
        [("b", 1), 1],
        [("b", 1), 2],
    ]
    for ix in izero:
        assert np.allclose(dat1.get_value(ix), 0.0)
    for ix in icopied:
        assert np.allclose(dat1.get_value(ix), dat0.get_value(ix))

import ctypes

import loopy as lp
import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import pmap

import pyop3 as op3
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


def test_scalar_copy_with_ragged_axis(scalar_copy_kernel):
    m = 5
    nnzdata = np.array([3, 2, 1, 3, 2], dtype=IntType)

    root = Axis(m)
    nnz = MultiArray(root, name="nnz", data=nnzdata, max_value=3)

    axes = AxisTree(root, {root.id: Axis(nnz)})
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=ScalarType)

    do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_with_two_ragged_axes(scalar_copy_kernel):
    m = 3
    nnzdata0 = np.asarray([3, 1, 2], dtype=IntType)
    nnzdata1 = np.asarray([1, 1, 5, 4, 2, 3], dtype=IntType)

    nnzaxes0 = AxisTree(Axis(m))
    nnz0 = MultiArray(
        nnzaxes0,
        name="nnz0",
        data=nnzdata0,
        max_value=3,
    )

    nnzaxes1 = nnzaxes0.add_subaxis(Axis(nnz0), *nnzaxes0.leaf)
    nnz1 = MultiArray(nnzaxes1, name="nnz1", data=nnzdata1, max_value=5)

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax2"), *nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_two_ragged_loops_with_fixed_loop_between(scalar_copy_kernel):
    m, n = 3, 2
    nnzdata0 = np.asarray([1, 3, 2], dtype=IntType)
    nnzdata1 = np.asarray(
        flatten([[[1, 2]], [[2, 1], [1, 1], [1, 1]], [[2, 3], [3, 1]]]), dtype=IntType
    )

    nnzaxes0 = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(nnzaxes0, name="nnz0", data=nnzdata0, max_value=3)

    nnzaxes1 = nnzaxes0.add_subaxis(
        Axis(nnz0, "ax1", id="ax1"), *nnzaxes0.leaf
    ).add_subaxis(Axis(n, "ax2"), "ax1")
    nnz1 = MultiArray(nnzaxes1, name="nnz1", data=nnzdata1, max_value=3)

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax3"), *nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    # do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    l = loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    l()
    print(l.loopy_code)
    print(dat1.data)
    # breakpoint()
    # assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_ragged_axis_inside_two_fixed_axes(scalar_copy_kernel):
    m, n = 2, 2
    nnzdata = np.asarray(flatten([[1, 2], [1, 2]]), dtype=IntType)

    nnzaxes = AxisTree(Axis(m, "ax0", id="root"), {"root": Axis(n, "ax1")})
    nnz = MultiArray(
        nnzaxes,
        name="nnz",
        data=nnzdata,
        max_value=max(nnzdata),
    )

    axes = nnzaxes.add_subaxis(Axis(nnz, "ax2"), *nnzaxes.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data, dat0.data)


@pytest.mark.skip(reason="passing parameters through to local kernel needs work")
def test_ragged_copy(ragged_copy_kernel):
    m = 5
    nnzdata = np.asarray([3, 2, 1, 3, 2], dtype=IntType)

    nnzaxes = AxisTree(Axis(m, "ax0"))
    nnz = MultiArray(
        nnzaxes,
        name="nnz",
        data=nnzdata,
        max_value=3,
    )

    axes = nnzaxes.add_subaxis(Axis([(nnz, "cpt0")], "ax1"), *nnzaxes.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=ScalarType)

    p = nnzaxes.index
    q = p.add_node(Index(Slice(axis="ax1", cpt="cpt0")), *p.leaf)
    do_loop(p, ragged_copy_kernel(dat0[q], dat1[q]))

    assert np.allclose(dat1.data, dat0.data)


@pytest.mark.xfail(reason="complex ragged temporary logic not implemented")
def test_nested_ragged_copy_with_independent_subaxes(nested_ragged_copy_kernel):
    m = 3
    nnzdata0 = np.asarray([3, 2, 1], dtype=IntType)
    nnzdata1 = np.asarray([2, 1, 2], dtype=IntType)
    npoints = sum(a * b for a, b in zip(nnzdata0, nnzdata1))

    nnzaxes = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(
        nnzaxes,
        name="nnz0",
        data=nnzdata0,
        max_value=3,
    )
    nnz1 = MultiArray(
        nnzaxes,
        name="nnz1",
        data=nnzdata1,
        max_value=2,
    )

    axes = AxisTree(Axis(m, "ax0"))
    axes = axes.add_subaxis(Axis(nnz0, "ax1"), axes.leaf)
    axes = axes.add_subaxis(Axis(nnz1, "ax2"), axes.leaf)

    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    q = p.copy()
    q = q.put_node(Index(Range("ax1", nnz0[p])), q.leaf)
    q = q.put_node(Index(Range("ax2", nnz1[p])), q.leaf)

    do_loop(p, nested_ragged_copy_kernel(dat0[q], dat1[q]))

    assert np.allclose(dat1.data, dat0.data)


@pytest.mark.xfail(reason="need to pass layout function through to the local kernel")
def test_nested_ragged_copy_with_dependent_subaxes(nested_dependent_ragged_copy_kernel):
    m = 3
    nnzdata0 = np.asarray([2, 0, 1], dtype=IntType)
    nnzdata1 = np.asarray(flatten([[2, 1], [], [2]]), dtype=IntType)
    npoints = sum(nnzdata1)

    nnzaxes0 = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(
        nnzaxes0,
        name="nnz0",
        data=nnzdata0,
        max_value=3,
    )

    nnzaxes1 = nnzaxes0.add_subaxis(Axis(nnz0, "ax1"), nnzaxes0.leaf)
    nnz1 = MultiArray(
        nnzaxes1,
        name="nnz1",
        data=nnzdata1,
        max_value=2,
    )

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax2"), nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    q = p.copy()
    q = q.put_node(Index(Range("ax1", nnz0[q])), q.leaf)
    q = q.put_node(Index(Range("ax2", nnz1[q])), q.leaf)

    do_loop(p, nested_dependent_ragged_copy_kernel(dat0[q], dat1[q]))

    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_of_ragged_component_in_multi_component_axis(scalar_copy_kernel):
    m0, m1, m2 = 4, 5, 6
    n0, n1 = 1, 2
    nnz_data = np.asarray([3, 2, 1, 2, 1], dtype=IntType)

    nnz_axis = op3.Axis({"pt1": m1}, "ax0")
    nnz = op3.Dat(
        nnz_axis,
        name="nnz",
        data=nnz_data,
        max_value=max(nnz_data),
    )

    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": m0, "pt1": m1, "pt2": m2}, "ax0"): [
                op3.Axis(n0),
                op3.Axis({"pt0": nnz}, "ax1"),
                op3.Axis(n1),
            ]
        }
    )

    dat0 = op3.Dat(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = op3.Dat(axes, name="dat1", dtype=dat0.dtype)

    iterset = op3.AxisTree.from_nest(
        {
            nnz_axis: op3.Axis({"pt0": nnz}, "ax1"),
        }
    )
    do_loop(p := iterset.index(), scalar_copy_kernel(dat0[p], dat1[p]))

    off = np.cumsum([m0 * n0, sum(nnz_data), m2 * n1])
    assert np.allclose(dat1.data_ro[: off[0]], 0)
    assert np.allclose(dat1.data_ro[off[0] : off[1]], dat0.data_ro[off[0] : off[1]])
    assert np.allclose(dat1.data_ro[off[1] :], 0)


def test_scalar_copy_of_permuted_axis_with_ragged_inner_axis(scalar_copy_kernel):
    m = 3
    nnzdata = np.asarray([2, 0, 4], dtype=IntType)
    npoints = sum(nnzdata)
    numbering = np.asarray([2, 1, 0], dtype=IntType)

    nnzaxis = Axis(m, "ax0")
    nnz = MultiArray(
        nnzaxis,
        name="nnz",
        data=nnzdata,
        max_value=4,
    )

    axes = AxisTree(nnzaxis, {nnzaxis.id: Axis(nnz, "ax1")})
    paxes = axes.with_modified_node(axes.root, numbering=numbering)

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=dat0.dtype)

    do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_of_permuted_then_ragged_then_permuted_axes(scalar_copy_kernel):
    m, n = 3, 2
    nnzdata = np.asarray([2, 1, 3], dtype=IntType)
    num0 = np.asarray([2, 1, 0], dtype=IntType)
    num1 = np.asarray([1, 0], dtype=IntType)
    npoints = sum(nnzdata) * n

    nnzaxis = Axis(m)
    nnz = MultiArray(
        nnzaxis,
        name="nnz",
        data=nnzdata,
        max_value=max(nnzdata),
    )

    axes = op3.AxisTree(
        nnzaxis, {nnzaxis.id: op3.Axis(nnz, id="ax0"), "ax0": op3.Axis(n)}
    )

    # axes = op3.AxisTree.from_nested(
    #     {
    #         nnzaxis:
    #     }
    # )

    paxes = axes.with_modified_node(axes.root, numbering=num0).with_modified_node(
        axes.leaf[0], numbering=num1
    )

    dat0 = op3.Dat(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = op3.Dat(paxes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))
    assert np.allclose(dat1.data_ro, dat0.data_ro)

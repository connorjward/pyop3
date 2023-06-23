import ctypes

import loopy as lp
import numpy as np
import pymbolic as pym
import pytest

from pyop3.axis import Axis, AxisComponent, AxisTree
from pyop3.codegen import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.distarray import MultiArray
from pyop3.dtypes import IntType, ScalarType
from pyop3.index import AffineMap, IdentityMap, Index, IndexTree, Range, TabulatedMap
from pyop3.loopexpr import INC, READ, WRITE, LoopyKernel, do_loop
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


@pytest.fixture
def vector_inc_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="vector_inc",
        lang_version=(2018, 2),
    )
    return LoopyKernel(code, [READ, INC])


@pytest.fixture
def ragged_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, shape=None, is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, shape=None, is_input=False, is_output=True),
            lp.ValueArg("n", dtype=IntType),
        ],
        assumptions="n <= 3",
        target=LOOPY_TARGET,
        name="ragged_copy",
        lang_version=(2018, 2),
    )
    return LoopyKernel(code, [READ, WRITE])


@pytest.fixture
def ragged_inc_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, shape=None, is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, shape=(1,), is_input=True, is_output=True),
            lp.ValueArg("n", dtype=IntType),
        ],
        assumptions="n <= 3",
        target=LOOPY_TARGET,
        name="ragged_inc",
        lang_version=(2018, 2),
    )
    return LoopyKernel(code, [READ, INC])


@pytest.fixture
def nested_dependent_ragged_copy_kernel():
    code = lp.make_kernel(
        [
            "{ [i]: 0 <= i < n0 }",
            "{ [j]: 0 <= j < layout0[n0] }",
        ],
        "y[i*n1+j] = x[i*n1+j]",
        [
            lp.GlobalArg("x", ScalarType, shape=None, is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, shape=None, is_input=False, is_output=True),
            lp.GlobalArg(
                "layout0", IntType, shape=None, is_input=True, is_output=False
            ),
            lp.ValueArg("n0", dtype=IntType),
        ],
        assumptions="n0 <= 3 and n1 <= 3",
        target=LOOPY_TARGET,
        name="ragged_copy",
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
        data=np.zeros(m, dtype=ScalarType),
    )
    do_loop(p := Index(Range(axis.label, m)), scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data, dat0.data)


def test_vector_copy(vector_copy_kernel):
    m, n = 10, 3

    axes = AxisTree(
        root := Axis(m),
        {
            root.id: Axis(n),
        },
    )
    dat1 = MultiArray(
        axes,
        name="dat1",
        data=np.arange(m * n, dtype=np.float64),
    )
    dat2 = MultiArray(
        axes,
        name="dat2",
        data=np.zeros(m * n, dtype=np.float64),
    )

    do_loop(p := Index(Range(axes.root.label, m)), vector_copy_kernel(dat1[p], dat2[p]))

    assert np.allclose(dat2.data, dat1.data)


def test_multi_component_vector_copy(vector_copy_kernel):
    m, n, a, b = 4, 6, 2, 3

    axes = AxisTree(
        root := Axis([m, n]),
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
        data=np.arange(m * a + n * b, dtype=np.float64),
        dtype=np.float64,
    )
    dat1 = MultiArray(
        axes,
        name="dat1",
        data=np.zeros(m * a + n * b, dtype=np.float64),
        dtype=np.float64,
    )
    do_loop(
        p := Index(Range((axes.root.label, 1), n)), vector_copy_kernel(dat0[p], dat1[p])
    )

    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])


def test_multi_component_scalar_copy_with_two_outer_loops(scalar_copy_kernel):
    m, n, a, b = 8, 6, 2, 3

    axes = AxisTree(
        root := Axis(
            [
                m,
                n,
            ],
            "ax_label0",
        ),
        {
            root.id: [
                Axis(a),
                Axis(b, "ax_label1"),
            ]
        },
    )
    dat0 = MultiArray(
        axes, name="dat0", data=np.arange(m * a + n * b, dtype=np.float64)
    )
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(m * a + n * b, dtype=np.float64))
    p = IndexTree(
        root := Index(Range(("ax_label0", 1), n)),
        {root.id: Index(Range("ax_label1", b))},
    )
    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert all(dat1.data[: m * a] == 0)
    assert all(dat1.data[m * a :] == dat0.data[m * a :])


def test_vector_copy_with_permuted_axis(vector_copy_kernel):
    m, n = 6, 3
    perm = np.asarray([3, 2, 0, 5, 4, 1], dtype=IntType)

    axes = AxisTree(root := Axis(m), {root.id: Axis(n)})
    paxes = axes.with_modified_node(
        axes.root,
        permutation=perm,
    )
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m * n, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", data=np.zeros(m * n, dtype=ScalarType))

    do_loop(p := Index(Range(axes.root.label, m)), vector_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data.reshape((m, n))[perm].flatten(), dat0.data)


def test_vector_copy_with_two_permuted_axes(vector_copy_kernel):
    a, b, c = 4, 2, 3
    perm0 = [3, 1, 0, 2]
    perm1 = [1, 0]

    axis0 = Axis(a, "ax_label0")
    axis1 = Axis(b, "ax_label1")
    axes = AxisTree(
        axis0,
        {
            axis0.id: axis1,
            axis1.id: Axis(c),
        },
    )
    paxes = axes.with_modified_node(axis0, permutation=perm0).with_modified_node(
        axis1, permutation=perm1
    )

    dat0 = MultiArray(axes, name="dat0", data=np.arange(a * b * c, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", data=np.zeros(a * b * c, dtype=ScalarType))

    p = IndexTree(
        root := Index(Range("ax_label0", a)), {root.id: Index(Range("ax_label1", b))}
    )
    do_loop(p, vector_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(
        dat1.data.reshape((a, b, c))[perm0][:, perm1].flatten(), dat0.data
    )


def test_vector_copy_with_permuted_inner_axis(vector_copy_kernel):
    a, b, c = 5, 4, 3
    perm = [3, 1, 0, 2]

    inner_axis = Axis(b)
    inner_paxis = inner_axis.copy(permutation=perm)

    axes = AxisTree(
        root := Axis(a, "ax_label0"), {root.id: inner_axis, inner_axis.id: Axis(c)}
    )
    paxes = AxisTree(
        root := Axis(a, "ax_label0"), {root.id: inner_paxis, inner_paxis.id: Axis(c)}
    )

    dat0 = MultiArray(axes, name="dat0", data=np.arange(a * b * c, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", data=np.zeros(a * b * c, dtype=ScalarType))

    p = IndexTree(
        root := Index(Range("ax_label0", a)),
        {
            root.id: Index(Range(inner_axis.label, b)),
        },
    )
    do_loop(p, vector_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data.reshape((a, b, c))[:, perm].flatten(), dat0.data)


def test_vector_copy_with_permuted_multi_component_axes(vector_copy_kernel):
    m, n, a, b = 3, 2, 2, 3
    perm = [4, 2, 0, 3, 1]

    fullperm = [10, 11] + [5, 6] + [0, 1] + [7, 8, 9] + [2, 3, 4]

    axes = AxisTree(root := Axis([m, n]), {root.id: [Axis(a), Axis(b)]})
    paxes = axes.with_modified_node(axes.root, permutation=perm)

    dat0 = MultiArray(
        axes, name="dat0", data=np.arange(m * a + n * b, dtype=ScalarType)
    )
    dat1 = MultiArray(
        paxes, name="dat1", data=np.zeros(m * a + n * b, dtype=ScalarType)
    )

    p = IndexTree(Index(Range((axes.root.label, 1), n)))
    do_loop(p, vector_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data[fullperm][: m * a], 0)
    assert np.allclose(dat1.data[fullperm][m * a :], dat0.data[m * a :])


def test_scalar_copy_with_ragged_axis(scalar_copy_kernel):
    m = 5
    nnzdata = np.array([3, 2, 1, 3, 2], dtype=IntType)
    npoints = sum(nnzdata)

    nnzaxes = AxisTree(Axis(5, "ax_label0"))
    nnz = MultiArray(nnzaxes, name="nnz", data=nnzdata, max_value=3)

    axes = nnzaxes.add_subaxis(Axis(AxisComponent(nnz), "ax_label1"), nnzaxes.root)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax_label0", m)))
    p = p.put_node(Index(Range("ax_label1", nnz[p])), p.root)
    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_with_two_ragged_axes(scalar_copy_kernel):
    m = 3
    nnzdata0 = np.asarray([3, 1, 2], dtype=IntType)
    nnzdata1 = np.asarray([1, 1, 5, 4, 2, 3], dtype=IntType)
    npoints = sum(nnzdata1)

    nnzaxes0 = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(
        nnzaxes0,
        name="nnz0",
        data=nnzdata0,
        max_value=3,
    )

    nnzaxes1 = nnzaxes0.add_subaxis(Axis(nnz0, "ax1"), nnzaxes0.root)
    nnz1 = MultiArray(nnzaxes1, name="nnz1", data=nnzdata1, max_value=5)

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax2"), nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", 3)))
    p = p.put_node(Index(Range("ax1", nnz0[p])), p.leaf)
    p = p.put_node(Index(Range("ax2", nnz1[p])), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_two_ragged_loops_with_fixed_loop_between(scalar_copy_kernel):
    m, n = 3, 2
    nnzdata0 = np.asarray([1, 3, 2], dtype=IntType)
    nnzdata1 = np.asarray(
        flatten([[[1, 2]], [[2, 1], [1, 1], [1, 1]], [[2, 3], [3, 1]]]), dtype=IntType
    )
    npoints = sum(nnzdata1)

    nnzaxes0 = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(nnzaxes0, name="nnz0", data=nnzdata0, max_value=3)

    nnzaxes1 = nnzaxes0.add_subaxis(
        subaxis := Axis(nnz0, "ax1"), nnzaxes0.leaf
    ).add_subaxis(Axis(n, "ax2"), subaxis)
    nnz1 = MultiArray(nnzaxes1, name="nnz1", data=nnzdata1, max_value=3)

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax3"), nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    p = p.put_node(Index(Range("ax1", nnz0[p])), p.leaf)
    p = p.put_node(Index(Range("ax2", n)), p.leaf)
    p = p.put_node(Index(Range("ax3", nnz1[p])), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data, dat0.data)


def test_scalar_copy_ragged_axis_inside_two_fixed_axes(scalar_copy_kernel):
    m, n = 2, 2
    nnzdata = np.asarray(flatten([[1, 2], [1, 2]]), dtype=IntType)
    npoints = sum(nnzdata)

    nnzaxes = AxisTree(root := Axis(m, "ax0"), {root.id: Axis(n, "ax1")})
    nnz = MultiArray(
        nnzaxes,
        name="nnz",
        data=nnzdata,
        max_value=2,
    )

    axes = nnzaxes.add_subaxis(Axis(nnz, "ax2"), nnzaxes.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    p = p.put_node(Index(Range("ax1", n)), p.leaf)
    p = p.put_node(Index(Range("ax2", nnz[p])), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data, dat0.data)


def test_ragged_copy(ragged_copy_kernel):
    m = 5
    nnzdata = np.asarray([3, 2, 1, 3, 2], dtype=IntType)
    npoints = sum(nnzdata)

    nnzaxes = AxisTree(Axis(m, "ax0"))
    nnz = MultiArray(
        nnzaxes,
        name="nnz",
        data=nnzdata,
        max_value=3,
    )

    axes = nnzaxes.add_subaxis(Axis(nnz, "ax1"), nnzaxes.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    q = p.put_node(Index(Range("ax1", nnz[p])), p.root)

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
    nnzdata = np.asarray([3, 2, 1, 2, 1], dtype=IntType)
    npoints = m0 * n0 + sum(nnzdata) + m2 * n1

    nnzaxes = AxisTree(Axis(m1, "ax0"))
    nnz = MultiArray(
        nnzaxes,
        name="nnz",
        data=nnzdata,
        max_value=3,
    )

    axes = AxisTree(
        root := Axis(
            [
                m0,
                m1,
                m2,
            ],
            "ax0",
        ),
        {
            root.id: [Axis(n0), Axis(nnz, "ax1"), Axis(n1)],
        },
    )
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    # TODO cleanup
    # Need to make sure that we index things with the same label here
    range0 = Range(("ax0", 0), m1)
    range1 = Range(("ax0", 1), m1)
    idx0 = Index(range0)
    idx1 = idx0.copy(components=range1)

    p = IndexTree(idx1)
    p = p.put_node(Index(Range("ax1", nnz[idx0])), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    off = np.cumsum([m0 * n0, sum(nnzdata), m2 * n1])
    assert np.allclose(dat1.data[: off[0]], 0)
    assert np.allclose(dat1.data[off[0] : off[1]], dat0.data[off[0] : off[1]])
    assert np.allclose(dat1.data[off[1] :], 0)


def test_scalar_copy_of_permuted_axis_with_ragged_inner_axis(scalar_copy_kernel):
    m = 3
    nnzdata = np.asarray([2, 0, 4], dtype=IntType)
    npoints = sum(nnzdata)
    perm = np.asarray([2, 1, 0], dtype=IntType)

    fullperm = [4, 5] + [] + [0, 1, 2, 3]
    assert len(fullperm) == npoints

    nnzaxis = Axis(m, "ax0")
    nnz = MultiArray(
        nnzaxis,
        name="nnz",
        data=nnzdata,
        max_value=4,
    )

    axes = AxisTree(root := nnzaxis, {root.id: Axis(nnz, "ax1")})
    paxes = axes.with_modified_node(axes.root, permutation=perm)

    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(root := Index(Range("ax0", m)))
    p = p.put_node(Index(Range("ax1", nnz[p])), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data[fullperm], dat0.data)


def test_scalar_copy_of_permuted_then_ragged_then_permuted_axes(scalar_copy_kernel):
    m, n = 3, 2
    nnzdata = np.asarray([2, 1, 3], dtype=IntType)
    perm0 = np.asarray([2, 1, 0], dtype=IntType)
    perm1 = np.asarray([1, 0], dtype=IntType)
    npoints = sum(nnzdata) * n

    fullperm = [9, 8, 11, 10] + [7, 6] + [1, 0, 3, 2, 5, 4]
    assert len(fullperm) == npoints

    nnzaxes = AxisTree(Axis(m, "ax0"))
    nnz = MultiArray(
        nnzaxes,
        name="nnz",
        data=nnzdata,
        max_value=3,
    )

    axes = nnzaxes.copy()
    axes = axes.add_subaxis(Axis(nnz, "ax1"), axes.leaf)
    axes = axes.add_subaxis(Axis(n, "ax2"), axes.leaf)

    paxes = axes.copy()
    paxes = paxes.with_modified_node(paxes.root, permutation=perm0)
    paxes = paxes.with_modified_node(paxes.leaf, permutation=perm1)

    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    p = p.put_node(Index(Range("ax1", nnz[p])), p.leaf)
    p = p.put_node(Index(Range("ax2", n)), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data[fullperm], dat0.data)


def test_scalar_copy_with_permuted_inner_axis(scalar_copy_kernel):
    m, n = 4, 3
    perm = np.asarray([2, 0, 1], dtype=IntType)
    npoints = m * n

    axes = AxisTree(root := Axis(m, "ax0"), {root.id: Axis(n, "ax1", permutation=perm)})
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(root := Index(Range("ax0", m)), {root.id: Index(Range("ax1", n))})
    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat0.data, dat1.data)


def test_scalar_copy_of_subset(scalar_copy_kernel):
    m, n = 6, 4
    sdata = np.asarray([2, 3, 5, 0], dtype=IntType)
    untouched = [1, 4]

    axes = AxisTree(Axis(m, "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(m, dtype=ScalarType))

    # a subset is really a map from a small set into a larger one
    saxes = AxisTree(root := Axis(n, "sax0"), {root.id: Axis(1)})
    subset = MultiArray(saxes, name="subset0", data=sdata)

    p = IndexTree(Index(Range("sax0", n)))
    p = p.put_node(
        Index(TabulatedMap([("sax0", 0)], [("ax0", 0)], arity=1, data=subset[p])),
        p.leaf,
    )

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data[sdata], dat0.data[sdata])
    assert np.allclose(dat1.data[untouched], 0)


def test_inc_from_tabulated_map(vector_inc_kernel):
    m, n = 4, 3
    mapdata = np.asarray([[1, 2, 0], [2, 0, 1], [3, 2, 3], [2, 0, 1]], dtype=IntType)

    axes = AxisTree(Axis(m, "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(m, dtype=ScalarType))

    maxes = axes.add_subaxis(Axis(n), axes.leaf)
    map0 = MultiArray(
        maxes,
        name="map0",
        data=mapdata.flatten(),
    )

    p = IndexTree(Index(Range("ax0", m)))
    q = p.put_node(
        Index(TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=n, data=map0[p])), p.leaf
    )

    do_loop(p, vector_inc_kernel(dat0[q], dat1[p]))

    # Since dat0.data is simply arange, accessing index i of it will also just be i.
    # Therefore, accessing multiple entries and storing the sum of them is
    # equivalent to summing the indices from the map.
    assert np.allclose(dat1.data, np.sum(mapdata, axis=1))


def test_inc_from_multi_component_temporary(vector_inc_kernel):
    m, n = 3, 4
    arity = 2
    mapdata = np.asarray([[1, 2], [0, 1], [3, 2]], dtype=IntType)

    axes0 = AxisTree(Axis([m, n], "ax0"))
    axes1 = AxisTree(Axis(m, "ax0"))

    dat0 = MultiArray(axes0, name="dat0", data=np.arange(m + n, dtype=ScalarType))
    dat1 = MultiArray(axes1, name="dat1", data=np.zeros(m, dtype=ScalarType))

    maxes = AxisTree(root := Axis(m, "ax0"), {root.id: Axis(arity)})
    map0 = MultiArray(maxes, name="map0", data=mapdata.flatten())

    p = IndexTree(Index(Range("ax0", m)))
    q = p.put_node(
        Index(
            [
                IdentityMap([("ax0", 0)], [("ax0", 0)], arity=1),
                TabulatedMap([("ax0", 0)], [("ax0", 1)], arity=arity, data=map0[p]),
            ]
        ),
        p.leaf,
    )

    do_loop(p, vector_inc_kernel(dat0[q], dat1[p]))

    # The expected value is the current index (from the identity map), plus the values
    # from the map. Since the indices in the map are offset in the actual array we
    # also need to add this.
    assert np.allclose(dat1.data, np.arange(m) + np.sum(mapdata + m, axis=1))


def test_copy_multi_component_temporary(vector_copy_kernel):
    m = 4
    n0, n1 = 2, 1
    npoints = m * n0 + m * n1

    axes = AxisTree(root := Axis(m, "ax0"), {root.id: Axis([n0, n1], "ax1")})
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    q = p.put_node(Index([Range(("ax1", 0), n0), Range(("ax1", 1), n1)]), p.leaf)

    do_loop(p, vector_copy_kernel(dat0[q], dat1[q]))

    assert np.allclose(dat1.data, dat0.data)


def test_inc_from_index_function(vector_inc_kernel):
    m, n = 3, 4

    axes0 = AxisTree(Axis([m, n], "ax0"))
    axes1 = AxisTree(Axis(m, "ax0"))

    dat0 = MultiArray(axes0, name="dat0", data=np.arange(m + n, dtype=ScalarType))
    dat1 = MultiArray(axes1, name="dat1", data=np.zeros(m, dtype=ScalarType))

    j0 = pym.var("j0")
    j1 = pym.var("j1")
    vars = (j0, j1)
    mapexpr = (vars, j0 + j1)

    p = IndexTree(Index(Range("ax0", m)))
    q = p.put_node(
        Index(
            [
                IdentityMap([("ax0", 0)], [("ax0", 0)], arity=1),
                AffineMap([("ax0", 0)], [("ax0", 1)], arity=2, expr=mapexpr),
            ]
        ),
        p.leaf,
    )

    do_loop(p, vector_inc_kernel(dat0[q], dat1[p]))

    bit0 = np.arange(m)  # from the identity map
    bit1 = np.arange(m, m + n - 1) + np.arange(m + 1, m + n)  # from the affine map
    assert np.allclose(dat1.data, bit0 + bit1)


def test_inc_with_multiple_maps(vector_inc_kernel):
    m = 5
    arity0, arity1 = 2, 1
    mapdata0 = np.asarray([[1, 2], [0, 2], [0, 1], [3, 4], [2, 1]], dtype=IntType)
    mapdata1 = np.asarray([[1], [1], [3], [0], [2]], dtype=IntType)

    axes = AxisTree(Axis(m, "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(m, dtype=ScalarType))

    maxes0 = axes.add_subaxis(Axis(arity0), axes.leaf)
    maxes1 = axes.add_subaxis(Axis(arity1), axes.leaf)

    map0 = MultiArray(
        maxes0,
        name="map0",
        data=mapdata0.flatten(),
    )
    map1 = MultiArray(
        maxes1,
        name="map1",
        data=mapdata1.flatten(),
    )

    p = IndexTree(Index(Range("ax0", m)))
    q = p.put_node(
        Index(
            [
                TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=arity0, data=map0[p]),
                TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=arity1, data=map1[p]),
            ]
        ),
        p.leaf,
    )

    do_loop(p, vector_inc_kernel(dat0[q], dat1[p]))

    assert np.allclose(dat1.data, np.sum(mapdata0, axis=1) + np.sum(mapdata1, axis=1))


def test_inc_with_map_composition(vector_inc_kernel):
    m = 5
    arity0, arity1 = 1, 3
    mapdata0 = np.asarray([[2], [3], [1], [0], [0]], dtype=IntType)
    mapdata1 = np.asarray(
        [[0, 4, 1], [2, 1, 3], [4, 2, 4], [0, 1, 2], [4, 2, 3]], dtype=IntType
    )

    axes = AxisTree(Axis(m, "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(m, dtype=ScalarType))

    maxes0 = axes.add_subaxis(Axis(arity0), axes.root)
    maxes1 = axes.add_subaxis(Axis(arity1), axes.root)

    map0 = MultiArray(maxes0, name="map0", data=mapdata0.flatten())
    map1 = MultiArray(maxes1, name="map1", data=mapdata1.flatten())

    p = IndexTree(Index(Range("ax0", m)))

    q = p.copy()
    q = q.put_node(
        Index(TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=arity0, data=map0[q])),
        q.leaf,
    )
    q = q.put_node(
        Index(TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=arity1, data=map1[q])),
        q.leaf,
    )

    do_loop(p, vector_inc_kernel(dat0[q], dat1[p]))

    expected = np.sum(np.sum(np.arange(m)[mapdata1], axis=1)[mapdata0], axis=1)
    assert np.allclose(dat1.data, expected)


def test_inc_with_variable_arity_map(ragged_inc_kernel):
    m = 3
    nnzdata = np.asarray([3, 2, 1], dtype=IntType)
    mapdata = [[2, 1, 0], [2, 1], [2]]

    axes = AxisTree(Axis(m, "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(m, dtype=ScalarType))

    nnz = MultiArray(axes, name="nnz", data=nnzdata, max_value=3)

    maxes = axes.add_subaxis(Axis(nnz, "ax1"), axes.leaf)
    map0 = MultiArray(
        maxes, name="map0", data=np.asarray(flatten(mapdata), dtype=IntType)
    )

    p = IndexTree(Index(Range("ax0", m)))
    q = p.put_node(
        Index(TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=nnz[p], data=map0[p])),
        p.leaf,
    )

    do_loop(p, ragged_inc_kernel(dat0[q], dat1[p]))

    assert np.allclose(dat1.data, [sum(xs) for xs in mapdata])


def test_loop_over_map(vector_inc_kernel):
    m = 5
    arity0 = 2
    arity1 = 3
    mapdata0 = np.asarray([[1, 2], [0, 2], [0, 1], [3, 4], [2, 1]], dtype=IntType)
    mapdata1 = np.asarray(
        [[3, 2, 4], [0, 2, 3], [3, 0, 2], [1, 4, 2], [1, 1, 3]], dtype=IntType
    )

    axes = AxisTree(Axis(m, "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(m, dtype=ScalarType))

    maxes0 = axes.add_subaxis(Axis(arity0), axes.leaf)
    maxes1 = axes.add_subaxis(Axis(arity1), axes.leaf)

    map0 = MultiArray(
        maxes0,
        name="map0",
        data=mapdata0.flatten(),
    )
    map1 = MultiArray(
        maxes1,
        name="map1",
        data=mapdata1.flatten(),
    )

    p = IndexTree(Index(Range("ax0", m)))
    p = p.put_node(
        Index(TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=arity0, data=map0[p])),
        p.leaf,
    )

    q = p.put_node(
        Index(TabulatedMap([("ax0", 0)], [("ax0", 0)], arity=arity1, data=map1[p])),
        p.leaf,
    )

    do_loop(p, vector_inc_kernel(dat0[q], dat1[p]))

    expected = np.zeros(m)
    for i0 in range(m):
        for i1 in range(arity0):
            expected[mapdata0[i0, i1]] += sum(dat0.data[mapdata1[mapdata0[i0, i1]]])
    assert np.allclose(dat1.data, expected)


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


def test_different_axis_orderings_do_not_change_packing_order():
    m0, m1, m2 = 5, 2, 2
    npoints = m0 * m1 * m2

    code = lp.make_kernel(
        [f"{{ [i]: 0 <= i < {m1} }}", f"{{ [j]: 0 <= j < {m2} }}"],
        "y[i, j] = x[i, j]",
        [
            lp.GlobalArg("x", np.float64, (m1, m2), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (m1, m2), is_input=False, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="copy",
        lang_version=(2018, 2),
    )
    copy_kernel = LoopyKernel(code, [READ, WRITE])

    axis0 = Axis(m0, "ax0")
    axis1 = Axis(m1, "ax1")
    axis2 = Axis(m2, "ax2")

    axes0 = AxisTree(axis0, {axis0.id: [axis1], axis1.id: [axis2]})
    axes1 = AxisTree(axis0, {axis0.id: [axis2], axis2.id: [axis1]})

    data0 = np.arange(npoints, dtype=ScalarType).reshape((m0, m1, m2))
    data1 = data0.swapaxes(1, 2)

    dat0_0 = MultiArray(
        axes0,
        name="dat0_0",
        data=data0.flatten(),
    )
    dat0_1 = MultiArray(axes1, name="dat0_1", data=data1.flatten())
    dat1 = MultiArray(axes0, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m0)))

    q = p.copy()
    q = q.put_node(Index(Range("ax1", m1)), q.leaf)
    q = q.put_node(Index(Range("ax2", m2)), q.leaf)

    do_loop(p, copy_kernel(dat0_0[q], dat1[q]))
    assert np.allclose(dat1.data, dat0_0.data)

    dat1.data[...] = 0

    do_loop(p, copy_kernel(dat0_1[q], dat1[q]))
    assert np.allclose(dat1.data, dat0_0.data)

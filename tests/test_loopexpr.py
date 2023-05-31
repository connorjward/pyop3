import ctypes

import loopy as lp
import numpy as np
import pytest

from pyop3.axis import Axis, AxisComponent, AxisTree
from pyop3.distarray import MultiArray
from pyop3.dtypes import IntType, ScalarType
from pyop3.index import Index, IndexTree, Range, TabulatedMap
from pyop3.loopexpr import READ, WRITE, LoopyKernel, do_loop
from pyop3.utils import flatten


@pytest.fixture
def scalar_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
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
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (3,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="vector_copy",
        lang_version=(2018, 2),
    )
    return LoopyKernel(code, [READ, WRITE])


@pytest.fixture
def ragged_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, shape=None, is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, shape=None, is_input=False, is_output=True),
            lp.ValueArg("n", dtype=np.int32),
        ],
        assumptions="n <= 3",
        target=lp.CTarget(),
        name="ragged_copy",
        lang_version=(2018, 2),
    )
    return pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])


def test_scalar_copy(scalar_copy_kernel):
    m = 10

    axis = Axis(m)
    dat1 = MultiArray(
        axis,
        name="dat1",
        data=np.arange(m, dtype=np.float64),
    )
    dat2 = MultiArray(
        axis,
        name="dat2",
        data=np.zeros(m, dtype=np.float64),
    )
    do_loop(p := Range(axis.label, m), scalar_copy_kernel(dat1[p], dat2[p]))

    assert np.allclose(dat2.data, dat1.data)


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

    do_loop(p := Range(axes.root.label, m), vector_copy_kernel(dat1[p], dat2[p]))

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
    do_loop(p := Range((axes.root.label, 1), n), vector_copy_kernel(dat0[p], dat1[p]))

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

    do_loop(p := Range(axes.root.label, m), vector_copy_kernel(dat0[p], dat1[p]))

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

    fullperm = (
        [10, 11] + [5, 6] + [0, 1] + [7, 8, 9] + [2, 3, 4]  # 4  # 2  # 0  # 3  # 1
    )

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


def test_ragged_loop(scalar_copy_kernel):
    m = 5
    nnzdata = np.array([3, 2, 1, 3, 2], dtype=IntType)
    npoints = sum(nnzdata)

    nnzaxes = AxisTree(Axis(5, "ax_label0"))
    nnz = MultiArray(nnzaxes, name="nnz", data=nnzdata)

    axes = nnzaxes.add_subaxis(Axis(AxisComponent(nnz), "ax_label1"), nnzaxes.root)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax_label0", m)))
    p = p.put_node(Index(Range("ax_label1", nnz[p])), p.root)
    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data, dat0.data)


def test_two_ragged_loops(scalar_copy_kernel):
    m = 3
    nnzdata0 = np.asarray([3, 1, 2], dtype=IntType)
    nnzdata1 = np.asarray([1, 1, 5, 4, 2, 3], dtype=IntType)
    npoints = sum(nnzdata1)

    nnzaxes0 = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(
        nnzaxes0,
        name="nnz0",
        data=nnzdata0,
    )

    nnzaxes1 = nnzaxes0.add_subaxis(Axis(nnz0, "ax1"), nnzaxes0.root)
    nnz1 = MultiArray(
        nnzaxes1,
        name="nnz1",
        data=nnzdata1,
    )

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax2"), nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", 3)))
    p = p.put_node(Index(Range("ax1", nnz0[p])), p.leaf)
    p = p.put_node(Index(Range("ax2", nnz1[p])), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data, dat0.data)


def test_two_ragged_loops_with_fixed_loop_between(scalar_copy_kernel):
    m, n = 3, 2
    nnzdata0 = np.asarray([1, 3, 2], dtype=IntType)
    nnzdata1 = np.asarray(
        flatten([[[1, 2]], [[2, 1], [1, 1], [1, 1]], [[2, 3], [3, 1]]]), dtype=IntType
    )
    npoints = sum(nnzdata1)

    nnzaxes0 = AxisTree(Axis(m, "ax0"))
    nnz0 = MultiArray(nnzaxes0, name="nnz0", data=nnzdata0)

    nnzaxes1 = nnzaxes0.add_subaxis(
        subaxis := Axis(nnz0, "ax1"), nnzaxes0.leaf
    ).add_subaxis(Axis(n, "ax2"), subaxis)
    nnz1 = MultiArray(nnzaxes1, name="nnz1", data=nnzdata1)

    axes = nnzaxes1.add_subaxis(Axis(nnz1, "ax3"), nnzaxes1.leaf)
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = IndexTree(Index(Range("ax0", m)))
    p = p.put_node(Index(Range("ax1", nnz0[p])), p.leaf)
    p = p.put_node(Index(Range("ax2", n)), p.leaf)
    p = p.put_node(Index(Range("ax3", nnz1[p])), p.leaf)

    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat0.data, dat1.data)


def test_ragged_inside_two_standard_loops(scalar_inc_kernel):
    ax1 = MultiAxis([MultiAxisComponent(2, "a", id="p1")])
    ax2 = ax1.add_subaxis("p1", [MultiAxisComponent(2, "b", id="p2")])
    nnz = MultiArray(
        ax2.set_up(),
        name="nnz",
        max_value=2,
        data=np.array([1, 2, 1, 2], dtype=np.int32),
    )
    ax3 = ax2.copy().add_subaxis("p2", [MultiAxisComponent(nnz, "c", id="p3")])

    root = ax3.set_up()
    dat1 = MultiArray(root, name="dat1", data=np.ones(6, dtype=np.float64))
    dat2 = MultiArray(root, name="dat2", data=np.zeros(6, dtype=np.float64))

    p = IndexTree([RangeNode("a", 2, id="a")])
    p.add_node(RangeNode("b", 2, id="b"), "a")
    p.add_node(RangeNode("c", nnz[p.copy()]), "b")

    expr = pyop3.Loop(p, scalar_inc_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout1_0, layout0_0, dat1, dat2)
    layout0_0 = root.node("p3").layout_fn.start

    # TODO: this is affine here, should it generally be?
    layout1_0 = layout0_0.dim.leaf.layout_fn.start

    args = [nnz.data, layout1_0.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_ragged_inner(ragged_copy_kernel):
    raise NotImplementedError(
        "I want to only need to use max_count here and raise an error, dont default"
    )
    ax1 = MultiAxis([MultiAxisComponent(5, label="a", id="p1")])
    nnz = MultiArray(
        ax1.set_up(),
        name="nnz",
        max_value=3,
        data=np.array([3, 2, 1, 3, 2], dtype=np.int32),
    )
    ax2 = ax1.copy().add_subaxis("p1", [MultiAxisComponent(nnz, label="b", id="p2")])

    root = ax2.set_up()
    dat1 = MultiArray(root, name="dat1", data=np.ones(11, dtype=np.float64))
    dat2 = MultiArray(root, name="dat2", data=np.zeros(11, dtype=np.float64))

    p = IndexTree([RangeNode("a", 5)])
    expr = pyop3.Loop(p, ragged_copy_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, dat1, dat2)
    layout0_0 = root.node("p2").layout_fn.start
    args = [nnz.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_compute_double_loop_ragged_mixed(scalar_copy_kernel):
    ax1 = MultiAxis([MultiAxisComponent(5, label=1, id="p1")])
    nnz = MultiArray(
        ax1.set_up(),
        name="nnz",
        data=np.array([3, 2, 1, 2, 1], dtype=np.int32),
    )

    axes = (
        MultiAxis(
            [
                MultiAxisComponent(4, id="p1"),
                MultiAxisComponent(5, label=1, id="p2"),
                MultiAxisComponent(4, id="p3"),
            ]
        )
        .add_subaxis("p1", [MultiAxisComponent(1)])
        .add_subaxis("p2", [MultiAxisComponent(nnz, label=0, id="p4")])
        .add_subaxis("p3", [MultiAxisComponent(2)])
    ).set_up()

    dat1 = MultiArray(axes, name="dat1", data=np.ones(4 + 9 + 8, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(4 + 9 + 8, dtype=np.float64))

    p = IndexTree([RangeNode(1, 5, id="i0")])
    p.add_node(RangeNode(0, nnz[p.copy()]), "i0")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, dat1, dat2)
    layout0_0 = dat1.root.node("p4").layout_fn.start

    args = [nnz.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat2.data[:4], 0)
    assert np.allclose(dat1.data[4:13], dat2.data[4:13])
    assert np.allclose(dat2.data[13:], 0)


def test_compute_ragged_permuted(scalar_copy_kernel):
    nnz = MultiArray(
        MultiAxis([MultiAxisComponent(6, "a")]).set_up(),
        name="nnz",
        data=np.array([3, 2, 0, 1, 3, 2], dtype=np.int32),
    )

    axes = (
        MultiAxis(
            [MultiAxisComponent(6, id="p1", label="a", numbering=[3, 2, 5, 0, 4, 1])]
        ).add_subaxis("p1", [MultiAxisComponent(nnz, label="b")])
    ).set_up()

    dat1 = MultiArray(axes, name="dat1", data=np.ones(11, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(11, dtype=np.float64))

    p = IndexTree([RangeNode("a", 6, id="i0")])
    p.add_node(RangeNode("b", nnz[p.copy()]), "i0")

    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, dat1, dat2)
    layout0_0 = dat1.root.leaf.layout_fn.start
    args = [nnz.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_permuted_ragged_permuted(scalar_copy_kernel):
    nnz = MultiArray(
        MultiAxis([MultiAxisComponent(6, label="a")]).set_up(),
        name="nnz",
        data=np.array([3, 2, 0, 1, 3, 2], dtype=np.int32),
    )

    axes = (
        MultiAxis(
            [MultiAxisComponent(6, id="p1", label="a", numbering=[3, 2, 5, 0, 4, 1])]
        )
        .add_subaxis("p1", [MultiAxisComponent(nnz, id="p2", label="b")])
        .add_subaxis(
            "p2", [MultiAxisComponent(2, numbering=[1, 0], id="p3", label="c")]
        )
    ).set_up()

    dat1 = MultiArray(axes, name="dat1", data=np.ones(22, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(22, dtype=np.float64))

    p = IndexTree([RangeNode("a", 6, id="i0")])
    p.add_node(RangeNode("b", nnz[p.copy()], id="i1"), "i0")
    p.add_node(RangeNode("c", 2), "i1")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, layout1_0, dat1, dat2)
    layout0_0 = axes.node("p2").layout_fn.start
    layout1_0 = axes.node("p3").layout_fn.data

    args = [nnz.data, layout0_0.data, layout1_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_permuted_inner_and_ragged(scalar_copy_kernel):
    # N.B. Do not try to modify the axis labels here, they are supposed to match
    # this demonstrates that naming the multi-axis nodes does the right thing
    axes = MultiAxisTree.from_dict(
        {
            MultiAxis([MultiAxisComponent(2, "x")], "ax1", id="ax1"): None,
            MultiAxis([MultiAxisComponent(2, "x")], "ax2"): ("ax1", "x"),
        }
    )
    # breakpoint()
    nnz = MultiArray(
        axes.copy().set_up(),
        name="nnz",
        data=np.array([3, 2, 1, 1], dtype=np.int32),
    )

    # we currently need to do this because ragged things admit no numbering
    # probably want a .without_numbering() method or similar
    # also, we might want to store the numbering per MultiAxisNode instead of per
    # component. That would then match DMPlex.
    axes = MultiAxisTree.from_dict(
        {
            MultiAxis([MultiAxisComponent(2, "x")], "ax1", id="ax1"): None,
            MultiAxis(
                [MultiAxisComponent(2, "x", numbering=("ax2", [1, 0]))],
                "ax2",
                id="ax2",
            ): (
                "ax1",
                "x",
            ),
            MultiAxisNode([MultiAxisComponent(nnz, "z")], "ax3"): ("ax2", "x"),
        }
    )
    axes.set_up()

    dat1 = MultiArray(axes, name="dat1", data=np.ones(7, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(7, dtype=np.float64))

    p = IndexTree([RangeNode(("ax1", "x"), 2, id="i0")])
    p.add_node(RangeNode(("ax2", "x"), 2, id="i1"), "i0")
    p.add_node(RangeNode(("ax3", "z"), nnz[p.copy()]), "i1")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    layout0_0 = dat1.dim.leaf.components[0].layout_fn.start
    layout1_0 = layout0_0.root.leaf.components[0].layout_fn.start
    args = [nnz.data, layout1_0.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_permuted_inner(scalar_copy_kernel):
    axes = (
        MultiAxis([MultiAxisComponent(4, "a", id="p1")]).add_subaxis(
            "p1", [MultiAxisComponent(3, "b", numbering=[2, 0, 1])]
        )
    ).set_up()

    dat1 = MultiArray(axes, name="dat1", data=np.ones(12, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(12, dtype=np.float64))

    p = IndexTree([RangeNode("a", 4, id="i0")])
    p.add_node(RangeNode("b", 3), "i0")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    layout0_0 = dat1.root.leaf.layout_fn.data
    args = [layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_subset(scalar_copy_kernel):
    axes = MultiAxis([MultiAxisComponent(6, "a")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.ones(6, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(6, dtype=np.float64))

    # a subset is really a map
    subset_axes = MultiAxis([MultiAxisComponent(4, "b", id="p1")])
    subset_axes.add_node(MultiAxisComponent(1, "c"), "p1")
    subset_axes.set_up()
    subset_array = MultiArray(
        subset_axes, prefix="subset", data=np.array([2, 3, 5, 0], dtype=np.int32)
    )

    p = IndexTree([RangeNode("b", 4, id="i0")])
    p.add_node(
        TabulatedMapNode(("b",), ("a",), arity=1, data=subset_array[p.copy()]), "i0"
    )
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [subset_array.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat2.data[[2, 3, 5, 0]], 1)
    assert np.allclose(dat2.data[[1, 4]], 0)


def test_map():
    axes = MultiAxis([MultiAxisComponent(5, "a", id="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    map_axes = axes.copy().add_subaxis("p1", [MultiAxisComponent(2, "b")]).set_up()
    map_array = MultiArray(
        map_axes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=True, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.INC])

    p0 = IndexTree([RangeNode("a", 5, id="i0")])
    p1 = p0.copy()
    p1.add_node(TabulatedMapNode(("a",), ("a",), arity=2, data=map_array[p0]), "i0")

    expr = pyop3.Loop(p0, kernel(dat1[p1], dat2[p0]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map_array.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    assert all(
        dat2.data == np.array([1 + 2, 0 + 2, 0 + 1, 3 + 4, 2 + 1], dtype=np.int32)
    )


def test_closure_ish():
    axes1 = MultiAxis(
        [MultiAxisComponent(3, label="p1"), MultiAxisComponent(4, label="p2")]
    ).set_up()
    dat1 = MultiArray(axes1, name="dat1", data=np.arange(7, dtype=np.float64))
    axes2 = MultiAxis([MultiAxisComponent(3, label="p1")]).set_up()
    dat2 = MultiArray(axes2, name="dat2", data=np.zeros(3, dtype=np.float64))

    # create a map from each cell to 2 edges
    axes3 = (
        MultiAxis([MultiAxisComponent(3, id="p1", label="p1")])
        .add_subaxis("p1", [MultiAxisComponent(2)])
        .set_up()
    )
    map1 = MultiArray(
        axes3, name="map1", data=np.array([1, 2, 0, 1, 3, 2], dtype=np.int32)
    )

    # we have a loop of size 3 here because the temporary has 1 cell DoF and 2 edge DoFs
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=True, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.INC])

    p0 = IndexTree([RangeNode("p1", 3, id="i0")])
    p1 = p0.copy()
    p1.add_node(IdentityMapNode(("p1",), ("p1",), arity=1), "i0")
    p1.add_node(TabulatedMapNode(("p1",), ("p2",), arity=2, data=map1[p0]), "i0")

    expr = pyop3.Loop(p0, kernel(dat1[p1], dat2[p0]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, map1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 1, 3, 2] (-> [4, 5, 3, 4, 6, 5]) and [0, 1, 2]
    assert all(dat2.data == np.array([4 + 5 + 0, 3 + 4 + 1, 6 + 5 + 2], dtype=np.int32))


def test_multipart_inner():
    axes = MultiAxis([MultiAxisComponent(5, label="p1", id="p1")])
    axes.add_nodes(
        [MultiAxisComponent(3, label="p2_0"), MultiAxisComponent(2, label="p2_1")], "p1"
    )

    axes.set_up()

    dat1 = MultiArray(axes, name="dat1", data=np.ones(25, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(25, dtype=np.float64))

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 5 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, (5,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (5,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    p = IndexTree([RangeNode("p1", 5)])
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_index_function():
    """Imagine an interval mesh:

    3 0 4 1 5 2 6
    x---x---x---x
    """
    axes1 = MultiAxis(
        [MultiAxisComponent(3, label="p1"), MultiAxisComponent(4, label="p2")]
    ).set_up()
    dat1 = MultiArray(axes1, name="dat1", data=np.arange(7, dtype=np.float64))
    axes2 = MultiAxis([MultiAxisComponent(3, label="p1")]).set_up()
    dat2 = MultiArray(axes2, name="dat2", data=np.zeros(3, dtype=np.float64))

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    # import pdb; pdb.set_trace()

    # basically here we have "target multi-index is self and self+1"
    mapexpr = ((j0 := pym.var("j0"), j1 := pym.var("j1")), j0 + j1)

    i1 = IndexTree([RangeNode("p1", 3, id="i0")])  # loop over "cells"
    i2 = i1.copy()
    i2.add_nodes(
        [
            IdentityMapNode(("p1",), ("p1",), arity=1),  # "cell" data
            AffineMapNode(("p1",), ("p2",), arity=2, expr=mapexpr),  # "vert" data
        ],
        "i0",
    )

    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    # [0, 1, 2] + [3+4, 4+5, 5+6]
    assert np.allclose(dat2.data, [0 + 3 + 4, 1 + 4 + 5, 2 + 5 + 6])


def test_multimap():
    axes = MultiAxis([MultiAxisComponent(5, label="p1", id="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [MultiAxisComponent(2)]).set_up()
    map0 = MultiArray(
        mapaxes,
        name="map0",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 1, 3, 0, 2, 1, 4, 3, 0, 1], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 4 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (4,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 5, id="i1")])
    i2 = i1.copy()
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map0[i1]), "i1")
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[i1]), "i1")
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map0.data, dat1.data, map1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    # and [1, 1, 3, 0, 2, 1, 4, 3, 0, 1]
    assert all(
        dat2.data
        == np.array(
            [1 + 2 + 1 + 1, 0 + 2 + 3 + 0, 0 + 1 + 2 + 1, 3 + 4 + 4 + 3, 2 + 1 + 0 + 1],
            dtype=np.int32,
        )
    )


def test_multimap_with_scalar():
    axes = MultiAxis([MultiAxisComponent(5, id="p1", label="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [MultiAxisComponent(2)]).set_up()
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 5, id="i1")])
    i2 = i1.copy()
    i2.add_node(IdentityMapNode(("p1",), ("p1",), arity=1), "i1")
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[i1]), "i1")
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, map1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1] and [0, 1, 2, 3, 4]
    assert all(
        dat2.data
        == np.array(
            [1 + 2 + 0, 0 + 2 + 1, 0 + 1 + 2, 3 + 4 + 3, 2 + 1 + 4], dtype=np.int32
        )
    )


def test_map_composition():
    axes = MultiAxis([MultiAxisComponent(5, id="p1", label="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [MultiAxisComponent(2)]).set_up()
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )
    map2 = MultiArray(
        mapaxes,
        name="map2",
        data=np.array([3, 2, 4, 1, 0, 2, 4, 2, 1, 3], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 4 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, shape=(4,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, shape=(1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 5, id="i1")])
    i2 = i1.copy()
    i2.add_node(
        TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[i1], id="i2"), "i1"
    )
    i3 = i2.copy()
    i3.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map2[i2]), "i2")

    expr = pyop3.Loop(i1, kernel(dat1[i3], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map1.data, map2.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    ans = [4 + 1 + 0 + 2, 3 + 2 + 0 + 2, 3 + 2 + 4 + 1, 4 + 2 + 1 + 3, 0 + 2 + 4 + 1]
    assert all(dat2.data == np.array(ans, dtype=np.int32))


def test_mixed_arity_map():
    axes = MultiAxis([MultiAxisComponent(3, id="p1", label="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(1, 4, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(3, dtype=np.float64))

    nnz = MultiArray(
        axes, name="nnz", data=np.array([3, 2, 1], dtype=np.int32), max_value=3
    )

    mapaxes = axes.copy().add_subaxis("p1", [MultiAxisComponent(nnz)]).set_up()
    map1 = MultiArray(
        mapaxes, name="map1", data=np.array([2, 1, 0, 2, 1, 2], dtype=np.int32)
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, shape=None, is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, shape=None, is_input=False, is_output=True),
            lp.ValueArg("n", dtype=np.int32),
        ],
        assumptions="n <= 3",
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 3, id="i1")])
    i2 = i1.copy()
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=nnz[i1], data=map1[i1]), "i1")

    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # import pdb; pdb.set_trace()

    layout0_0 = map1.axes.leaf.layout_fn.start
    args = [nnz.data, layout0_0.data, map1.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == np.array([1 + 2 + 3, 2 + 3, 3], dtype=np.int32))


def test_iter_map_composition():
    axes = MultiAxis([MultiAxisComponent(5, label="p1", id="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [MultiAxisComponent(2)]).set_up()
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )
    map2 = MultiArray(
        mapaxes,
        name="map2",
        data=np.array([3, 2, 2, 3, 0, 2, 1, 2, 1, 3], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = y[i] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    p = IndexTree([RangeNode("p1", 5, id="i1")])
    p.add_node(
        TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[p.copy()], id="i2"), "i1"
    )
    p.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map2[p.copy()]), "i2")
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map1.data, map2.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    # data is just written to itself (but not the final one because it's not in map1)
    ans = [0, 1, 2, 3, 0]
    assert all(dat2.data == np.array(ans, dtype=np.int32))


def test_mixed_real_loop():
    axes = MultiAxis(
        [
            MultiAxisComponent(3, label="p1", id="p1"),  # regular part
            MultiAxisComponent(1, label="p2"),  # "real" part
        ]
    )
    axes.add_node(MultiAxisComponent(2), "p1")

    axes.set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(7))

    lpknl = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "x[i]  = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=True)],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(lpknl, [pyop3.INC])

    i1 = IndexTree([RangeNode("p1", 3, id="i1")])
    i2 = i1.copy()
    i2.add_node(IdentityMapNode(("p1",), ("p1",), arity=1), "i1")
    # it's a map from everything to zero
    i2.add_node(
        AffineMapNode(("p1",), ("p2",), arity=1, expr=(pym.variables("x y"), 0)), "i1"
    )

    expr = pyop3.Loop(i1, kernel(dat1[i2]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, [1, 1, 1, 1, 1, 1, 3])


def test_different_axis_orderings_do_not_change_packing_order():
    # FIXME
    # code = lp.make_kernel(
    #     "{ [i, j]: 0 <= i, j < 2 }",
    #     "y[i, j] = x[i, j]",
    #     [
    #         lp.GlobalArg("x", np.float64, (2, 2), is_input=True, is_output=False),
    #         lp.GlobalArg("y", np.float64, (2, 2), is_input=False, is_output=True),
    #     ],
    #     target=lp.CTarget(),
    #     name="copy",
    #     lang_version=(2018, 2),
    # )
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 4 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, (4,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (4,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="copy",
        lang_version=(2018, 2),
    )
    copy_kernel = LoopyKernel(code, [READ, WRITE])

    axis0 = Axis([AxisComponent(2)], "ax0")
    axis1 = Axis([AxisComponent(2)], "ax1")
    axis2 = Axis([AxisComponent(2)], "ax2")

    axes0 = AxisTree(axis0, {axis0.id: [axis1], axis1.id: [axis2]})
    axes1 = AxisTree(axis0, {axis0.id: [axis2], axis2.id: [axis1]})

    dat0_0 = MultiArray(
        axes0, name="dat0_0", data=np.arange(2 * 2 * 2, dtype=np.float64)
    )
    dat0_1 = MultiArray(
        axes1, name="dat0_1", data=np.array([0, 2, 1, 3, 4, 6, 5, 7], dtype=np.float64)
    )
    dat1 = MultiArray(axes0, name="dat1", data=np.zeros(2 * 2 * 2, dtype=np.float64))

    p = IndexTree(Index([Range(("ax0", 0), 2)]))
    q = IndexTree(
        p.root,
        {
            p.root.id: [Index([Range(("ax1", 0), 2)], id="idx_id0")],
            "idx_id0": [Index([Range(("ax2", 0), 2)])],
        },
    )

    do_loop(p, copy_kernel(dat0_0[q], dat1[q]))
    assert np.allclose(dat1.data, dat0_0.data)

    dat1.data[...] = 0

    do_loop(p, copy_kernel(dat0_1[q], dat1[q]))
    assert np.allclose(dat1.data, dat0_0.data)

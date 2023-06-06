import numpy as np
import pytest

from pyop3.axis import AffineLayout, Axis, AxisComponent, AxisTree, TabulatedLayout
from pyop3.distarray import MultiArray
from pyop3.dtypes import IntType
from pyop3.utils import flatten

# TODO make axis a subpackage and layouts a submodule


def check_offsets(axes, indices_and_offsets):
    for indices, offset in indices_and_offsets:
        assert axes.get_offset(indices) == offset


def check_invalid_indices(axes, indicess):
    for indices in indicess:
        with pytest.raises(IndexError):
            axes.get_offset(indices)


def test_1d_affine_layout():
    axes = AxisTree(Axis(5, id="id0"))

    layout0 = axes.layouts["id0", 0]
    assert isinstance(layout0, AffineLayout)
    assert layout0.start == 0
    assert layout0.step == 1

    check_offsets(
        axes,
        [
            ([0], 0),
            ([1], 1),
            ([2], 2),
            ([3], 3),
            ([4], 4),
            ([5], 5),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [-1],
            [6],
        ],
    )


def test_2d_affine_layout():
    axes = AxisTree(root := Axis(3, id="id0"), {root.id: Axis(2, id="id1")})

    layout0 = axes.layouts["id0", 0]
    assert isinstance(layout0, AffineLayout)
    assert layout0.start == 0
    assert layout0.step == 2

    layout1 = axes.layouts["id1", 0]
    assert isinstance(layout1, AffineLayout)
    assert layout1.start == 0
    assert layout1.step == 1

    check_offsets(
        axes,
        [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 2),
            ([1, 1], 3),
            ([2, 0], 4),
            ([2, 1], 5),
        ],
    )
    check_invalid_indices(axes, [[-1, 0], [3, 0], [0, -1], [0, 2]])


def test_1d_multi_component_layout():
    axes = AxisTree(Axis([AxisComponent(3), AxisComponent(2)], "ax0", id="id0"))

    layout0 = axes.layouts["id0", 0]
    assert isinstance(layout0, AffineLayout)
    assert layout0.start == 0
    assert layout0.step == 1

    layout1 = axes.layouts["id0", 1]
    assert isinstance(layout1, AffineLayout)
    assert layout1.start == 3
    assert layout1.step == 1

    check_offsets(
        axes,
        [
            ([(("ax0", 0), 0)], 0),
            ([(("ax0", 0), 1)], 1),
            ([(("ax0", 0), 2)], 2),
            ([(("ax0", 1), 0)], 3),
            ([(("ax0", 1), 1)], 4),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [(("ax0", 0), -1)],
            [(("ax0", 0), 3)],
            [(("ax0", 1), -1)],
            [(("ax0", 1), 2)],
        ],
    )


def test_1d_permuted_layout():
    axes = AxisTree(Axis(3, permutation=[1, 2, 0], id="id0"))

    layout0 = axes.layouts["id0", 0]
    assert isinstance(layout0, TabulatedLayout)
    assert np.allclose(layout0.data.data, [1, 2, 0])

    check_offsets(
        axes,
        [
            ([0], 1),
            ([1], 2),
            ([2], 0),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [-1],
            [3],
        ],
    )


def test_1d_multi_component_permuted_layout():
    axes = AxisTree(
        Axis(
            [AxisComponent(3), AxisComponent(2)],
            "ax0",
            permutation=[1, 4, 3, 2, 0],
            id="id0",
        )
    )

    layout0 = axes.layouts["id0", 0]
    assert isinstance(layout0, TabulatedLayout)
    assert np.allclose(layout0.data.data, [1, 4, 3])

    layout1 = axes.layouts["id0", 1]
    assert isinstance(layout1, TabulatedLayout)
    assert np.allclose(layout1.data.data, [2, 0])

    check_offsets(
        axes,
        [
            ([(("ax0", 0), 0)], 1),
            ([(("ax0", 0), 1)], 4),
            ([(("ax0", 0), 2)], 3),
            ([(("ax0", 1), 0)], 2),
            ([(("ax0", 1), 1)], 0),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [(("ax0", 0), -1)],
            [(("ax0", 0), 3)],
            [(("ax0", 1), -1)],
            [(("ax0", 1), 2)],
        ],
    )


def test_ragged_layout():
    nnz = MultiArray(
        AxisTree(Axis(3, "ax0")), data=np.asarray([2, 1, 2], dtype=IntType)
    )
    axes = AxisTree(root := Axis(3, "ax0", id="id0"), {root.id: Axis(nnz, id="id1")})

    layout0 = axes.layouts["id0", 0]
    assert isinstance(layout0, TabulatedLayout)
    assert np.allclose(layout0.data.data, [0, 2, 3])

    layout1 = axes.layouts["id1", 0]
    assert isinstance(layout1, AffineLayout)
    assert layout1.start == 0
    assert layout1.step == 1

    check_offsets(
        axes,
        [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 2),
            ([2, 0], 3),
            ([2, 1], 4),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [-1, 0],
            [0, -1],
            [0, 2],
            [1, -1],
            [1, 1],
            [2, -1],
            [2, 2],
            [3, 0],
        ],
    )


@pytest.mark.skip("TODO")
def test_ragged_layout_with_zeros():
    m = 3
    nnz = MultiArray(
        AxisTree(Axis(m, "ax0")), data=np.asarray([2, 0, 1], dtype=IntType)
    )
    axes = AxisTree(root := Axis(m, "ax0"), {root.id: Axis(nnz)})


def test_ragged_layout_with_two_outer_axes():
    outer = AxisTree(root := Axis(2, id="id0"), {root.id: Axis(2, id="id1")})
    nnzdata = np.asarray(flatten([[2, 1], [1, 2]]), dtype=IntType)
    nnz = MultiArray(outer, data=nnzdata)
    axes = outer.add_subaxis(Axis(nnz, id="id2"), outer.leaf)

    layout0 = axes.layouts["id0", 0]
    assert layout0 is None

    layout1 = axes.layouts["id1", 0]
    assert isinstance(layout1, TabulatedLayout)
    assert np.allclose(layout1.data.data, flatten([[0, 2], [3, 4]]))

    layout2 = axes.layouts["id2", 0]
    assert isinstance(layout2, AffineLayout)
    assert layout2.start == 0
    assert layout2.step == 1

    check_offsets(
        axes,
        [
            ([0, 0, 0], 0),
            ([0, 0, 1], 1),
            ([0, 1, 0], 2),
            ([1, 0, 0], 3),
            ([1, 1, 0], 4),
            ([1, 1, 1], 5),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [0, 0, 2],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 2],
            [1, 2, 0],
            [2, 0, 0],
        ],
    )


def test_independent_ragged_axes():
    nnzaxis0 = Axis(2, "ax0", id="id0")
    nnzaxis1 = Axis(2, "ax1", id="id1")
    nnzdata0 = np.asarray([2, 1], dtype=IntType)
    nnzdata1 = np.asarray([1, 2], dtype=IntType)
    nnz0 = MultiArray(nnzaxis0, data=nnzdata0)
    nnz1 = MultiArray(nnzaxis1, data=nnzdata1)
    axes = AxisTree(
        nnzaxis0,
        {nnzaxis0.id: nnzaxis1, nnzaxis1.id: Axis([nnz0, nnz1, 3], "ax2", id="id2")},
    )

    layout0 = axes.layouts["ax0", 0]
    layout1 = axes.layouts["ax1", 0]
    layout2_0 = axes.layouts["ax2", 0]
    layout2_1 = axes.layouts["ax2", 1]
    layout2_2 = axes.layouts["ax2", 2]

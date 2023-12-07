import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import freeze, pmap

import pyop3 as op3
from pyop3.utils import UniqueNameGenerator, just_one


class RenameMapper(pym.mapper.IdentityMapper):
    """Mapper that renames variables in layout expressions.

    This enables one to obtain a consistent string representation of
    an expression.

    """

    def __init__(self):
        self.name_generator = None

    def __call__(self, expr):
        # reset the counter each time the mapper is used
        self.name_generator = UniqueNameGenerator()
        return super().__call__(expr)

    def map_axis_variable(self, expr):
        return pym.var(self.name_generator("var"))

    def map_multi_array(self, expr):
        return pym.var(self.name_generator("array"))


class OrderedCollector(pym.mapper.CombineMapper):
    def combine(self, values):
        return sum(values, ())

    def map_constant(self, expr):
        return ()

    map_variable = map_constant
    map_wildcard = map_constant
    map_dot_wildcard = map_constant
    map_star_wildcard = map_constant
    map_function_symbol = map_constant

    def map_multi_array(self, expr):
        return (expr.array,)


_rename_mapper = RenameMapper()
_ordered_collector = OrderedCollector()


def as_str(layout):
    return str(_rename_mapper(layout))


def collect_multi_arrays(layout):
    return _ordered_collector(layout)


def check_offsets(axes, indices_and_offsets):
    for indices, offset in indices_and_offsets:
        assert axes.offset(indices) == offset


def check_invalid_indices(axes, indicess):
    for indices in indicess:
        with pytest.raises(IndexError):
            axes.offset(indices)


@pytest.mark.parametrize("numbering", [None, [2, 3, 0, 4, 1]])
def test_1d_affine_layout(numbering):
    # the numbering should not change the final layout
    axes = op3.AxisTree.from_nest(op3.Axis({"pt0": 5}, "ax0", numbering=numbering))

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]

    assert as_str(layout0) == "var_0"
    check_offsets(
        axes,
        [
            ([0], 0),
            ([1], 1),
            ([2], 2),
            ([3], 3),
            ([4], 4),
        ],
    )
    check_invalid_indices(axes, [[5]])


def test_2d_affine_layout():
    axes = op3.AxisTree.from_nest(
        {op3.Axis({"pt0": 3}, "ax0"): op3.Axis({"pt0": 2}, "ax1")},
    )

    layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0"})]

    assert as_str(layout0) == "var_0*2 + var_1"
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
    check_invalid_indices(axes, [[3, 0], [0, 2], [1, 2], [2, 2]])


def test_1d_multi_component_layout():
    axes = op3.AxisTree.from_nest(op3.Axis({"pt0": 3, "pt1": 2}, "ax0"))

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]
    layout1 = axes.layouts[pmap({"ax0": "pt1"})]

    assert as_str(layout0) == "var_0"
    assert as_str(layout1) == "var_0 + 3"
    check_offsets(
        axes,
        [
            ([("pt0", 0)], 0),
            ([("pt0", 1)], 1),
            ([("pt0", 2)], 2),
            ([("pt1", 0)], 3),
            ([("pt1", 1)], 4),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [],
            [("pt0", -1)],
            [("pt0", 3)],
            [("pt1", -1)],
            [("pt1", 2)],
            [("pt0", 0), 0],
        ],
    )


def test_1d_multi_component_permuted_layout():
    axes = op3.AxisTree.from_nest(
        op3.Axis(
            {"pt0": 3, "pt1": 2},
            "ax0",
            numbering=[4, 0, 3, 2, 1],
        )
    )

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]
    layout1 = axes.layouts[pmap({"ax0": "pt1"})]

    assert as_str(layout0) == "array_0"
    assert as_str(layout1) == "array_0"
    assert np.allclose(layout0.array.data_ro, [1, 3, 4])
    assert np.allclose(layout1.array.data_ro, [0, 2])
    check_offsets(
        axes,
        [
            ([("pt0", 0)], 1),
            ([("pt0", 1)], 3),
            ([("pt0", 2)], 4),
            ([("pt1", 0)], 0),
            ([("pt1", 1)], 2),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [("pt0", -1)],
            [("pt0", 3)],
            [("pt1", -1)],
            [("pt1", 2)],
        ],
    )


def test_1d_zero_sized_layout():
    axes = op3.AxisTree.from_nest(op3.Axis({"pt0": 0}, "ax0"))

    layout0 = axes.layouts[pmap({"ax0": "pt0"})]

    assert as_str(layout0) == "var_0"
    check_invalid_indices(axes, [[], [0]])


@pytest.mark.skip(reason="Need to tidy get_offset API")
def test_multi_component_layout_with_zero_sized_subaxis():
    axes = AxisTree(
        Axis([AxisComponent(2, "pt0"), AxisComponent(1, "pt1")], "ax0", id="root"),
        {
            "root": [
                Axis([AxisComponent(0, "pt0")], "ax1"),
                Axis([AxisComponent(3, "pt0")], "ax1"),
            ],
        },
    )

    assert axes.size == 3

    layout00, layout01 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0"})]
    assert isinstance(layout00, AffineLayout)
    assert layout00.start == 0
    assert layout00.step == 0
    assert isinstance(layout01, AffineLayout)
    assert layout01.start == 0
    assert layout01.step == 1

    layout10, layout11 = axes.layouts[pmap({"ax0": "pt1", "ax1": "pt0"})]
    assert isinstance(layout10, AffineLayout)
    assert layout10.start == 0
    assert layout10.step == 3
    assert isinstance(layout11, AffineLayout)
    assert layout11.start == 0
    assert layout11.step == 1

    check_offsets(
        axes,
        [
            ([("pt1", "pt0"), 0], 0),
            ([("pt1", "pt0"), 1], 1),
            ([("pt1", "pt0"), 2], 2),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [],
            [("pt0", "pt0"), 0],
            [("pt1", "pt0"), 3],
            [("pt1", "pt0"), 0, 0],
        ],
    )


def test_permuted_multi_component_layout_with_zero_sized_subaxis():
    axis0 = op3.Axis({"pt0": 3, "pt1": 2}, "ax0", numbering=[3, 1, 4, 2, 0])
    axis1 = op3.Axis({"pt0": 0}, "ax1")
    axis2 = op3.Axis({"pt0": 3}, "ax1")
    axes = op3.AxisTree.from_nest({axis0: {"pt0": axis1, "pt1": axis2}})

    assert axes.size == 6

    layout0 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0"})]
    layout1 = axes.layouts[freeze({"ax0": "pt1", "ax1": "pt0"})]

    assert as_str(layout0) == "array_0 + var_0"
    assert as_str(layout1) == "array_0 + var_0"

    array0 = just_one(collect_multi_arrays(layout0))
    array1 = just_one(collect_multi_arrays(layout1))
    assert (array0.data_ro == [3, 6, 6]).all()
    assert (array1.data_ro == [0, 3]).all()

    check_offsets(
        axes,
        [
            ([("pt1", 0), 0], 0),
            ([("pt1", 0), 1], 1),
            ([("pt1", 0), 2], 2),
            ([("pt1", 1), 0], 3),
            ([("pt1", 1), 1], 4),
            ([("pt1", 1), 2], 5),
        ],
    )
    check_invalid_indices(
        axes,
        [
            [("pt0", 0), 0],
            [("pt1", 0)],
            [("pt1", 2), 0],
            [("pt1", 0), 3],
            [("pt1", 0), 0, 0],
        ],
    )


def test_ragged_layout():
    nnz_axis = op3.Axis({"pt0": 3}, "ax0")
    nnz = op3.Dat(nnz_axis, data=np.asarray([2, 1, 2]), dtype=op3.IntType)

    axes = op3.AxisTree.from_nest({nnz_axis: op3.Axis({"pt0": nnz}, "ax1")}).freeze()

    layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0"})]
    array0 = just_one(collect_multi_arrays(layout0))

    assert as_str(layout0) == "array_0 + var_0"
    assert np.allclose(array0.data_ro, [0, 2, 3])
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


def test_ragged_layout_with_two_outer_axes():
    axis0 = op3.Axis({"pt0": 2}, "ax0")
    axis1 = op3.Axis({"pt0": 2}, "ax1")
    nnz_axes = op3.AxisTree.from_nest(
        {axis0: axis1},
    )
    nnz_data = np.asarray([[2, 1], [1, 2]])
    nnz = op3.Dat(nnz_axes, data=nnz_data.flatten(), dtype=op3.IntType)

    axes = op3.AxisTree.from_nest(
        {axis0: {axis1: op3.Axis({"pt0": nnz}, "ax2")}},
    )

    layout0 = axes.layouts[pmap({"ax0": "pt0", "ax1": "pt0", "ax2": "pt0"})]
    array0 = just_one(collect_multi_arrays(layout0))

    assert as_str(layout0) == "array_0 + var_0"
    assert np.allclose(array0.data_ro, np.asarray([[0, 2], [3, 4]]).flatten())
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
    axis0 = op3.Axis({"pt0": 2}, "ax0")
    nnz_data0 = np.asarray([2, 1])
    nnz0 = op3.Dat(axis0, name="nnz0", data=nnz_data0, dtype=op3.IntType)

    axis1 = op3.Axis({"pt0": 2}, "ax1")
    nnz_data1 = np.asarray([1, 2])
    nnz1 = op3.Dat(axis1, name="nnz1", data=nnz_data1, dtype=op3.IntType)

    axis2 = op3.Axis({"pt0": nnz0, "pt1": nnz1, "pt2": 3}, "ax2")
    axes = op3.AxisTree.from_nest({axis0: {axis1: axis2}})

    layout0 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt0"})]
    layout1 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt1"})]
    layout2 = axes.layouts[freeze({"ax0": "pt0", "ax1": "pt0", "ax2": "pt2"})]

    breakpoint()
    array0 = just_one(collect_multi_arrays(layout0))

    assert as_str(layout0) == "array_0 + var_0"
    assert np.allclose(array0.data_ro, np.asarray([[0, 2], [3, 4]]).flatten())
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

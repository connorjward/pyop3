import loopy as lp
import numpy as np
import pytest
from pyrsistent import pmap

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


@pytest.fixture
def vector_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(lpy_kernel, [op3.READ, op3.INC])


@pytest.fixture
def vec2_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = y[i] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (2,), is_input=True, is_output=True),
        ],
        name="vec2_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(lpy_kernel, [op3.READ, op3.INC])


@pytest.fixture
def vec6_inc_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 6 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (6,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(code, [op3.READ, op3.INC])


@pytest.fixture
def vec12_inc_kernel():
    code = lp.make_kernel(
        ["{ [i]: 0 <= i < 6 }", "{ [j]: 0 <= j < 2 }"],
        "y[j] = y[j] + x[i, j]",
        [
            lp.GlobalArg("x", op3.ScalarType, (6, 2), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (2,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return op3.Function(code, [op3.READ, op3.INC])


@pytest.mark.parametrize("nested", [True, False])
def test_inc_from_tabulated_map(scalar_inc_kernel, vector_inc_kernel, nested):
    m, n = 4, 3
    map_data = np.asarray([[1, 2, 0], [2, 0, 1], [3, 2, 3], [2, 0, 1]])

    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    map_axes = op3.AxisTree.from_nest({axis: op3.Axis(n)})
    map_dat = op3.Dat(
        map_axes,
        name="map0",
        data=map_data.flatten(),
        dtype=op3.IntType,
    )
    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat),
            ],
        },
        "map0",
    )

    if nested:
        op3.do_loop(
            p := axis.index(),
            op3.loop(q := map0(p).index(), scalar_inc_kernel(dat0[q], dat1[p])),
        )
    else:
        op3.do_loop(p := axis.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j in range(n):
            expected[i] += dat0.data_ro[map_data[i, j]]
    assert np.allclose(dat1.data_ro, expected)


def test_inc_from_multi_component_temporary(vector_inc_kernel):
    m, n = 3, 4
    arity = 2
    map_data = np.asarray([[1, 2], [0, 1], [3, 2]])

    axis0 = op3.Axis({"pt0": m, "pt1": n}, "ax0")
    axis1 = axis0["pt0"]

    dat0 = op3.MultiArray(
        axis0, name="dat0", data=np.arange(axis0.size), dtype=op3.ScalarType
    )
    dat1 = op3.MultiArray(axis1, name="dat1", dtype=dat0.dtype)

    # poor man's identity map
    map_axes0 = op3.AxisTree.from_nest({axis1: op3.Axis(1)})
    map_dat0 = op3.Dat(
        map_axes0,
        name="map0",
        data=np.arange(map_axes0.size),
        dtype=op3.IntType,
    )

    map_axes1 = op3.AxisTree.from_nest({axis1: op3.Axis(arity)})
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data.flatten(), dtype=op3.IntType
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
                op3.TabulatedMapComponent("ax0", "pt1", map_dat1),
            ],
        },
        "map0",
    )

    op3.do_loop(p := axis1.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        expected[i] += dat0.data_ro[i]  # identity
        for j in range(arity):
            # add offset of m to reads since we are indexing the second
            # component (stored contiguously)
            expected[i] += dat0.data_ro[map_data[i, j] + m]
    assert np.allclose(dat1.data, expected)


def test_inc_with_multiple_maps(vector_inc_kernel):
    m = 5
    arity0, arity1 = 2, 1
    map_data0 = np.asarray([[1, 2], [0, 2], [0, 1], [3, 4], [2, 1]])
    map_data1 = np.asarray([[1], [1], [3], [0], [2]])

    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    map_axes0 = op3.AxisTree.from_nest({axis: op3.Axis(arity0)})
    map_axes1 = op3.AxisTree.from_nest({axis: op3.Axis(arity1)})

    map_dat0 = op3.Dat(
        map_axes0,
        name="map0",
        data=map_data0.flatten(),
        dtype=op3.IntType,
    )
    map_dat1 = op3.Dat(
        map_axes1,
        name="map1",
        data=map_data1.flatten(),
        dtype=op3.IntType,
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
                op3.TabulatedMapComponent("ax0", "pt0", map_dat1),
            ],
        },
        "map0",
    )

    op3.do_loop(p := axis.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j0 in range(arity0):
            expected[i] += dat0.data_ro[map_data0[i, j0]]
        for j1 in range(arity1):
            expected[i] += dat0.data_ro[map_data0[i, j1]]
    assert np.allclose(dat1.data, expected)


@pytest.mark.parametrize("nested", [True, False])
def test_inc_with_map_composition(scalar_inc_kernel, vec6_inc_kernel, nested):
    m = 5
    arity0, arity1 = 2, 3
    map_data0 = np.asarray([[2, 1], [0, 3], [1, 4], [0, 0], [3, 2]])
    map_data1 = np.asarray(
        [[0, 4, 1], [2, 1, 3], [4, 2, 4], [0, 1, 2], [4, 2, 3]],
    )

    axis = op3.Axis({"pt0": m}, "ax0")
    dat0 = op3.Dat(axis, name="dat0", data=np.arange(m), dtype=op3.ScalarType)
    dat1 = op3.Dat(axis, name="dat1", dtype=dat0.dtype)

    map_axes0 = op3.AxisTree.from_nest({axis: op3.Axis(arity0)})
    map_axes1 = op3.AxisTree.from_nest({axis: op3.Axis(arity1)})

    map_dat0 = op3.Dat(
        map_axes0, name="map0", data=map_data0.flatten(), dtype=op3.IntType
    )
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
            ],
        },
        "map0",
    )
    map1 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat1),
            ],
        },
        "map1",
    )

    if nested:
        op3.do_loop(
            p := axis.index(),
            op3.loop(
                q := map0(p).index(),
                op3.loop(r := map1(q).index(), scalar_inc_kernel(dat0[r], dat1[p])),
            ),
        )
    else:
        op3.do_loop(p := axis.index(), vec6_inc_kernel(dat0[map1(map0(p))], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j in range(arity0):
            for k in range(arity1):
                expected[i] += dat0.data_ro[map_data1[map_data0[i, j], k]]
    assert np.allclose(dat1.data_ro, expected)


@pytest.mark.parametrize("nested", [True, False])
def test_vector_inc_with_map_composition(vec2_inc_kernel, vec12_inc_kernel, nested):
    m, n = 5, 2
    arity0, arity1 = 2, 3
    map_data0 = np.asarray([[2, 1], [0, 3], [1, 4], [0, 0], [3, 2]])
    map_data1 = np.asarray([[0, 4, 1], [2, 1, 3], [4, 2, 4], [0, 1, 2], [4, 2, 3]])

    axis = op3.Axis({"pt0": m}, "ax0")

    dat_axes = op3.AxisTree.from_nest({axis: op3.Axis({"pt0": n}, "ax1")})
    dat0 = op3.Dat(
        dat_axes, name="dat0", data=np.arange(dat_axes.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(dat_axes, name="dat1", dtype=dat0.dtype)

    map_axes0 = op3.AxisTree.from_nest({axis: op3.Axis(arity0)})
    map_axes1 = op3.AxisTree.from_nest({axis: op3.Axis(arity1)})

    map_dat0 = op3.Dat(
        map_axes0, name="map0", data=map_data0.flatten(), dtype=op3.IntType
    )
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )

    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat0),
            ],
        },
        "map0",
    )
    map1 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_dat1),
            ],
        },
        "map1",
    )

    if nested:
        op3.do_loop(
            p := axis.index(),
            op3.loop(
                q := map0(p).index(),
                op3.loop(r := map1(q).index(), vec2_inc_kernel(dat0[r, :], dat1[p, :])),
            ),
        )
    else:
        op3.do_loop(
            p := axis.index(), vec12_inc_kernel(dat0[map1(map0(p)), :], dat1[p, :])
        )

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        for j in range(arity0):
            for k in range(arity1):
                idx = map_data1[map_data0[i, j], k]
                for d in range(n):
                    expected[i * n + d] += dat0.data_ro[idx * n + d]
    assert np.allclose(dat1.data_ro, expected)


@pytest.mark.skip(
    reason="Passing ragged arguments through to the local is not yet supported"
)
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


def test_map_composition(vec2_inc_kernel):
    arity0, arity1 = 3, 2

    iterset = op3.Axis({"pt0": 2}, "ax0")
    dat_axis0 = op3.Axis(10)
    dat_axis1 = op3.Axis(arity1)

    map_axes0 = op3.AxisTree.from_nest({iterset: op3.Axis(arity0)})
    map_data0 = np.asarray([[2, 4, 0], [6, 7, 1]])
    map_dat0 = op3.Dat(
        map_axes0, name="map0", data=map_data0.flatten(), dtype=op3.IntType
    )
    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent(
                    dat_axis0.label, dat_axis0.component.label, map_dat0, label="a"
                ),
            ],
        },
        "map0",
    )

    # this map targets the entries in map0 so it can only contain 0s, 1s and 2s
    map_axes1 = op3.AxisTree.from_nest({iterset: op3.Axis(arity1)})
    map_data1 = np.asarray([[0, 2], [2, 1]])
    map_dat1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )
    map1 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("map0", "a", map_dat1),
            ],
        },
        "map1",
    )

    dat0 = op3.Dat(
        dat_axis0, name="dat0", data=np.arange(dat_axis0.size), dtype=op3.ScalarType
    )
    dat1 = op3.Dat(dat_axis1, name="dat1", dtype=dat0.dtype)

    op3.do_loop(p := iterset.index(), vec2_inc_kernel(dat0[map0(p)][map1(p)], dat1))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(iterset.size):
        temp = np.zeros(arity0, dtype=dat0.dtype)
        for j0 in range(arity0):
            temp[j0] = dat0.data_ro[map_data0[i, j0]]
        for j1 in range(arity1):
            expected[j1] += temp[map_data1[i, j1]]
    assert np.allclose(dat1.data_ro, expected)


def test_recursive_multi_component_maps():
    m, n = 5, 6
    arity0_0, arity0_1, arity1 = 3, 2, 1

    axis = op3.Axis(
        {"pt0": m, "pt1": n},
        "ax0",
    )

    # maps from pt0 so the array has size (m, arity0_0)
    map_axes0_0 = op3.AxisTree.from_nest({axis["pt0"]: op3.Axis(arity0_0)})
    # maps to ax0_cpt0 so the maximum possible index is m - 1
    map_data0_0 = np.asarray(
        [[2, 4, 0], [2, 3, 1], [0, 2, 3], [1, 3, 4], [3, 1, 0]],
    )
    assert np.prod(map_data0_0.shape) == map_axes0_0.size
    map_array0_0 = op3.Dat(
        map_axes0_0, name="map0_0", data=map_data0_0.flatten(), dtype=op3.IntType
    )

    # maps from ax0_cpt0 so the array has size (m, arity0_1)
    map_axes0_1 = op3.AxisTree.from_nest({axis["pt0"]: op3.Axis(arity0_1)})
    # maps to ax0_cpt1 so the maximum possible index is n - 1
    map_data0_1 = np.asarray([[4, 5], [2, 1], [0, 3], [5, 0], [3, 2]])
    assert np.prod(map_data0_1.shape) == map_axes0_1.size
    map_array0_1 = op3.Dat(
        map_axes0_1, name="map0_1", data=map_data0_1.flatten(), dtype=op3.IntType
    )

    # maps from ax0_cpt1 so the array has size (n, arity1)
    map_axes1 = op3.AxisTree.from_nest({axis["pt1"]: op3.Axis(arity1)})
    # maps to ax0_cpt1 so the maximum possible index is n - 1
    map_data1 = np.asarray([[4], [5], [2], [3], [0], [1]])
    assert np.prod(map_data1.shape) == map_axes1.size
    map_array1 = op3.Dat(
        map_axes1, name="map1", data=map_data1.flatten(), dtype=op3.IntType
    )

    # map from cpt0 -> {cpt0, cpt1} and from cpt1 -> {cpt1}
    map0 = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent("ax0", "pt0", map_array0_0),
                op3.TabulatedMapComponent("ax0", "pt1", map_array0_1),
            ],
            pmap({"ax0": "pt1"}): [
                op3.TabulatedMapComponent("ax0", "pt1", map_array1),
            ],
        },
        "map0",
    )
    map1 = map0.copy(name="map1")

    dat0 = op3.Dat(axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(axis["pt0"], name="dat1", dtype=dat0.dtype)

    # the temporary from the maps will look like:
    # Axis([3, 2], label=map0)
    # ├──➤ Axis([3, 2], label=map1)
    # │    ├──➤ None
    # │    └──➤ None
    # └──➤ Axis(1, label=map1)
    #      └──➤ None
    # which has 17 entries
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 17 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", op3.ScalarType, (17,), is_input=True, is_output=False),
            lp.GlobalArg("y", op3.ScalarType, (1,), is_input=False, is_output=True),
        ],
        name="sum_kernel",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    sum_kernel = op3.Function(lpy_kernel, [op3.READ, op3.WRITE])

    op3.do_loop(p := axis["pt0"].index(), sum_kernel(dat0[map1(map0(p))], dat1[p]))

    expected = np.zeros_like(dat1.data_ro)
    for i in range(m):
        # cpt0, cpt0 (9 entries)
        packed00 = dat0.data_ro[:5][map_data0_0[map_data0_0[i]]]
        # cpt0, cpt1 (6 entries)
        packed01 = dat0.data_ro[5:][map_data0_1[map_data0_0[i]]]
        # cpt1, cpt1 (2 entries)
        packed11 = dat0.data_ro[5:][map_data1[map_data0_1[i]]]

        # in the local kernel we sum all the entries together
        expected[i] = np.sum(packed00) + np.sum(packed01) + np.sum(packed11)
    assert np.allclose(dat1.data_ro, expected)


def test_sum_with_consecutive_maps():
    iterset_size = 5
    dat_sizes = 10, 4
    arity0 = 3
    arity1 = 2

    iterset = AxisTree(Axis([AxisComponent(iterset_size, "iter_ax0_cpt0")], "iter_ax0"))
    dat_axes0 = AxisTree(
        Axis([AxisComponent(dat_sizes[0], "dat0_ax0_cpt0")], "dat0_ax0", id="root"),
        {"root": Axis([AxisComponent(dat_sizes[1], "dat0_ax1_cpt0")], "dat0_ax1")},
    )

    dat0 = MultiArray(
        dat_axes0, name="dat0", data=np.arange(dat_axes0.size, dtype=ScalarType)
    )
    dat1 = MultiArray(iterset, name="dat1", dtype=dat0.dtype)

    # map0 maps from the iterset to dat0_ax0
    map_axes0 = iterset.add_node(Axis(arity0), *iterset.leaf)
    map_data0 = np.asarray(
        [[2, 9, 0], [6, 7, 1], [5, 3, 8], [9, 3, 2], [2, 4, 6]], dtype=IntType
    )
    map_array0 = MultiArray(map_axes0, name="map0", data=map_data0.flatten())
    map0 = Map(
        {
            pmap({"iter_ax0": "iter_ax0_cpt0"}): [
                TabulatedMapComponent("dat0_ax0", "dat0_ax0_cpt0", map_array0),
            ],
        },
        "map0",
    )

    # map1 maps from the iterset to dat0_ax1
    map_axes1 = iterset.add_node(Axis(arity1), *iterset.leaf)
    map_data1 = np.asarray([[0, 2], [2, 1], [3, 1], [0, 0], [1, 2]], dtype=IntType)
    map_array1 = MultiArray(map_axes1, name="map1", data=map_data1.flatten())
    map1 = Map(
        {
            pmap({"iter_ax0": "iter_ax0_cpt0"}): [
                TabulatedMapComponent("dat0_ax1", "dat0_ax1_cpt0", map_array1),
            ],
        },
        "map1",
    )

    # create the local kernel
    lpy_kernel = lp.make_kernel(
        f"{{ [i]: 0 <= i < {arity0*arity1} }}",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg(
                "x", ScalarType, (arity0 * arity1,), is_input=True, is_output=False
            ),
            lp.GlobalArg("y", ScalarType, (1,), is_input=False, is_output=True),
        ],
        name="sum",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    sum_kernel = Function(lpy_kernel, [READ, WRITE])

    do_loop(p := iterset.index(), sum_kernel(dat0[map0(p), map1(p)], dat1[p]))

    expected = np.zeros_like(dat1.data)
    for i in range(iterset.size):
        expected[i] = np.sum(
            dat0.data.reshape(dat_sizes)[map_data0[i]][:, map_data1[i]]
        )
    assert np.allclose(dat1.data, expected)

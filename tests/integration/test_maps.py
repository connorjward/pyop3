import loopy as lp
import numpy as np
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
    Index,
    IndexTree,
    IntType,
    Map,
    MultiArray,
    ScalarType,
    Slice,
    SliceComponent,
    TabulatedMapComponent,
    do_loop,
    loop,
)
from pyop3.codegen.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


@pytest.fixture
def vector_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return Function(lpy_kernel, [READ, INC])


@pytest.fixture
def vec2_inc_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = y[i] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (2,), is_input=True, is_output=True),
        ],
        name="vec2_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return Function(lpy_kernel, [READ, INC])


@pytest.fixture
def vec6_inc_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 6 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, (6,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return Function(code, [READ, INC])


@pytest.fixture
def vec12_inc_kernel():
    code = lp.make_kernel(
        ["{ [i]: 0 <= i < 6 }", "{ [j]: 0 <= j < 2 }"],
        "y[j] = y[j] + x[i, j]",
        [
            lp.GlobalArg("x", ScalarType, (6, 2), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (2,), is_input=True, is_output=True),
        ],
        name="vector_inc",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return Function(code, [READ, INC])


def test_inc_from_tabulated_map(vector_inc_kernel):
    m, n = 4, 3
    mapdata = np.asarray([[1, 2, 0], [2, 0, 1], [3, 2, 3], [2, 0, 1]], dtype=IntType)

    axes = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    maxes = axes.add_node(Axis(n), *axes.leaf)
    maparray = MultiArray(
        maxes,
        name="map0",
        data=mapdata.flatten(),
    )
    map0 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                TabulatedMapComponent("ax0", "cpt0", maparray),
            ],
        },
        "map0",
    )

    do_loop(p := axes.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))

    # Since dat0.data is simply arange, accessing index i of it will also just be i.
    # Therefore, accessing multiple entries and storing the sum of them is
    # equivalent to summing the indices from the map.
    assert np.allclose(dat1.data, np.sum(mapdata, axis=1))


def test_inc_from_multi_component_temporary(vector_inc_kernel):
    m, n = 3, 4
    arity = 2
    mapdata = np.asarray([[1, 2], [0, 1], [3, 2]], dtype=IntType)

    axes0 = AxisTree(Axis([AxisComponent(m, "cpt0"), AxisComponent(n, "cpt1")], "ax0"))
    axes1 = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))

    dat0 = MultiArray(axes0, name="dat0", data=np.arange(m + n, dtype=ScalarType))
    dat1 = MultiArray(axes1, name="dat1", data=np.zeros(m, dtype=ScalarType))

    # this is an identity map
    maxes0 = axes1.add_subaxis(Axis(1), *axes1.leaf)
    maparray0 = MultiArray(
        maxes0, name="map0", data=np.arange(maxes0.size, dtype=IntType)
    )

    maxes1 = axes1.add_subaxis(Axis(arity), *axes1.leaf)
    maparray1 = MultiArray(maxes1, name="map1", data=mapdata.flatten())

    map0 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                TabulatedMapComponent("ax0", "cpt0", maparray0),
                TabulatedMapComponent("ax0", "cpt1", maparray1),
            ],
        },
        "map0",
    )

    do_loop(p := axes1.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))

    # The expected value is the current index (from the identity map), plus the values
    # from the map. Since the indices in the map are offset in the actual array we
    # also need to add this.
    assert np.allclose(dat1.data, np.arange(m) + np.sum(mapdata + m, axis=1))


@pytest.mark.skip(reason="Affine maps not yet supported")
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

    axes = AxisTree(Axis([AxisComponent(m, "pt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    maxes0 = axes.add_subaxis(Axis(arity0), *axes.leaf)
    maxes1 = axes.add_subaxis(Axis(arity1), *axes.leaf)

    maparray0 = MultiArray(
        maxes0,
        name="map0",
        data=mapdata0.flatten(),
    )
    maparray1 = MultiArray(
        maxes1,
        name="map1",
        data=mapdata1.flatten(),
    )

    map0 = Map(
        {
            pmap({"ax0": "pt0"}): [
                TabulatedMapComponent("ax0", "pt0", maparray0),
                TabulatedMapComponent("ax0", "pt0", maparray1),
            ],
        },
        "map0",
    )

    do_loop(p := axes.index(), vector_inc_kernel(dat0[map0(p)], dat1[p]))
    assert np.allclose(dat1.data, np.sum(mapdata0, axis=1) + np.sum(mapdata1, axis=1))


def test_inc_with_map_composition(vec6_inc_kernel):
    m = 5
    arity0, arity1 = 2, 3
    mapdata0 = np.asarray([[2, 1], [0, 3], [1, 4], [0, 0], [3, 2]], dtype=IntType)
    mapdata1 = np.asarray(
        [[0, 4, 1], [2, 1, 3], [4, 2, 4], [0, 1, 2], [4, 2, 3]], dtype=IntType
    )

    axes = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    maxes0 = axes.add_subaxis(Axis(arity0), axes.root)
    maxes1 = axes.add_subaxis(Axis(arity1), axes.root)

    maparray0 = MultiArray(maxes0, name="map0", data=mapdata0.flatten())
    maparray1 = MultiArray(maxes1, name="map1", data=mapdata1.flatten())

    map0 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                TabulatedMapComponent("ax0", "cpt0", maparray0),
            ],
        },
        "map0",
    )
    map1 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                TabulatedMapComponent("ax0", "cpt0", maparray1),
            ],
        },
        "map1",
    )

    do_loop(p := axes.index(), vec6_inc_kernel(dat0[map1(map0(p))], dat1[p]))

    expected = np.sum(np.sum(np.arange(m)[mapdata1], axis=1)[mapdata0], axis=1)
    assert np.allclose(dat1.data, expected)


def test_vector_inc_with_map_composition(vec12_inc_kernel):
    m, n = 5, 2
    arity0, arity1 = 2, 3
    mapdata0 = np.asarray([[2, 1], [0, 3], [1, 4], [0, 0], [3, 2]], dtype=IntType)
    mapdata1 = np.asarray(
        [[0, 4, 1], [2, 1, 3], [4, 2, 4], [0, 1, 2], [4, 2, 3]], dtype=IntType
    )

    axes = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))

    daxes = axes.add_subaxis(Axis([AxisComponent(n, "cpt0")], "ax1"), axes.root)
    dat0 = MultiArray(daxes, name="dat0", data=np.arange(daxes.size, dtype=ScalarType))
    dat1 = MultiArray(daxes, name="dat1", dtype=dat0.dtype)

    maxes0 = axes.add_subaxis(Axis(arity0), axes.root)
    maxes1 = axes.add_subaxis(Axis(arity1), axes.root)

    maparray0 = MultiArray(maxes0, name="map0", data=mapdata0.flatten())
    maparray1 = MultiArray(maxes1, name="map1", data=mapdata1.flatten())

    map0 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                TabulatedMapComponent("ax0", "cpt0", maparray0),
            ],
        },
        "map0",
    )
    map1 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                TabulatedMapComponent("ax0", "cpt0", maparray1),
            ],
        },
        "map1",
    )

    do_loop(p := axes.index(), vec12_inc_kernel(dat0[map1(map0(p)), :], dat1[p, :]))

    expected = np.sum(
        np.sum(np.arange(m * n).reshape((m, n))[mapdata1, :], axis=1)[mapdata0, :],
        axis=1,
    )
    assert np.allclose(dat1.data.reshape((m, n)), expected)


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


@pytest.mark.skip(reason="Looping over maps is not yet supported")
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


def test_map_composition(vec2_inc_kernel):
    arity0, arity1 = 3, 2

    iterset = AxisTree(Axis([(2, "cpt0")], "ax0"))

    daxes0 = AxisTree(Axis([(10, "cpt0")]))
    daxes1 = AxisTree(Axis([AxisComponent(arity1, "cpt0")], "ax0"))

    mapaxes0 = iterset.add_node(Axis(arity0), *iterset.leaf)
    mapdata0 = np.asarray([[2, 4, 0], [6, 7, 1]], dtype=int)
    maparray0 = MultiArray(mapaxes0, name="map0", data=flatten(mapdata0))
    map0 = Map(
        {
            pmap({iterset.root.label: "cpt0"}): [
                TabulatedMapComponent(daxes0.root.label, "cpt0", maparray0, label="a"),
            ],
        },
        "map0",
    )

    # this map targets the entries in mapdata0 so it can only contain 0s, 1s and 2s
    mapaxes1 = iterset.add_node(Axis(arity1), *iterset.leaf)
    mapdata1 = np.asarray([[0, 2], [2, 1]], dtype=int)
    maparray1 = MultiArray(mapaxes1, name="map1", data=mapdata1.flatten())
    map1 = Map(
        {
            pmap({iterset.root.label: "cpt0"}): [
                TabulatedMapComponent("map0", "a", maparray1),
            ],
        },
        "map1",
    )

    dat0 = MultiArray(
        daxes0, name="dat0", data=np.arange(daxes0.size, dtype=ScalarType)
    )
    dat1 = MultiArray(daxes1, name="dat1", dtype=dat0.dtype)

    do_loop(p := iterset.index(), vec2_inc_kernel(dat0[map0(p)][map1(p)], dat1[:]))

    expected = np.zeros_like(dat1.data)
    for i in range(iterset.size):
        expected += dat0.data[mapdata0[i]][mapdata1[i]]
        # i = 0
        # [2, 4, 0][[0, 2]]
        # [2, 0]
        # i = 1
        # [6, 7, 1][[2, 1]]
        # [1, 7]
        # so dat1 is [0, 0] then [2, 0] then [3, 7], it is not indexed by i
    assert np.allclose(dat1.data, expected)


def test_recursive_multi_component_maps():
    dat_sizes = 5, 6
    arity0_0, arity0_1, arity1 = 3, 2, 1

    dat0_axes = AxisTree(
        Axis(
            [
                AxisComponent(dat_sizes[0], "dat0_ax0_cpt0"),
                AxisComponent(dat_sizes[1], "dat0_ax0_cpt1"),
            ],
            "dat0_ax0",
        )
    )
    dat1_axes = AxisTree(
        Axis([AxisComponent(dat_sizes[0], "dat0_ax0_cpt0")], "dat0_ax0")
    )

    # maps from ax0_cpt0 so the array has size (dat_sizes[0], arity0_0)
    map_axes0_0 = AxisTree(
        Axis([AxisComponent(dat_sizes[0], "dat0_ax0_cpt0")], "dat0_ax0", id="root"),
        {"root": Axis(arity0_0)},
    )
    # maps to ax0_cpt0 so the maximum possible index is dat_sizes[0] - 1
    map_data0_0 = np.asarray(
        [[2, 4, 0], [2, 3, 1], [0, 2, 3], [1, 3, 4], [3, 1, 0]], dtype=IntType
    )
    assert np.prod(map_data0_0.shape) == map_axes0_0.size
    map_array0_0 = MultiArray(map_axes0_0, name="map0_0", data=map_data0_0.flatten())

    # maps from ax0_cpt0 so the array has size (dat_sizes[0], arity0_1)
    map_axes0_1 = AxisTree(
        Axis([AxisComponent(dat_sizes[0], "dat0_ax0_cpt0")], "dat0_ax0", id="root"),
        {"root": Axis(arity0_1)},
    )
    # maps to ax0_cpt1 so the maximum possible index is dat_sizes[1] - 1
    map_data0_1 = np.asarray([[4, 5], [2, 1], [0, 3], [5, 0], [3, 2]], dtype=IntType)
    assert np.prod(map_data0_1.shape) == map_axes0_1.size
    map_array0_1 = MultiArray(map_axes0_1, name="map0_1", data=map_data0_1.flatten())

    # maps from ax0_cpt1 so the array has size (dat_sizes[1], arity1)
    map_axes1 = AxisTree(
        Axis([AxisComponent(dat_sizes[1], "dat0_ax0_cpt1")], "dat0_ax0", id="root"),
        {"root": Axis(arity1)},
    )
    # maps to ax0_cpt1 so the maximum possible index is dat_sizes[1] - 1
    map_data1 = np.asarray([[4], [5], [2], [3], [0], [1]], dtype=IntType)
    assert np.prod(map_data1.shape) == map_axes1.size
    map_array1 = MultiArray(map_axes1, name="map1", data=map_data1.flatten())

    # map from cpt0 -> {cpt0, cpt1} and from cpt1 -> {cpt1}
    map0 = Map(
        {
            pmap({"dat0_ax0": "dat0_ax0_cpt0"}): [
                TabulatedMapComponent("dat0_ax0", "dat0_ax0_cpt0", map_array0_0),
                TabulatedMapComponent("dat0_ax0", "dat0_ax0_cpt1", map_array0_1),
            ],
            pmap({"dat0_ax0": "dat0_ax0_cpt1"}): [
                TabulatedMapComponent("dat0_ax0", "dat0_ax0_cpt1", map_array1),
            ],
        },
        "map0",
    )
    map1 = map0.copy(name="map1")

    dat0 = MultiArray(
        dat0_axes, name="dat0", data=np.arange(dat0_axes.size, dtype=ScalarType)
    )
    dat1 = MultiArray(dat1_axes, name="dat1", dtype=dat0.dtype)

    # create the local kernel
    # the temporary from the maps will look like:
    # Axis([{count=3}, {count=2}], label=map0)
    # ├──➤ Axis([{count=3}, {count=2}], label=map1)
    # │    ├──➤ None
    # │    └──➤ None
    # └──➤ Axis([{count=1}], label=map1)
    #      └──➤ None
    # which has 17 entries
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 17 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", ScalarType, (17,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (1,), is_input=False, is_output=True),
        ],
        name="sum_kernel",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    sum_kernel = Function(lpy_kernel, [READ, WRITE])

    do_loop(p := dat1_axes.index(), sum_kernel(dat0[map1(map0(p))], dat1[p]))

    expected = np.zeros_like(dat1.data)
    for i in range(dat_sizes[0]):
        # cpt0, cpt0 (9 entries)
        packed00 = dat0.data[:5][map_data0_0[map_data0_0[i]]]
        # cpt0, cpt1 (6 entries)
        packed01 = dat0.data[5:][map_data0_1[map_data0_0[i]]]
        # cpt1, cpt1 (2 entries)
        packed11 = dat0.data[5:][map_data1[map_data0_1[i]]]

        # in the local kernel we sum all the entries together
        expected[i] = np.sum(packed00) + np.sum(packed01) + np.sum(packed11)
    assert np.allclose(dat1.data, expected)


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

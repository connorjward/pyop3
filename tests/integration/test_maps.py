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
    Index,
    IndexTree,
    LoopyKernel,
    Map,
    MultiArray,
    ScalarType,
    Slice,
    TabulatedMap,
    do_loop,
    loop,
)
from pyop3.codegen import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import flatten


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
    return LoopyKernel(lpy_kernel, [READ, INC])


def test_scalar_copy_of_subset(scalar_copy_kernel):
    m, n = 6, 4
    sdata = np.asarray([2, 3, 5, 0], dtype=IntType)
    untouched = [1, 4]

    axes = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    # a subset is really a map from a small set into a larger one
    saxes = AxisTree(
        Axis([AxisComponent(n, "scpt0")], "sax0", id="root"), {"root": Axis(1)}
    )
    subset = MultiArray(saxes, name="subset0", data=sdata)

    p = (Index(TabulatedMap("sax0", "scpt0", "ax0", "cpt0", arity=1, data=subset)),)
    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data[sdata], dat0.data[sdata])
    assert np.allclose(dat1.data[untouched], 0)


def test_inc_from_tabulated_map(vector_inc_kernel):
    m, n = 4, 3
    mapdata = np.asarray([[1, 2, 0], [2, 0, 1], [3, 2, 3], [2, 0, 1]], dtype=IntType)

    axes = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    maxes = axes.add_node(Axis(n), *axes.leaf)
    map0 = MultiArray(
        maxes,
        name="map0",
        data=mapdata.flatten(),
    )

    p = axes.index
    q = p.add_node(
        Index(TabulatedMap("ax0", "cpt0", "ax0", "cpt0", arity=n, data=map0)), *p.leaf
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
                ("a", maparray0, arity0, "ax0", "cpt0"),
            ],
        },
        "map0",
    )
    map1 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                ("a", maparray1, arity1, "ax0", "cpt0"),
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
                ("a", maparray0, arity0, "ax0", "cpt0"),
            ],
        },
        "map0",
    )
    map1 = Map(
        {
            pmap({"ax0": "cpt0"}): [
                ("a", maparray1, arity1, "ax0", "cpt0"),
            ],
        },
        "map1",
    )

    p = axes.index()
    iroot = map1(map0(p))
    itree = IndexTree(iroot, {iroot.id: Slice([("ax1", "cpt0", 0, None, 1)])})

    itree1 = IndexTree(p, {p.id: Slice([("ax1", "cpt0", 0, None, 1)])})

    do_loop(p, vec12_inc_kernel(dat0[itree], dat1[itree1]))

    expected = np.sum(
        np.sum(np.arange(m * n).reshape((m, n))[mapdata1, :], axis=1)[mapdata0, :],
        axis=1,
    )
    assert np.allclose(dat1.data.reshape((m, n)), expected)


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
                ("a", maparray0, arity0, daxes0.root.label, "cpt0"),
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
                ("b", maparray1, arity1, "map0", "a"),
            ],
        },
        "map1",
    )

    dat0 = MultiArray(
        daxes0, name="dat0", data=np.arange(daxes0.size, dtype=ScalarType)
    )
    dat1 = MultiArray(daxes1, name="dat1", dtype=dat0.dtype)

    p = iterset.index()
    itree0 = IndexTree(map0(p))
    itree1 = IndexTree(map1(p))
    itree2 = IndexTree(Slice([("ax0", "cpt0", 0, None, 1)]))

    do_loop(p, vec2_inc_kernel(dat0[itree0][itree1], dat1[itree2]))

    expected = np.zeros_like(dat1.data)
    for i in range(2):
        expected += dat0.data[mapdata0[i]][mapdata1[i]]
    assert np.allclose(dat1.data, expected)


def test_multi_map_composition():
    raise NotImplementedError
    mmap0 = MultiMap(maps0)
    mmap1 = MultiMap(maps1)
    do_loop(p := Axis(2).index(), copy_kernel(dat0[mmap0(p)][mmap1(p)], dat1[...]))

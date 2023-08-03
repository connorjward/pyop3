import loopy as lp
import numpy as np
import pytest
from pyrsistent import pmap

from pyop3 import (
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
def copy_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (2,), is_input=False, is_output=True),
        ],
        name="copy",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(lpy_kernel, [READ, WRITE])


@pytest.fixture
def debug_kernel():
    lpy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 6 }",
        "",
        [
            lp.GlobalArg("x", ScalarType, (6,), is_input=True, is_output=False),
        ],
        name="debug",
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return LoopyKernel(lpy_kernel, [READ])


def test_1d_slice_composition(copy_kernel):
    m, n = 10, 2
    dat0 = MultiArray(
        AxisTree(Axis([(m, "cpt0")], "ax0")),
        name="dat0",
        data=np.arange(m, dtype=ScalarType),
    )
    dat1 = MultiArray(Axis([(n, "cpt0")], "ax0"), name="dat1", dtype=ScalarType)

    # equivalent to dat1[...] = dat0[::2][1:3] (== [2, 4])
    do_loop(
        Axis(1).index,  # for _ in range(1)
        copy_kernel(
            dat0[Index(Slice(None, None, 2, axis="ax0", cpt="cpt0"))][
                Index(Slice(1, 3, axis="ax0", cpt="cpt0"))
            ],
            dat1[Index(Slice(axis="ax0", cpt="cpt0"))],
        ),
    )

    assert np.allclose(dat1.data, dat0.data[::2][1:3])


def test_2d_slice_composition(copy_kernel):
    m0, m1, n = 10, 3, 2

    axes0 = AxisTree(
        Axis([(m0, "cpt0")], "ax0", id="root"), {"root": Axis([(m1, "cpt0")], "ax1")}
    )
    axes1 = AxisTree(Axis([(n, "cpt0")], "ax0"))

    dat0 = MultiArray(axes0, name="dat0", data=np.arange(axes0.size, dtype=ScalarType))
    dat1 = MultiArray(axes1, name="dat1", dtype=dat0.dtype)

    # equivalent to dat0.data[::2, 1:][2:4, 1]
    p0 = IndexTree(
        Index(Slice(None, None, 2, axis="ax0", cpt="cpt0"), id="idx0"),
        {"idx0": Index(Slice(1, None, axis="ax1", cpt="cpt0"))},
    )
    p1 = IndexTree(
        Index(Slice(2, 4, axis="ax0", cpt="cpt0"), id="idx1"),
        {"idx1": Index(Slice(1, 2, axis="ax1", cpt="cpt0"))},
    )

    do_loop(
        Axis(1).index,  # for _ in range(1)
        copy_kernel(
            dat0[p0][p1],
            dat1[...],
        ),
    )

    assert np.allclose(dat1.data, dat0.data.reshape((m0, m1))[::2, 1:][2:4, 1])


def test_map_composition(copy_kernel):
    arity0, arity1 = 3, 2

    iterset = AxisTree(Axis([(2, "cpt0")], "ax0"))
    axes = AxisTree(Axis([(10, "cpt0")]))

    mapaxes0 = iterset.add_node(Axis(arity0), *iterset.leaf)
    mapdata0 = np.asarray(flatten([[2, 4, 0], [6, 7, 1]]), dtype=int)
    maparray0 = MultiArray(mapaxes0, name="map0", data=mapdata0)
    map0 = Map(
        {
            pmap({iterset.root.label: "cpt0"}): [
                ("a", maparray0, arity0, axes.root.label, "cpt0"),
            ],
        },
        "map0",
    )

    # this map targets the entries in mapdata0 so it can only contain 0s, 1s and 2s
    mapaxes1 = iterset.add_node(Axis(arity1), *iterset.leaf)
    mapdata1 = np.asarray(flatten([[0, 2], [2, 1]]), dtype=int)
    maparray1 = MultiArray(mapaxes1, name="map1", data=mapdata1)
    map1 = Map(
        {
            pmap({iterset.root.label: "cpt0"}): [
                ("b", maparray1, arity1, "map0", "a"),
            ],
        },
        "map1",
    )

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(Axis(arity1), name="dat1", dtype=dat0.dtype)

    do_loop(p := iterset.index(), copy_kernel(dat0[map0(p)][map1(p)], dat1[...]))

    assert False, "TODO"


def test_multi_map_composition():
    raise NotImplementedError
    mmap0 = MultiMap(maps0)
    mmap1 = MultiMap(maps1)
    do_loop(p := Axis(2).index(), copy_kernel(dat0[mmap0(p)][mmap1(p)], dat1[...]))

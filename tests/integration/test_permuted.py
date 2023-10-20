import ctypes

import loopy as lp
import numpy as np
import pymbolic as pym
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
    IntType,
    MultiArray,
    ScalarType,
    do_loop,
    loop,
)
from pyop3.codegen.ir import loopy_lang_version, loopy_target
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
        target=loopy_target(),
        name="scalar_copy",
        lang_version=loopy_lang_version(),
    )
    return Function(code, [READ, WRITE])


@pytest.fixture
def vector_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", ScalarType, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", ScalarType, (3,), is_input=False, is_output=True),
        ],
        target=loopy_target(),
        name="vector_copy",
        lang_version=loopy_lang_version(),
    )
    return Function(code, [READ, WRITE])


def test_vector_copy_with_permuted_axis(vector_copy_kernel):
    m, n = 6, 3
    perm = np.asarray([3, 2, 0, 5, 4, 1], dtype=IntType)

    axes = AxisTree(Axis(m, id="root"), {"root": Axis(n)})
    paxes = axes.with_modified_node(
        axes.root,
        permutation=perm,
    )
    dat0 = MultiArray(axes, name="dat0", data=np.arange(m * n, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    do_loop(p := axes.root.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))
    assert np.allclose(dat1.data.reshape((m, n))[perm].flatten(), dat0.data)


def test_vector_copy_with_two_permuted_axes(vector_copy_kernel):
    a, b, c = 4, 2, 3
    perm0 = [3, 1, 0, 2]
    perm1 = [1, 0]

    axis0 = Axis(a, "ax0")
    axis1 = Axis(b, "ax1")
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

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    iterset = AxisTree(axis0, {axis0.id: axis1})
    do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))

    assert np.allclose(
        dat1.data.reshape((a, b, c))[perm0][:, perm1].flatten(), dat0.data
    )


def test_vector_copy_with_permuted_inner_axis(vector_copy_kernel):
    a, b, c = 5, 4, 3
    perm = [3, 1, 0, 2]

    root = Axis(a, "ax0")
    inner_axis = Axis(b)
    inner_paxis = inner_axis.copy(permutation=perm)

    axes = AxisTree(root, {root.id: inner_axis, inner_axis.id: Axis(c)})
    paxes = AxisTree(root, {root.id: inner_paxis, inner_paxis.id: Axis(c)})

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    iterset = AxisTree(root, {root.id: inner_axis})
    do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))

    assert np.allclose(dat1.data.reshape((a, b, c))[:, perm].flatten(), dat0.data)


def test_vector_copy_with_permuted_multi_component_axes(vector_copy_kernel):
    m, n, a, b = 3, 2, 2, 3
    perm = [4, 2, 0, 3, 1]

    fullperm = [10, 11] + [5, 6] + [0, 1] + [7, 8, 9] + [2, 3, 4]

    root = Axis([m, n])
    axes = AxisTree(root, {root.id: [Axis(a), Axis(b)]})
    paxes = axes.with_modified_node(axes.root, permutation=perm)

    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(paxes, name="dat1", dtype=ScalarType)

    iterset = AxisTree(Axis([root.components[1]], root.label))
    do_loop(p := iterset.index(), vector_copy_kernel(dat0[p, :], dat1[p, :]))

    assert np.allclose(dat1.data[fullperm][: m * a], 0)
    assert np.allclose(dat1.data[fullperm][m * a :], dat0.data[m * a :])


def test_scalar_copy_with_permuted_inner_axis(scalar_copy_kernel):
    m, n = 4, 3
    perm = np.asarray([2, 0, 1], dtype=IntType)
    npoints = m * n

    axes = AxisTree(root := Axis(m, "ax0"), {root.id: Axis(n, "ax1", permutation=perm)})
    dat0 = MultiArray(axes, name="dat0", data=np.arange(npoints, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    do_loop(p := axes.index(), scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat0.data, dat1.data)

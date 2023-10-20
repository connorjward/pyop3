import numpy as np

import pyop3 as op3


def test_transpose(scalar_copy_kernel):
    npoints = 5
    axis0 = op3.Axis(npoints)
    axis1 = op3.Axis(npoints)
    axes0 = op3.AxisTree(axis0, {axis0.id: axis1})
    axes1 = op3.AxisTree(axis1, {axis1.id: axis0})

    array0 = op3.MultiArray(
        axes0, name="array0", data=np.arange(axes0.size, dtype=op3.ScalarType)
    )
    array1 = op3.MultiArray(axes1, name="array1", dtype=array0.dtype)

    op3.do_loop(
        p := axis0.index(),
        op3.loop(q := axis1.index(), scalar_copy_kernel(array0[p, q], array1[q, p])),
    )
    assert np.allclose(
        array1.data.reshape((npoints, npoints)),
        array0.data.reshape((npoints, npoints)).T,
    )


def test_nested_multi_component_loops(scalar_copy_kernel):
    a, b, c, d = 2, 3, 4, 5
    axis0 = op3.Axis([op3.AxisComponent(a, "a"), op3.AxisComponent(b, "b")], "ax0")
    axis1 = op3.Axis([op3.AxisComponent(c, "c"), op3.AxisComponent(d, "d")], "ax1")
    axis1_dup = axis1.copy(id=op3.Axis.unique_id())
    axes = op3.AxisTree(axis0, {axis0.id: [axis1, axis1_dup]})

    array0 = op3.MultiArray(
        axes, name="array0", data=np.arange(axes.size, dtype=op3.ScalarType)
    )
    array1 = op3.MultiArray(axes, name="array1", dtype=array0.dtype)

    op3.do_loop(
        p := axis0.index(),
        op3.loop(q := axis1.index(), scalar_copy_kernel(array0[p, q], array1[p, q])),
    )
    assert np.allclose(array1.data_ro, array0.data_ro)

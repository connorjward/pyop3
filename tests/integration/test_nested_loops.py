import numpy as np

from pyop3 import Axis, AxisTree, MultiArray, ScalarType, do_loop, loop


def test_transpose(scalar_copy_kernel):
    npoints = 5
    axis0 = Axis(npoints)
    axis1 = Axis(npoints)

    axes0 = AxisTree(axis0, {axis0.id: axis1})
    axes1 = AxisTree(axis1, {axis1.id: axis0})

    array0 = MultiArray(
        axes0, name="array0", data=np.arange(axes0.size, dtype=ScalarType)
    )
    array1 = MultiArray(axes1, name="array1", dtype=array0.dtype)

    do_loop(
        p := axis0.index(),
        loop(q := axis1.index(), scalar_copy_kernel(array0[p, q], array1[q, p])),
    )
    assert np.allclose(
        array1.data.reshape((npoints, npoints)),
        array0.data.reshape((npoints, npoints)).T,
    )

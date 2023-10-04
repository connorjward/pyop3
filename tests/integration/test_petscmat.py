import numpy as np
import pytest
from pyrsistent import pmap

import pyop3 as op3


def test_map_compression(scalar_copy_kernel_int):
    # Produce a point-to-DoF map from a point-to-point map. This should be
    # automated by Mats (but not PetscMats).
    npoints = 5
    ndofs = 3
    arity = 2

    points_axis = op3.Axis([op3.AxisComponent(npoints, "pt0")], "ax0")
    dofs_axis = op3.Axis(ndofs)
    arity_axis = op3.Axis([op3.AxisComponent(arity, "map_pt0")], "map0")

    data_axes = op3.AxisTree(points_axis, {points_axis.id: dofs_axis})

    point_to_points_axes = op3.AxisTree(points_axis, {points_axis.id: arity_axis})
    pt_to_pts_data = np.asarray(
        [[0, 2], [4, 3], [1, 1], [4, 0], [2, 3]], dtype=op3.IntType
    )
    point_to_points_array = op3.MultiArray(
        point_to_points_axes, name="map0", data=pt_to_pts_data.flatten()
    )
    pt_to_pts_map = op3.Map(
        {
            pmap({"ax0": "pt0"}): [
                op3.TabulatedMapComponent(
                    "ax0", "pt0", point_to_points_array, label="map_pt0"
                )
            ]
        },
        "map0",
    )

    pt_to_dofs_axes = op3.AxisTree(
        points_axis, {points_axis.id: arity_axis, arity_axis.id: dofs_axis}
    )
    pt_to_dofs = op3.MultiArray(pt_to_dofs_axes, dtype=op3.IntType)

    # op3.do_loop(
    loop = op3.loop(
        p := points_axis.index(),
        op3.loop(
            q := pt_to_pts_map(p).index(),
            op3.loop(
                d := data_axes[p, :].index(),
                # the offset bit is currently using the wrong thing
                scalar_copy_kernel_int(
                    op3.offset(data_axes, [q, d]), pt_to_dofs[p, q.i, d]
                ),
            ),
        ),
    )
    loop()

    expected = np.zeros((npoints, arity, ndofs))
    for i0 in range(npoints):
        for i1 in range(arity):
            for i2 in range(ndofs):
                offset = pt_to_pts_data[i0, i1] * ndofs + i2
                expected[i0, i1, i2] = offset
    assert np.allclose(pt_to_dofs.data_ro, expected.flatten())

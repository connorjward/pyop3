import ctypes

import loopy as lp
import numpy as np
import pymbolic as pym
import pytest
from pyrsistent import pmap

from pyop3 import AffineSliceComponent, IndexTree, Slice, SplitLoopIndex
from pyop3.axis import Axis, AxisComponent, AxisTree
from pyop3.codegen import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.distarray import MultiArray
from pyop3.dtypes import IntType, ScalarType
from pyop3.index import SplitIndexTree
from pyop3.loopexpr import INC, READ, WRITE, LoopyKernel, do_loop, loop
from pyop3.utils import flatten, just_one


def test_different_axis_orderings_do_not_change_packing_order():
    m0, m1, m2 = 5, 2, 2
    npoints = m0 * m1 * m2

    lpy_kernel = lp.make_kernel(
        [f"{{ [i]: 0 <= i < {m1} }}", f"{{ [j]: 0 <= j < {m2} }}"],
        "y[i, j] = x[i, j]",
        [
            lp.GlobalArg("x", np.float64, (m1, m2), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (m1, m2), is_input=False, is_output=True),
        ],
        target=LOOPY_TARGET,
        name="copy",
        lang_version=(2018, 2),
    )
    copy_kernel = LoopyKernel(lpy_kernel, [READ, WRITE])

    axis0 = Axis(m0, "ax0")
    axis1 = Axis(m1, "ax1")
    axis2 = Axis(m2, "ax2")

    axes0 = AxisTree(axis0, {axis0.id: [axis1], axis1.id: [axis2]})
    axes1 = AxisTree(axis0, {axis0.id: [axis2], axis2.id: [axis1]})

    data0 = np.arange(npoints, dtype=ScalarType).reshape((m0, m1, m2))
    data1 = data0.swapaxes(1, 2)

    dat0_0 = MultiArray(
        axes0,
        name="dat0_0",
        data=data0.flatten(),
    )
    dat0_1 = MultiArray(axes1, name="dat0_1", data=data1.flatten())
    dat1 = MultiArray(axes0, name="dat1", data=np.zeros(npoints, dtype=ScalarType))

    p = axis0.index()
    path = just_one(axis0.target_paths)
    q = IndexTree(SplitLoopIndex(p, path))
    q = q.put_node(
        Slice("ax1", [AffineSliceComponent(axis1.component_labels[0])]), *q.leaf
    )
    q = q.put_node(
        Slice("ax2", [AffineSliceComponent(axis2.component_labels[0])]), *q.leaf
    )
    q = SplitIndexTree({pmap({p: path}): q})

    do_loop(p, copy_kernel(dat0_0[q], dat1[q]))
    assert np.allclose(dat1.data, dat0_0.data)

    dat1.data[...] = 0

    do_loop(p, copy_kernel(dat0_1[q], dat1[q]))
    assert np.allclose(dat1.data, dat0_0.data)

import numpy as np

from pyop3.axis import Axis, AxisTree
from pyop3.distarray import MultiArray
from pyop3.dtypes import IntType
from pyop3.extras.viz import view_axes

ntest = 0

if ntest == 0:
    nnzaxes = AxisTree(Axis(5))
    nnzdata = np.asarray([3, 2, 1, 4, 2], dtype=IntType)
    nnz = MultiArray(nnzaxes, data=nnzdata)
    axes = nnzaxes.add_subaxis(Axis(nnz), nnzaxes.leaf)

elif ntest == 1:
    axes = AxisTree(
        Axis([3, 2], permutation=[2, 3, 0, 4, 1], id="ax0"),
        {
            "ax0": [Axis(3, id="ax1"), Axis(2)],
            "ax1": Axis(2),
        },
    )

else:
    raise AssertionError

view_axes(axes)

import pytest


from pyop3 import *
from pyop3.meshdata.dat import ConstrainedMultiAxis, order_axes


# TODO parametrise this
def test_axis_ordering():
    ax1 = MultiAxis([AxisPart(3)], id="ax1")
    ax2 = MultiAxis([AxisPart(2)], id="ax2")

    axes = [
        ConstrainedMultiAxis(ax1),
        ConstrainedMultiAxis(ax2)]
    ax = order_axes(axes)
    assert ax.id == "ax1"
    assert ax.part.subaxis.id == "ax2"

    axes = [
        ConstrainedMultiAxis(ax1),
        ConstrainedMultiAxis(ax2, priority=0)]
    ax = order_axes(axes)
    assert ax.id == "ax2"
    assert ax.part.subaxis.id == "ax1"

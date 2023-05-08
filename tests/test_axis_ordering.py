import pytest


from pyop3 import *
from pyop3.meshdata.dat import ConstrainedMultiAxis, order_axes


def id_tuple(nodes):
    return tuple(node.id for node in nodes)


def label_tuple(nodes):
    return tuple(node.label for node in nodes)


# TODO parametrise this
def test_axis_ordering():
    # should we enforce that we always have a label?
    ax1 = AxisPart(3, id="ax1", label="ax1")
    ax2 = AxisPart(2, id="ax2", label="ax2")

    layout = [ConstrainedMultiAxis([ax1]), ConstrainedMultiAxis([ax2])]
    axtree = order_axes(layout)

    assert axtree.rootless_depth == 2
    assert axtree.root_axes == (ax1,)
    assert axtree.children(ax1) == (ax2,)
    assert axtree.children(ax2) == ()

    axes = [ConstrainedMultiAxis([ax1]), ConstrainedMultiAxis([ax2], priority=0)]
    axtree = order_axes(axes)
    assert axtree.rootless_depth == 2
    assert axtree.root_axes == (ax2,)
    assert axtree.children(ax2) == (ax1,)
    assert axtree.children(ax1) == ()


def test_multicomponent_constraints():
    axes1 = [
        MultiAxisComponent(3, label="cpt1", id="cpt1"),
        MultiAxisComponent(3, label="cpt2", id="cpt2"),
    ]
    axes2 = [MultiAxisComponent(3, label="cpt3", id="cpt3")]

    ###

    layout = [ConstrainedMultiAxis(axes1, within_labels={"cpt3"}), ConstrainedMultiAxis(axes2)]
    axtree = order_axes(layout)

    assert axtree.rootless_depth == 2
    assert id_tuple(axtree.root_axes) == ("cpt3",)
    assert id_tuple(axtree.children("cpt3")) == ("cpt1", "cpt2")
    assert all(axtree.children(node) == () for node in ["cpt1", "cpt2"])

    ###

    with pytest.raises(ValueError):
        layout = [
            ConstrainedMultiAxis(axes1, within_labels={"cpt3"}),
            ConstrainedMultiAxis(axes2, within_labels={"cpt1"}),
        ]
        axtree = order_axes(layout)

    ###

    layout = [
        ConstrainedMultiAxis(axes1),
        ConstrainedMultiAxis(axes2, within_labels={"cpt2"}),
    ]
    axtree = order_axes(layout)

    assert axtree.rootless_depth == 2
    assert id_tuple(axtree.root_axes) == ("cpt1", "cpt2")
    assert axtree.children("cpt1") == ()
    assert id_tuple(axtree.children("cpt2")) == ("cpt3",)

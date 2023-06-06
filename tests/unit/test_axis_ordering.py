import pytest

from pyop3 import *
from pyop3.multiaxis import ConstrainedMultiAxis


def id_tuple(nodes):
    return tuple(node.id for node in nodes)


def label_tuple(nodes):
    return tuple(node.label for node in nodes)


def test_axis_ordering():
    axis0 = MultiAxis([MultiAxisComponent(3, "cpt0")], "ax0")
    axis1 = MultiAxis([MultiAxisComponent(1, "cpt0")], "ax1")

    layout = [ConstrainedMultiAxis(axis0), ConstrainedMultiAxis(axis1)]
    axes = MultiAxisTree.from_layout(layout)

    assert axes.depth == 2
    assert axes.root == axis0
    assert just_one(axes.children({"ax0": "cpt0"})).label == "ax1"
    assert not axes.children({"ax0": "cpt0", "ax1": "cpt0"})

    ###

    layout = [ConstrainedMultiAxis(axis0), ConstrainedMultiAxis(axis1, priority=0)]
    axes = MultiAxisTree.from_layout(layout)
    assert axes.depth == 2
    assert axes.root.label == "ax1"
    assert just_one(axes.children({"ax1": "cpt0"})).label == "ax0"
    assert not axes.children({"ax0": "cpt0", "ax1": "cpt0"})


def test_multicomponent_constraints():
    axis0 = MultiAxis(
        [MultiAxisComponent(3, "cpt0"), MultiAxisComponent(3, "cpt1")], "ax0"
    )
    axis1 = MultiAxis([MultiAxisComponent(3, "cpt0")], "ax1")

    ###

    layout = [
        ConstrainedMultiAxis(axis0, within_labels={("ax1", "cpt0")}),
        ConstrainedMultiAxis(axis1),
    ]
    axes = order_axes(layout)

    assert axes.depth == 2
    assert axes.root.label == "ax1"
    assert just_one(axes.children({"ax1": "cpt0"})).label == "ax0"
    assert not axes.children({"ax1": "cpt0", "ax0": "cpt0"})
    assert not axes.children({"ax1": "cpt0", "ax0": "cpt1"})

    ###

    with pytest.raises(ValueError):
        layout = [
            ConstrainedMultiAxis(axis0, within_labels={("ax1", "cpt0")}),
            ConstrainedMultiAxis(axis1, within_labels={("ax0", "cpt0")}),
        ]
        order_axes(layout)

    ###

    layout = [
        ConstrainedMultiAxis(axis0),
        ConstrainedMultiAxis(axis1, within_labels={("ax0", "cpt1")}),
    ]
    axes = order_axes(layout)

    assert axes.depth == 2
    assert axes.root.label == "ax0"
    assert not axes.children({"ax0": "cpt0"})
    assert just_one(axes.children({"ax0": "cpt1"})).label == "ax1"
    assert not axes.children({"ax0": "cpt1", "ax1": "cpt0"})

    ###


def test_multicomponent_constraints_more():
    # ax0
    # ├──➤ cpt0 : ax1
    # │           └──➤ cpt0
    # └──➤ cpt1 : ax2
    #             └──➤ cpt0 : ax1
    #                         └──➤ cpt0

    axis0 = MultiAxis(
        [
            MultiAxisComponent(3, "cpt0"),
            MultiAxisComponent(3, "cpt1"),
        ],
        "ax0",
    )
    axis1 = MultiAxis([MultiAxisComponent(3, "cpt0")], "ax1")
    axis2 = MultiAxis([MultiAxisComponent(3, "cpt0")], "ax2")

    layout = [
        ConstrainedMultiAxis(axis0, priority=0),
        ConstrainedMultiAxis(axis1, priority=20),
        ConstrainedMultiAxis(axis2, within_labels={("ax0", "cpt1")}, priority=10),
    ]
    axes = order_axes(layout)

    assert axes.depth == 3
    assert axes.root.label == "ax0"
    assert just_one(axes.children({"ax0": "cpt0"})).label == "ax1"
    assert just_one(axes.children({"ax0": "cpt1"})).label == "ax2"
    assert not axes.children({"ax0": "cpt0", "ax1": "cpt0"})
    assert just_one(axes.children({"ax0": "cpt1", "ax2": "cpt0"})).label == "ax1"
    assert not axes.children({"ax0": "cpt1", "ax2": "cpt0", "ax1": "cpt0"})

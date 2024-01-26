import pytest
from pyrsistent import freeze, pmap

import pyop3 as op3


def test_axes_iter_flat():
    iterset = op3.Axis({"pt0": 5}, "ax0")
    for i, p in enumerate(iterset.iter()):
        assert p.source_path == freeze({"ax0": "pt0"})
        assert p.target_path == p.source_path
        assert p.source_exprs == freeze({"ax0": i})
        assert p.target_exprs == p.source_exprs


def test_axes_iter_nested():
    iterset = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": 5}, "ax0"): op3.Axis({"pt0": 3}, "ax1"),
        },
    )

    iterator = iterset.iter()
    for i in range(5):
        for j in range(3):
            p = next(iterator)
            assert p.source_path == freeze({"ax0": "pt0", "ax1": "pt0"})
            assert p.target_path == p.source_path
            assert p.source_exprs == freeze({"ax0": i, "ax1": j})
            assert p.target_exprs == p.source_exprs

    # make sure that the iterator is empty
    try:
        next(iterator)
        assert False
    except StopIteration:
        pass


def test_axes_iter_multi_component():
    iterset = op3.Axis({"pt0": 3, "pt1": 3}, "ax0")

    iterator = iterset.iter()
    for i in range(3):
        p = next(iterator)
        assert p.source_path == freeze({"ax0": "pt0"})
        assert p.target_path == p.source_path
        assert p.source_exprs == freeze({"ax0": i})
        assert p.target_exprs == p.source_exprs

    for i in range(3):
        p = next(iterator)
        assert p.source_path == freeze({"ax0": "pt1"})
        assert p.target_path == p.source_path
        assert p.source_exprs == freeze({"ax0": i})
        assert p.target_exprs == p.source_exprs

    # make sure that the iterator is empty
    try:
        next(iterator)
        assert False
    except StopIteration:
        pass


def test_index_forest_inserts_extra_slices():
    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": 5}, "ax0"): op3.Axis({"pt0": 3}, "ax1"),
        },
    )
    iforest = op3.itree.as_index_forest(slice(None), axes=axes)

    # since there are no loop indices, the index forest should contain a single entry
    assert len(iforest) == 1
    assert pmap() in iforest.keys()

    itree = iforest[pmap()]
    assert itree.depth == 2


@pytest.mark.xfail(reason="Index tree.leaves currently broken")
def test_multi_component_index_forest_inserts_extra_slices():
    axes = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": 5, "pt1": 4}, "ax0"): {
                "pt0": op3.Axis({"pt0": 3}, "ax1"),
                "pt1": op3.Axis({"pt0": 2}, "ax1"),
            }
        },
    )
    iforest = op3.itree.as_index_forest(
        op3.Slice("ax1", [op3.AffineSliceComponent("pt0")]), axes=axes
    )

    # since there are no loop indices, the index forest should contain a single entry
    assert len(iforest) == 1
    assert pmap() in iforest.keys()

    itree = iforest[pmap()]
    assert itree.depth == 2
    assert itree.root.label == "ax1"

    # FIXME this currently fails because itree.leaves does not work.
    # This is because it is difficult for loop indices to advertise component labels.
    # Perhaps they should be an index component themselves? I have made some notes
    # on this.
    assert all(index.label == "ax0" for index, _ in itree.leaves)
    assert len(itree.leaves) == 2

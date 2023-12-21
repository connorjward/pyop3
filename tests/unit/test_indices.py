import numpy as np
import pytest
from pyrsistent import freeze

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

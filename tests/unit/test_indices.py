import numpy as np
import pytest
from pyrsistent import freeze

import pyop3 as op3


def test_loop_index_iter_flat():
    iterset = op3.AxisTree(op3.Axis([op3.AxisComponent(5, "pt0")], "ax0"))
    expected = [(freeze({"ax0": "pt0"}), freeze({"ax0": i})) for i in range(5)]
    assert list(iterset.index().iter()) == expected


def test_loop_index_iter_nested():
    iterset = op3.AxisTree(
        op3.Axis([op3.AxisComponent(5, "pt0")], "ax0", id="root"),
        {
            "root": op3.Axis([op3.AxisComponent(3, "pt0")], "ax1"),
        },
    )

    path = freeze({"ax0": "pt0", "ax1": "pt0"})
    expected = [
        (path, freeze({"ax0": i, "ax1": j})) for i in range(5) for j in range(3)
    ]
    assert list(iterset.index().iter()) == expected


def test_loop_index_iter_multi_component():
    iterset = op3.AxisTree(
        op3.Axis([op3.AxisComponent(3, "pt0"), op3.AxisComponent(3, "pt1")], "ax0"),
    )

    path0 = freeze({"ax0": "pt0"})
    path1 = freeze({"ax0": "pt1"})
    expected = [(path0, freeze({"ax0": i})) for i in range(3)] + [
        (path1, freeze({"ax0": i})) for i in range(3)
    ]
    assert list(iterset.index().iter()) == expected

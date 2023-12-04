import numpy as np
import pytest
from pyrsistent import freeze

import pyop3 as op3


def test_loop_index_iter_flat():
    iterset = op3.AxisTree.from_nest(op3.Axis({"pt0": 5}, "ax0"))
    expected = [
        (freeze({"ax0": "pt0"}),) * 2 + (freeze({"ax0": i}),) * 2 for i in range(5)
    ]
    assert list(iterset.index().iter()) == expected


def test_loop_index_iter_nested():
    iterset = op3.AxisTree.from_nest(
        {
            op3.Axis({"pt0": 5}, "ax0"): op3.Axis({"pt0": 3}, "ax1"),
        },
    )

    path = freeze({"ax0": "pt0", "ax1": "pt0"})
    expected = [
        (path,) * 2 + (freeze({"ax0": i, "ax1": j}),) * 2
        for i in range(5)
        for j in range(3)
    ]
    assert list(iterset.index().iter()) == expected


def test_loop_index_iter_multi_component():
    iterset = op3.AxisTree.from_nest(
        op3.Axis({"pt0": 3, "pt1": 3}, "ax0"),
    )

    path0 = freeze({"ax0": "pt0"})
    path1 = freeze({"ax0": "pt1"})
    expected = [(path0,) * 2 + (freeze({"ax0": i}),) * 2 for i in range(3)] + [
        (path1,) * 2 + (freeze({"ax0": i}),) * 2 for i in range(3)
    ]
    assert list(iterset.index().iter()) == expected

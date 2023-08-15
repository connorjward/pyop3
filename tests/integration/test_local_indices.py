import numpy as np
import pytest

from pyop3 import Axis, AxisTree, MultiArray, ScalarType, do_loop

# TODO
pytest.skip(allow_module_level=True)


def test_copy_with_local_indices(scalar_copy_kernel):
    size = 10
    axes = AxisTree(Axis(size))
    dat0 = MultiArray(axes, data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, dtype=dat0.dtype)

    do_loop(p := axes.enumerate(), scalar_copy_kernel(dat0[p.index], dat1[p.count]))
    assert np.allclose(dat1.data, dat0.data)


def test_inc_into_small_array(scalar_copy_kernel):
    size = 10
    dim = 3

    big = MultiArray(big_axes)
    small = MultiArray(small_axes)

    # The following is equivalent to
    # for p in big_axes.root:
    #   for i, q in enumerate(big_axes[p]):
    #     small[i] = big[p, q]
    do_loop(
        p := big_axes.root.index(),
        loop(
            # NOTE: This could be a namedtuple to make things clearer
            q := big_axes[p, :].enumerate(),
            scalar_copy_kernel(big[p, q[1]], small[q[0]]),
        ),
    )

    assert False, "TODO"

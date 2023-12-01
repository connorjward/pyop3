import numpy as np
import pytest

import pyop3 as op3


def test_copy_with_local_indices(scalar_copy_kernel):
    axis = op3.Axis(10)
    dat0 = op3.Dat(axis, data=np.arange(axis.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(axis, dtype=dat0.dtype)

    op3.do_loop(
        p := axes.index(),
        scalar_copy_kernel(dat0[p], dat1[p.i]),
    )
    assert np.allclose(dat1.data_ro, dat0.data_ro)


def test_copy_slice(scalar_copy_kernel):
    axis = op3.Axis(10)
    dat0 = op3.Dat(axis, name="dat0", data=np.arange(axis.size), dtype=op3.ScalarType)
    dat1 = op3.Dat(axis[:5], name="dat1", dtype=dat0.dtype)

    # op3.do_loop(
    loop = op3.loop(
        p := axis[::2].index(),
        scalar_copy_kernel(dat0[p], dat1[p.i]),
    )
    loop()
    breakpoint()
    assert np.allclose(dat1.data_ro, dat0.data_ro[::2])

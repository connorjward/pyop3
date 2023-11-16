import loopy as lp
import numpy as np
import pytest

import pyop3 as op3
from pyop3.extras.debug import print_with_rank
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET


@pytest.mark.parallel(nprocs=2)
# @pytest.mark.parametrize("intent", [op3.INC, op3.MIN, op3.MAX])
@pytest.mark.parametrize(["intent", "fill_value"], [(op3.WRITE, 0), (op3.INC, 0)])
def test_parallel_loop(comm, paxis, intent, fill_value):
    assert comm.size == 2
    rank = comm.rank
    other_rank = (comm.rank + 1) % 2

    print_with_rank(paxis)
    print_with_rank(paxis.axes.freeze().owned)

    dat = op3.Dat(paxis, data=np.full(paxis.size, fill_value, dtype=int))

    knl = op3.Function(
        lp.make_kernel(
            "{ [i]: 0 <= i < 1 }",
            f"x[i] = {rank} + 1",
            [lp.GlobalArg("x", shape=(1,), dtype=int)],
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        ),
        [intent],
    )

    op3.do_loop(
        p := paxis.index(),
        knl(dat[p]),
    )

    assert np.equal(dat.array._data[: paxis.owned_count], rank + 1).all()
    assert np.equal(dat.array._data[paxis.owned_count :], fill_value).all()

    # since we do not modify ghost points no reduction is needed
    assert dat.array._pending_reduction is None

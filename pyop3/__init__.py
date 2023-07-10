# Many pyop3 objects inherit from pytools.RecordWithoutPickling.
# RecordWithoutPickling sets __getattr__ for linting purposes but this breaks
# tracebacks for @property methods so we remove it here.
import pytools

del pytools.RecordWithoutPickling.__getattr__
del pytools


from pyop3.axis import Axis, AxisTree  # noqa: F401
from pyop3.distarray import MultiArray  # noqa: F401
from pyop3.dtypes import ScalarType  # noqa: F401
from pyop3.index import Index, IndexTree, Slice, TabulatedMap  # noqa: F401
from pyop3.loopexpr import (  # noqa: F401
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    WRITE,
    Loop,
    LoopyKernel,
    do_loop,
    loop,
)
from pyop3.meshdata import Const, Dat, Mat  # noqa: F401
from pyop3.space import ConstrainedAxis, Space  # noqa: F401

# from pyop3.mesh import *  # noqa: F401
# from pyop3.multiaxis import *  # noqa: F401

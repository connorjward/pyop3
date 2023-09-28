# Many pyop3 objects inherit from pytools.RecordWithoutPickling.
# RecordWithoutPickling sets __getattr__ for linting purposes but this breaks
# tracebacks for @property methods so we remove it here.
import pytools

del pytools.RecordWithoutPickling.__getattr__
del pytools


import pyop3.transforms
from pyop3.axes import Axis, AxisComponent, AxisTree  # noqa: F401
from pyop3.distarray import MultiArray  # noqa: F401
from pyop3.dtypes import IntType, ScalarType  # noqa: F401
from pyop3.indices import (  # noqa: F401
    AffineSliceComponent,
    Index,
    IndexTree,
    Map,
    Slice,
    SliceComponent,
    Subset,
    TabulatedMapComponent,
)
from pyop3.lang import (  # noqa: F401
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    WRITE,
    Function,
    Loop,
    do_loop,
    loop,
    offset,
)
from pyop3.meshdata import Const, Dat, Mat  # noqa: F401
from pyop3.space import ConstrainedAxis, Space  # noqa: F401

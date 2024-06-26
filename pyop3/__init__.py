# Many pyop3 objects inherit from pytools.RecordWithoutPickling.
# RecordWithoutPickling sets __getattr__ for linting purposes but this breaks
# tracebacks for @property methods so we remove it here.
import pytools

try:
    del pytools.RecordWithoutPickling.__getattr__
except AttributeError:
    pass
del pytools


import pyop3.ir
import pyop3.transform
from pyop3.array import Array, FancyIndexWriteException, HierarchicalArray, MultiArray
from pyop3.array.petsc import Mat, Sparsity  # noqa: F401
from pyop3.axtree import (  # noqa: F401
    Axis,
    AxisComponent,
    AxisTree,
    AxisVariable,
    IndexedAxisTree,
)
from pyop3.buffer import DistributedBuffer, NullBuffer  # noqa: F401
from pyop3.dtypes import IntType, ScalarType  # noqa: F401
from pyop3.itree import (  # noqa: F401
    AffineSliceComponent,
    Index,
    IndexTree,
    LoopIndex,
    Map,
    Slice,
    SliceComponent,
    Subset,
    TabulatedMapComponent,
)
from pyop3.itree.tree import ScalarIndex, as_index_forest
from pyop3.lang import (  # noqa: F401
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    NA,
    READ,
    RW,
    WRITE,
    AddAssignment,
    DummyKernelArgument,
    Function,
    Loop,
    OpaqueKernelArgument,
    Pack,
    ReplaceAssignment,
    do_loop,
    loop,
)
from pyop3.sf import StarForest, serial_forest, single_star

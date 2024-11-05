# arguably put this directly in pyop3/__init__.py
# no use namespacing here really

from .base import Array  # noqa: F401
from .harray import (  # noqa: F401
    FancyIndexWriteException,
    Dat,
    _ConcretizedDat,
)
from .petsc import Mat, Sparsity, AbstractMat  # noqa: F401

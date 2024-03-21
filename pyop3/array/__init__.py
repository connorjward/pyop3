# arguably put this directly in pyop3/__init__.py
# no use namespacing here really

from .base import Array  # noqa: F401
from .harray import (  # noqa: F401
    ContextSensitiveMultiArray,
    HierarchicalArray,
    MultiArray,
)
from .petsc import PetscMat, PetscMatAIJ  # noqa: F401

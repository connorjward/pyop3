# TODO: This module is quite redundant. Most transformations do not fit this paradigm.
# Only reshapes.

import abc

from pyop3.array.base import Array
from pyop3.array.harray import Dat
from pyop3.array.petsc import AbstractMat


class ArrayTransformation(abc.ABC):
    """A reversible transformation that acts on an array."""
    def __init__(self, initial: Array) -> None:
        self.initial = initial

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.initial!r})"

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.initial})"


class DatTransformation(ArrayTransformation, abc.ABC):
    def __init__(self, initial: Dat) -> None:
        super().__init__(initial)


class MatTransformation(ArrayTransformation, abc.ABC):
    def __init__(self, initial: AbstractMat) -> None:
        super().__init__(initial)


class DatReshape(DatTransformation):
    pass


class MatReshape(MatTransformation):
    pass

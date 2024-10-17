from __future__ import annotations

import abc


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
    def __init__(self, initial: Mat) -> None:
        super().__init__(initial)


class Reshape(DatTransformation):
    pass

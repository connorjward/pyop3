import abc

from pyop3.lang import FunctionArgument, ReplaceAssignment
from pyop3.utils import UniqueNameGenerator


class Array(FunctionArgument, abc.ABC):
    _prefix = "array"
    _name_generator = UniqueNameGenerator()

    def __init__(self, name=None, *, prefix=None) -> None:
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")
        self.name = name or self._name_generator(prefix or self._prefix)

    def assign(self, other, eager=True):
        expr = ReplaceAssignment(self, other)
        return expr() if eager else expr

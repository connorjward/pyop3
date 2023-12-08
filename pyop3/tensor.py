import abc

from pyop3.array import Array
from pyop3.utils import UniqueNameGenerator


class Tensor(abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    _prefix = "tensor"
    _name_generator = UniqueNameGenerator()

    def __init__(self, array: Array, name=None, *, prefix=None) -> None:
        if self.rank not in array.valid_ranks:
            raise TypeError("Unsuitable array provided")
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")

        self.array = array
        self.name = name or self._name_generator(prefix or self._prefix)

    @property
    @abc.abstractmethod
    def rank(self) -> int:
        pass


class Global(Tensor):
    @property
    def rank(self) -> int:
        return 0


class Dat(Tensor):
    @property
    def rank(self) -> int:
        return 1


class Mat(Tensor):
    @property
    def rank(self) -> int:
        return 2

import abc

import pytools

from pyop3.utils import UniqueNameGenerator


# don't make a Record, copy() should be reserved for other stuff
# also adding "orig_array" causes a recursion issue with __repr__
class Tensor(abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    _prefix = "array"
    _name_generator = UniqueNameGenerator()

    def __init__(self, name=None, *, prefix=None) -> None:
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")

        self.name = name or self._name_generator(prefix or self._prefix)

    # hack for now, just dont make a pytool.Record
    def __repr__(self):
        return self.name

    def copy(self):
        raise NotImplementedError("Deepcopy stuff")

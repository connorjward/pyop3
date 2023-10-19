import abc

import pytools

from pyop3.utils import UniqueNameGenerator


# don't make a Record, copy() should be reserved for other stuff
# also adding "orig_array" causes a recursion issue with __repr__
class DistributedArray(pytools.RecordWithoutPickling, abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    fields = {"name", "orig_array"}
    prefix = "array"
    name_generator = UniqueNameGenerator()

    def __init__(self, name=None, prefix=None, orig_array=None) -> None:
        super().__init__()

        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")
        self.name = name or self.name_generator(prefix or self.prefix)
        self.orig_array = orig_array or self

    # hack for now, just dont make a pytool.Record
    def __repr__(self):
        return self.name

    def copy_record(self, **kwargs):
        return super().copy(**kwargs)

    def copy(self):
        raise NotImplementedError("Deepcopy stuff")

    # @abc.abstractmethod
    # def sync(self):
    #     pass

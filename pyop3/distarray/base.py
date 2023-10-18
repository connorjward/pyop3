import abc

import pytools

from pyop3.utils import UniqueNameGenerator


# don't make a Record, copy() should be reserved for other stuff
class DistributedArray(pytools.RecordWithoutPickling, abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    fields = {"name"}
    prefix = "array"
    name_generator = UniqueNameGenerator()

    def __init__(self, name=None, prefix=None) -> None:
        super().__init__()

        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")
        self.name = name or self.name_generator(prefix or self.prefix)

    def copy_record(self, **kwargs):
        return super().copy(**kwargs)

    def copy(self):
        raise NotImplementedError("Deepcopy stuff")

    # @abc.abstractmethod
    # def sync(self):
    #     pass

import abc

import pytools


# don't make a Record, copy() should be reserved for other stuff
class DistributedArray(abc.ABC, pytools.RecordWithoutPickling):
    """Base class for all :mod:`pyop3` parallel objects."""

    fields = {"name"}

    def __init__(self, name: str) -> None:
        pytools.Record.__init__(self)
        self.name = name

    def copy_record(self, **kwargs):
        return super().copy(**kwargs)

    def copy(self):
        raise NotImplementedError("Deepcopy stuff")

    # @abc.abstractmethod
    # def sync(self):
    #     pass

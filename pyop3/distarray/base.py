import abc

import pytools


# don't make a Record, copy() should be reserved for other stuff
# class DistributedArray(abc.ABC, pytools.Record):
class DistributedArray(abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    fields = {"name"}

    def __init__(self, name: str) -> None:
        pytools.Record.__init__(self)
        self.name = name

    # @abc.abstractmethod
    # def sync(self):
    #     pass

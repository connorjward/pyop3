import abc

import pytools


class DistributedArray(abc.ABC, pytools.Record):
    """Base class for all :mod:`pyop3` parallel objects."""

    fields = {"name"}

    def __init__(self, name: str) -> None:
        pytools.Record.__init__(self)
        self.name = name

    # @abc.abstractmethod
    # def sync(self):
    #     pass

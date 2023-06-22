import abc
import pytools


class DistributedArray(abc.ABC, pytools.ImmutableRecord):
    """Base class for all :mod:`pyop3` parallel objects."""

    fields = set()

    # @abc.abstractmethod
    # def sync(self):
    #     pass


# TODO make MultiArray inherit from this too

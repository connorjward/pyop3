import abc

import pytools


class DistributedArray(abc.ABC, pytools.Record):
    """Base class for all :mod:`pyop3` parallel objects."""

    fields = set()

    # @abc.abstractmethod
    # def sync(self):
    #     pass


# TODO make MultiArray inherit from this too

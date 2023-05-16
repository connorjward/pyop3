# TODO: Should this file be named base.py instead?
import abc


class DistributedArray(abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    # @abc.abstractmethod
    # def sync(self):
    #     pass


# TODO make MultiArray inherit from this too

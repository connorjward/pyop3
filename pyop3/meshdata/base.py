from __future__ import annotations
import abc
class MeshDataCarrier(abc.ABC):

    def __getitem__(self, indices: IndexTree) -> IndexedMultiArray:
        # TODO fail if we don't fully index the dat, this is because spaces can have
        # variable orderings so the resulting temporary would have undefined shape
        # if not is_fully_indexed(self.array.axes, indices):
        #     raise ValueError("Dats must be fully indexed")
        return self.array[indices]

    @property
    @abc.abstractmethod
    def array(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @property
    @abc.abstractmethod
    def comm(self):
        pass

    @property
    @abc.abstractmethod
    def internal_comm(self):
        pass

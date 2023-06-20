from mpi4py import MPI
from pyop3.mpi import internal_comm
from pyop3.space import order_axes
from pyop3.distarray import MultiArray
from pyop3.meshdata.base import MeshDataCarrier


class Const(MeshDataCarrier):
    def __init__(self, layout, data=None, *, dtype=None, name=None, comm=None):
        array = MultiArray(order_axes(layout), data=data, dtype=dtype, name=name)

        self._array = array
        self._comm = comm or MPI.COMM_SELF
        self._internal_comm = internal_comm(comm)

    @property
    def array(self):
        return self._array

    @property
    def name(self):
        return self.array.name

    @property
    def comm(self):
        return self._comm

    @property
    def internal_comm(self):
        return self._internal_comm

    # this function only exists for backwards compat, could we raise a FutureWarning?
    # in general users should not be accessing this data
    # this would let us have a more consistent API across Const, Dat and Mat
    @property
    def data(self):
        return self.array.data_rw

    @property
    def data_ro(self):
        return self.array.data_ro

    @property
    def data_wo(self):
        return self.array.data_wo

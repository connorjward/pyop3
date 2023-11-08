from functools import cached_property

from petsc4py import PETSc

from pyop3.dtypes import get_mpi_dtype


class StarForest:
    """Convenience wrapper for a `petsc4py.SF`."""

    def __init__(self, sf, size: int):
        self.sf = sf
        self.size = size

    @classmethod
    def from_graph(cls, size: int, nroots: int, ilocal, iremote, comm=None):
        sf = PETSc.SF().create(comm or PETSc.Sys.getDefaultComm())
        sf.setGraph(nroots, ilocal, iremote)
        return cls(sf, size)

    @cached_property
    def iroot(self):
        """Return the indices of roots on the current process."""
        # mark leaves and reduce
        buffer = np.full(self.size, False, dtype=bool)
        buffer[ilocal] = True
        self.reduce(buffer, MPI.REPLACE)

        # now clear the leaf indices, the remaining marked indices are roots
        buffer[ilocal] = False
        return just_one(np.nonzero(buffer))

    @property
    def ileaf(self):
        return self.ilocal

    @property
    def nroots(self):
        return self._graph[0]

    @property
    def nleaves(self):
        return len(self.ileaf)

    @property
    def ilocal(self):
        return self._graph[1]

    @property
    def iremote(self):
        return self._graph[2]

    def broadcast(self, *args):
        self.broadcast_begin(*args)
        self.broadcast_end(*args)

    def broadcast_begin(self, *args):
        bcast_args = self._prepare_args(*args)
        self.sf.bcastBegin(*bcast_args)

    def broadcast_end(self, *args):
        bcast_args = self._prepare_args(*args)
        self.sf.bcastEnd(*bcast_args)

    def reduce(self, *args):
        self.reduce_begin(*args)
        self.reduce_end(*args)

    def reduce_begin(self, *args):
        reduce_args = self._prepare_args(*args)
        self.sf.reduceBegin(*reduce_args)

    def reduce_end(self, *args):
        reduce_args = self._prepare_args(*args)
        self.sf.reduceEnd(*reduce_args)

    @cached_property
    def _graph(self):
        return self.sf.getGraph()

    @staticmethod
    def _prepare_args(*args):
        if len(args) == 3:
            from_buffer, to_buffer, op = args
        elif len(args) == 2:
            from_buffer, op = args
            to_buffer = from_buffer
        else:
            raise ValueError

        # what about cdim?
        dtype, _ = get_mpi_dtype(from_buffer.dtype)
        return (dtype, from_buffer, to_buffer, op)

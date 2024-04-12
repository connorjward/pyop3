from __future__ import annotations

import abc
import contextlib
import numbers
from functools import cached_property

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.dtypes import ScalarType
from pyop3.lang import READ, RW, WRITE, KernelArgument
from pyop3.sf import StarForest
from pyop3.utils import UniqueNameGenerator, as_tuple, deprecated, readonly


class IncompatibleStarForestException(Exception):
    pass


class DataTransferInFlightException(Exception):
    pass


class BadOrderingException(Exception):
    pass


def not_in_flight(fn):
    """Ensure that a method cannot be called when a transfer is in progress."""

    def wrapper(self, *args, **kwargs):
        if self._transfer_in_flight:
            raise DataTransferInFlightException(
                f"Not valid to call {fn.__name__} with messages in-flight, "
                f"please call {self._finalizer.__name__} first"
            )
        return fn(self, *args, **kwargs)

    return wrapper


class Buffer(KernelArgument, abc.ABC):
    DEFAULT_DTYPE = ScalarType

    @property
    @abc.abstractmethod
    def dtype(self):
        pass

    @property
    @abc.abstractmethod
    def datamap(self):
        pass

    @property
    def kernel_dtype(self):
        return self.dtype


class NullBuffer(Buffer):
    """A buffer that does not carry data.

    This is useful for handling temporaries when we generate code. For much
    of the compilation we want to treat temporaries like ordinary arrays but
    they are not passed as kernel arguments nor do they have any parallel
    semantics.

    """

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = self.DEFAULT_DTYPE
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def datamap(self):
        return pmap()


class DistributedBuffer(Buffer):
    """An array distributed across multiple processors with ghost values."""

    # NOTE: When GPU support is added, the host-device awareness and
    # copies should live in this class.

    # NOTE: It is probably easiest to treat the data as being "moved into" the
    # DistributedArray. But copies should ideally be avoided?

    _prefix = "array"
    _name_generator = UniqueNameGenerator()

    def __init__(
        self,
        shape,
        sf_or_comm,  # Should just be SF, or None?
        dtype=None,
        *,
        name=None,
        prefix=None,
        data=None,
    ):
        shape = as_tuple(shape)

        if not all(isinstance(s, numbers.Integral) for s in shape):
            raise TypeError

        if dtype is None:
            dtype = self.DEFAULT_DTYPE

        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")

        if data is not None:
            if data.shape != shape:
                raise ValueError
            if data.dtype != dtype:
                raise ValueError

        if isinstance(sf_or_comm, StarForest):
            sf = sf_or_comm
            comm = sf.comm
            # TODO I don't really like having shape as an argument...
            if sf and shape[0] != sf.size:
                raise IncompatibleStarForestException
        else:
            sf = None
            comm = sf_or_comm

        self.shape = shape
        self._dtype = dtype
        self._lazy_data = data
        self.sf = sf

        assert comm is not None
        self.comm = comm

        self.name = name or self._name_generator(prefix or self._prefix)

        # counter used to keep track of modifications
        self.state = 0

        # flags for tracking parallel correctness
        self._leaves_valid = True
        self._pending_reduction = None
        self._finalizer = None

    # @classmethod
    # def from_array(cls, array: np.ndarray, **kwargs):
    #     return cls(array.shape, array.dtype, data=array, **kwargs)

    @property
    def dtype(self):
        return self._dtype

    @property
    @not_in_flight
    @deprecated(".data_rw")
    def data(self):
        return self.data_rw

    @property
    @not_in_flight
    def data_rw(self):
        self.state += 1

        if not self._roots_valid:
            self._reduce_leaves_to_roots()

        # modifying owned values invalidates ghosts
        self._leaves_valid = False
        return self._owned_data

    @property
    @not_in_flight
    def data_ro(self):
        if not self._roots_valid:
            self._reduce_leaves_to_roots()
        return readonly(self._owned_data)

    @property
    @not_in_flight
    def data_wo(self):
        """
        Have to be careful. If not setting all values (i.e. subsets) should call
        `reduce_leaves_to_roots` first.

        When this is called we set roots_valid, claiming that any (lazy) 'in-flight' writes
        can be dropped.
        """
        self.state += 1

        # pending writes can be dropped
        self._pending_reduction = None
        self._leaves_valid = False
        return self._owned_data

    @property
    def is_distributed(self) -> bool:
        return self.comm.size > 1

    @property
    def leaves_valid(self) -> bool:
        return self._leaves_valid

    @property
    def datamap(self):
        return freeze({self.name: self})

    @property
    def _data(self):
        if self._lazy_data is None:
            self._lazy_data = np.zeros(self.shape, dtype=self.dtype)
        return self._lazy_data

    @property
    def _owned_data(self):
        if self.is_distributed and self.sf.nleaves > 0:
            return self._data[: -self.sf.nleaves]
        else:
            return self._data

    @property
    def _roots_valid(self) -> bool:
        return self._pending_reduction is None

    @property
    def _transfer_in_flight(self) -> bool:
        return self._finalizer is not None

    @cached_property
    def _reduction_ops(self):
        # TODO Move this import out, requires moving location of these intents
        from pyop3.lang import INC, WRITE

        return {
            WRITE: MPI.REPLACE,
            INC: MPI.SUM,
        }

    @not_in_flight
    def _reduce_leaves_to_roots(self):
        self._reduce_leaves_to_roots_begin()
        self._reduce_leaves_to_roots_end()

    @not_in_flight
    def _reduce_leaves_to_roots_begin(self):
        if not self._roots_valid:
            self.sf.reduce_begin(
                self._data, self._reduction_ops[self._pending_reduction]
            )
            self._leaves_valid = False
        self._finalizer = self._reduce_leaves_to_roots_end

    def _reduce_leaves_to_roots_end(self):
        if self._finalizer is None:
            raise BadOrderingException(
                "Should not call _reduce_leaves_to_roots_end without first calling "
                "_reduce_leaves_to_roots_begin"
            )
        if self._finalizer != self._reduce_leaves_to_roots_end:
            raise DataTransferInFlightException("Wrong finalizer called")

        if not self._roots_valid:
            self.sf.reduce_end(self._data, self._reduction_ops[self._pending_reduction])
        self._pending_reduction = None
        self._finalizer = None

    @not_in_flight
    def _broadcast_roots_to_leaves(self):
        self._broadcast_roots_to_leaves_begin()
        self._broadcast_roots_to_leaves_end()

    @not_in_flight
    def _broadcast_roots_to_leaves_begin(self):
        if not self._roots_valid:
            raise RuntimeError("Cannot broadcast invalid roots")

        if not self._leaves_valid:
            self.sf.broadcast_begin(self._data, MPI.REPLACE)
        self._finalizer = self._broadcast_roots_to_leaves_end

    def _broadcast_roots_to_leaves_end(self):
        if self._finalizer is None:
            raise BadOrderingException(
                "Should not call _broadcast_roots_to_leaves_end without first "
                "calling _broadcast_roots_to_leaves_begin"
            )
        if self._finalizer != self._broadcast_roots_to_leaves_end:
            raise DataTransferInFlightException("Wrong finalizer called")

        if not self._leaves_valid:
            self.sf.broadcast_end(self._data, MPI.REPLACE)
        self._leaves_valid = True
        self._finalizer = None

    @not_in_flight
    def _reduce_then_broadcast(self):
        self._reduce_leaves_to_roots()
        self._broadcast_roots_to_leaves()


class PackedBuffer(Buffer):
    """Abstract buffer originating from a function call.

    For example, the buffer returned from ``MatGetValues`` is such a "packed"
    buffer.

    """

    def __init__(self, array):
        self.array = array

    # needed?
    @property
    def dtype(self):
        return self.array.dtype

    @property
    def is_distributed(self) -> bool:
        return False

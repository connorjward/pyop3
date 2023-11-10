# TODO Rename this file to distarray.py and distarray/ to tensor/

from __future__ import annotations

import numbers
from functools import cached_property

import numpy as np
from mpi4py import MPI

from pyop3.dtypes import ScalarType
from pyop3.utils import UniqueNameGenerator, deprecated, readonly


class IncompatibleStarForestException(Exception):
    pass


class DistributedArray:
    """An array distributed across multiple processors with ghost values."""

    # NOTE: When GPU support is added, the host-device awareness and
    # copies should live in this class.

    DEFAULT_DTYPE = ScalarType

    _prefix = "array"
    _name_generator = UniqueNameGenerator()

    def __init__(self, data_or_shape, dtype=None, *, name=None, prefix=None, sf=None):
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")

        # option 1: passed shape
        if isinstance(data_or_shape, (numbers.Integral, tuple)):
            data = None
            shape = (
                data_or_shape if isinstance(data_or_shape, tuple) else (data_or_shape,)
            )
            if not dtype:
                dtype = self.DEFAULT_DTYPE
        # option 2: passed a numpy array
        elif isinstance(data_or_shape, np.ndarray):
            data = data_or_shape
            shape = data.shape
            if len(shape) > 1:
                raise NotImplementedError
            if not dtype:
                dtype = data.dtype
            data = np.asarray(data, dtype)
        else:
            raise TypeError(f"Unexpected type passed to data_or_shape")

        if sf and shape[0] != sf.size:
            raise IncompatibleStarForestException

        self.shape = shape
        self.dtype = dtype
        self._lazy_data = data
        self.sf = sf

        self.name = name or self._name_generator(prefix or self._prefix)

        # counter used to keep track of modifications
        self.state = 0

        # flags for tracking parallel correctness
        self._pending_reduction = None
        self._leaves_valid = True

        # TODO
        # self._sync_thread = None

    @property
    @deprecated(".data_rw")
    def data(self):
        return self.data_rw

    @property
    def data_rw(self):
        self.state += 1

        if not self._roots_valid:
            self._reduce_leaves_to_roots()

        # modifying owned values invalidates ghosts
        self._leaves_valid = False
        return self._owned_data

    @property
    def data_ro(self):
        if not self._roots_valid:
            self._reduce_leaves_to_roots()
        return readonly(self._owned_data)

    @property
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
        return self.sf is not None

    @property
    def _data(self):
        if self._lazy_data is None:
            self._lazy_data = np.zeros(self.shape, dtype=self.dtype)
        return self._lazy_data

    @property
    def _owned_data(self):
        if self.is_distributed:
            return self._data[: -self.sf.nleaves]
        else:
            return self._data

    @property
    def _roots_valid(self) -> bool:
        return self._pending_reduction is None

    @cached_property
    def _reduction_ops(self):
        # TODO Move this import out, requires moving location of these intents
        from pyop3.lang import INC

        return {
            INC: MPI.SUM,
        }

    # TODO should have begin and end methods
    def _reduce_leaves_to_roots(self):
        if self._roots_valid:
            return

        pending_write_op = self._pending_reduction

        mpi_op = self._reduction_ops[pending_write_op]
        self.sf.reduce(self._data, mpi_op)

        self._leaves_valid = False
        self._pending_reduction = None

    def _broadcast_roots_to_leaves(self):
        if not self._roots_valid:
            # or do the reduction?
            raise RuntimeError
        self.sf.broadcast(self._data, MPI.REPLACE)
        self._leaves_valid = True

    def _reduce_then_broadcast(self):
        self._reduce_leaves_to_roots()
        self._broadcast_roots_to_leaves()

    # def sync_begin(self, need_halo_values=False):
    #     """Begin synchronizing shared data."""
    #     self._sync_thread = threading.Thread(
    #         target=self.__class__.sync,
    #         args=(self,),
    #         kwargs={"need_halo_values": need_halo_values},
    #     )
    #     self._sync_thread.start()
    #
    # def sync_end(self):
    #     """Finish synchronizing shared data."""
    #     if not self._sync_thread:
    #         raise RuntimeError(
    #             "Cannot call sync_end without a prior call to sync_begin"
    #         )
    #     self._sync_thread.join()
    #     self._sync_thread = None
    #
    # # TODO create Synchronizer object for encapsulation?
    # def sync(self, need_halo_values=False):
    #     """Perform halo exchanges to ensure that all ranks store up-to-date values.
    #
    #     Parameters
    #     ----------
    #     need_halo_values : bool
    #         Whether or not halo values also need to be synchronized.
    #
    #     Notes
    #     -----
    #     This is a blocking operation. F.labelor the non-blocking alternative use
    #     :meth:`sync_begin` and :meth:`sync_end` (FIXME)
    #
    #     Note that this method should only be called when one needs to read from
    #     the array.
    #     """
    #     # 1. Reduce leaf values to roots if they have been written to.
    #     # (this is basically local-to-global)
    #     if self._pending_write_op:
    #         assert (
    #             not self._halo_valid
    #         ), "If a write is pending the halo cannot be valid"
    #         # If halo entries have also been written to then we need to use the
    #         # full SF containing both shared and halo points. If the halo has not
    #         # been modified then we only need to reduce with shared points.
    #         if self._halo_modified:
    #             self.reduce_leaves_to_roots(self.root.sf, self._pending_write_op)
    #         else:
    #             # only reduce with values from owned points
    #             self.reduce_leaves_to_roots(self.root.shared_sf, self._pending_write_op)
    #
    #     # implicit barrier? can only broadcast reduced values
    #
    #     # 3. at this point only one of the owned points knows the correct result which
    #     # now needs to be scattered back to some (but not necessarily all) of the other ranks.
    #     # (this is basically global-to-local)
    #
    #     # only send to halos if we want to read them and they are out-of-date
    #     if need_halo_values and not self._halo_valid:
    #         # send the root value back to all the points
    #         self.broadcast_roots_to_leaves(self.root.sf)
    #         self._halo_valid = True  # all the halo points are now up-to-date
    #     else:
    #         # only need to update owned points if we did a reduction earlier
    #         if self._pending_write_op:
    #             # send the root value back to just the owned points
    #             self.broadcast_roots_to_leaves(self.root.shared_sf)
    #             # (the halo is still dirty)
    #
    #     # set self.last_op to None here? what if halo is still off?
    #     # what if we read owned values and then owned+halo values?
    #     # just perform second step
    #     self._pending_write_op = None
    #     self._halo_modified = False

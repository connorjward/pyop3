import abc
import itertools
import numbers

import numpy as np
from petsc4py import PETSc

from pyop3.utils import strictly_all, print_with_rank, single_valued


class DistributedArray(abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    # @abc.abstractmethod
    # def sync(self):
    #     pass


# TODO make MultiArray inherit from this too


class PetscVec(DistributedArray, abc.ABC):
    def __new__(cls, *args, **kwargs):
        # dispatch to different vec types based on -vec_type
        raise NotImplementedError


class PetscVecStandard(PetscVec):
    ...


class PetscVecNest(PetscVec):
    ...


class PetscMat(DistributedArray, abc.ABC):
    # def __new__(cls, *args, **kwargs):
        # dispatch to different mat types based on -mat_type
        # raise NotImplementedError

    # TODO: What is the best way to instantiate these things?
    @classmethod
    def new(cls, axes, maps):
        raxis, caxis = axes
        rmap, cmap = maps
        # only certain types of axes can form particular matrix types (e.g. MATNEST
        # requires a mixed space "outside")
        # need to "flatten" axes and maps s.t. there is a constant inner shape

        # or similar
        can_be_mat_nest = all(part.size == 1 for part in itertools.chain(raxis.parts, caxis.parts))


        """
        * The axes reference only the nodes, not the cells
        * The maps map from cells (e.g.) to the nodes

        This makes matrices specific to the loop in which they are used as they
        depend on the indices. This is not dissimilar to how vector overlaps depend
        on the sizes of the stencils.

        This is not a pleasing abstraction. Can we do things lazily?

        I suppose that matrix allocation is dependent upon the discretisation (i.e.
        the overlap). Therefore it should be allocat-able prior to a loop. The parloop
        should check that the operations are being provided with valid data structures,
        not allocating the data structures to conform to the operation.
        """
        


class PetscMatDense(PetscMat):
    ...


class PetscMatAIJ(PetscMat):
    def __init__(self, raxes, caxes, sparsity, *, comm=None):
        # TODO: Remove this import
        from pyop3.tensors import overlap_axes

        if comm is None:
            comm = PETSc.Sys.getDefaultComm()

        if any(ax.nparts > 1 for ax in [raxes, caxes]):
            raise ValueError(
                "Cannot construct a PetscMat with multi-part axes")

        if not all(ax.part.has_partitioned_halo for ax in [raxes, caxes]):
            raise ValueError(
                "Multi-axes must store halo points in a contiguous block "
                "after owned points")

        # axes = overlap_axes(raxes.local_view, caxes.local_view, sparsity)
        axes = overlap_axes(raxes, caxes, sparsity)
        # axes = overlap_axes(raxes.local_view, caxes, sparsity)

        # rsize and csize correspond to the local dimensions
        # rsize = raxes.local_view.part.count
        # csize = caxes.local_view.part.count
        rsize = raxes.part.nowned
        csize = caxes.part.nowned
        sizes = ((rsize, None), (csize, None))

        row_part = axes.part
        col_part = row_part.subaxis.part

        col_indices = col_part.indices.data
        row_ptrs = np.concatenate(
            [col_part.layout_fn.start.data, [len(col_indices)]],
            dtype=col_indices.dtype)

        # row_ptrs = 

        # build the local to global maps from the provided star forests
        if strictly_all(ax.part.is_distributed for ax in [raxes, caxes]):
            rlgmap = _create_lgmap(raxes)
            clgmap = _create_lgmap(caxes)
        else:
            rlgmap = clgmap = None

        print_with_rank(clgmap.indices)

        # convert column indices into global numbering
        if strictly_all([rlgmap, clgmap]):
            col_indices = clgmap.indices[col_indices]

        # csr is a 2-tuple of row pointers and column indices
        csr = (row_ptrs, col_indices)
        print_with_rank(csr)
        petscmat = PETSc.Mat().createAIJ(sizes, csr=csr, comm=comm)

        if strictly_all([rlgmap, clgmap]):
            petscmat.setLGMap(rlgmap, clgmap)

        petscmat.setUp()

        self.axes = axes
        self.petscmat = petscmat


class PetscMatBAIJ(PetscMat):
    ...


class PetscMatNest(PetscMat):
    ...


class PetscMatPython(PetscMat):
    ...


def _create_lgmap(axes):
    # TODO: Fix imports
    from pyop3.tensors import Owned, Shared, MPI, get_mpi_dtype

    if axes.nparts > 1:
        raise NotImplementedError
    if axes.part.overlap is None:
        raise ValueError("axes is expected to have a specified overlap")
    if not isinstance(axes.part.count, numbers.Integral):
        raise NotImplementedError("Expecting an integral axis size")

    # 1. Globally number all owned processes
    sendbuf = np.array([axes.part.nowned], dtype=PETSc.IntType)
    recvbuf = np.zeros_like(sendbuf)
    axes.sf.comm.tompi4py().Exscan(sendbuf, recvbuf)
    global_num = single_valued(recvbuf)
    indices = np.full(axes.part.count, -1, dtype=PETSc.IntType)
    for i, olabel in enumerate(axes.part.overlap):
        if (isinstance(olabel, Owned)
            or isinstance(olabel, Shared) and olabel.root is None):
            indices[i] = global_num
            global_num += 1

    # 2. Broadcast the global numbering to SF leaves
    mpi_dtype, _ = get_mpi_dtype(indices.dtype)
    mpi_op = MPI.REPLACE
    args = (mpi_dtype, indices, indices, mpi_op)
    axes.sf.bcastBegin(*args)
    axes.sf.bcastEnd(*args)

    assert not any(indices == -1)

    return PETSc.LGMap().create(indices, comm=axes.sf.comm)

    # nroots, local, _ = sf.getGraph();
    # nleaves = len(local)
    # npoints = nroots + nleaves
    #
    # sendbuf = np.array([nroots])
    # recvbuf = np.zeros(1, dtype=sendbuf.dtype)
    #
    # # sf.comm.Exscan(&nroots, &start, 1, MPIU_INT, MPI_SUM, comm));
    # sf.comm.Exscan(sendbuf, recvbuf);
    # start = single_valued(recvbuf)
    #
    # indices = np.empty(npoints, dtype=PETSc.IntType)
    #
    # PetscCall(PetscSFGetLeafRange(sf, NULL, &maxlocal));
    # ++maxlocal;
    # PetscCall(PetscMalloc1(nroots, &globals));
    # PetscCall(PetscMalloc1(maxlocal, &ltog));
    # for (i = 0; i < nroots; i++) globals[i] = start + i;
    # for (i = 0; i < maxlocal; i++) ltog[i] = -1;
    # PetscCall(PetscSFBcastBegin(sf, MPIU_INT, globals, ltog, MPI_REPLACE));
    # PetscCall(PetscSFBcastEnd(sf, MPIU_INT, globals, ltog, MPI_REPLACE));
    # PetscCall(ISLocalToGlobalMappingCreate(comm, 1, maxlocal, ltog, PETSC_OWN_POINTER, mapping));
    # PetscCall(PetscFree(globals));
    # PetscFunctionReturn(PETSC_SUCCESS);

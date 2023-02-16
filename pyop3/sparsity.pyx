# cython: language_level=3

# TODO: What to do about this copyright notice?
# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc
from petsc4py import PETSc
from pyop2.datatypes import IntType

np.import_array()

cdef extern from "petsc.h":
    ctypedef long PetscInt
    ctypedef double PetscScalar
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE
    ctypedef enum PetscInsertMode "InsertMode":
        PETSC_INSERT_VALUES "INSERT_VALUES"
    int PetscCalloc1(size_t, void*)
    int PetscMalloc1(size_t, void*)
    int PetscMalloc2(size_t, void*, size_t, void*)
    int PetscFree(void*)
    int PetscFree2(void*,void*)
    int MatSetValuesBlockedLocal(PETSc.PetscMat, PetscInt, PetscInt*, PetscInt, PetscInt*,
                                 PetscScalar*, PetscInsertMode)
    int MatSetValuesLocal(PETSc.PetscMat, PetscInt, PetscInt*, PetscInt, PetscInt*,
                          PetscScalar*, PetscInsertMode)
    int MatPreallocatorPreallocate(PETSc.PetscMat, PetscBool, PETSc.PetscMat)
    int MatXAIJSetPreallocation(PETSc.PetscMat, PetscInt, const PetscInt[], const PetscInt[],
                                const PetscInt[], const PetscInt[])

cdef extern from "petsc/private/matimpl.h":
    struct _p_Mat:
        void *data

ctypedef struct Mat_Preallocator:
    void *ht
    PetscInt *dnz
    PetscInt *onz

cdef extern from *:
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError

cdef object PetscError = <object>PyExc_RuntimeError

cdef inline int SETERR(int ierr) with gil:
    if (<void*>PetscError) != NULL:
        PyErr_SetObject(PetscError, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return ierr

cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == 0:
        return 0 # no error
    else:
        SETERR(ierr)
        return -1

cdef object set_writeable(map):
     flag = map.values_with_halo.flags['WRITEABLE']
     map.values_with_halo.setflags(write=True)
     return flag

cdef void restore_writeable(map, flag):
     map.values_with_halo.setflags(write=flag)


def get_preallocation(PETSc.Mat preallocator, PetscInt nrow):
    cdef:
        _p_Mat *A = <_p_Mat *>(preallocator.mat)
        Mat_Preallocator *p = <Mat_Preallocator *>(A.data)

    if p.dnz != NULL:
        dnz = <PetscInt[:nrow]>p.dnz
        dnz = np.asarray(dnz).copy()
    else:
        dnz = np.zeros(0, dtype=IntType)
    if p.onz != NULL:
        onz = <PetscInt[:nrow]>p.onz
        onz = np.asarray(onz).copy()
    else:
        onz = np.zeros(0, dtype=IntType)
    return dnz, onz


# TODO: Wrap this in a cache outside
def compute_nonzero_pattern(axes, maps, comm):
    """TODO"""
    raxis, caxis = axes
    rmap, cmap = maps
    nrows = raxis.count
    ncols = caxis.count

    preallocator = PETSc.Mat().create(comm=comm)
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
    # preallocator.setLGMap(rmap=rset.scalar_lgmap, cmap=cset.scalar_lgmap)
    # maybe bsize is bigger? - no since we expand to account for the block size at the end
    preallocator.setSizes(size=(nrows, ncols), bsize=1)
    preallocator.setUp()

    fill_with_zeros(preallocator, (1, 1), sparsity.maps,
                    iteration_regions, set_diag=sparsity._has_diagonal)
    preallocator.assemble()

    nnz, onnz = get_preallocation(preallocator, nrows)
    # if not (sparsity._block_sparse and rset.cdim == cset.cdim):
    #     # We only build baij for the the square blocks, so unwind if we didn't
    #     nnz = nnz * cset.cdim
    #     nnz = np.repeat(nnz, rset.cdim)
    #     onnz = onnz * cset.cdim
    #     onnz = np.repeat(onnz, rset.cdim)


    preallocator.destroy()
    return diag_nnz, offdiag_nnz


    # we are only expecting depth of 1 or 2 - nesting happens outside and we flatten
    # in advance.

    # rset, cset = sparsity.dsets
    # should not happen, flatten outside
    # mixed = len(rset) > 1 or len(cset) > 1
    # nest = sparsity.nested
    # if mixed and sparsity.nested:
    #     raise ValueError("Can't build sparsity on mixed nest, build the sparsity on the blocks")
    # if mixed:
    #     # Sparsity is the dof sparsity.
    #     nrows = sum(s.size*s.cdim for s in rset)
    #     ncols = sum(s.size*s.cdim for s in cset)
    #     preallocator.setLGMap(rmap=rset.unblocked_lgmap, cmap=cset.unblocked_lgmap)
    # else:
        # Sparsity is the block sparsity
        # nrows = rset.size
        # ncols = cset.size
        # preallocator.setLGMap(rmap=rset.scalar_lgmap, cmap=cset.scalar_lgmap)

    # iteration_regions = sparsity.iteration_regions
    # if mixed:
    #     for i, r in enumerate(rset):
    #         for j, c in enumerate(cset):
    #             maps = list(zip((m.split[i] for m in sparsity.rmaps),
    #                             (m.split[j] for m in sparsity.cmaps)))
    #             mat = preallocator.getLocalSubMatrix(isrow=rset.local_ises[i],
    #                                                  iscol=cset.local_ises[j])
    #             fill_with_zeros(mat, (r.cdim, c.cdim),
    #                             maps,
    #                             iteration_regions,
    #                             set_diag=((i == j) and sparsity._has_diagonal))
    #             mat.assemble()
    #             preallocator.restoreLocalSubMatrix(isrow=rset.local_ises[i],
    #                                                iscol=cset.local_ises[j],
    #                                                submat=mat)
    #     preallocator.assemble()
    #     nnz, onnz = get_preallocation(preallocator, nrows)
    # else:
        # fill_with_zeros(preallocator, (1, 1), sparsity.maps,
        #                 iteration_regions, set_diag=sparsity._has_diagonal)
        # preallocator.assemble()
        # nnz, onnz = get_preallocation(preallocator, nrows)
        # if not (sparsity._block_sparse and rset.cdim == cset.cdim):
        #     # We only build baij for the the square blocks, so unwind if we didn't
        #     nnz = nnz * cset.cdim
        #     nnz = np.repeat(nnz, rset.cdim)
        #     onnz = onnz * cset.cdim
        #     onnz = np.repeat(onnz, rset.cdim)


def fill_with_zeros(mat, maps, bsizes):
    """Fill a PETSc matrix with zeros in all slots we might end up inserting into

    :arg mat: the PETSc Mat (must already be preallocated)
    :arg dims: the dimensions of the sparsity (block size)
    :arg maps: the pairs of maps defining the sparsity pattern

    You must call ``mat.assemble()`` after this call.
    """
    rmap, cmap = maps
    rbsize, cbsize = bsizes

    # Iterate over row map values including value entries
    assert cmap.from_set == rmap.from_set
    
    rmap = pair[0].values_with_halo
    cmap = pair[1].values_with_halo
    rarity = pair[0].arity
    carity = pair[1].arity

    values = np.zeros(rarity*carity*rdim*cdim, dtype=PETSc.ScalarType)
    for i in range(rmap.from_set.count):
        mat.setValuesBlockedLocal(
            mat.mat, rmap.data[i], cmap.data[i], values
        )

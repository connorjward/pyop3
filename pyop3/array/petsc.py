from __future__ import annotations

import abc
import enum
import itertools
import numbers
from functools import cached_property

import numpy as np
import pymbolic as pym
from petsc4py import PETSc
from pyrsistent import freeze

from pyop3.array.base import Array
from pyop3.array.harray import ContextSensitiveMultiArray, HierarchicalArray
from pyop3.axtree import AxisTree
from pyop3.axtree.tree import ContextFree, ContextSensitive, as_axis_tree
from pyop3.buffer import PackedBuffer
from pyop3.dtypes import ScalarType
from pyop3.itree import IndexTree
from pyop3.itree.tree import (
    _compose_bits,
    _index_axes,
    as_index_forest,
    as_index_tree,
    index_axes,
)
from pyop3.utils import just_one, merge_dicts, single_valued, strictly_all


# don't like that I need this
class PetscVariable(pym.primitives.Variable):
    def __init__(self, obj: PetscObject):
        super().__init__(obj.name)
        self.obj = obj


class PetscObject(Array, abc.ABC):
    dtype = ScalarType


class PetscVec(PetscObject):
    def __new__(cls, *args, **kwargs):
        # dispatch to different vec types based on -vec_type
        raise NotImplementedError


class PetscVecStandard(PetscVec):
    ...


class PetscVecNest(PetscVec):
    ...


class MatType(enum.Enum):
    AIJ = "aij"
    BAIJ = "baij"


# TODO Better way to specify a default? config?
DEFAULT_MAT_TYPE = MatType.AIJ


class PetscMat(PetscObject, abc.ABC):
    prefix = "mat"

    def __new__(cls, *args, **kwargs):
        mat_type = kwargs.pop("mat_type", DEFAULT_MAT_TYPE)
        if mat_type == MatType.AIJ:
            return object.__new__(PetscMatAIJ)
        elif mat_type == MatType.BAIJ:
            return object.__new__(PetscMatBAIJ)
        else:
            raise AssertionError

    # like Dat, bad name? handle?
    @property
    def array(self):
        return self.petscmat


class MonolithicPetscMat(PetscMat, abc.ABC):
    def __getitem__(self, indices):
        if len(indices) != 2:
            raise ValueError

        # TODO also support context-free (see MultiArray.__getitem__)
        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=self.axes):
            # make a temporary of the right shape
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                target_paths,
                index_exprs,
                layout_exprs_per_indexed_cpt,
                domain_index_exprs,
            ) = _index_axes(self.axes, index_tree, loop_context)

            indexed_axes = indexed_axes.set_up()

            packed = PackedBuffer(self)

            array_per_context[loop_context] = HierarchicalArray(
                indexed_axes,
                data=packed,
                target_paths=target_paths,
                index_exprs=index_exprs,
                domain_index_exprs=domain_index_exprs,
                name=self.name,
            )

        return ContextSensitiveMultiArray(array_per_context)

    @cached_property
    def datamap(self):
        return freeze({self.name: self})


# is this required?
class ContextSensitiveIndexedPetscMat(ContextSensitive):
    pass


class PetscMatAIJ(MonolithicPetscMat):
    def __init__(self, raxes, caxes, sparsity, *, name: str = None):
        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)

        super().__init__(name)
        if any(axes.depth > 1 for axes in [raxes, caxes]):
            # TODO, good exceptions
            # raise InvalidDimensionException("Cannot instantiate PetscMats with nested axis trees")
            raise RuntimeError
        if any(len(axes.root.components) > 1 for axes in [raxes, caxes]):
            # TODO, good exceptions
            raise RuntimeError

        self.petscmat = _alloc_mat(raxes, caxes, sparsity)

        self.raxis = raxes.root
        self.caxis = caxes.root
        self.sparsity = sparsity

        self.axes = AxisTree.from_nest({self.raxis: self.caxis})


class PetscMatBAIJ(MonolithicPetscMat):
    def __init__(self, raxes, caxes, sparsity, bsize, *, name: str = None):
        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)

        if isinstance(bsize, numbers.Integral):
            bsize = (bsize, bsize)

        super().__init__(name)
        if any(axes.depth > 1 for axes in [raxes, caxes]):
            # TODO, good exceptions
            # raise InvalidDimensionException("Cannot instantiate PetscMats with nested axis trees")
            raise RuntimeError
        if any(len(axes.root.components) > 1 for axes in [raxes, caxes]):
            # TODO, good exceptions
            raise RuntimeError

        self.petscmat = _alloc_mat(raxes, caxes, sparsity, bsize)

        self.raxis = raxes.root
        self.caxis = caxes.root
        self.sparsity = sparsity
        self.bsize = bsize

        # TODO include bsize here?
        self.axes = AxisTree.from_nest({self.raxis: self.caxis})


class PetscMatNest(PetscMat):
    ...


class PetscMatDense(PetscMat):
    ...


class PetscMatPython(PetscMat):
    ...


# TODO cache this function and return a copy if possible
def _alloc_mat(raxes, caxes, sparsity, bsize=None):
    comm = single_valued([raxes.comm, caxes.comm])

    sizes = (raxes.leaf_component.count, caxes.leaf_component.count)
    nnz = sparsity.axes.leaf_component.count

    if bsize is None:
        mat = PETSc.Mat().createAIJ(sizes, nnz=nnz.data, comm=comm)
    else:
        mat = PETSc.Mat().createBAIJ(sizes, bsize, nnz=nnz.data, comm=comm)

    # fill with zeros (this should be cached)
    # this could be done as a pyop3 loop (if we get ragged local working) or
    # explicitly in cython
    raxis, rcpt = raxes.leaf
    caxis, ccpt = caxes.leaf
    # e.g.
    # map_ = Map({pmap({raxis.label: rcpt.label}): [TabulatedMapComponent(caxes.label, ccpt.label, sparsity)]})
    # do_loop(p := raxes.index(), write(zeros, mat[p, map_(p)]))

    # but for now do in Python...
    assert nnz.max_value is not None
    if bsize is None:
        shape = (nnz.max_value,)
        set_values = mat.setValuesLocal
    else:
        rbsize, _ = bsize
        shape = (nnz.max_value, rbsize)
        set_values = mat.setValuesBlockedLocal
    zeros = np.zeros(shape, dtype=PetscMat.dtype)
    for row_idx in range(rcpt.count):
        cstart = sparsity.axes.offset([row_idx, 0])
        try:
            cstop = sparsity.axes.offset([row_idx + 1, 0])
        except IndexError:
            # catch the last one
            cstop = len(sparsity.data_ro)
        set_values([row_idx], sparsity.data_ro[cstart:cstop], zeros[: cstop - cstart])
    mat.assemble()
    return mat

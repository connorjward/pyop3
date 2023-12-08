from __future__ import annotations

import abc
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
    collect_loop_contexts,
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

    def as_var(self):
        return PetscVariable(self)


class PetscVec(PetscObject):
    def __new__(cls, *args, **kwargs):
        # dispatch to different vec types based on -vec_type
        raise NotImplementedError

    @property
    def valid_ranks(self):
        return frozenset({0, 1})


class PetscVecStandard(PetscVec):
    ...


class PetscVecNest(PetscVec):
    ...


class PetscMat(PetscObject):
    prefix = "mat"

    def __new__(cls, *args, **kwargs):
        # TODO dispatch to different mat types based on -mat_type
        return object.__new__(PetscMatAIJ)

    @property
    def valid_ranks(self):
        return frozenset({2})

    @cached_property
    def datamap(self):
        return freeze({self.name: self})


# is this required?
class ContextSensitiveIndexedPetscMat(ContextSensitive):
    pass


# Not a super important class, could just inspect type of .array instead?
class PackedPetscMatAIJ(PackedBuffer):
    pass


class PetscMatAIJ(PetscMat):
    def __init__(self, raxes, caxes, sparsity, *, comm=None, name: str = None):
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

        sizes = (raxes.leaf_component.count, caxes.leaf_component.count)
        nnz = sparsity.axes.leaf_component.count
        mat = PETSc.Mat().createAIJ(sizes, nnz=nnz.data, comm=comm)

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
        zeros = np.zeros(nnz.max_value, dtype=self.dtype)
        for row_idx in range(rcpt.count):
            cstart = sparsity.axes.offset([row_idx, 0])
            try:
                cstop = sparsity.axes.offset([row_idx + 1, 0])
            except IndexError:
                # catch the last one
                cstop = len(sparsity.data_ro)
            # truncate zeros
            mat.setValuesLocal(
                [row_idx], sparsity.data_ro[cstart:cstop], zeros[: cstop - cstart]
            )
        mat.assemble()

        self.raxis = raxes.root
        self.caxis = caxes.root
        self.sparsity = sparsity

        self.axes = AxisTree.from_nest({self.raxis: self.caxis})

        # copy only needed if we reuse the zero matrix
        self.petscmat = mat.copy()

    def __getitem__(self, indices):
        # TODO also support context-free (see MultiArray.__getitem__)
        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=self.axes):
            # make a temporary of the right shape
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                # target_path_per_indexed_cpt,
                # index_exprs_per_indexed_cpt,
                target_paths,
                index_exprs,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.axes, index_tree, loop_context)

            # is this needed? Just use the defaults?
            # (
            #     target_paths,
            #     index_exprs,
            #     layout_exprs,
            # ) = _compose_bits(
            #     self.axes,
            #     # use the defaults because Mats can only be indexed once
            #     # (then they turn into Dats)
            #     self.axes._default_target_paths(),
            #     self.axes._default_index_exprs(),
            #     None,
            #     indexed_axes,
            #     target_path_per_indexed_cpt,
            #     index_exprs_per_indexed_cpt,
            #     layout_exprs_per_indexed_cpt,
            # )

            # "freeze" the indexed_axes, we want to tabulate the layout of them
            # (when usually we don't)
            indexed_axes = indexed_axes.set_up()

            packed = PackedPetscMatAIJ(self)

            array_per_context[loop_context] = HierarchicalArray(
                indexed_axes,
                data=packed,
                target_paths=target_paths,
                index_exprs=index_exprs,
                name=self.name,
            )

        return ContextSensitiveMultiArray(array_per_context)

    # like Dat, bad name? handle?
    @property
    def array(self):
        return self.petscmat


class PetscMatBAIJ(PetscMat):
    ...


class PetscMatNest(PetscMat):
    ...


class PetscMatDense(PetscMat):
    ...


class PetscMatPython(PetscMat):
    ...

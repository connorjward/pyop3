from __future__ import annotations

import abc
import functools
import itertools
import numbers

import numpy as np
import pymbolic as pym
from petsc4py import PETSc
from pyrsistent import freeze

from pyop3.axtree import AxisTree
from pyop3.axtree.tree import ContextFree, ContextSensitive, as_axis_tree
from pyop3.distarray.base import Tensor
from pyop3.distarray.multiarray import ContextSensitiveMultiArray, Dat, MultiArray
from pyop3.dtypes import ScalarType
from pyop3.itree import IndexTree
from pyop3.utils import just_one, merge_dicts, single_valued, strictly_all


# don't like that I need this
class PetscVariable(pym.primitives.Variable):
    def __init__(self, obj: PetscObject):
        super().__init__(obj.name)
        self.obj = obj


class PetscObject(abc.ABC):
    dtype = ScalarType

    def as_var(self):
        return PetscVariable(self)


class PetscVec(Tensor, PetscObject):
    def __new__(cls, *args, **kwargs):
        # dispatch to different vec types based on -vec_type
        raise NotImplementedError


class PetscVecStandard(PetscVec):
    ...


class PetscVecNest(PetscVec):
    ...


class PetscMat(Tensor, PetscObject):
    prefix = "mat"

    def __new__(cls, *args, **kwargs):
        # TODO dispatch to different mat types based on -mat_type
        return object.__new__(PetscMatAIJ)

    @functools.cached_property
    def datamap(self):
        return freeze({self.name: self})


# this is needed because
class ContextSensitiveIndexedPetscMat(ContextSensitive):
    pass


class IndexedPetscMat(ContextFree):
    def __init__(self, getvalues, axes, orig_mat):
        self.getvalues = getvalues
        self.axes = axes
        self.orig_mat = orig_mat

    @property
    def array(self):
        return self.orig_mat.petscmat

    @property
    def name(self):
        return self.getvalues.parameters[0].name

    @property
    def dtype(self):
        return PetscObject.dtype

    @functools.cached_property
    def datamap(self):
        # this is ugly
        return self.orig_mat.datamap | merge_dicts(
            self.getvalues.parameters[i].function.map_component.datamap for i in (1, 2)
        )

    def materialize(self):
        axes = AxisTree(self.axes.parent_to_children)
        return Dat(axes, dtype=self.dtype)


class PetscMatDense(PetscMat):
    ...


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
            cstart = sparsity.axes.get_offset([row_idx, 0])
            try:
                cstop = sparsity.axes.get_offset([row_idx + 1, 0])
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

        # old code below
        # if any(ax.nparts > 1 for ax in [raxes, caxes]):
        #     raise ValueError("Cannot construct a PetscMat with multi-part axes")
        #
        # if not all(ax.part.has_partitioned_halo for ax in [raxes, caxes]):
        #     raise ValueError(
        #         "Multi-axes must store halo points in a contiguous block "
        #         "after owned points"
        #     )
        #
        # # axes = overlap_axes(raxes.local_view, caxes.local_view, sparsity)
        # axes = overlap_axes(raxes, caxes, sparsity)
        # # axes = overlap_axes(raxes.local_view, caxes, sparsity)
        #
        # # rsize and csize correspond to the local dimensions
        # # rsize = raxes.local_view.part.count
        # # csize = caxes.local_view.part.count
        # rsize = raxes.part.nowned
        # csize = caxes.part.nowned
        # sizes = ((rsize, None), (csize, None))
        #
        # row_part = axes.part
        # col_part = row_part.subaxis.part
        #
        # # drop the last few because these are below the rows PETSc cares about
        # row_ptrs = col_part.layout_fn.start.data
        # col_indices = col_part.indices.data[: row_ptrs[rsize]]
        # # row_ptrs = np.concatenate(
        # #     [col_part.layout_fn.start.data, [len(col_indices)]],
        # #     dtype=col_indices.dtype)
        #
        # # row_ptrs =
        #
        # # build the local to global maps from the provided star forests
        # if strictly_all(ax.part.is_distributed for ax in [raxes, caxes]):
        #     # rlgmap = _create_lgmap(raxes)
        #     # clgmap = _create_lgmap(caxes)
        #     rlgmap = raxes.part.lgmap
        #     clgmap = caxes.part.lgmap
        # else:
        #     rlgmap = clgmap = None
        #
        # # convert column indices into global numbering
        # if strictly_all(lgmap is not None for lgmap in [rlgmap, clgmap]):
        #     col_indices = clgmap[col_indices]
        #
        # # csr is a 2-tuple of row pointers and column indices
        # csr = (row_ptrs, col_indices)
        # petscmat = PETSc.Mat().createAIJ(sizes, csr=csr, comm=comm)
        #
        # if strictly_all(lgmap is not None for lgmap in [rlgmap, clgmap]):
        #     rlgmap = PETSc.LGMap().create(rlgmap, comm=comm)
        #     clgmap = PETSc.LGMap().create(clgmap, comm=comm)
        #     petscmat.setLGMap(rlgmap, clgmap)
        #
        # petscmat.setUp()
        #
        # self.axes = axes
        # self.petscmat = petscmat

    def __getitem__(self, indices):
        from pyop3.itree.tree import (
            _compose_bits,
            _index_axes,
            as_index_forest,
            as_index_tree,
            collect_loop_contexts,
            index_axes,
        )

        # TODO also support context-free (see MultiArray.__getitem__)
        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=self.axes):
            # make a temporary of the right shape
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.axes, index_tree, loop_context)

            (
                target_paths,
                index_exprs,
                layout_exprs,
            ) = _compose_bits(
                self.axes,
                # use the defaults because Mats can only be indexed once
                # (then they turn into Dats)
                self.axes._default_target_paths(),
                self.axes._default_index_exprs(),
                None,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            # TODO is IndexedPetscMat required? How is it different from
            # a Dat? The layout functions are somehow not the same.

            new_axes = AxisTree(
                indexed_axes.parent_to_children,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            # not a layout!
            rindex, cindex = indices
            (iraxis, ircpt), (icaxis, iccpt) = new_axes.path_with_nodes(
                *new_axes.leaf, ordered=True
            )
            rkey = (iraxis.id, ircpt)
            ckey = (icaxis.id, iccpt)

            rlayout_expr = index_exprs_per_indexed_cpt[rkey][
                just_one(target_path_per_indexed_cpt[rkey])
            ]
            clayout_expr = index_exprs_per_indexed_cpt[ckey][
                just_one(target_path_per_indexed_cpt[ckey])
            ]

            getvalues = pym.var("MatGetValues")(
                self.as_var(), rlayout_expr, clayout_expr
            )

            # not sure that this is quite correct
            layout_axes = AxisTree(
                new_axes.parent_to_children,
                # target_paths=target_paths,
                # index_exprs=None,
            )
            array_per_context[loop_context] = IndexedPetscMat(
                getvalues, layout_axes, self
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


class PetscMatPython(PetscMat):
    ...

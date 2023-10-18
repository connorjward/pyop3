import abc
import itertools
import numbers

import numpy as np
import pymbolic as pym
from petsc4py import PETSc
from pyrsistent import freeze

from pyop3.axes import AxisTree
from pyop3.axes.tree import FrozenAxisTree
from pyop3.distarray.base import DistributedArray
from pyop3.distarray.multiarray import ContextSensitiveMultiArray, MultiArray
from pyop3.dtypes import ScalarType
from pyop3.indices import IndexTree
from pyop3.indices.tree import IndexedAxisTree
from pyop3.utils import just_one, single_valued, strictly_all


class PetscObject(DistributedArray, abc.ABC):
    dtype = ScalarType


class PetscVec(PetscObject):
    def __new__(cls, *args, **kwargs):
        # dispatch to different vec types based on -vec_type
        raise NotImplementedError


class PetscVecStandard(PetscVec):
    ...


class PetscVecNest(PetscVec):
    ...


class PetscMat(PetscObject):
    def __new__(cls, *args, **kwargs):
        # TODO dispatch to different mat types based on -mat_type
        return object.__new__(PetscMatAIJ)


class PetscMatDense(PetscMat):
    ...


class PetscMatAIJ(PetscMat):
    def __init__(self, raxes, caxes, sparsity, *, comm=None):
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

        # bit unpleasant
        self.layout_axes = AxisTree(self.raxis, {self.raxis.id: self.caxis}).freeze()

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
        from pyop3.indices.tree import (
            _compose_bits,
            _index_axes,
            as_index_forest,
            as_index_tree,
            collect_loop_contexts,
            index_axes,
        )

        # TODO also support context-free (see MultiArray.__getitem__)
        array_per_context = {}
        for index_tree in as_index_forest(indices, axes=self.layout_axes):
            # make a temporary of the right shape
            loop_context = index_tree.loop_context
            (
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            ) = _index_axes(self.layout_axes, index_tree, loop_context)

            (
                target_paths,
                index_exprs,
                layout_exprs,
            ) = _compose_bits(
                self.layout_axes,
                indexed_axes,
                target_path_per_indexed_cpt,
                index_exprs_per_indexed_cpt,
                layout_exprs_per_indexed_cpt,
            )

            new_axes = IndexedAxisTree(
                indexed_axes.root,
                indexed_axes.parent_to_children,
                target_paths,
                index_exprs,
                layout_exprs,
            )

            # but the layouts are different here!
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

            new_layouts = freeze(
                {
                    new_axes.path(*new_axes.leaf): pym.var("MatGetValues")(
                        rlayout_expr, clayout_expr
                    )
                }
            )

            layout_axes = FrozenAxisTree(
                new_axes.root,
                new_axes.parent_to_children,
                target_paths=target_paths,
                index_exprs=index_exprs,
                layouts=new_layouts,
            )
            array_per_context[loop_context] = MultiArray(layout_axes, dtype=self.dtype)
        return ContextSensitiveMultiArray(array_per_context)


class PetscMatBAIJ(PetscMat):
    ...


class PetscMatNest(PetscMat):
    ...


class PetscMatPython(PetscMat):
    ...

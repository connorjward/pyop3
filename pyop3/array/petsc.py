from __future__ import annotations

import abc
import collections
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
from pyop3.axtree.layout import collect_external_loops
from pyop3.axtree.tree import (
    ContextFree,
    ContextSensitive,
    PartialAxisTree,
    as_axis_tree,
    relabel_axes,
)
from pyop3.buffer import PackedBuffer
from pyop3.cache import cached
from pyop3.dtypes import IntType, ScalarType
from pyop3.itree.tree import CalledMap, LoopIndex, _index_axes, as_index_forest
from pyop3.lang import PetscMatStore, do_loop, loop
from pyop3.mpi import hash_comm
from pyop3.utils import deprecated, just_one, merge_dicts, single_valued, strictly_all


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


class PetscMat(PetscObject, abc.ABC):
    DEFAULT_MAT_TYPE = PETSc.Mat.Type.AIJ

    prefix = "mat"

    def __new__(cls, *args, **kwargs):
        # If the user called PetscMat(...), as opposed to PetscMatAIJ(...) etc
        # then inspect mat_type and return the right object.
        if cls is PetscMat:
            mat_type = kwargs.pop("mat_type", cls.DEFAULT_MAT_TYPE)
            if mat_type == PETSc.Mat.Type.AIJ:
                return object.__new__(PetscMatAIJ)
            # elif mat_type == PETSc.Mat.Type.BAIJ:
            #     return object.__new__(PetscMatBAIJ)
            else:
                raise AssertionError
        else:
            return object.__new__(cls)

    # like Dat, bad name? handle?
    @property
    def array(self):
        return self.mat

    @property
    def values(self):
        if self.raxes.size * self.caxes.size > 1e6:
            raise ValueError(
                "Printing a dense matrix with more than 1 million "
                "entries is not allowed"
            )

        self.assemble()
        return self.mat[:, :]

    def assemble(self):
        self.mat.assemble()

    def assign(self, other):
        return PetscMatStore(self, other)

    def eager_zero(self):
        self.mat.zeroEntries()


class MonolithicPetscMat(PetscMat, abc.ABC):
    _row_suffix = "_row"
    _col_suffix = "_col"

    def __init__(self, raxes, caxes, sparsity=None, *, name=None):
        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)

        if sparsity is not None:
            mat = sparsity.materialize(self.mat_type)
        else:
            mat = self._make_mat(raxes, caxes, self.mat_type)

        super().__init__(name)
        self.raxes = raxes
        self.caxes = caxes
        self.mat = mat

    def __getitem__(self, indices):
        # TODO also support context-free (see MultiArray.__getitem__)
        if len(indices) != 2:
            raise ValueError

        # Combine the loop contexts of the row and column indices. Consider
        # a loop over a multi-component axis with components "a" and "b":
        #
        #   loop(p, mat[p, p])
        #
        # The row and column index forests with "merged" loop contexts would
        # look like:
        #
        #   {
        #     {p: "a"}: [rtree0, ctree0],
        #     {p: "b"}: [rtree1, ctree1]
        #   }
        #
        # By contrast, distinct loop indices are combined as a product, not
        # merged. For example, the loop
        #
        #   loop(p, loop(q, mat[p, q]))
        #
        # with p still a multi-component loop over "a" and "b" and q the same
        # over "x" and "y". This would give the following combined set of
        # index forests:
        #
        #   {
        #     {p: "a", q: "x"}: [rtree0, ctree0],
        #     {p: "a", q: "y"}: [rtree0, ctree1],
        #     {p: "b", q: "x"}: [rtree1, ctree0],
        #     {p: "b", q: "y"}: [rtree1, ctree1],
        #   }

        rtrees = as_index_forest(indices[0], axes=self.raxes)
        ctrees = as_index_forest(indices[1], axes=self.caxes)
        rcforest = {}
        for rctx, rtree in rtrees.items():
            for cctx, ctree in ctrees.items():
                # skip if the row and column contexts are incompatible
                if any(idx in rctx and rctx[idx] != path for idx, path in cctx.items()):
                    continue
                rcforest[rctx | cctx] = (rtree, ctree)

        arrays = {}
        for ctx, (rtree, ctree) in rcforest.items():
            indexed_raxes = _index_axes(rtree, ctx, self.raxes)
            indexed_caxes = _index_axes(ctree, ctx, self.caxes)

            if indexed_raxes.alloc_size() == 0 or indexed_caxes.alloc_size() == 0:
                continue
            router_loops = indexed_raxes.outer_loops
            couter_loops = indexed_caxes.outer_loops

            # rmap_axes = AxisTree(indexed_raxes.layout_axes.parent_to_children)
            # cmap_axes = AxisTree(indexed_caxes.layout_axes.parent_to_children)

            """
            
            KEY POINTS
            ----------

            * These maps require new layouts. Typically when we index something
              we want to use the prior layout, here we want to materialise them.
              This is basically what we always want for temporaries but this time
              we actually want to materialise data.
            * We then have to use the default target paths and index exprs. If these
              are the "indexed" ones then they don't work. For instance the target
              paths target non-existent layouts since we are using new layouts.

            """

            rmap = HierarchicalArray(
                indexed_raxes,
                # indexed_raxes.layout_axes,
                # rmap_axes,
                # target_paths=indexed_raxes.target_paths,
                index_exprs=indexed_raxes.index_exprs,
                target_paths=indexed_raxes._default_target_paths(),
                # index_exprs=indexed_raxes._default_index_exprs(),
                layouts=indexed_raxes.layouts,
                # target_paths=indexed_raxes.layout_axes.target_paths,
                # index_exprs=indexed_raxes.layout_axes.index_exprs,
                # layouts=indexed_raxes.layout_axes.layouts,
                outer_loops=router_loops,
                dtype=IntType,
            )
            cmap = HierarchicalArray(
                indexed_caxes,
                # indexed_caxes.layout_axes,
                # cmap_axes,
                # target_paths=indexed_caxes.target_paths,
                index_exprs=indexed_caxes.index_exprs,
                target_paths=indexed_caxes._default_target_paths(),
                # index_exprs=indexed_caxes._default_index_exprs(),
                layouts=indexed_caxes.layouts,
                # target_paths=indexed_caxes.layout_axes.target_paths,
                # index_exprs=indexed_caxes.layout_axes.index_exprs,
                # layouts=indexed_caxes.layout_axes.layouts,
                outer_loops=couter_loops,
                dtype=IntType,
            )

            from pyop3.axtree.layout import my_product

            # so these are now failing BADLY because I have no real idea what
            # I'm doing here...
            # So the issue is that cmap is having values set in the wrong place
            # when we are building a sparsity.

            for idxs in my_product(router_loops):
                # I don't think that source_indices is currently required because
                # we express layouts in terms of the LoopIndexVariable instead of
                # LocalLoopIndexVariable (which we should fix).
                source_indices = {idx.index.id: idx.source_exprs for idx in idxs}
                target_indices = {idx.index.id: idx.target_exprs for idx in idxs}
                for p in indexed_raxes.iter(idxs):
                    offset = self.raxes.offset(
                        p.target_exprs, p.target_path, loop_exprs=target_indices
                    )
                    rmap.set_value(
                        # p.source_exprs, offset, p.source_path, loop_exprs=source_indices
                        p.source_exprs,
                        offset,
                        p.source_path,
                        loop_exprs=target_indices,
                    )

            for idxs in my_product(couter_loops):
                source_indices = {idx.index.id: idx.source_exprs for idx in idxs}
                target_indices = {idx.index.id: idx.target_exprs for idx in idxs}
                for p in indexed_caxes.iter(idxs):
                    offset = self.caxes.offset(
                        p.target_exprs, p.target_path, loop_exprs=target_indices
                    )
                    cmap.set_value(
                        # p.source_exprs, offset, p.source_path, loop_exprs=source_indices
                        p.source_exprs,
                        offset,
                        p.source_path,
                        loop_exprs=target_indices,
                    )

            shape = (indexed_raxes.size, indexed_caxes.size)
            packed = PackedPetscMat(self, rmap, cmap, shape)

            # Since axes require unique labels, relabel the row and column axis trees
            # with different suffixes. This allows us to create a combined axis tree
            # without clashes.
            raxes_relabel = relabel_axes(indexed_raxes, self._row_suffix)
            caxes_relabel = relabel_axes(indexed_caxes, self._col_suffix)

            axes = PartialAxisTree(raxes_relabel.parent_to_children)
            for leaf in raxes_relabel.leaves:
                axes = axes.add_subtree(caxes_relabel, *leaf, uniquify_ids=True)
            axes = axes.set_up()

            outer_loops = list(router_loops)
            all_ids = [l.id for l in router_loops]
            for ol in couter_loops:
                if ol.id not in all_ids:
                    outer_loops.append(ol)

            my_target_paths = indexed_raxes.target_paths | indexed_caxes.target_paths
            my_index_exprs = indexed_raxes.index_exprs | indexed_caxes.index_exprs

            arrays[ctx] = HierarchicalArray(
                axes,
                data=packed,
                target_paths=my_target_paths,
                index_exprs=my_index_exprs,
                # TODO ordered set?
                outer_loops=outer_loops,
                name=self.name,
            )
        return ContextSensitiveMultiArray(arrays)

    @property
    @abc.abstractmethod
    def mat_type(self) -> str:
        pass

    @staticmethod
    def _make_mat(raxes, caxes, mat_type):
        # TODO: Internal comm?
        comm = single_valued([raxes.comm, caxes.comm])
        mat = PETSc.Mat().create(comm)
        mat.setType(mat_type)
        # None is for the global size, PETSc will determine it
        mat.setSizes(((raxes.owned.size, None), (caxes.owned.size, None)))

        rlgmap = PETSc.LGMap().create(raxes.global_numbering(), comm=comm)
        clgmap = PETSc.LGMap().create(caxes.global_numbering(), comm=comm)
        mat.setLGMap(rlgmap, clgmap)

        return mat

    @cached_property
    def datamap(self):
        return freeze({self.name: self})

    @property
    def kernel_dtype(self):
        raise NotImplementedError("opaque type?")


class PetscMatAIJ(MonolithicPetscMat):
    def __init__(self, raxes, caxes, sparsity=None, *, name: str = None):
        super().__init__(raxes, caxes, sparsity, name=name)

    @property
    def mat_type(self) -> str:
        return PETSc.Mat.Type.AIJ


# class PetscMatBAIJ(MonolithicPetscMat):
#     ...


class PetscMatPreallocator(MonolithicPetscMat):
    def __init__(self, raxes, caxes, *, name: str = None):
        super().__init__(raxes, caxes, name=name)
        self._lazy_template = None

    @property
    def mat_type(self) -> str:
        return PETSc.Mat.Type.PREALLOCATOR

    def materialize(self, mat_type: str) -> PETSc.Mat:
        if self._lazy_template is None:
            self.assemble()

            template = self._make_mat(self.raxes, self.caxes, mat_type)
            template.preallocateWithMatPreallocator(self.mat)
            # We can safely set these options since by using a sparsity we
            # are asserting that we know where the non-zeros are going.
            template.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
            template.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
            self._lazy_template = template
        return self._lazy_template.copy()


# class PetscMatDense(MonolithicPetscMat):
#     ...


# class PetscMatNest(PetscMat):
#     ...


# class PetscMatPython(PetscMat):
#     ...


class PackedPetscMat(PackedBuffer):
    def __init__(self, mat, rmap, cmap, shape):
        super().__init__(mat)
        self.rmap = rmap
        self.cmap = cmap
        self.shape = shape

    @property
    def mat(self):
        return self.array

    @cached_property
    def datamap(self):
        datamap_ = self.mat.datamap | self.rmap.datamap | self.cmap.datamap
        for s in self.shape:
            if isinstance(s, HierarchicalArray):
                datamap_ |= s.datamap
        return datamap_

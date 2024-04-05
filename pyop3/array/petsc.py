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
from pyrsistent import freeze, pmap

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
from pyop3.itree.tree import (
    CalledMap,
    LoopIndex,
    _index_axes,
    as_index_forest,
    iter_axis_tree,
)
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


class PetscMat(PetscObject, ContextFree, abc.ABC):
    DEFAULT_MAT_TYPE = PETSc.Mat.Type.AIJ

    prefix = "mat"

    # make abstract property of some parent class?
    constant = False

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
        if isinstance(other, HierarchicalArray):
            # TODO: Check axes match between self and other
            return PetscMatStore(self, other)
        elif isinstance(other, numbers.Number):
            static = HierarchicalArray(
                self.axes,
                data=np.full(self.axes.size, other, dtype=self.dtype),
                constant=True,
            )
            return PetscMatStore(self, static)
        else:
            raise NotImplementedError

    def eager_zero(self):
        self.mat.zeroEntries()


class MonolithicPetscMat(PetscMat, abc.ABC):
    _row_suffix = "_row"
    _col_suffix = "_col"

    # def __init__(self, raxes, caxes, mat=None, *, name=None):
    # TODO: target paths and index exprs should be part of raxes, caxes
    def __init__(
        self,
        raxes,
        caxes,
        mat=None,
        *,
        name=None,
        rtarget_paths=None,
        rindex_exprs=None,
        orig_raxes=None,
        router_loops=None,
        ctarget_paths=None,
        cindex_exprs=None,
        orig_caxes=None,
        couter_loops=None,
    ):
        # TODO: Remove
        if strictly_all(
            x is None
            for x in [rtarget_paths, rindex_exprs, ctarget_paths, cindex_exprs]
        ):
            rtarget_paths = raxes._default_target_paths()
            rindex_exprs = raxes._default_index_exprs()
            orig_raxes = raxes
            router_loops = ()
            ctarget_paths = caxes._default_target_paths()
            cindex_exprs = caxes._default_index_exprs()
            orig_caxes = caxes
            couter_loops = ()

        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)

        if mat is None:
            mat = self._make_mat(raxes, caxes, self.mat_type)

        super().__init__(name)
        self.raxes = raxes
        self.caxes = caxes
        self.mat = mat

        # TODO: delete
        self.rtarget_paths = rtarget_paths
        self.rindex_exprs = rindex_exprs
        self.orig_raxes = orig_raxes
        self.router_loops = router_loops
        self.ctarget_paths = ctarget_paths
        self.cindex_exprs = cindex_exprs
        self.orig_caxes = orig_caxes
        self.couter_loops = couter_loops

    @classmethod
    def from_sparsity(cls, raxes, caxes, sparsity, *, name=None):
        mat = sparsity.materialize(cls.mat_type)
        return cls(raxes, caxes, mat, name=name)

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    def getitem(self, indices, *, strict=False):
        from pyop3.itree.tree import _compose_bits, _index_axes, as_index_forest

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

        rtrees = as_index_forest(indices[0], axes=self.raxes, strict=strict)
        ctrees = as_index_forest(indices[1], axes=self.caxes, strict=strict)
        rcforest = {}
        for rctx, rtree in rtrees.items():
            for cctx, ctree in ctrees.items():
                # skip if the row and column contexts are incompatible
                if any(idx in rctx and rctx[idx] != path for idx, path in cctx.items()):
                    continue
                rcforest[rctx | cctx] = (rtree, ctree)

        # If there are no outer loops then we can return a context-free array.
        if rcforest.keys() == {pmap()}:
            rtree, ctree = rcforest[pmap()]

            indexed_raxes = _index_axes(rtree, pmap(), self.raxes)
            indexed_caxes = _index_axes(ctree, pmap(), self.caxes)

            rtarget_paths, rindex_exprs, _ = _compose_bits(
                self.raxes,
                self.rtarget_paths,
                self.rindex_exprs,
                None,
                indexed_raxes,
                indexed_raxes.target_paths,
                indexed_raxes.index_exprs,
                {},
            )
            ctarget_paths, cindex_exprs, _ = _compose_bits(
                self.caxes,
                self.ctarget_paths,
                self.cindex_exprs,
                None,
                indexed_caxes,
                indexed_caxes.target_paths,
                indexed_caxes.index_exprs,
                {},
            )

            return type(self)(
                indexed_raxes,
                indexed_caxes,
                self.mat,
                name=self.name,
                # delete below
                rtarget_paths=rtarget_paths,
                rindex_exprs=rindex_exprs,
                orig_raxes=self.orig_raxes,
                router_loops=indexed_raxes.outer_loops,
                ctarget_paths=ctarget_paths,
                cindex_exprs=cindex_exprs,
                orig_caxes=self.orig_caxes,
                couter_loops=indexed_caxes.outer_loops,
            )

        # Otherwise we are context-sensitive
        arrays = {}
        for ctx, (rtree, ctree) in rcforest.items():
            indexed_raxes = _index_axes(rtree, ctx, self.raxes)
            indexed_caxes = _index_axes(ctree, ctx, self.caxes)

            if indexed_raxes.alloc_size() == 0 or indexed_caxes.alloc_size() == 0:
                continue

            rtarget_paths, rindex_exprs, _ = _compose_bits(
                self.raxes,
                self.rtarget_paths,
                self.rindex_exprs,
                None,
                indexed_raxes,
                indexed_raxes.target_paths,
                indexed_raxes.index_exprs,
                {},
            )
            ctarget_paths, cindex_exprs, _ = _compose_bits(
                self.caxes,
                self.ctarget_paths,
                self.cindex_exprs,
                None,
                indexed_caxes,
                indexed_caxes.target_paths,
                indexed_caxes.index_exprs,
                {},
            )

            arrays[ctx] = type(self)(
                indexed_raxes,
                indexed_caxes,
                self.mat,
                name=self.name,
                # delete below
                rtarget_paths=rtarget_paths,
                rindex_exprs=rindex_exprs,
                orig_raxes=self.orig_raxes,
                router_loops=indexed_raxes.outer_loops,
                ctarget_paths=ctarget_paths,
                cindex_exprs=cindex_exprs,
                orig_caxes=self.orig_caxes,
                couter_loops=indexed_caxes.outer_loops,
            )
        # But this is now a PetscMat...
        return ContextSensitiveMultiArray(arrays)

    @cached_property
    def maps(self):
        from pyop3.axtree.layout import my_product

        """
        KEY POINTS
        ----------

        * These maps require new layouts. Typically when we index something
          we want to use the prior layout, here we want to materialise them.
          This is basically what we always want for temporaries but this time
          we actually want to materialise data.
        """

        rmap = HierarchicalArray(
            self.raxes,
            index_exprs=self.raxes.index_exprs,
            target_paths=self.raxes._default_target_paths(),
            layouts=self.raxes.layouts,
            outer_loops=self.router_loops,
            dtype=IntType,
        )
        cmap = HierarchicalArray(
            self.caxes,
            index_exprs=self.caxes.index_exprs,
            target_paths=self.caxes._default_target_paths(),
            layouts=self.caxes.layouts,
            outer_loops=self.couter_loops,
            dtype=IntType,
        )

        for idxs in my_product(self.router_loops):
            target_indices = {idx.index.id: idx.target_exprs for idx in idxs}

            # TODO: We use iter_axis_tree here because the target_paths and
            # index_exprs are not tied to raxes.
            riter = iter_axis_tree(
                self.raxes.index(),
                self.raxes,
                self.rtarget_paths,
                self.rindex_exprs,
                idxs,
            )
            # for p in self.raxes.iter(idxs):
            for p in riter:
                offset = self.orig_raxes.offset(
                    p.target_exprs, p.target_path, loop_exprs=target_indices
                )
                rmap.set_value(
                    p.source_exprs,
                    offset,
                    p.source_path,
                    loop_exprs=target_indices,
                )

        for idxs in my_product(self.couter_loops):
            target_indices = {idx.index.id: idx.target_exprs for idx in idxs}

            # TODO: as above, replace with .iter()
            citer = iter_axis_tree(
                self.caxes.index(),
                self.caxes,
                self.ctarget_paths,
                self.cindex_exprs,
                idxs,
            )
            # for p in self.caxes.iter(idxs):
            for p in citer:
                offset = self.orig_caxes.offset(
                    p.target_exprs, p.target_path, loop_exprs=target_indices
                )
                cmap.set_value(
                    p.source_exprs,
                    offset,
                    p.source_path,
                    loop_exprs=target_indices,
                )

        return rmap, cmap

    @property
    def rmap(self):
        return self.maps[0]

    @property
    def cmap(self):
        return self.maps[1]

    @property
    def shape(self):
        return (self.raxes.size, self.caxes.size)

    @cached_property
    def axes(self):
        # Since axes require unique labels, relabel the row and column axis trees
        # with different suffixes. This allows us to create a combined axis tree
        # without clashes.
        raxes_relabel = relabel_axes(self.raxes, self._row_suffix)
        caxes_relabel = relabel_axes(self.caxes, self._col_suffix)

        axes = PartialAxisTree(raxes_relabel.parent_to_children)
        for leaf in raxes_relabel.leaves:
            axes = axes.add_subtree(caxes_relabel, *leaf, uniquify_ids=True)
        axes = axes.set_up()
        return axes

    # @property
    # @abc.abstractmethod
    # def mat_type(self) -> str:
    #     pass

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
        return freeze({self.name: self}) | self.rmap.datamap | self.cmap.datamap

    @property
    def kernel_dtype(self):
        raise NotImplementedError("opaque type?")


class PetscMatAIJ(MonolithicPetscMat):
    def __init__(self, raxes, caxes, mat=None, *, name: str = None, **kwargs):
        super().__init__(raxes, caxes, mat, name=name, **kwargs)

    # @property
    # def mat_type(self) -> str:
    #     return PETSc.Mat.Type.AIJ
    mat_type = PETSc.Mat.Type.AIJ


# class PetscMatBAIJ(MonolithicPetscMat):
#     ...


class PetscMatPreallocator(MonolithicPetscMat):
    # TODO: Delete kwargs, not required when raxes, caxes are IndexedAxisTree
    # def __init__(self, raxes, caxes, *, name: str = None):
    def __init__(self, raxes, caxes, mat=None, *, name: str = None, **kwargs):
        super().__init__(raxes, caxes, mat, name=name, **kwargs)
        self._lazy_template = None

    # @property
    # def mat_type(self) -> str:
    #     return PETSc.Mat.Type.PREALLOCATOR
    mat_type = PETSc.Mat.Type.PREALLOCATOR

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

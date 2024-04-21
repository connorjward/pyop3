from __future__ import annotations

import abc
import collections
import numbers
from functools import cached_property
from itertools import product

import numpy as np
import pymbolic as pym
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.array.base import Array
from pyop3.array.harray import ContextSensitiveMultiArray, HierarchicalArray
from pyop3.axtree.tree import (
    AxisTree,
    ContextFree,
    IndexedAxisTree,
    as_axis_tree,
    relabel_axes,
)
from pyop3.dtypes import IntType, ScalarType
from pyop3.itree.tree import (
    IndexExpressionReplacer,
    as_index_forest,
    compose_axes,
    index_axes,
)
from pyop3.lang import PetscMatStore
from pyop3.utils import (
    deprecated,
    just_one,
    merge_dicts,
    single_valued,
    strictly_all,
    unique,
)


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


class AbstractMat(Array, ContextFree):
    DEFAULT_MAT_TYPE = PETSc.Mat.Type.AIJ

    prefix = "mat"
    dtype = PETSc.ScalarType

    # Make abstract property of some parent class?
    constant = False

    _row_suffix = "_row"
    _col_suffix = "_col"

    # TODO: target paths and index exprs should be part of raxes, caxes
    # def __init__(self, raxes, caxes, mat=None, *, name=None):
    def __init__(
        self,
        raxes,
        caxes,
        mat_type=None,
        mat=None,
        *,
        name=None,
    ):
        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)

        if mat_type is None:
            mat_type = self.DEFAULT_MAT_TYPE

        if mat_type == "baij":
            raise NotImplementedError("Need to give Mats a shape")

        if mat is None:
            mat = self._make_mat(raxes, caxes, mat_type)

        super().__init__(name)
        self.raxes = raxes
        self.caxes = caxes
        self.mat_type = mat_type
        self.mat = mat

        # self._cache = {}

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    def getitem(self, indices, *, strict=False):
        # does not work as indices may not be hashable, parse first?
        # cache_key = (indices, strict)
        # if cache_key in self._cache:
        #     return self._cache[cache_key]

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

            indexed_raxes = index_axes(rtree, pmap(), self.raxes)
            raxes = compose_axes(indexed_raxes, self.raxes)

            indexed_caxes = index_axes(ctree, pmap(), self.caxes)
            caxes = compose_axes(indexed_caxes, self.caxes)

            mat = type(self)(
                raxes,
                caxes,
                mat_type=self.mat_type,
                mat=self.mat,
                name=self.name,
            )
            # self._cache[cache_key] = mat
            return mat

        # Otherwise we are context-sensitive
        arrays = {}
        for ctx, (rtree, ctree) in rcforest.items():
            indexed_raxes = index_axes(rtree, ctx, self.raxes)
            indexed_caxes = index_axes(ctree, ctx, self.caxes)

            if indexed_raxes.alloc_size() == 0 or indexed_caxes.alloc_size() == 0:
                continue

            raxes = compose_axes(indexed_raxes, self.raxes)
            caxes = compose_axes(indexed_caxes, self.caxes)

            arrays[ctx] = type(self)(
                raxes,
                caxes,
                self.mat_type,
                self.mat,
                name=self.name,
            )
        # But this is now a PetscMat...
        mat = ContextSensitiveMultiArray(arrays)
        # self._cache[cache_key] = mat
        return mat

    # like Dat, bad name? handle?
    @property
    def array(self):
        return self.mat

    # old alias, deprecate?
    @property
    def handle(self):
        return self.mat

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

    @property
    def nested(self):
        return isinstance(self.mat_type, collections.abc.Mapping)

    @cached_property
    def nest_labels(self):
        if self.nested:
            return tuple(self._iter_nest_labels())
        else:
            return ((None, None),)

    def _iter_nest_labels(
        self, raxis=None, caxis=None, mat_type=None, rlabel_acc=None, clabel_acc=None
    ):
        assert self.nested

        if strictly_all(
            x is None for x in {raxis, caxis, mat_type, rlabel_acc, clabel_acc}
        ):
            raxis = self.raxes.unindexed.root
            caxis = self.caxes.unindexed.root
            mat_type = self.mat_type
            rlabel_acc = ()
            clabel_acc = ()

        if not strictly_all(x is None for x, _ in mat_type.keys()):
            rroot = self.raxes.root
            rlabels = unique(
                clabel
                for c in rroot.components
                for axlabel, clabel in self.raxes.target_paths[
                    rroot.id, c.label
                ].items()
                if axlabel == raxis.label
            )
            assert len(rlabels) in {0, 1}

            if len(rlabels) == 0:
                rlabels = tuple(c.label for c in raxis.components)
        else:
            rlabels = (None,)

        if not strictly_all(x is None for _, x in mat_type.keys()):
            croot = self.caxes.root
            clabels = unique(
                clabel
                for c in croot.components
                for axlabel, clabel in self.caxes.target_paths[
                    croot.id, c.label
                ].items()
                if axlabel == caxis.label
            )
            assert len(clabels) in {0, 1}

            if len(clabels) == 0:
                clabels = tuple(c.label for c in caxis.components)
        else:
            clabels = (None,)

        for rlabel, clabel in product(rlabels, clabels):
            rlabel_acc_ = rlabel_acc + (rlabel,)
            clabel_acc_ = clabel_acc + (clabel,)

            submat_type = mat_type[rlabel, clabel]
            if isinstance(submat_type, collections.abc.Mapping):
                rsubaxis = self.raxes.unindexed.child(raxis, rlabel)
                csubaxis = self.caxes.unindexed.child(caxis, clabel)
                yield from self._iter_nest_labels(
                    rsubaxis, csubaxis, submat_type, rlabel_acc_, clabel_acc_
                )
            else:
                yield (rlabel_acc_, clabel_acc_)

    @cached_property
    @PETSc.Log.EventDecorator()
    def maps(self):
        from pyop3.axtree.layout import my_product

        # TODO: Don't think these need to be lists here.
        # FIXME: This will only work for singly-nested matrices
        if self.nested:
            rfield_axis = self.raxes.unindexed.root
            cfield_axis = self.caxes.unindexed.root

            if strictly_all(c.unit for c in rfield_axis.components):
                # This weird trick is because the right target path for the field
                # is actually tied to the root of the axis tree, rather than None.
                # This seems like a limitation of the _compose_bits function.
                rfield = single_valued(
                    cpt
                    for mycpt in self.raxes.root.components
                    for ax, cpt in self.raxes.target_paths[
                        self.raxes.root.id, mycpt.label
                    ].items()
                    if ax == rfield_axis.label
                )
                orig_raxes = AxisTree(self.raxes.unindexed[rfield].node_map)
                orig_raxess = [orig_raxes]
                dropped_rkeys = {rfield_axis.label}
            else:
                orig_raxess = [self.raxes.unindexed]
                dropped_rkeys = frozenset()

            if strictly_all(c.unit for c in cfield_axis.components):
                cfield = single_valued(
                    cpt
                    for mycpt in self.caxes.root.components
                    for ax, cpt in self.caxes.target_paths[
                        self.caxes.root.id, mycpt.label
                    ].items()
                    if ax == cfield_axis.label
                )
                orig_caxes = AxisTree(self.caxes.unindexed[cfield].node_map)
                orig_caxess = [orig_caxes]
                dropped_ckeys = {cfield_axis.label}
            else:
                orig_caxess = [self.caxes.unindexed]
                dropped_ckeys = set()
        else:
            orig_raxess = [self.raxes.unindexed]
            orig_caxess = [self.caxes.unindexed]
            dropped_rkeys = set()
            dropped_ckeys = set()

        # TODO: are dropped_rkeys and dropped_ckeys still needed?

        loop_index = just_one(self.raxes.outer_loops)
        iterset = AxisTree(loop_index.iterset.node_map)

        rmap_axes = iterset.add_subtree(self.raxes, *iterset.leaf)
        rmap = HierarchicalArray(rmap_axes, dtype=IntType)
        rmap = rmap[loop_index.local_index]

        loop_index = just_one(self.caxes.outer_loops)
        iterset = AxisTree(loop_index.iterset.node_map)

        cmap_axes = iterset.add_subtree(self.caxes, *iterset.leaf)
        cmap = HierarchicalArray(cmap_axes, dtype=IntType)
        cmap = cmap[loop_index.local_index]

        # TODO: Make the code below go into a separate function distinct
        # from mat_type logic. Then can also share code for rmap and cmap.
        for orig_raxes in orig_raxess:
            for idxs in my_product(self.raxes.outer_loops):
                # target_indices = {idx.index.id: idx.target_exprs for idx in idxs}
                target_indices = merge_dicts([idx.replace_map for idx in idxs])

                # for p in self.raxes.iter(idxs):
                for p in self.raxes.iter(idxs, include_ghost_points=True):  # seems to fix things
                    target_path = p.target_path
                    target_exprs = p.target_exprs
                    for key in dropped_rkeys:
                        target_path = target_path.remove(key)
                        target_exprs = target_exprs.remove(key)

                    offset = orig_raxes.offset(
                        target_exprs, target_path, loop_exprs=target_indices
                    )

                    rmap.set_value(
                        p.source_exprs,
                        offset,
                        p.source_path,
                        loop_exprs=target_indices,
                    )

        for orig_caxes in orig_caxess:
            for idxs in my_product(self.caxes.outer_loops):
                # target_indices = {idx.index.id: idx.target_exprs for idx in idxs}
                target_indices = merge_dicts([idx.replace_map for idx in idxs])

                # for p in self.caxes.iter(idxs):
                for p in self.caxes.iter(idxs, include_ghost_points=True):  # seems to fix things
                    target_path = p.target_path
                    target_exprs = p.target_exprs
                    for key in dropped_ckeys:
                        target_path = target_path.remove(key)
                        target_exprs = target_exprs.remove(key)

                    offset = orig_caxes.offset(
                        target_exprs, target_path, loop_exprs=target_indices
                    )
                    cmap.set_value(
                        p.source_exprs,
                        offset,
                        p.source_path,
                        loop_exprs=target_indices,
                    )

        # breakpoint()

        return (rmap, cmap)

    @property
    def rmap(self):
        return self.maps[0]

    @property
    def cmap(self):
        return self.maps[1]

    @cached_property
    def row_lgmap_dat(self):
        if self.nested or self.mat_type == "baij":
            raise NotImplementedError("Use a smaller set of axes here")
        return HierarchicalArray(self.raxes, data=self.raxes.unindexed.global_numbering)

    @cached_property
    def column_lgmap_dat(self):
        if self.nested or self.mat_type == "baij":
            raise NotImplementedError("Use a smaller set of axes here")
        return HierarchicalArray(self.caxes, data=self.caxes.unindexed.global_numbering)

    @cached_property
    def comm(self):
        return single_valued([self.raxes.comm, self.caxes.comm])

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

        # might not be needed
        axes = AxisTree(raxes_relabel.node_map)
        for leaf in raxes_relabel.leaves:
            axes = axes.add_subtree(caxes_relabel, *leaf, uniquify_ids=True)
        return axes

    @classmethod
    def _make_mat(cls, raxes, caxes, mat_type):
        if isinstance(mat_type, collections.abc.Mapping):
            # TODO: This is very ugly
            rsize = max(x or 0 for x, _ in mat_type.keys()) + 1
            csize = max(y or 0 for _, y in mat_type.keys()) + 1
            submats = np.empty((rsize, csize), dtype=object)
            for (rkey, ckey), submat_type in mat_type.items():
                subraxes = raxes[rkey] if rkey is not None else raxes
                subcaxes = caxes[ckey] if ckey is not None else caxes
                submat = cls._make_mat(subraxes, subcaxes, submat_type)
                submats[rkey, ckey] = submat

            # TODO: Internal comm? Set as mat property (then not a classmethod)?
            comm = single_valued([raxes.comm, caxes.comm])
            return PETSc.Mat().createNest(submats, comm=comm)
        else:
            return cls._make_monolithic_mat(raxes, caxes, mat_type)

    @cached_property
    def datamap(self):
        return freeze({self.name: self}) | self.rmap.datamap | self.cmap.datamap

    @property
    def kernel_dtype(self):
        raise NotImplementedError("opaque type?")


class Sparsity(AbstractMat):
    def materialize(self) -> PETSc.Mat:
        if not hasattr(self, "_lazy_template"):
            self.assemble()

            template = Mat._make_mat(self.raxes, self.caxes, self.mat_type)
            self._preallocate(self.mat, template, self.mat_type)
            # template.preallocateWithMatPreallocator(self.mat)
            # We can safely set these options since by using a sparsity we
            # are asserting that we know where the non-zeros are going.
            # NOTE: These may already get set by PETSc.
            template.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
            template.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

            template.assemble()
            self._lazy_template = template
        return self._lazy_template.copy()

    def _preallocate(self, preallocator, template, mat_type):
        if isinstance(mat_type, collections.abc.Mapping):
            for (ridx, cidx), submat_type in mat_type.items():
                if ridx is None:
                    ridx = 0
                if cidx is None:
                    cidx = 0
                subpreallocator = preallocator.getNestSubMatrix(ridx, cidx)
                submat = template.getNestSubMatrix(ridx, cidx)
                self._preallocate(subpreallocator, submat, submat_type)
        else:
            if mat_type != "dat":
                template.preallocateWithMatPreallocator(preallocator)

    @classmethod
    def _make_monolithic_mat(cls, raxes, caxes, mat_type: str):
        # TODO: Internal comm?
        comm = single_valued([raxes.comm, caxes.comm])

        if mat_type == "dat":
            matdat = _MatDat(raxes, caxes)
            if matdat.is_row_matrix:
                assert not matdat.is_column_matrix
                sizes = ((raxes.owned.size, None), (None, 1))
            elif matdat.is_column_matrix:
                sizes = ((None, 1), (caxes.owned.size, None))
            else:
                # 1x1 block
                sizes = ((None, 1), (None, 1))
            mat = PETSc.Mat().createPython(sizes, comm=comm)
            mat.setPythonContext(matdat)
        else:
            mat = PETSc.Mat().create(comm)
            mat.setType(PETSc.Mat.Type.PREALLOCATOR)

            # None is for the global size, PETSc will figure it out for us
            sizes = ((raxes.owned.size, None), (caxes.owned.size, None))
            mat.setSizes(sizes)

            rlgmap = PETSc.LGMap().create(raxes.global_numbering, comm=comm)
            clgmap = PETSc.LGMap().create(caxes.global_numbering, comm=comm)
            mat.setLGMap(rlgmap, clgmap)

        mat.setUp()
        return mat


class Mat(AbstractMat):
    @classmethod
    def from_sparsity(cls, sparsity, *, name=None):
        mat = sparsity.materialize()
        return cls(sparsity.raxes, sparsity.caxes, sparsity.mat_type, mat, name=name)

    def eager_zero(self):
        self.mat.zeroEntries()

    @property
    def values(self):
        if self.raxes.size * self.caxes.size > 1e6:
            raise ValueError(
                "Printing a dense matrix with more than 1 million "
                "entries is not allowed"
            )

        self.assemble()

        # TODO: Should use something similar to buffer_indices to select the
        # right indices.
        if self.nested:
            if len(self.nest_labels) > 1:
                raise NotImplementedError("Cannot display mat nests")

            ridx, cidx = map(_zero_if_none, map(just_one, just_one(self.nest_labels)))
            mat = self.mat.getNestSubMatrix(ridx, cidx)
        else:
            mat = self.mat

        if mat.getType() == PETSc.Mat.Type.PYTHON:
            return mat.getPythonContext().dat.data_ro
        else:
            return mat[:, :]

    # TODO: Almost identical code to Sparsity
    @classmethod
    def _make_monolithic_mat(cls, raxes, caxes, mat_type: str):
        # TODO: Internal comm?
        comm = single_valued([raxes.comm, caxes.comm])

        if mat_type == "dat":
            matdat = _MatDat(raxes, caxes)
            if matdat.is_row_matrix:
                assert not matdat.is_column_matrix
                sizes = ((raxes.owned.size, None), (None, 1))
            elif matdat.is_column_matrix:
                sizes = ((None, 1), (caxes.owned.size, None))
            else:
                # 1x1 block
                sizes = ((None, 1), (None, 1))
            mat = PETSc.Mat().createPython(sizes, comm=comm)
            mat.setPythonContext(matdat)
        else:
            mat = PETSc.Mat().create(comm)
            mat.setType(mat_type)

            # None is for the global size, PETSc will figure it out for us
            sizes = ((raxes.owned.size, None), (caxes.owned.size, None))
            mat.setSizes(sizes)

            rlgmap = PETSc.LGMap().create(raxes.global_numbering, comm=comm)
            clgmap = PETSc.LGMap().create(caxes.global_numbering, comm=comm)
            mat.setLGMap(rlgmap, clgmap)

        mat.setUp()
        return mat


class _MatDat:
    dtype = Mat.dtype

    # NOTE: This dat should potentially just be a buffer.
    def __init__(self, raxes, caxes, dat=None):
        self.raxes = raxes
        self.caxes = caxes

        self._lazy_dat = dat

    @property
    def dat(self):
        if self._lazy_dat is None:
            if self.is_row_matrix:
                assert not self.is_column_matrix
                axes = self.raxes
            elif self.is_column_matrix:
                axes = self.caxes
            else:
                axes = AxisTree()

            dat = HierarchicalArray(axes, dtype=self.dtype)
            self._lazy_dat = dat
        return self._lazy_dat

    @property
    def is_row_matrix(self):
        root = self.raxes.root
        return len(root.components) != 1 or not root.component.unit

    @property
    def is_column_matrix(self):
        root = self.caxes.root
        return len(root.components) != 1 or not root.component.unit

    # def __getitem__(self, key):
    #     shape = [s[0] or 1 for s in self.sizes]
    #     return self.dat.data_ro.reshape(*shape)[key]

    def zeroEntries(self, mat):
        self.dat.eager_zero()

    def mult(self, A, x, y):
        """Set y = A * x (where A is self)."""
        print("AAAAAAAAAAAAAAAAAAA", flush=True)
        with self.dat.vec_ro as v:
            if self.is_row_matrix:
                y.setValue(0, v.dot(x))
            else:
                assert self.is_column_matrix
                # this is really slow, but what is already there is unclear

                # print("BBBBBBBBBBBBBBBBBBB", flush=True)
                # breakpoint()
                # for row in
                # if x.sizes[1] == 1:
                #     v.copy(y)
                #     a = np.zeros(1, dtype=dtypes.ScalarType)
                #     if x.comm.rank == 0:
                #         a[0] = x.array_r
                #     else:
                #         x.array_r
                #     with mpi.temp_internal_comm(x.comm) as comm:
                #         comm.bcast(a)
                #     return y.scale(a)
                # else:
                #     return v.pointwiseMult(x, y)

    # def multTranspose(self, mat, x, y):
    #     with self.dat.vec_ro as v:
    #         if self.sizes[0][0] is None:
    #             # Row matrix
    #             if x.sizes[1] == 1:
    #                 v.copy(y)
    #                 a = np.zeros(1, dtype=dtypes.ScalarType)
    #                 if x.comm.rank == 0:
    #                     a[0] = x.array_r
    #                 else:
    #                     x.array_r
    #                 with mpi.temp_internal_comm(x.comm) as comm:
    #                     comm.bcast(a)
    #                 y.scale(a)
    #             else:
    #                 v.pointwiseMult(x, y)
    #         else:
    #             # Column matrix
    #             out = v.dot(x)
    #             if y.comm.rank == 0:
    #                 y.array[0] = out
    #             else:
    #                 y.array[...]
    #
    # def multTransposeAdd(self, mat, x, y, z):
    #     ''' z = y + mat^Tx '''
    #     with self.dat.vec_ro as v:
    #         if self.sizes[0][0] is None:
    #             # Row matrix
    #             if x.sizes[1] == 1:
    #                 v.copy(z)
    #                 a = np.zeros(1, dtype=dtypes.ScalarType)
    #                 if x.comm.rank == 0:
    #                     a[0] = x.array_r
    #                 else:
    #                     x.array_r
    #                 with mpi.temp_internal_comm(x.comm) as comm:
    #                     comm.bcast(a)
    #                 if y == z:
    #                     # Last two arguments are aliased.
    #                     tmp = y.duplicate()
    #                     y.copy(tmp)
    #                     y = tmp
    #                 z.scale(a)
    #                 z.axpy(1, y)
    #             else:
    #                 if y == z:
    #                     # Last two arguments are aliased.
    #                     tmp = y.duplicate()
    #                     y.copy(tmp)
    #                     y = tmp
    #                 v.pointwiseMult(x, z)
    #                 return z.axpy(1, y)
    #         else:
    #             # Column matrix
    #             out = v.dot(x)
    #             y = y.array_r
    #             if z.comm.rank == 0:
    #                 z.array[0] = out + y[0]
    #             else:
    #                 z.array[...]

    def duplicate(self, mat, copy=True):
        # debug, this is not the problem
        return mat
        if copy:
            # arguably duplicate is a better name for this function
            context = type(self)(self.raxes, self.caxes, dat=self.dat.copy())
        else:
            context = type(self)(self.raxes, self.caxes)

        mat = PETSc.Mat().createPython(mat.getSizes(), comm=mat.comm)
        mat.setPythonContext(context)
        mat.setUp()
        return mat


def _zero_if_none(value):
    return value if value is not None else 0

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
from pyop3.axtree.tree import (
    ContextFree,
    ContextSensitive,
    PartialAxisTree,
    as_axis_tree,
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


class MatType(enum.Enum):
    AIJ = "aij"
    BAIJ = "baij"
    PREALLOCATOR = "preallocator"


class PetscMat(PetscObject, abc.ABC):
    DEFAULT_MAT_TYPE = MatType.AIJ

    prefix = "mat"

    def __new__(cls, *args, **kwargs):
        if cls is PetscMat:
            mat_type_str = kwargs.pop("mat_type", cls.DEFAULT_MAT_TYPE)
            mat_type = MatType(mat_type_str)
            if mat_type == MatType.AIJ:
                return object.__new__(PetscMatAIJ)
            elif mat_type == MatType.BAIJ:
                return object.__new__(PetscMatBAIJ)
            else:
                raise AssertionError
        else:
            return object.__new__(cls)

    # like Dat, bad name? handle?
    @property
    def array(self):
        return self.petscmat

    def assemble(self):
        self.mat.assemble()

    def assign(self, other):
        return PetscMatStore(self, other)

    def zero(self):
        self.mat.zeroEntries()


class MonolithicPetscMat(PetscMat, abc.ABC):
    def __init__(self, raxes, caxes, *, name=None):
        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)

        super().__init__(name)

        self.raxes = raxes
        self.caxes = caxes

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

            full_raxes = _index_axes(
                rtree, ctx, self.raxes, include_loop_index_shape=True
            )
            full_caxes = _index_axes(
                ctree, ctx, self.caxes, include_loop_index_shape=True
            )

            if full_raxes.size == 0 or full_caxes.size == 0:
                continue

            ###

            # Build the flattened row and column maps
            # rindex = just_one(rtree.nodes)
            # rloop_index = rtree
            # while isinstance(rloop_index, CalledMap):
            #     rloop_index = rloop_index.from_index
            # assert isinstance(rloop_index, LoopIndex)
            #
            # # build the map
            # riterset = rloop_index.iterset
            # my_raxes = self.raxes[rindex]
            # rmap_axes = PartialAxisTree(riterset.parent_to_children)
            # # if len(rmap_axes.leaves) > 1:
            # #     raise NotImplementedError
            # for leaf in rmap_axes.leaves:
            #     # TODO the leaves correspond to the paths/contexts, cleanup
            #     # FIXME just do this for now since we only have one leaf
            #     axes_to_add = just_one(my_raxes.context_map.values())
            #     rmap_axes = rmap_axes.add_subtree(axes_to_add, *leaf)
            # rmap_axes = rmap_axes.set_up()
            # rmap_axes = full_raxes.set_up()
            rmap_axes = full_raxes
            rlayouts = AxisTree(rmap_axes.parent_to_children).layouts
            rdiexpr = rmap_axes.domain_index_exprs
            rmap = HierarchicalArray(
                rmap_axes, dtype=IntType, layouts=rlayouts, domain_index_exprs=rdiexpr
            )
            # cmap_axes = full_caxes.set_up()
            cmap_axes = full_caxes
            clayouts = AxisTree(cmap_axes.parent_to_children).layouts
            cdiexpr = cmap_axes.domain_index_exprs
            cmap = HierarchicalArray(
                cmap_axes, dtype=IntType, layouts=clayouts, domain_index_exprs=cdiexpr
            )

            # do_loop(
            #     p := rloop_index,
            #     loop(
            #         q := rindex,
            #         rmap[p, q.i].assign(TODO)
            #     ),
            # )

            # for p in riterset.iter(loop_index=rloop_index):
            #     for q in rindex.iter({p}):
            #         for q_ in (
            #             self.raxes[q.index]
            #             .with_context(p.loop_context | q.loop_context)
            #             .iter({q})
            #         ):
            #             path = p.source_path | q.source_path | q_.source_path
            #             indices = p.source_exprs | q.source_exprs | q_.source_exprs
            #             offset = self.raxes.offset(
            #                 q_.target_path, q_.target_exprs, insert_zeros=True
            #             )
            #             rmap.set_value(path, indices, offset)
            for p in rmap_axes.iter():
                path = p.source_path
                indices = p.source_exprs
                offset = self.raxes.offset(
                    p.target_exprs,
                    p.target_path,
                )
                rmap.set_value(indices, offset, path)

            for p in cmap_axes.iter():
                path = p.source_path
                indices = p.source_exprs
                offset = self.caxes.offset(p.target_exprs, p.target_path)
                cmap.set_value(indices, offset, path)

            ###

            shape = (indexed_raxes.size, indexed_caxes.size)
            packed = PackedPetscMat(self, rmap, cmap, shape)

            indexed_axes = PartialAxisTree(indexed_raxes.parent_to_children)
            for leaf_axis, leaf_cpt in indexed_raxes.leaves:
                indexed_axes = indexed_axes.add_subtree(
                    indexed_caxes, leaf_axis, leaf_cpt, uniquify=True
                )
            indexed_axes = indexed_axes.set_up()

            arrays[ctx] = HierarchicalArray(
                indexed_axes,
                data=packed,
                target_paths=indexed_axes.target_paths,
                index_exprs=indexed_axes.index_exprs,
                # domain_index_exprs=indexed_axes.domain_index_exprs,
                domain_index_exprs=indexed_raxes.domain_index_exprs
                | indexed_caxes.domain_index_exprs,
                name=self.name,
            )
        return ContextSensitiveMultiArray(arrays)

    @cached_property
    def datamap(self):
        return freeze({self.name: self})


# is this required?
class ContextSensitiveIndexedPetscMat(ContextSensitive):
    pass


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


class PetscMatAIJ(MonolithicPetscMat):
    def __init__(self, points, adjacency, raxes, caxes, *, name: str = None):
        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)
        mat = _alloc_mat(points, adjacency, raxes, caxes)

        super().__init__(raxes, caxes, name=name)
        self.mat = mat

    @property
    # @deprecated("mat") ???
    def petscmat(self):
        return self.mat


class PetscMatBAIJ(MonolithicPetscMat):
    def __init__(self, raxes, caxes, sparsity, bsize, *, name: str = None):
        raise NotImplementedError
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


class PetscMatPreallocator(MonolithicPetscMat):
    def __init__(self, points, adjacency, raxes, caxes, *, name: str = None):
        # TODO internal comm?
        comm = single_valued([raxes.comm, caxes.comm])
        mat = PETSc.Mat().create(comm)
        mat.setType(PETSc.Mat.Type.PREALLOCATOR)
        mat.setSizes((raxes.size, caxes.size))
        mat.setUp()

        super().__init__(raxes, caxes, name=name)
        self.mat = mat


class PetscMatNest(PetscMat):
    ...


class PetscMatDense(PetscMat):
    ...


class PetscMatPython(PetscMat):
    ...


# TODO is there a better name? It does a bit more than allocate

# TODO Perhaps tie this cache to the mesh with a context manager?


def _alloc_mat(points, adjacency, raxes, caxes, bsize=None):
    template_mat = _alloc_template_mat(points, adjacency, raxes, caxes, bsize)
    return template_mat.copy()


_sparsity_cache = {}


def _alloc_template_mat_cache_key(points, adjacency, raxes, caxes, bsize=None):
    # TODO include comm in cache key, requires adding internal comm stuff
    # comm = single_valued([raxes._comm, caxes._comm])
    # return (hash_comm(comm), points, adjacency, raxes, caxes, bsize)
    return (points, adjacency, raxes, caxes, bsize)


@cached(_sparsity_cache, key=_alloc_template_mat_cache_key)
def _alloc_template_mat(points, adjacency, raxes, caxes, bsize=None):
    if bsize is not None:
        raise NotImplementedError

    # Determine the nonzero pattern by filling a preallocator matrix
    prealloc_mat = PetscMatPreallocator(points, adjacency, raxes, caxes)

    do_loop(
        p := points.index(),
        prealloc_mat[p, adjacency(p)].assign(666),
    )

    # for p in points.iter():
    #     for q in adjacency(p.index).iter({p}):
    #         for p_ in raxes[p.index].with_context(p.loop_context).iter({p}):
    #             for q_ in (
    #                 caxes[q.index]
    #                 .with_context(p.loop_context | q.loop_context)
    #                 .iter({q})
    #             ):
    #                 # NOTE: It is more efficient (but less readable) to
    #                 # compute this higher up in the loop nest
    #                 row = raxes.offset(p_.target_path, p_.target_exprs)
    #                 col = caxes.offset(q_.target_path, q_.target_exprs)
    #                 prealloc_mat.setValue(row, col, 666)
    prealloc_mat.assemble()

    # Now build the matrix from this preallocator
    sizes = (raxes.size, caxes.size)
    comm = single_valued([raxes.comm, caxes.comm])
    mat = PETSc.Mat().createAIJ(sizes, comm=comm)
    mat.preallocateWithMatPreallocator(prealloc_mat.mat)
    mat.assemble()
    return mat

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

    def eager_zero(self):
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

            # breakpoint()

            if indexed_raxes.alloc_size() == 0 or indexed_caxes.alloc_size() == 0:
                continue
            router_loops = indexed_raxes.outer_loops
            couter_loops = indexed_caxes.outer_loops

            rmap = HierarchicalArray(
                indexed_raxes,
                target_paths=indexed_raxes.target_paths,
                index_exprs=indexed_raxes.index_exprs,
                # is this right?
                # outer_loops=(),
                outer_loops=router_loops,
                dtype=IntType,
            )
            cmap = HierarchicalArray(
                indexed_caxes,
                target_paths=indexed_caxes.target_paths,
                index_exprs=indexed_caxes.index_exprs,
                # outer_loops=(),
                outer_loops=couter_loops,
                dtype=IntType,
            )

            from pyop3.axtree.layout import my_product

            for idxs in my_product(router_loops):
                indices = {
                    idx.index.id: (idx.source_exprs, idx.target_exprs) for idx in idxs
                }
                for p in indexed_raxes.iter(idxs):
                    offset = self.raxes.offset(p.target_exprs, p.target_path)
                    rmap.set_value(p.source_exprs | indices, offset, p.source_path)

            for idxs in my_product(couter_loops):
                indices = {
                    idx.index.id: (idx.source_exprs, idx.target_exprs) for idx in idxs
                }
                for p in indexed_caxes.iter(idxs):
                    offset = self.caxes.offset(p.target_exprs, p.target_path)
                    cmap.set_value(p.source_exprs | indices, offset, p.source_path)

            shape = (indexed_raxes.size, indexed_caxes.size)
            # breakpoint()
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
                # TODO ordered set?
                outer_loops=router_loops
                + tuple(filter(lambda l: l not in router_loops, couter_loops)),
                name=self.name,
            )
        return ContextSensitiveMultiArray(arrays)

    @cached_property
    def datamap(self):
        return freeze({self.name: self})

    @property
    def kernel_dtype(self):
        raise NotImplementedError("opaque type?")


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
        # None is for the global size, PETSc will determine it
        mat.setSizes(((raxes.size, None), (caxes.size, None)))

        # ah, is the problem here???
        if comm.size > 1:
            raise NotImplementedError

        # rlgmap = PETSc.LGMap().create(raxes.root.global_numbering(), comm=comm)
        # clgmap = PETSc.LGMap().create(caxes.root.global_numbering(), comm=comm)
        rlgmap = np.arange(raxes.size, dtype=IntType)
        clgmap = np.arange(raxes.size, dtype=IntType)
        rlgmap = PETSc.LGMap().create(rlgmap, comm=comm)
        clgmap = PETSc.LGMap().create(clgmap, comm=comm)
        mat.setLGMap(rlgmap, clgmap)

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

    # this one is tough because the temporary can have wacky shape
    # do_loop(
    #     p := points.index(),
    #     prealloc_mat[p, adjacency(p)].assign(666),
    # )
    do_loop(
        p := points.index(),
        loop(
            q := adjacency(p).index(),
            prealloc_mat[p, q].assign(666),
        ),
    )
    prealloc_mat.assemble()

    # Now build the matrix from this preallocator

    # None is for the global size, PETSc will determine it
    # sizes = ((raxes.owned.size, None), (caxes.owned.size, None))
    sizes = ((raxes.size, None), (caxes.size, None))
    # breakpoint()
    comm = single_valued([raxes.comm, caxes.comm])
    mat = PETSc.Mat().createAIJ(sizes, comm=comm)
    mat.preallocateWithMatPreallocator(prealloc_mat.mat)

    if comm.size > 1:
        raise NotImplementedError
    rlgmap = np.arange(raxes.size, dtype=IntType)
    clgmap = np.arange(raxes.size, dtype=IntType)
    # rlgmap = PETSc.LGMap().create(raxes.root.global_numbering(), comm=comm)
    # clgmap = PETSc.LGMap().create(caxes.root.global_numbering(), comm=comm)
    rlgmap = PETSc.LGMap().create(rlgmap, comm=comm)
    clgmap = PETSc.LGMap().create(clgmap, comm=comm)

    mat.setLGMap(rlgmap, clgmap)
    mat.assemble()

    # from PyOP2
    mat.setOption(mat.Option.NEW_NONZERO_LOCATION_ERR, True)
    mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)

    return mat

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
from pyop3.dtypes import IntType, ScalarType
from pyop3.itree.tree import CalledMap, LoopIndex, _index_axes, as_index_forest
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


class PetscMat(PetscObject, abc.ABC):
    DEFAULT_MAT_TYPE = MatType.AIJ

    prefix = "mat"

    def __new__(cls, *args, **kwargs):
        mat_type_str = kwargs.pop("mat_type", cls.DEFAULT_MAT_TYPE)
        mat_type = MatType(mat_type_str)

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

    def assemble(self):
        self.mat.assemble()


class MonolithicPetscMat(PetscMat, abc.ABC):
    def __getitem__(self, indices):
        # TODO also support context-free (see MultiArray.__getitem__)
        if len(indices) != 2:
            raise ValueError

        rindex, cindex = indices

        # Build the flattened row and column maps
        rloop_index = rindex
        while isinstance(rloop_index, CalledMap):
            rloop_index = rloop_index.from_index
        assert isinstance(rloop_index, LoopIndex)

        # build the map
        riterset = rloop_index.iterset
        my_raxes = self.raxes[rindex]
        rmap_axes = PartialAxisTree(riterset.parent_to_children)
        if len(rmap_axes.leaves) > 1:
            raise NotImplementedError
        for leaf in rmap_axes.leaves:
            # TODO the leaves correspond to the paths/contexts, cleanup
            # FIXME just do this for now since we only have one leaf
            axes_to_add = just_one(my_raxes.context_map.values())
            rmap_axes = rmap_axes.add_subtree(axes_to_add, *leaf)
        rmap_axes = rmap_axes.set_up()
        rmap = HierarchicalArray(rmap_axes, dtype=IntType)

        for p in riterset.iter(loop_index=rloop_index):
            for q in rindex.iter({p}):
                for q_ in (
                    self.raxes[q.index]
                    .with_context(p.loop_context | q.loop_context)
                    .iter({q})
                ):
                    # leaf_axis = rmap_axes.child(*rmap_axes._node_from_path(p.source_path))
                    # leaf_clabel = str(q.target_path)
                    # path = p.source_path | {leaf_axis.label: leaf_clabel}
                    # path = p.source_path | q_.target_path
                    path = p.source_path | q.source_path | q_.source_path
                    # indices = p.source_exprs | {leaf_axis.label: next(counters[q_.target_path])}
                    indices = p.source_exprs | q.source_exprs | q_.source_exprs
                    offset = self.raxes.offset(
                        q_.target_path, q_.target_exprs, insert_zeros=True
                    )
                    rmap.set_value(path, indices, offset)

        # FIXME being extremely lazy, rmap and cmap are NOT THE SAME
        cmap = rmap

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
        rcforest = {}
        for rctx, rtree in as_index_forest(rindex, axes=self.raxes).items():
            for cctx, ctree in as_index_forest(cindex, axes=self.caxes).items():
                # skip if the row and column contexts are incompatible
                for idx, path in cctx.items():
                    if idx in rctx and rctx[idx] != path:
                        continue
                rcforest[rctx | cctx] = (rtree, ctree)

        arrays = {}
        for ctx, (rtree, ctree) in rcforest.items():
            indexed_raxes = _index_axes(rtree, ctx, self.raxes)
            indexed_caxes = _index_axes(ctree, ctx, self.caxes)

            packed = PackedPetscMat(self, rmap, cmap)

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
                domain_index_exprs=indexed_axes.domain_index_exprs,
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
    def __init__(self, mat, rmap, cmap):
        super().__init__(mat)
        self.rmap = rmap
        self.cmap = cmap

    @property
    def mat(self):
        return self.array

    @cached_property
    def datamap(self):
        return self.mat.datamap | self.rmap.datamap | self.cmap.datamap


class PetscMatAIJ(MonolithicPetscMat):
    def __init__(self, points, adjacency, raxes, caxes, *, name: str = None):
        raxes = as_axis_tree(raxes)
        caxes = as_axis_tree(caxes)
        mat = _alloc_mat(points, adjacency, raxes, caxes)

        # TODO this is quite ugly
        # axes = PartialAxisTree(raxes.parent_to_children)
        # for leaf_axis, leaf_cpt in raxes.leaves:
        #     axes = axes.add_subtree(caxes, leaf_axis, leaf_cpt, uniquify=True)
        # breakpoint()

        super().__init__(name)

        self.mat = mat
        self.raxes = raxes
        self.caxes = caxes
        # self.axes = axes

    @property
    @deprecated("mat")
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


class PetscMatNest(PetscMat):
    ...


class PetscMatDense(PetscMat):
    ...


class PetscMatPython(PetscMat):
    ...


# TODO cache this function and return a copy if possible
# TODO is there a better name? It does a bit more than allocate
def _alloc_mat(points, adjacency, raxes, caxes, bsize=None):
    if bsize is not None:
        raise NotImplementedError

    comm = single_valued([raxes.comm, caxes.comm])

    # sizes = (raxes.leaf_component.count, caxes.leaf_component.count)
    # nnz = sparsity.axes.leaf_component.count
    sizes = (raxes.size, caxes.size)

    # 1. Determine the nonzero pattern by filling a preallocator matrix
    prealloc_mat = PETSc.Mat().create(comm)
    prealloc_mat.setType(PETSc.Mat.Type.PREALLOCATOR)
    prealloc_mat.setSizes(sizes)
    prealloc_mat.setUp()

    for p in points.iter():
        for q in adjacency(p.index).iter({p}):
            for p_ in raxes[p.index, :].with_context(p.loop_context).iter({p}):
                for q_ in (
                    caxes[q.index, :]
                    .with_context(p.loop_context | q.loop_context)
                    .iter({q})
                ):
                    # NOTE: It is more efficient (but less readable) to
                    # compute this higher up in the loop nest
                    row = raxes.offset(p_.target_path, p_.target_exprs)
                    col = caxes.offset(q_.target_path, q_.target_exprs)
                    prealloc_mat.setValue(row, col, 666)

    prealloc_mat.assemble()

    mat = PETSc.Mat().createAIJ(sizes, comm=comm)
    mat.preallocateWithMatPreallocator(prealloc_mat)
    mat.assemble()
    return mat

    mat.view()

    raise NotImplementedError

    ###

    # NOTE: A lot of this code is very similar to op3.transforms.compress
    # In fact, it is almost exactly identical and the outputs are the same!
    # The only difference, I think, is that one produces a big array
    # whereas the other produces a map. This needs some more thought.
    # ---
    # I think it might be fair to say that a sparsity and adjacency maps are
    # completely equivalent to each other. Constructing the indices explicitly
    # isn't actually very helpful.

    # currently unused
    # inc_lpy_kernel = lp.make_kernel(
    #     "{ [i]: 0 <= i < 1 }",
    #     "x[i] = x[i] + 1",
    #     [lp.GlobalArg("x", shape=(1,), dtype=utils.IntType)],
    #     name="inc",
    #     target=op3.ir.LOOPY_TARGET,
    #     lang_version=op3.ir.LOOPY_LANG_VERSION,
    # )
    # inc_kernel = op3.Function(inc_lpy_kernel, [op3.INC])

    iterset = mesh.points.as_tree()

    # prepare nonzero arrays
    sizess = {}
    for leaf_axis, leaf_clabel in iterset.leaves:
        iterset_path = iterset.path(leaf_axis, leaf_clabel)

        # bit unpleasant to have to create a loop index for this
        sizes = {}
        index = iterset.index()
        cf_map = adjacency(index).with_context({index.id: iterset_path})
        for target_path in cf_map.leaf_target_paths:
            if iterset.depth != 1:
                # TODO For now we assume iterset to have depth 1
                raise NotImplementedError
            # The axes of the size array correspond only to the specific
            # components selected from iterset by iterset_path.
            clabels = (op3.utils.just_one(iterset_path.values()),)
            subiterset = iterset[clabels]

            # subiterset is an axis tree with depth 1, we only want the axis
            assert subiterset.depth == 1
            subiterset = subiterset.root

            sizes[target_path] = op3.HierarchicalArray(
                subiterset, dtype=utils.IntType, prefix="nnz"
            )
        sizess[iterset_path] = sizes
    sizess = freeze(sizess)

    # count nonzeros
    # TODO Currently a Python loop because nnz is context sensitive and things get
    # confusing. I think context sensitivity might be better not tied to a loop index.
    # op3.do_loop(
    #     p := mesh.points.index(),
    #     op3.loop(
    #         q := adjacency(p).index(),
    #         inc_kernel(nnz[p])  # TODO would be nice to support __setitem__ for this
    #     ),
    # )
    for p in iterset.iter():
        counter = collections.defaultdict(lambda: 0)
        for q in adjacency(p.index).iter({p}):
            counter[q.target_path] += 1

        for target_path, npoints in counter.items():
            nnz = sizess[p.source_path][target_path]
            nnz.set_value(p.source_path, p.source_exprs, npoints)

    # now populate the sparsity
    # unused
    # set_lpy_kernel = lp.make_kernel(
    #     "{ [i]: 0 <= i < 1 }",
    #     "y[i] = x[i]",
    #     [lp.GlobalArg("x", shape=(1,), dtype=utils.IntType),
    #      lp.GlobalArg("y", shape=(1,), dtype=utils.IntType)],
    #     name="set",
    #     target=op3.ir.LOOPY_TARGET,
    #     lang_version=op3.ir.LOOPY_LANG_VERSION,
    # )
    # set_kernel = op3.Function(set_lpy_kernel, [op3.READ, op3.WRITE])

    # prepare sparsity, note that this is different to how we produce the maps since
    # the result is a single array
    subaxes = {}
    for iterset_path, sizes in sizess.items():
        axlabel, clabel = op3.utils.just_one(iterset_path.items())
        assert axlabel == mesh.name
        subaxes[clabel] = op3.Axis(
            [
                op3.AxisComponent(nnz, label=str(target_path))
                for target_path, nnz in sizes.items()
            ],
            "inner",
        )
    sparsity_axes = op3.AxisTree.from_nest(
        {mesh.points.copy(numbering=None, sf=None): subaxes}
    )
    sparsity = op3.HierarchicalArray(
        sparsity_axes, dtype=utils.IntType, prefix="sparsity"
    )

    # The following works if I define .enumerate() (needs to be a counter, not
    # just a loop index).
    # op3.do_loop(
    #     p := mesh.points.index(),
    #     op3.loop(
    #         q := adjacency(p).enumerate(),
    #         set_kernel(q, indices[p, q.i])
    #     ),
    # )
    for p in iterset.iter():
        # this is needed because a simple enumerate cannot distinguish between
        # different labels
        counters = collections.defaultdict(itertools.count)
        for q in adjacency(p.index).iter({p}):
            leaf_axis = sparsity.axes.child(
                *sparsity.axes._node_from_path(p.source_path)
            )
            leaf_clabel = str(q.target_path)
            path = p.source_path | {leaf_axis.label: leaf_clabel}
            indices = p.source_exprs | {leaf_axis.label: next(counters[q.target_path])}
            # we expect maps to only output a single target index
            q_value = op3.utils.just_one(q.target_exprs.values())
            sparsity.set_value(path, indices, q_value)

    return sparsity

    ###

    # 2. Create the actual matrix to use

    # 3. Insert zeros

    if bsize is None:
        mat = PETSc.Mat().createAIJ(sizes, nnz=nnz.data, comm=comm)
    else:
        mat = PETSc.Mat().createBAIJ(sizes, bsize, nnz=nnz.data, comm=comm)

    # fill with zeros (this should be cached)
    # this could be done as a pyop3 loop (if we get ragged local working) or
    # explicitly in cython
    raxis = raxes.leaf_axis
    caxis = caxes.leaf_axis
    rcpt = raxes.leaf_component
    ccpt = caxes.leaf_component

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

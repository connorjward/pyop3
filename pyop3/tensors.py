import functools
import numpy as np
import operator
import abc
import itertools
import collections
import dataclasses
from typing import Tuple, Union, Any, Optional, Sequence
import numbers
import pyrsistent

import pymbolic as pym

import pytools
import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple, checked_zip, Tree, NameGenerator


@dataclasses.dataclass(frozen=True)
class Node:
    value: Any
    children: Sequence = ()


# TODO expunge pyop3.utils.Tree in favour of subdims here
class Dim(pytools.ImmutableRecord):
    fields = {"sizes", "permutation", "labels"}

    _label_generator = NameGenerator("dim")

    def __init__(self, sizes=(), *, permutation=None, labels=None):
        if not isinstance(sizes, collections.abc.Sequence):
            sizes = (sizes,)
        if not labels:
            labels = tuple(self._label_generator.next() for _ in sizes)

        assert len(labels) == len(sizes)

        self.sizes = sizes
        self.permutation = permutation
        self.labels = labels
        super().__init__()

    def __bool__(self):
        assert len(self.sizes) == len(self.labels)
        return bool(self.sizes)

    # Could I tie sizes and subdims together?

    @property
    def strata(self):
        return tuple(range(len(self.sizes)))

    @property
    def size(self):
        return pytools.single_valued(self.sizes)

    @property
    def offsets(self):
        return tuple(sum(self.sizes[:i]) for i in range(len(self.sizes)))


# TODO replace `within` with `LoopIndex`
# TODO delete `id`
class Index(pytools.ImmutableRecord, abc.ABC):
    """Does it make sense to index a tensor with this object?"""
    fields = {"label", "within", "id"}

    _id_generator = NameGenerator("idx")

    def __init__(self, label, within=False, *, id=None):
        # self.size = size
        self.label = label
        self.within = within
        self.id = id or self._id_generator.next()
        # if id == "idx3":
        #     import pdb; pdb.set_trace()
        super().__init__()

    # @property
    # @abc.abstractmethod
    # def size(self):
    #     ...


class ScalarIndex(Index):
    """Not a fancy index (scalar-valued)"""


class FancyIndex(Index):
    """Name inspired from numpy. This allows you to slice something with a
    list of indices."""


class Slice(FancyIndex):
    fields = FancyIndex.fields | {"size", "start", "step", "offset"}

    def __init__(self, size, start=0, step=1, offset=0, **kwargs):
        self.size = size
        if isinstance(size, Tensor):
            assert size.indices is not None
        self.start = start
        self.step = step
        self.offset = offset
        super().__init__(**kwargs)

    @classmethod
    def from_dim(cls, dim, subdim_id, *, parent_indices=None, **kwargs):
        # size = cls._as_pym_var(dim.sizes[subdim_id])
        # index size with the right indices
        if isinstance(size := dim.sizes[subdim_id], pym.primitives.Expression):
            if not isinstance(size, Tensor):
                raise NotImplementedError
            if size.indices is None:
                raise NotImplementedError
                size = size[StencilGroup([Stencil([parent_indices[-size.order:]])])]
        label = dim.labels[subdim_id]
        offset = dim.offsets[subdim_id]
        return cls(size=size, label=label, offset=offset, **kwargs)

    # @functools.singledispatch
    @staticmethod
    def _as_pym_var(param):
        if pym.primitives.is_valid_operand(param):
            return param
        elif isinstance(param, Tensor):
            return pym.var(param.name)
        else:
            raise TypeError

    # @_as_pym_var.register
    # def _(param: "Tensor"):
    #     return pym.var(param.name)


# class Stencil(tuple):
#     pass
#
#
# class StencilGroup(tuple):
#     def __mul__(self, other):
#         """Do some interleaving magic - needed for matrices."""
#         if isinstance(other, StencilGroup):
#             return StencilGroup(
#                 Stencil(
#                     tuple(
#                         idx for pair in itertools.zip_longest(idxs1, idxs2)
#                         for idx in pair if idx is not None
#                     )
#                     for idxs1, idxs2 in itertools.product(stcl1, stcl2)
#                 )
#                 for stcl1, stcl2 in itertools.product(self, other)
#             )
#         else:
#             return super().__mul__(other)


class NonAffineMap(Index):
    fields = Index.fields | {"tensor"}

    # TODO is this ever not valid?
    offset = 0

    def __init__(self, tensor, **kwargs):
        self.tensor = tensor
        if "label" in kwargs:
            assert kwargs["label"] == self.tensor.linear_indices[-1].label
        else:
            kwargs["label"] = self.tensor.linear_indices[-1].label
        super().__init__(**kwargs)

    @property
    def map(self):
        return self.tensor

    # @property
    # def arity(self):
    #     dims = self.tensor.dim
    #     dim = dims.root
    #     while subdim := dims.get_child(dim):
    #         dim = subdim
    #     return dim.size

    # @property
    # def start(self):
    #     return 0
    #
    # @property
    # def stop(self):
    #     return self.arity
    #
    # @property
    # def step(self):
    #     return 1



class Tensor(pym.primitives.Variable, pytools.ImmutableRecordWithoutPickling):

    fields = {"dim", "indices", "dtype", "mesh", "name", "data", "max_value"}

    name_generator = pyop3.utils.MultiNameGenerator()
    prefix = "ten"

    # def __new__(cls, dim=None, stencils=frozenset(), **kwargs):
    #     # FIXME what is the right way to think about this now?
    #     # if (len(dims) == 2 and isinstance(dims[0], IndexedDimension) and not isinstance(dims[1], IndexedDimension)
    #     if False:
    #         return NonAffineMap(dims, **kwargs)
    #     else:
    #         return super().__new__(cls)

    def __init__(self, dim=None, indices=None, dtype=None, *, mesh = None, name: str = None, prefix: str=None, data=None, max_value=32):
        name = name or self.name_generator.next(prefix or self.prefix)
        self.data = data
        self.params = {}
        self._param_namer = NameGenerator(f"{name}_p")
        assert dtype is not None

        self.dim = dim
        # if not self._is_valid_indices(indices, dim.root):
        assert self._is_valid_indices(indices, dim.root, dim)
        self.indices = indices # self._parse_indices(dim.root, indices)
        # self.linear_indices
        # import pdb; pdb.set_trace()

        self.mesh = mesh
        self.dtype = dtype
        self.max_value = max_value
        super().__init__(name)

    @classmethod
    def new(cls, dim=None, indices=None, *args, **kwargs):
        indices = cls._parse_indices(dim.root, dim, indices)
        return cls(dim, indices, *args, **kwargs)

    def __getitem__(self, indices):
        """The plan of action here is as follows:

        - if a tensor is indexed by a set of stencils then that's great.
        - if it is indexed by a set of slices and integers then we convert
          that to a set of stencils.
        - if passed a combination of stencil groups and integers/slices then
          the integers/slices are first converted to stencil groups and then
          the groups are concatenated, multiplying where required.

        N.B. for matrices, we want to take a tensor product of the stencil groups
        rather than concatenate them. For example:

        mat[f(p), g(q)]

        where f(p) is (0, map[i]) and g(q) is (0, map[j]) would produce the wrong
        thing because we would get mat[0, map[i], 0, map[j]] where what we really
        want is mat[0, 0, map[i], map[j]]. Therefore we instead write:

        mat[f(p)*g(q)]

        to get the correct behaviour.
        """

        # TODO Add support for already indexed items
        # This is complicated because additional indices should theoretically index
        # pre-existing slices, rather than get appended/prepended as is currently
        # assumed.
        if self.is_indexed:
            raise NotImplementedError("Needs more thought")

        # if stencils == "fill":
        #     stencils = StencilGroup([
        #         Stencil([
        #             _construct_indices((), self.dim, self.dim.root)
        #         ])
        #     ])
            # import pdb; pdb.set_trace()
        # elif not isinstance(stencils, StencilGroup):
        #     empty = StencilGroup([Stencil([()])])
        #     stencils = functools.reduce(self._merge_stencils, as_tuple(stencils), empty)
        # else:
        #
        #     # import pdb; pdb.set_trace()
        #
        #     # fill final indices with full slices
        #     stencils = StencilGroup([
        #         Stencil([
        #             _construct_indices(indices, self.dim, self.dim.root)
        #             for indices in stencil
        #         ])
        #         for stencil in stencils
        #     ])

        # import pdb; pdb.set_trace()
        indices = self._parse_indices(self.dim.root, self.dim, indices)
        return self.copy(indices=indices)

    @classmethod
    def _is_valid_indices(cls, indices, dim, dtree):
        if dim.sizes and not indices:
            return False

        # scalar case
        if not dim.sizes and not indices:
            return True

        for idx, children in indices:
            assert idx.label in dim.labels

            subdim_id = dim.labels.index(idx.label)
            if subdims := dtree.get_children(dim):
                subdim = subdims[subdim_id]
                if not cls._is_valid_indices(children, subdim, dtree):
                    return False
        return True

    def __str__(self):
        return self.name

    @property
    def is_indexed(self):
        return self._check_indexed(self.dim.root, self.indices)

    def _check_indexed(self, dim, indices):
        for label, size in zip(dim.labels, dim.sizes):
            try:
                (index, subindices), = [(idx, subidxs) for idx, subidxs in indices if idx.label == label]

                subdim_id = dim.labels.index(index.label)

                if subdims := self.dim.get_children(dim):
                    subdim = subdims[subdim_id]
                    return self._check_indexed(subdim, subindices)
                else:
                    return index.size != size
            except:
                return True

    @classmethod
    def _parse_indices(cls, dim, dtree, indices, parent_indices=None):
        # import pdb; pdb.set_trace()
        if not parent_indices:
            parent_indices = []

        # TODO firm up the idea of an empty dim/index
        if not indices or indices == [[]]:
            if dim.sizes:
                indices = [(Slice.from_dim(dim, i), []) for i, _ in enumerate(dim.sizes)]
            else:
                return None

        # import pdb; pdb.set_trace()
        new_indices = []
        for idx, subidxs in indices:
            if isinstance(idx, NonAffineMap):
                new_indices.append((idx, subidxs))
            else:
                # reindex dim.size s.t. it references the correct parent indices
                if isinstance(idx.size, pym.primitives.Expression):
                    if not isinstance(idx.size, Tensor):
                        raise NotImplementedError
                    myidxs = []
                    for myidx in reversed(parent_indices[-idx.size.order:]):
                        if not myidxs:
                            myidxs = [myidx, myidxs]
                        else:
                            myidxs = [myidx, [myidxs]]
                    # import pdb; pdb.set_trace()
                    idx = idx.copy(size=idx.size[[myidxs]])

                subdim_id = dim.labels.index(idx.label)
                if subdims := dtree.get_children(dim):
                    subdim = subdims[subdim_id]
                    new_indices.append((idx, cls._parse_indices(subdim, dtree, subidxs, parent_indices+[idx])))
                else:
                    new_indices.append((idx, subidxs))

        return new_indices

    @property
    def linear_indices(self):
        try:
            idxs, = self.linear_indicess
            return idxs
        except ValueError:
            raise RuntimeError

    @property
    def linear_indicess(self):
        # import pdb; pdb.set_trace()
        if not self.indices:
            return [[]]
        return [val for item in self.indices for val in self._linearize(item)]

    def _linearize(self, item):
        # import pdb; pdb.set_trace()
        value, children = item

        if children:
            return [[value] + result for child in children for result in self._linearize(child)]
        else:
            return [[value]]

    @property
    def indexed_shape(self):
        try:
            sh, = self.indexed_shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def indexed_shapes(self):
        return indexed_shapes(self)

    @property
    def indexed_size(self):
        return functools.reduce(operator.mul, self.indexed_shape, 1)

    @property
    def shape(self):
        try:
            sh, = self.shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def shapes(self):
        return self._compute_shapes(self.dim.root)

    @property
    def size(self):
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def order(self):
        return self._compute_order(self.dim.root)

    def _parametrise_if_needed(self, value):
        if isinstance(value, Tensor):
            if (param := pym.var(value.name)) in self.params:
                assert self.params[param] == value
            else:
                self.params[param] = value
            return param
        else:
            return value

    def _compute_order(self, dim):
        subdims = self.dim.get_children(dim)
        ords = {self._compute_order(subdim) for subdim in subdims}

        if len(ords) == 0:
            return 1
        elif len(ords) == 1:
            return 1 + ords.pop()
        if len(ords) > 1:
            raise Exception("tensor order cannot be established (subdims are different depths)")

    def _merge_stencils(self, stencils1, stencils2):
        return _merge_stencils(stencils1, stencils2, self.dim)

    def _compute_shapes(self, dim):
        # import pdb; pdb.set_trace()
        if not dim:
            return ((),)

        if subdims := self.dim.get_children(dim):
            return tuple(
                (dim.size, *sh) for subdim in subdims for sh in self._compute_shape(subdim)
            )
        else:
            return ((dim.size,),)


def indexed_shapes(tensor):
    if not tensor.indices:
        return ((),)
    return tuple(v for item in tensor.indices for v in _compute_indexed_shape(item))


def _compute_indexed_shape(item):
    # if not indices:
    #     return ()
    #
    # index, *subindices = indices
    #
    # return index_shape(index) + _compute_indexed_shape(subindices)
    # import pdb; pdb.set_trace()
    # if not dim:
    #     return ((),)
    index, children = item

    if children:
        return tuple(
            index_shape(index) + sh for child in children for sh in _compute_indexed_shape(child)
        )
    else:
        return (index_shape(index),)


def _compute_indexed_shape2(flat_indices):
    shape = ()
    for index in flat_indices:
        shape += index_shape(index)
    return shape


@functools.singledispatch
def index_shape(index):
    raise TypeError

@index_shape.register(Slice)
def _(index):
    # import pdb; pdb.set_trace()
    if index.within:
        return ()
    return (index.size,)

@index_shape.register(NonAffineMap)
def _(index):
    if index.within:
        return ()
    else:
        return indexed_shape(index.tensor)


def _merge_stencils(stencils1, stencils2, dims):
    stencils1 = as_stencil_group(stencils1, dims)
    stencils2 = as_stencil_group(stencils2, dims)

    return StencilGroup(
        Stencil(
            idxs1+idxs2
            for idxs1, idxs2 in itertools.product(stc1, stc2)
        )
        for stc1, stc2 in itertools.product(stencils1, stencils2)
    )

def as_stencil_group(stencils, dims):
    if isinstance(stencils, StencilGroup):
        return stencils

    is_sequence = lambda seq: isinstance(seq, collections.abc.Sequence)
    # case 1: dat[x]
    if not is_sequence(stencils):
        return StencilGroup([
            Stencil([
                _construct_indices([stencils], dims, dims.root)
            ])
        ])
    # case 2: dat[x, y]
    elif not is_sequence(stencils[0]):
        return StencilGroup([
            Stencil([
                _construct_indices(stencils, dims, dims.root)
            ])
        ])
    # case 3: dat[(a, b), (c, d)]
    elif not is_sequence(stencils[0][0]):
        return StencilGroup([
            Stencil([
                _construct_indices(idxs, dims, dims.root)
                for idxs in stencils
            ])
        ])
    # case 4: dat[((a, b), (c, d)), ((e, f), (g, h))]
    elif not is_sequence(stencils[0][0][0]):
        return StencilGroup([
            Stencil([
                _construct_indices(idxs, dims, dims.root)
                for idxs in stencil
            ])
            for stencil in stencils
        ])
    # default
    else:
        raise ValueError


def _construct_indices(input_indices, dims, current_dim, parent_indices=None):
    if not parent_indices:
        parent_indices = []
    # import pdb; pdb.set_trace()
    if not current_dim:
        return ()

    if not input_indices:
        if len(dims.get_children(current_dim)) > 1:
            raise RuntimeError("Ambiguous subdim_id")
        input_indices = [Slice.from_dim(current_dim, 0, parent_indices=parent_indices)]

    index, *subindices = input_indices

    subdim_id = current_dim.labels.index(index.label)

    if subdims := dims.get_children(current_dim):
        subdim = subdims[subdim_id]
    else:
        subdim = None

    return (index,) + _construct_indices(subindices, dims, subdim, parent_indices + [index.copy(within=True)])



def index(indices):
    """wrap all slices and maps in loop index objs."""
    return tuple(_myindex(index) for index in indices)


def _myindex(item):
    value, children = item

    return value.copy(within=True), tuple(_myindex(child) for child in children)


def _break_mixed_slices(stencils, dtree):
    return tuple(
        tuple(idxs
            for indices in stencil
            for idxs in _break_mixed_slices_per_indices(indices, dtree)
        )
        for stencil in stencils
    )


def _break_mixed_slices_per_indices(indices, dtree):
    """Every slice over a mixed dim should branch the indices."""
    if not indices:
        yield ()
    else:
        index, *subindices = indices
        for i, idx in _partition_slice(index, dtree):
            subtree = dtree.children[i]
            for subidxs in _break_mixed_slices_per_indices(subindices, subtree):
                yield (idx, *subidxs)


"""
so I like it if we could go dat[:mesh.ncells, 2] to access the right part of the mixed dim. How to do multiple stencils?

dat[(), ()] for a single stencil
or dat[((), ()), ((), ())] for stencils

also dat[:] should return multiple indicess (for each part of the mixed dim) (but same stencil/temp)

how to handle dat[2:mesh.ncells] with raggedness? Global offset required.

N.B. dim tree no longer used for codegen - can probably get removed. Though a tree still needed since dims should
be independent of their children.

What about LoopIndex?

N.B. it is really difficult to support partial indexing (inc. integers) because, for ragged tensors, the initial offset
is non-trivial and needs to be computed by summing all preceding entries.
"""


def _partition_slice(slice_, dtree):
    if isinstance(slice_, slice):
        ptr = 0
        for i, child in enumerate(dtree.children):
            dsize = child.value.size
            dstart = ptr
            dstop = ptr + dsize

            # check for overlap
            if ((slice_.stop is None or dstart < slice_.stop)
                    and (slice_.start is None or dstop > slice_.start)):
                start = max(dstart, slice_.start) if slice_.start is not None else dstart
                stop = min(dstop, slice_.stop) if slice_.stop is not None else dstop
                yield i, slice(start, stop)
            ptr += dsize
    else:
        yield 0, slice_


class Map(FancyIndex, abc.ABC):
    fields = FancyIndex.fields | {"from_index", "arity", "name"}

    _name_generator = NameGenerator("map")

    def __init__(self, from_index, dim, arity, *, name=None, **kwargs):
        if not name:
            name = self._name_generator.next()

        self.from_index = from_index
        # self.arity = arity
        self.name = name
        super().__init__(dim=dim, **kwargs)

    @property
    def index(self):
        return LoopIndex(self)

    # @property
    # def size(self):
    #     return self.arity


class IndexFunction(Map):
    """The idea here is that we provide an expression, say, "2*x0 + x1 - 3"
    and then use pymbolic maps to replace the xN with the correct inames for the
    outer domains. We could also possibly use pN (or pym.var subclass called Parameter)
    to describe parameters."""
    def __init__(self, expr):
        self.expr = expr


class AffineMap(Map):
    pass




class Section:
    def __init__(self, dofs):
        # dofs is a list of dofs per stratum in the mesh (for now)
        self.dofs = dofs


def Global(*, name: str = None):
    return Tensor(name=name, prefix="glob")


def Dat(mesh, dofs: Section, *, prefix="dat", **kwargs) -> Tensor:
    dims = mesh.dim_tree.copy()
    for i, _ in enumerate(dims.root.sizes):
        dims = dims.add_child(dims.root, Dim(dofs.dofs[i]))
    return Tensor(dims, mesh=mesh, prefix=prefix, **kwargs)


def VectorDat(mesh, dofs, count, **kwargs):
    dim = MixedDim(
        mesh.tdim,
        tuple(
            UniformDim(
                mesh.strata_sizes[stratum],
                UniformDim(dofs.dofs[stratum], UniformDim(count))
            )
            for stratum in range(mesh.tdim)
        )
    )
    return Tensor(dim, **kwargs)


def ExtrudedDat(mesh, dofs, **kwargs):
    # dim = MixedDim(
    #     2,
    #     (
    #         UniformDim(  # base edges
    #             mesh.strata_sizes[0],
    #             MixedDim(
    #                 2,
    #                 (
    #                     UniformDim(mesh.layer_count),  # extr cells
    #                     UniformDim(mesh.layer_count),  # extr 'inner' edges
    #                 )
    #             )
    #         ),
    #         UniformDim(  # base verts
    #             mesh.strata_sizes[1],
    #             MixedDim(
    #                 2,
    #                 (
    #                     UniformDim(mesh.layer_count),  # extr 'outer' edges
    #                     UniformDim(mesh.layer_count),  # extr verts
    #                 )
    #             )
    #         )
    #     )
    # )
    # dim = mesh.dim.copy(children=(
    #     mesh.dim.children[0].copy(children=(
    #         
    #     ))
    # ))
    # TODO Actually attach a section to it.
    return Tensor(mesh.dim_tree, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    raise NotImplementedError

import functools
import abc
import itertools
import collections
import dataclasses
from typing import Tuple, Union, Any, Optional, Sequence
import numbers
import pyrsistent

import pymbolic.primitives as pym

import pytools
import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple, checked_zip, Tree, NameGenerator


class Dim(pytools.ImmutableRecord):
    fields = {"size", "name"}
    name_generator = NameGenerator("dim")

    def __init__(self, size, *, name=None):
        if not name:
            name = self.name_generator.next()

        self.size = size
        self.name = name
        super().__init__()


class UniformDim(Dim):
    pass


class MixedDim(Dim):
    pass


class Index(pytools.ImmutableRecord):
    """Does it make sense to index a tensor with this object?"""
    fields = {"dim", "within"}

    def __init__(self, dim, within=False):
        self.dim = dim
        self.within = within
        super().__init__()


class ScalarIndex(Index):
    """Not a fancy index (scalar-valued)"""


class FancyIndex(Index):
    """Name inspired from numpy. This allows you to slice something with a
    list of indices."""


class Slice(FancyIndex):
    fields = FancyIndex.fields | {"start", "stop", "step"}


    def __init__(self, dim, start, stop, step, within=False):
        # start, stop, step = None, None, None
        # if len(args) == 0:
        #     pass
        # elif len(args) == 1:
        #     stop, = args
        # elif len(args) == 2:
        #     start, stop = args
        # elif len(args) == 3:
        #     start, stop, step = args
        # else:
        #     raise ValueError

        self.start = start
        self.stop = stop
        self.step = step
        super().__init__(dim, within)
        self.map = Tensor(name="slice_map")

    @property
    def size(self):
        return self.dim.size


class Stencil(tuple):
    pass


class StencilGroup(tuple):
    def __mul__(self, other):
        """Do some interleaving magic - needed for matrices."""
        if isinstance(other, StencilGroup):
            return StencilGroup(
                Stencil(
                    tuple(
                        idx for pair in itertools.zip_longest(idxs1, idxs2)
                        for idx in pair if idx is not None
                    )
                    for idxs1, idxs2 in itertools.product(stcl1, stcl2)
                )
                for stcl1, stcl2 in itertools.product(self, other)
            )
        else:
            return super().__mul__(other)


def indexed_shape(stencil):
    size = 0
    for indices in stencil:
        size += indexed_size_per_index_group(indices)
    return (size,)


def indexed_size_per_index_group(indices):
    index, *subindices = indices
    if subindices:
        return index_size(index) * indexed_size_per_index_group(subindices)
    else:
        return index_size(index)


class Tensor(pytools.ImmutableRecordWithoutPickling):

    name_generator = pyop3.utils.MultiNameGenerator()
    prefix = "ten"

    def __new__(cls, dim=None, stencils=frozenset(), **kwargs):
        # FIXME what is the right way to think about this now?
        # if (len(dims) == 2 and isinstance(dims[0], IndexedDimension) and not isinstance(dims[1], IndexedDimension)
        if False:
            return NonAffineMap(dims, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, dim=None, stencils=None, *, mesh = None, name: str = None, prefix: str=None):
        name = name or self.name_generator.generate(prefix or self.prefix)
        super().__init__(dim=dim, stencils=stencils, mesh=mesh, name=name)

    def __getitem__(self, stencils):
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
        if self.stencils:
            raise NotImplementedError("Needs more thought")

        if not isinstance(stencils, StencilGroup):
            empty = StencilGroup([Stencil([()])])
            stencils = functools.reduce(self._merge_stencils, as_tuple(stencils), empty)

        # fill final indices with full slices
        stencils = StencilGroup([
            Stencil([
                _construct_indices(indices, self.dim, self.dim.root)
                for indices in stencil
            ])
            for stencil in stencils
        ])

        return self.copy(stencils=stencils)


    def __str__(self):
        return self.name

    def _merge_stencils(self, stencils1, stencils2):
        return _merge_stencils(stencils1, stencils2, self.dim)


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


def _construct_indices(input_indices, dims, current_dim):
    if not current_dim:
        return ()

    if not input_indices:
        input_indices = [slice(0, current_dim.size, 1)]

    index, *subindices = input_indices

    if isinstance(current_dim, MixedDim):
        # assert isinstance(index, numbers.Integral)
        if isinstance(index, numbers.Integral):
            subdim = dims.get_children(current_dim)[index]
        else:
            subdim = dims.get_children(current_dim)[index.start]
    else:
        subdim = dims.get_child(current_dim)

    if isinstance(index, Index):
        new_index = index
    elif isinstance(index, slice):
        new_index = Slice(current_dim, index.start, index.stop, index.step)
    elif isinstance(index, numbers.Integral):
        new_index = Slice(current_dim, index, index+1, 1)
    else:
        raise NotImplementedError

    return (new_index,) + _construct_indices(subindices, dims, subdim)



def index(stencils):
    """wrap all slices and maps in loop index objs."""
    return StencilGroup([
        Stencil([
            tuple(index.copy(within=True) for index in indices)
            for indices in stencil
        ])
        for stencil in stencils
    ])


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


def _calc_size(dtree):
    dim = dtree.value

    if isinstance(dim, MixedDim):
        raise NotImplementedError

    if isinstance(dim.size, Tensor):
        raise NotImplementedError
    else:
        if dtree.is_leaf:
            return dim.size
        else:
            return dim.size * _calc_size(dtree.child)


class Map(FancyIndex, abc.ABC):
    fields = FancyIndex.fields | {"from_index", "arity", "name"}

    _name_generator = NameGenerator("map")

    def __init__(self, from_index, dim, arity, *, name=None, **kwargs):
        if not name:
            name = self._name_generator.next()

        self.from_index = from_index
        self.arity = arity
        self.name = name
        super().__init__(dim=dim, **kwargs)

    @property
    def index(self):
        return LoopIndex(self)

    @property
    def size(self):
        return self.arity


class NonAffineMap(Index):
    start = 0
    stop = None
    step = 1
    def __init__(self, tensor, dim):
        self.tensor = tensor
        super().__init__(dim)

    @property
    def map(self):
        return self.tensor

    @property
    def arity(self):
        dims = self.tensor.dim
        return dims.get_child(dims.root).size


class AffineMap(Map):
    pass


@functools.singledispatch
def index_size(index):
    raise TypeError


# @index_size.register(IntIndex)
# @index_size.register(LoopIndex)
# def _(index):
#     return 1


@index_size.register
def _(index: Slice):
    if index.within:
        return 1
    start = index.start or 0
    stop = index.stop or index.dim.size
    return stop - start


@index_size.register
def _(index: NonAffineMap):
    # FIXME
    # This doesn't quite work - need to have indexed the map beforehand (different to offset tensors)
    return index.arity  # * indexed_size_per_index_group(index.tensor.stencils...)


def full_dim_size(dim, dtree):
    if child := dtree.get_child(dim):
        return dim.size * full_dim_size(child, dtree)
    else:
        return dim.size


class Section:
    def __init__(self, dofs):
        # dofs is a list of dofs per stratum in the mesh (for now)
        self.dofs = dofs


def Global(*, name: str = None):
    return Tensor(name=name, prefix="glob")


def Dat(mesh, dofs: Section, *, prefix="dat", **kwargs) -> Tensor:
    dims = mesh.dim_tree.copy()
    for i, child in enumerate(mesh.dim_tree.get_children(mesh.dim_tree.root)):
        dims = dims.add_child(child, UniformDim(dofs.dofs[i]))
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

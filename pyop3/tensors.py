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
    fields = {"dim"}

    def __init__(self, dim):
        self.dim = dim
        super().__init__()


class BasicIndex(Index):
    """Not a fancy index (scalar-valued)"""
    size = 1


class LoopIndex(BasicIndex):
    fields = BasicIndex.fields | {"domain"}

    def __init__(self, domain):
        self.domain = domain
        super().__init__(domain.dim)


class FancyIndex(Index):
    """Name inspired from numpy. This allows you to slice something with a
    list of indices."""

    @property
    def index(self) -> LoopIndex:
        return LoopIndex(self)


class IntIndex(BasicIndex):
    fields = BasicIndex.fields | {"value"}

    def __init__(self, value):
        self.value = value
        super().__init__()


class Slice(FancyIndex):
    fields = FancyIndex.fields | {"start", "stop", "step"}

    def __init__(self, dim, *args):
        start, stop, step = None, None, None
        if len(args) == 0:
            pass
        elif len(args) == 1:
            stop, = args
        elif len(args) == 2:
            start, stop = args
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise ValueError

        self.start = start
        self.stop = stop
        self.step = step
        super().__init__(dim)

    @property
    def size(self):
        return self.dim.size


class Range(FancyIndex):
    fields = FancyIndex.fields | {"start", "stop", "step"}

    def __init__(self, dim, *args):
        if len(args) == 1:
            start, stop, step = 0, *args, 1
        elif len(args) == 2:
            start, stop, step = *args, 1
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise ValueError

        self.start = start
        self.stop = stop
        self.step = step
        super().__init__(dim)

    @property
    def size(self):
        if self.step != 1:
            raise NotImplementedError
        return self.dim.size
        return self.stop - self.start


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


class StencilGroup(pytools.ImmutableRecord):
    def __init__(self, stencils, mesh=None):
        super().__init__(stencils=stencils, mesh=mesh)

    def __iter__(self):
        return iter(self.stencils)

    @property
    def index(self):
        new_stencils = set()
        for stencil in self.stencils:
            new_stencil = []
            for indices in stencil:
                new_indices = []
                for index in indices:
                    if isinstance(index, FancyIndex):
                        index = index.index
                    new_indices.append(index)
                new_stencil.append(tuple(new_indices))
            new_stencils.add(tuple(new_stencil))
        new_stencils = frozenset(new_stencils)
        return self.copy(stencils=new_stencils)


# To avoid annoying errors for now
class IndexTree:
    ...


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

    def __init__(self, dim=None, stencils=frozenset(), *, name: str = None, prefix: str=None):
        name = name or self.name_generator.generate(prefix or self.prefix)
        super().__init__(dim=dim, stencils=stencils, name=name)

    def __getitem__(self, stencils):
        # TODO Add support for already indexed items
        if self.stencils:
            raise NotImplementedError("Needs more thought")

        stencils = _extract_stencils(stencils)
        stencils = _break_mixed_slices(stencils, self.dim)

        return self.copy(stencils=stencils)

    def __str__(self):
        return self.name


def _extract_stencils(stencils):
    is_sequence = lambda seq: isinstance(seq, collections.abc.Sequence)

    # case 1: dat[x]
    if not is_sequence(stencils):
        return (((stencils,),),)
    # case 2: dat[x, y]
    elif not is_sequence(stencils[0]):
        return ((stencils,),)
    # case 3: dat[(a, b), (c, d)]
    elif not is_sequence(stencils[0][0]):
        return (stencils,)
    # case 4: dat[((a, b), (c, d)), ((e, f), (g, h))]
    elif not is_sequence(stencils[0][0][0]):
        return stencils
    # default
    else:
        raise ValueError


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

# TODO I should calculate the global offset here...
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


def _get_child(tree, item):
    if tree.value == item:
        try:
            return tree.child
        except ValueError:
            return None
    else:
        for child in tree.children:
            if (res := _get_child(child, item)) is not None:
                return res
        return None


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


# class NonAffineMap(Tensor, Map):
class NonAffineMap(Map):
    pass


class AffineMap(Map):
    pass


@functools.singledispatch
def index_size(index):
    raise TypeError


@index_size.register(IntIndex)
@index_size.register(LoopIndex)
def _(index):
    return 1


@index_size.register
def _(slice_: Slice):
    start = slice_.start or 0
    stop = slice_.stop or slice_.dim.size
    return stop - start


@index_size.register
def _(map_: Map):
    return index_size(map_.from_index) * map_.arity


class Section:
    def __init__(self, dofs):
        # dofs is a list of dofs per stratum in the mesh (for now)
        self.dofs = dofs


def Global(*, name: str = None):
    return Tensor(name=name, prefix="glob")


def Dat(mesh, dofs: Section, *, prefix="dat", **kwargs) -> Tensor:
    new_children = []
    for i, subtree in enumerate(mesh.dim_tree.children):
        new_subchildren = Tree(UniformDim(dofs.dofs[i]))
        new_children.append(subtree.copy(children=new_subchildren))
    new_children = tuple(new_children)
    dim_tree = mesh.dim_tree.copy(children=new_children)
    return Tensor(dim_tree, prefix=prefix, **kwargs)


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

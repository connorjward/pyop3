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
from pyop3.utils import as_tuple, checked_zip


class Dim(pytools.ImmutableRecord):
    size: int

    @property
    @abc.abstractmethod
    def is_leaf(self):
        pass


class UniformDim(Dim):
    def __init__(self, size, subdim=None):
        super().__init__(size=size, subdim=subdim)

    @property
    def is_leaf(self):
        return not self.subdim


class MixedDim(Dim):
    def __init__(self, size, subdims=None):
        super().__init__(size=size, subdims=subdims)

    @property
    def is_leaf(self):
        return not self.subdims


class Index(abc.ABC):
    """Does it make sense to index a tensor with this object?"""


class BasicIndex(Index):
    """Not a fancy index (scalar-valued)"""


class ScalarIndex(BasicIndex):
    def __init__(self, value):
        self.value = value


class LoopIndex(BasicIndex):
    def __init__(self, domain):
        self.domain = domain


class FancyIndex(Index, abc.ABC):
    """Name inspired from numpy. This allows you to slice something with a
    list of indices."""

    @property
    @abc.abstractmethod
    def index(self) -> ScalarIndex:
        pass


class PythonicIndex:
    def __init__(self, value):
        self.value = value


class IntIndex(PythonicIndex):
    ...


class Slice(PythonicIndex, FancyIndex):
    def __init__(self, *args, mesh=None):
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

    @property
    def index(self):
        return LoopIndex(self)


def indexed_shape(dim, stencil):
    size = 0
    for indices in stencil:
        size += indexed_size_per_index_group(dim, indices)
    return (size,)


def indexed_size_per_index_group(dim, indices):
    if not dim:
        raise AssertionError

    if not indices:
        indices = (Slice(),)

    index, *subindices = indices
    if isinstance(dim, MixedDim):
        if dim.subdims:
            subdim = dim.subdims[index.value]
        else:
            subdim = None
    else:
        subdim = dim.subdim

    size = index_size(index, dim)
    if subdim:
        return size * indexed_size_per_index_group(subdim, subindices)
    else:
        return size


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

    name_generator = pyop3.utils.UniqueNameGenerator()
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

        # convert indices to a tuple of tuples
        # if isinstance(indices, collections.abc.Sequence):
        #     if all(isinstance(idx, collections.abc.Sequence) for idx in indices):
        #         pass
        #     else:
        #         indices = (indices,)
        # else:
        #     indices = ((indices,),)

        # now set index types
        # def to_index(index):
        #     if isinstance(index, Index):
        #         return index
        #     elif isinstance(index, numbers.Integral):
        #         return ScalarIndex(index)
        #     else:
        #         raise ValueError

        # indices = tuple(tuple(to_index(idx) for idx in idxs) for idxs in indices)

        return self.copy(stencils=stencils)

    def __str__(self):
        return self.name


class Map(FancyIndex, abc.ABC):

    @property
    def index(self):
        return self.to_dim.index


class NonAffineMap(Tensor, Map):

    prefix = "map"

    def __init__(self, *dims):
        self.from_index, self.to_dim = dims
        super().__init__(dims, name="map")


class AffineMap(Map):
    def __init__(self, from_index, arity):
        self.from_index = from_index
        self.arity = arity
        self.to_dim = UniformDim(arity)


@functools.singledispatch
def index_size(index, dim):
    raise TypeError


@index_size.register(IntIndex)
@index_size.register(LoopIndex)
def _(slice_, dim):
    return 1


@index_size.register
def _(slice_: Slice, dim):
    start = slice_.start or 0
    stop = slice_.stop or dim.size
    return stop - start


@index_size.register
def _(map_: Map, dim):
    return index_size(map_.from_index, dim) * map_.to_dim.size


class Section:
    def __init__(self, dofs):
        # dofs is a list of dofs per stratum in the mesh (for now)
        self.dofs = dofs


def Global(*, name: str = None):
    return Tensor(name=name, prefix="glob")


def Dat(mesh, dofs: Section, *, prefix="dat", **kwargs) -> Tensor:
    dim = MixedDim(
        mesh.tdim,
        tuple(
            UniformDim(mesh.strata_sizes[stratum], UniformDim(dofs.dofs[stratum]))
            for stratum in range(mesh.tdim)
        )
    )
    return Tensor(dim, prefix=prefix, **kwargs)


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
    dim = MixedDim(
        2,
        (
            UniformDim(  # base edges
                mesh.strata_sizes[0],
                MixedDim(
                    2,
                    (
                        UniformDim(mesh.layer_count),  # extr cells
                        UniformDim(mesh.layer_count),  # extr 'inner' edges
                    )
                )
            ),
            UniformDim(  # base verts
                mesh.strata_sizes[1],
                MixedDim(
                    2,
                    (
                        UniformDim(mesh.layer_count),  # extr 'outer' edges
                        UniformDim(mesh.layer_count),  # extr verts
                    )
                )
            )
        )
    )
    return Tensor(dim, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    raise NotImplementedError

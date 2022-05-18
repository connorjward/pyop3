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
from pyop3.utils import as_tuple


class Dim(pytools.ImmutableRecord):
    def __init__(self, size, children=None):
        if isinstance(size, numbers.Integral) and isinstance(children, collections.abc.Sequence) and len(children) != size:
            raise ValueError
        super().__init__(size=size, children=children)

    @property
    def has_single_child_dim_type(self):
        return isinstance(self.children, Dim)

    @property
    def child(self):
        if not self.has_single_child_dim_type:
            raise TypeError
        return self.children


# class FixedSizeDim(Dim):
#     """Dimension whose size is known at compile-time (e.g. a mesh's topological dimension or something mixed).
#
#     Such tensor dimensions can have variable children types.
#     """
#     def __init__(self, size: numbers.Integral, children: Union[Dim, Sequence]):
#         # this class can have either a tuple of children or a single child
#         if isinstance(children, collections.abc.Sequence) and len(children) != size:
#             raise ValueError
#         super().__init__(size, children)


# class VariableDim(Dim):
#     """Dimension whose size is unknown at compile-time (e.g. # cells in a mesh)."""
#     def __init__(self, size, children: Dim):
#         super().__init__(size, children)


class Index(abc.ABC):
    """Does it make sense to index a tensor with this object?"""


class BasicIndex(Index):
    """Not a fancy index (scalar-valued)"""
    size = 1


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
    size = 1

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
    def value(self):
        return slice(self.start, self.stop, self.step)

    @property
    def index(self):
        raise Exception("This doesn't make sense. Can only do this over a range (with extent)")


class Range(PythonicIndex, FancyIndex):
    def __init__(self, *args, mesh=None):
        if len(args) == 1:
            start, stop, step = 0, *args, 1
        elif len(args) == 2:
            start, stop, step = *args, 1
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise ValueError

        if isinstance(start, str):
            start = pym.Variable(start)
        if isinstance(stop, str):
            stop = pym.Variable(stop)

        self.start = start
        self.stop = stop
        self.step = step

    @property
    def size(self):
        return (self.stop - self.start) // self.step

    @property
    def value(self):
        return slice(self.start, self.stop, self.step)

    @property
    def index(self):
        return LoopIndex(self)


class IndexTree:
    def __init__(self, indices, mesh=None):
        indices = self._replace_integers(indices)
        indices = pyrsistent.pmap(indices)

        # super().__init__(indices=indices, mesh=mesh, size=size)
        self.indices = indices
        self.mesh = mesh

    def compute_size(self, dim):
        size = 0
        for index, subtree in self.indices.items():
            if not subtree and dim.children:
                if dim.has_single_child_dim_type:
                    size += index.size * dim.child.size
                else:
                    for child in dim.children:
                        size += index.size * child.size
            else:
                if isinstance(index, PythonicIndex):
                    if isinstance(index, Slice):
                        subdims = dim.children[index.value]
                    else:
                        subdims = (dim.children[index.value],)
                else:
                    subdims = dim.children

                for subdim in subdims:
                    size += index.size * subtree.compute_size(subdim)
        return size

    @property
    def index(self):
        new_indices = {}
        for idx, itree in self.indices.items():
            if itree:
                new_indices[idx] = itree.index
            elif isinstance(idx, FancyIndex):
                new_indices[LoopIndex(idx)] = None
            else:
                new_indices[idx] = IndexTree({Slice(): None})
        return type(self)(indices=new_indices, mesh=self.mesh)

    @staticmethod
    def _replace_integers(indices):
        new_indices = {}
        for idx, itree in indices.items():
            if isinstance(idx, numbers.Number):
                new_indices[IntIndex(idx)] = itree
            else:
                new_indices[idx] = itree
        return new_indices


 # class IndexTreeBag(frozenset):
#     """Collection of IndexTuples."""
#     def __init__(self, args):
#         if not all(isinstance(arg, IndexTree) for arg in args):
#             raise ValueError
#
#     @property
#     def index(self):
#         if not any(index_tree.is_fancy for index_tree in self):
#             raise TypeError("All already scalar indices")
#
#         return 
#         new_args = []
#         for arg in self:
#             try:
#                 new_args.append(arg.index)
#             except TypeError:
#                 new_args.append(arg)
#         return type(self)(new_args)
#
#     @staticmethod
#     def _replace_fancy_index_tree(index_tree):
#         return index_tree.index if index, FancyIndex) else index


def index(indices):
    # if not any(isinstance(idx, FancyIndex) for idx in itertools.chain(*indices)):
    #     raise ValueError("Already scalar")

    def replace_with_scalar_index(index):
        if isinstance(index, FancyIndex):
            return index.index
        elif isinstance(index, slice):
            return ChildIndex(slice)
        else:
            return index

    return tuple(tuple(map(replace_with_scalar_index, idxs)) for idxs in indices)


class Tensor(pytools.ImmutableRecordWithoutPickling):

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "ten"

    def __new__(cls, dim=None, indices=(), **kwargs):
        # FIXME what is the right way to think about this now?
        # if (len(dims) == 2 and isinstance(dims[0], IndexedDimension) and not isinstance(dims[1], IndexedDimension)
        if False:
            return NonAffineMap(dims, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, dim=None, indices=(), *, name: str = None, prefix: str=None):
        name = name or self.name_generator.generate(prefix or self.prefix)
        super().__init__(dim=dim, indices=indices, name=name)

    def __getitem__(self, indices):
        # TODO Add support for already indexed items
        if self.indices:
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

        return self.copy(indices=indices)

    def __str__(self):
        if self.indices:
            return f"{self.name}[{','.join(str(idx) for idx in self.indices)}]"
        else:
            return self.name

    @property
    def broadcast_domains(self):
        domains = set()
        for dim in self.dims:
            domains |= self._get_broadcast_domains(dim)
        return frozenset(domains)

    @classmethod
    def _get_broadcast_domains(cls, dim):
        if isinstance(dim, Index):
            return frozenset()
        elif isinstance(dim, Slice):
            return frozenset({dim})
        elif isinstance(dim, Map):
            return frozenset({dim.to_space}) | cls._get_broadcast_domains(dim.from_space)
        else:
            raise AssertionError

    @property
    def orig_shape(self):
        shape = []
        for dim in self.dims:
            space = dim.space if isinstance(dim, Index) else dim
            while isinstance(space, Map):
                space = space.from_space

            # this is because a maps space can either be an index or slice
            space = space.space if isinstance(space, Index) else space

            # FIXME This requires some thought about bin-ops
            # shape.append(space.size)
            shape.append(space.stop)
        return tuple(shape)

    # @property
    # def domain(self):
    #     try:
    #         (dom,) = self.shape
    #         return dom
    #     except ValueError:
    #         raise TypeError

    @property
    def order(self):
        return len(self.spaces)

    @property
    def is_scalar(self):
        return self.order == 0

    @property
    def is_vector(self):
        return self.order == 1


class Map(FancyIndex, abc.ABC):

    @property
    def index(self):
        return self.to_dim.index


class NonAffineMap(Tensor, Map):

    prefix = "map"

    def __init__(self, *dims):
        self.from_dim, self.to_dim = dims
        super().__init__(dims, name="map")

    @property
    def size(self):
        return self.from_dim.size * self.to_dim.size


class AffineMap(Map):
    def __init__(self, from_dim, offsets=1, strides=1):
        self.from_dim = from_dim
        self.offsets = as_tuple(offsets)
        self.strides = as_tuple(strides)

    @property
    def arity(self):
        return pytools.single_valued([len(self.offsets), len(self.strides)])

    # FIXME This will break but do I want to set this in super().__init__ or here?
    @property
    def from_dim(self):
        return self.from_dim

    @property
    def to_dim(self):
        return Slice(self.arity)


class Section:
    def __init__(self, dofs):
        # dofs is a list of dofs per stratum in the mesh (for now)
        self.dofs = dofs


def Global(*, name: str = None):
    return Tensor(name=name, prefix="glob")


def Dat(mesh, dofs: Section, *, prefix="dat", **kwargs) -> Tensor:
    dim = Dim(
        mesh.tdim,
        children=tuple(Dim(mesh.strata_sizes[stratum], Dim(dofs.dofs[stratum]))
                           for stratum in range(mesh.tdim))
    )
    return Tensor(dim, prefix=prefix, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(shape, name=name)

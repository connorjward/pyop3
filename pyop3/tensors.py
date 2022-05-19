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
        assert len(subdims) == size
        super().__init__(size=size, subdims=subdims)

    @property
    def is_leaf(self):
        return not self.subdims


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


def indexed_shape(dim, itree):
    if not dim:
        return ()

    if not itree:
        itree = IndexTree(Range(dim.size))

    if isinstance(dim, UniformDim):
        if itree.children:
            child, = itree.children
            shape = (itree.index.size,) + indexed_shape(dim.subdim, child)
        else:
            shape = (itree.index.size,)
    else:
        shapes = tuple(
            indexed_shape(subdim, subtree)
            for subdim, subtree in checked_zip(dim.subdims, itree.children)
            if subtree
        )

        # This is quite confusing. The idea here is that a mixed dim will produce
        # multiple children and these need to be combined. As an example, consider
        # a case where the child shapes are (1, 3), (3, 3) and (3, 3). We want the
        # final shape to be (7, 3) so we sum the first value and assert that all
        # subdim shapes are equal.
        outer = sum(shape[0] for shape in shapes)
        inner = pytools.single_valued(shape[1:] for shape in shapes)
        shape = (outer,) + inner
    return shape


class IndexTree(pytools.ImmutableRecord):
    def __init__(self, index, children=(), mesh=None):
        if isinstance(index, numbers.Integral):
            index = IntIndex(index)
        super().__init__(index=index, children=children, mesh=mesh)

    @property
    def loop_index(self):
        # if indexing a mixed object
        if self.index is None:
            new_children = []
            for child in self.children:
                if child:
                    new_children.append(child.loop_index)
                else:
                    new_children.append(None)
            new_children = tuple(new_children)
            return self.copy(children=new_children)

        if isinstance(self.index, BasicIndex) and not self.children:
            new_children = type(self)(Slice())
            return self.copy(children=new_children)

        if isinstance(self.index, FancyIndex):
            new_index = LoopIndex(self.index)
        else:
            new_index = self.index

        if self.children:
            new_children = tuple(child.loop_index for child in self.children)
        else:
            new_children = ()

        return self.copy(index=new_index, children=new_children)


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
    dim = MixedDim(
        mesh.tdim,
        tuple(UniformDim(mesh.strata_sizes[stratum], UniformDim(dofs.dofs[stratum]))
                           for stratum in range(mesh.tdim))
    )
    return Tensor(dim, prefix=prefix, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(shape, name=name)

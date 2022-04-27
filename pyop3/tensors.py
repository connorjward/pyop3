import abc
import collections
import dataclasses
from typing import Tuple, Union
import numbers

import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple


def as_domain(domain):
    if isinstance(domain, Domain):
        return domain
    elif isinstance(domain, numbers.Integral):
        return Domain(0, domain)
    else:
        raise ValueError


class Tensor:

    name_generator = pyop3.utils.UniqueNameGenerator()
    prefix = "ten"

    def __init__(self, shape=(), indices=(), *, mesh=None, name: str = None, prefix: str=None, parent=None):
        self.shape = tuple(as_domain(dom) for dom in as_tuple(shape))
        self.indices = as_tuple(indices)
        self.mesh = mesh
        self.name = name or self.name_generator.generate(prefix or self.prefix)


        self.shape_domains = frozenset(self.shape)
        self.broadcast_domains = frozenset((idx.domain for idx in self.indices if not isinstance(idx, IndexLiteral) and idx.is_vector)) | self.shape_domains

        # need to traverse to collect all domains
        def myrecursion(index):
            result = {index.domain}
            for idx in index.indices:
                result |= myrecursion(idx)
            return result

        all_domains = set()
        for index in self.indices:
            all_domains |= myrecursion(index)
        self.all_domains = frozenset(all_domains) | self.shape_domains
        self.within_domains = self.all_domains - self.broadcast_domains

        self.parent = parent

    def __getitem__(self, indices):
        # FIXME You cannot index over the top of slices since these are considered indices
        indices = as_tuple(indices)

        if len(indices) > len(self.shape):
            raise ValueError

        return self.copy(
            shape=self.shape[len(indices):],
            indices=self.indices + indices,
            parent=self
        )

    def __str__(self):
        return f"{self.name}[{','.join(str(idx) for idx in self.indices)}]"

    @property
    def orig_shape(self):
        tensor = self
        while tensor.parent:
            tensor = tensor.parent
        return tensor.shape

    @property
    def index(self):
        return self[IndexLiteral(self.shape[0], mesh=self.mesh)]

    @property
    def domain(self):
        try:
            (dom,) = self.shape
            return dom
        except ValueError:
            raise TypeError

    @property
    def order(self):
        return len(self.broadcast_domains)

    @property
    def is_scalar(self):
        return self.order == 0

    @property
    def is_vector(self):
        return self.order == 1

    def copy(self, **kwargs):
        shape = kwargs.get("shape", self.shape)
        indices = kwargs.get("indices", self.indices)
        mesh = kwargs.get("mesh", self.mesh)
        name = kwargs.get("name", self.name)
        parent = kwargs.get("parent", self.parent)
        return type(self)(shape=shape, indices=indices, mesh=mesh, name=name, parent=parent)


@dataclasses.dataclass(frozen=True)
class Domain:
    start: Tensor
    stop: Tensor

    def __len__(self):
        return self.stop - self.start

    def to_range(self):
        return Range(self.start, self.stop)


@dataclasses.dataclass(frozen=True)
class IndexLiteral:
    domain: Domain
    mesh: int=None

    indices = ()
    is_vector = False
    is_scalar = True



# this just isn't a tensor unfortunately
# range.index should return i
# whereas
# tensor.index should return tensor[i]
class Range:
    def __init__(self, *args, mesh=None):
        try:
            self.start, self.stop = args
        except ValueError:
            self.start = 0
            (self.stop,) = args
        self.shape = Domain(self.start, self.stop)
        self.domain = self.shape
        self.mesh = mesh

        self.index = IndexLiteral(self.shape, mesh=self.mesh)
        self.indices = self.index,

    is_vector = True
    is_scalar = False


def Global(*, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="global")
    return Tensor(name=name)


def Dat(shape: Tuple[int, ...], *, prefix="dat", **kwargs) -> Tensor:
    return Tensor(shape, prefix=prefix, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    if not name:
        name = Tensor.name_generator.generate(prefix="mat")
    return Tensor(shape, name=name)

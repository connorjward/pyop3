import abc
import collections
import dataclasses
import enum
import functools
import operator
from typing import Iterable, Sequence, Tuple
from weakref import WeakValueDictionary

import numpy as np
import pytools

from pyop3.distarray import DistributedArray, MultiArray
from pyop3.utils import NameGenerator, as_tuple, merge_dicts


class AccessDescriptor(enum.Enum):
    READ = "read"
    WRITE = "write"
    RW = "rw"
    INC = "inc"
    MIN = "min"
    MAX = "max"


READ = AccessDescriptor.READ
WRITE = AccessDescriptor.WRITE
INC = AccessDescriptor.INC
RW = AccessDescriptor.RW
MIN = AccessDescriptor.MIN
MAX = AccessDescriptor.MAX


class LoopExpr(pytools.ImmutableRecord, abc.ABC):
    fields = set()

    @property
    @abc.abstractmethod
    def datamap(self) -> WeakValueDictionary[str, DistributedArray]:
        """Map from names to arrays.

        weakref since we don't want to hold a reference to these things?
        """
        pass

    # nice for drawing diagrams
    # @property
    # @abc.abstractmethod
    # def operands(self) -> tuple["LoopExpr"]:
    #     pass


class Loop(LoopExpr):
    fields = LoopExpr.fields | {"indices", "statements", "id", "depends_on"}

    id_generator = NameGenerator("loop")

    def __init__(
        self, indices, statements: Sequence[LoopExpr], id=None, depends_on=frozenset()
    ):
        # FIXME
        # assert isinstance(index, pyop3.tensors.Indexed)
        if not id:
            id = self.id_generator.next()

        self.indices = indices
        self.statements = as_tuple(statements)
        self.id = id
        self.depends_on = depends_on

        super().__init__()

    @functools.cached_property
    def datamap(self):
        return merge_dicts(stmt.datamap for stmt in self.statements)

    @property
    def index(self):
        # deprecated alias for indices
        return self.indices

    def __str__(self):
        return f"for {self.index} ∊ {self.index.point_set}"

    def __call__(self, **kwargs):
        from pyop3.codegen.loopexpr2loopy import compile
        from pyop3.codegen.loopy2exe import compile_loopy

        kernel = compile(self)
        exe = compile_loopy(kernel)

        args = [
            _as_pointer(self.datamap[arg.name])
            for arg in kernel.default_entrypoint.args
        ]

        # TODO parse kwargs

        exe(*args)


@dataclasses.dataclass(frozen=True)
class ArgumentSpec:
    access: AccessDescriptor
    dtype: np.dtype
    space: Tuple[int]


@dataclasses.dataclass(frozen=True)
class FunctionArgument:
    tensor: "IndexedTensor"
    spec: ArgumentSpec

    # alias

    @property
    def name(self):
        return self.tensor.name

    @property
    def access(self) -> AccessDescriptor:
        return self.spec.access

    @property
    def dtype(self):
        # assert self.tensor.dtype == self.spec.dtype
        return self.spec.dtype

    @property
    def indices(self):
        return self.tensor.indices


class LoopyKernel:
    """A callable function."""

    def __init__(self, loopy_kernel, access_descrs):
        self.code = loopy_kernel
        self._access_descrs = access_descrs

    def __call__(self, *args):
        if len(args) != len(self.argspec):
            raise ValueError(
                f"Wrong number of arguments provided, expected {len(self.argspec)} "
                f"but received {len(args)}"
            )
        return FunctionCall(self, args)

    @property
    def argspec(self):
        return tuple(
            ArgumentSpec(access, arg.dtype, arg.shape)
            for access, arg in zip(
                self._access_descrs, self.code.default_entrypoint.args
            )
        )

    @property
    def name(self):
        return self.code.default_entrypoint.name


class Terminal(LoopExpr):
    pass


class FunctionCall(Terminal):
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

    @functools.cached_property
    def datamap(self) -> dict[str, DistributedArray]:
        return functools.reduce(
            operator.or_, [arg.datamap for arg in self.arguments], {}
        )

    # @property
    # def operands(self):
    #     ...

    @property
    def name(self):
        return self.function.name

    @property
    def argspec(self):
        return self.function.argspec

    @property
    def inputs(self):
        return tuple(
            arg
            for arg, spec in zip(self.arguments, self.argspec)
            if spec.access == AccessDescriptor.READ
        )

    @property
    def outputs(self):
        return tuple(
            arg
            for arg, spec in zip(self.arguments, self.argspec)
            if spec.access in {AccessDescriptor.WRITE, AccessDescriptor.INC}
        )

    @property
    def output_specs(self):
        return tuple(
            filter(
                lambda spec: spec.access
                in {AccessDescriptor.WRITE, AccessDescriptor.INC},
                self.argspec,
            )
        )


def loop(*args, **kwargs):
    return Loop(*args, **kwargs)


def do_loop(*args, **kwargs):
    loop(*args, **kwargs)()


@functools.singledispatch
def _as_pointer(array: DistributedArray) -> int:
    raise NotImplementedError


@_as_pointer.register
def _(array: MultiArray):
    return array.data.ctypes.data

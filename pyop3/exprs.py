import abc
import collections
import dataclasses
import enum
import functools
from typing import Iterable, Tuple

import numpy as np
import pytools

# import pyop3.tensors
from pyop3.utils import NameGenerator, as_tuple


class Expr:
    pass


class Operator(Expr):
    pass


class Loop(pytools.ImmutableRecord, Operator):
    fields = {"indices", "statements", "id", "depends_on"}

    id_generator = NameGenerator("loop")

    def __init__(self, indices, statements, id=None, depends_on=frozenset()):
        # FIXME
        # assert isinstance(index, pyop3.tensors.Indexed)
        if not id:
            id = self.id_generator.next()

        self.indices = indices
        self.statements = as_tuple(statements)
        self.id = id
        self.depends_on = depends_on

        super().__init__()

    @property
    def index(self):
        # deprecated alias for indices
        return self.indices

    @property
    def unrolled_migs(self):
        from pyop3.tensors import Map

        def unroll(mig):
            if isinstance(mig, Map):
                return tuple([*unroll(mig.from_index), mig])
            else:
                return (mig,)

        return unroll(self.indices)

    # @property
    # def flat_indices(self):
    #     def flatten(item):
    #         value, children = item
    #         return (value,) + tuple(val for child in children for val in flatten(child))
    #
    #     return tuple(v for it in self.index for v in flatten(it))

    def __str__(self):
        return f"for {self.index} ∊ {self.index.point_set}"

    def __call__(self, *args, **kwargs):
        from pyop3.codegen.loopy import to_c

        code = to_c(self)

        breakpoint()
        from pyop2.compilation import load

        exe = load(code)
        exe()


class AccessDescriptor(enum.Enum):
    READ = enum.auto()
    WRITE = enum.auto()
    INC = enum.auto()
    RW = enum.auto()
    MIN = enum.auto()
    MAX = enum.auto()


READ = AccessDescriptor.READ
WRITE = AccessDescriptor.WRITE
INC = AccessDescriptor.INC
RW = AccessDescriptor.RW
MIN = AccessDescriptor.MIN
MAX = AccessDescriptor.MAX


@dataclasses.dataclass(frozen=True)
class ArgumentSpec:
    access: AccessDescriptor
    dtype: np.dtype
    space: Tuple[int]


@dataclasses.dataclass(frozen=True)
class FunctionArgument:
    tensor: "IndexedTensor"
    spec: ArgumentSpec

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
        return FunctionCall(
            self,
            tuple(
                FunctionArgument(tensor, spec)
                for tensor, spec in zip(args, self.argspec)
            ),
        )

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


class Terminal(Expr):
    pass


class FunctionCall(Terminal):
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

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

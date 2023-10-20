from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import operator
from typing import Iterable, Sequence, Tuple
from weakref import WeakValueDictionary

import loopy as lp
import numpy as np
import pymbolic as pym
import pytools
from pyrsistent import freeze, pmap

from pyop3.axes import as_axis_tree
from pyop3.axes.tree import (
    ContextFree,
    ContextSensitive,
    FrozenAxisTree,
    MultiArrayCollector,
)
from pyop3.distarray import DistributedArray, MultiArray, PetscMat
from pyop3.distarray.multiarray import IndexExpressionReplacer, substitute_layouts
from pyop3.dtypes import IntType
from pyop3.indices.tree import (
    IndexedAxisTree,
    _compose_bits,
    _index_axes,
    as_index_forest,
)
from pyop3.utils import as_tuple, checked_zip, merge_dicts


# TODO I don't think that this belongs in this file, it belongs to the kernel?
# create a kernel.py file?
class Access(enum.Enum):
    READ = "read"
    WRITE = "write"
    RW = "rw"
    INC = "inc"
    MIN_WRITE = "min_write"
    MIN_RW = "min_rw"
    MAX_WRITE = "max_write"
    MAX_RW = "max_rw"


READ = Access.READ
WRITE = Access.WRITE
INC = Access.INC
RW = Access.RW
MIN_RW = Access.MIN_RW
MIN_WRITE = Access.MIN_WRITE
MAX_RW = Access.MAX_RW
MAX_WRITE = Access.MAX_WRITE


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
    fields = LoopExpr.fields | {"index", "statements", "id", "depends_on"}

    # doubt that I need an ID here
    id_generator = pytools.UniqueNameGenerator()

    def __init__(
        self,
        index: IndexTree,
        statements: Sequence[LoopExpr],
        id=None,
        depends_on=frozenset(),
    ):
        # FIXME
        # assert isinstance(index, pyop3.tensors.Indexed)
        if not id:
            id = self.id_generator("loop")

        super().__init__()

        self.index = index
        self.statements = as_tuple(statements)
        self.id = id
        # I think this can go if I generate code properly
        self.depends_on = depends_on

    # maybe these should not exist? backwards compat
    @property
    def axes(self):
        return self.index.axes

    @property
    def indices(self):
        return self.index.indices

    @functools.cached_property
    def datamap(self):
        return self.index.datamap | merge_dicts(
            stmt.datamap for stmt in self.statements
        )

    def __str__(self):
        return f"for {self.index} ∊ {self.index.point_set}"

    def __call__(self, **kwargs):
        args = [
            _as_pointer(self.datamap[arg.name])
            for arg in self.loopy_code.default_entrypoint.args
        ]

        # TODO parse kwargs
        # breakpoint()

        self.executable(*args)

    @functools.cached_property
    def loopy_code(self):
        from pyop3.codegen.ir import compile

        return compile(self)

    @functools.cached_property
    def c_code(self):
        from pyop3.codegen.exe import compile_loopy

        return compile_loopy(self.loopy_code, stop_at_c=True)

    @functools.cached_property
    def executable(self):
        from pyop3.codegen.exe import compile_loopy

        return compile_loopy(self.loopy_code)


@dataclasses.dataclass(frozen=True)
class ArgumentSpec:
    access: Intent
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
    def access(self) -> Intent:
        return self.spec.access

    @property
    def dtype(self):
        # assert self.tensor.dtype == self.spec.dtype
        return self.spec.dtype

    @property
    def indices(self):
        return self.tensor.indices


class Function:
    """A callable function."""

    def __init__(self, loopy_kernel, access_descrs):
        lpy_args = loopy_kernel.default_entrypoint.args
        if len(lpy_args) != len(access_descrs):
            raise ValueError("Wrong number of access descriptors given")
        for lpy_arg, access in zip(lpy_args, access_descrs):
            if access in {MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE} and lpy_arg.shape != (
                1,
            ):
                raise ValueError("Reduction operations are only valid for scalars")

        self.code = fix_intents(loopy_kernel, access_descrs)
        self._access_descrs = access_descrs

    def __call__(self, *args):
        if len(args) != len(self.argspec):
            raise ValueError(
                f"Wrong number of arguments provided, expected {len(self.argspec)} "
                f"but received {len(args)}"
            )
        if any(
            spec.dtype.numpy_dtype != arg.dtype
            for spec, arg in checked_zip(self.argspec, args)
        ):
            raise ValueError("Arguments to the kernel have the wrong dtype")
        return CalledFunction(self, args)

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


class CalledFunction(LoopExpr):
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

    @functools.cached_property
    def datamap(self) -> dict[str, DistributedArray]:
        return merge_dicts([arg.datamap for arg in self.arguments])

    # @property
    # def operands(self):
    #     ...

    @property
    def name(self):
        return self.function.name

    @property
    def argspec(self):
        return self.function.argspec

    # @property
    # def inputs(self):
    #     return tuple(
    #         arg
    #         for arg, spec in zip(self.arguments, self.argspec)
    #         if spec.access == READ
    #     )
    #
    # @property
    # def outputs(self):
    #     return tuple(
    #         arg
    #         for arg, spec in zip(self.arguments, self.argspec)
    #         if spec.access in {AccessDescriptor.WRITE, AccessDescriptor.INC}
    #     )

    # @property
    # def output_specs(self):
    #     return tuple(
    #         filter(
    #             lambda spec: spec.access
    #             in {AccessDescriptor.WRITE, AccessDescriptor.INC},
    #             self.argspec,
    #         )
    #     )


class Offset(LoopExpr, ContextSensitive):
    """Terminal containing the offset of some axis tree given some multi-index."""

    def __init__(self, per_context):
        LoopExpr.__init__(self)
        ContextSensitive.__init__(self, per_context)

    # FIXME
    @property
    def name(self):
        return "my_offset"

    @property
    def dtype(self):
        return IntType

    @functools.cached_property
    def datamap(self):
        return merge_dicts(axes.datamap for axes in self.context_map.values())


def offset(axes, indices):
    axes = as_axis_tree(axes).freeze()
    axes_per_context = {}
    for index_tree in as_index_forest(indices, axes=axes):
        loop_context = index_tree.loop_context
        (
            indexed_axes,
            target_path_per_indexed_cpt,
            index_exprs_per_indexed_cpt,
            layout_exprs_per_indexed_cpt,
        ) = _index_axes(axes, index_tree, loop_context)

        (
            target_paths,
            index_exprs,
            layout_exprs,
        ) = _compose_bits(
            axes,
            indexed_axes,
            target_path_per_indexed_cpt,
            index_exprs_per_indexed_cpt,
            layout_exprs_per_indexed_cpt,
        )

        new_axes = IndexedAxisTree(
            indexed_axes.root,
            indexed_axes.parent_to_children,
            target_paths,
            index_exprs,
            layout_exprs,
        )

        new_layouts = substitute_layouts(
            axes,
            new_axes,
            target_path_per_indexed_cpt,
            index_exprs_per_indexed_cpt,
        )
        layout_axes = FrozenAxisTree(
            new_axes.root,
            new_axes.parent_to_children,
            target_paths,
            index_exprs,
            new_layouts,
        )
        axes_per_context[loop_context] = layout_axes
    return Offset(axes_per_context)


class Instruction(pytools.ImmutableRecord):
    fields = set()


class Assignment(Instruction):
    fields = Instruction.fields | {"tensor", "temporary", "shape"}

    def __init__(self, tensor, temporary, shape, **kwargs):
        self.tensor = tensor
        self.temporary = temporary
        self.shape = shape
        super().__init__(**kwargs)

    # better name
    @property
    def array(self):
        return self.tensor


class Read(Assignment):
    @property
    def lhs(self):
        return self.temporary

    @property
    def rhs(self):
        return self.tensor


class Write(Assignment):
    @property
    def lhs(self):
        return self.tensor

    @property
    def rhs(self):
        return self.temporary


class Increment(Assignment):
    @property
    def lhs(self):
        return self.tensor

    @property
    def rhs(self):
        return self.temporary


class Zero(Assignment):
    @property
    def lhs(self):
        return self.temporary

    # FIXME
    @property
    def rhs(self):
        # return 0
        return self.tensor


def loop(*args, **kwargs):
    return Loop(*args, **kwargs)


def do_loop(*args, **kwargs):
    loop(*args, **kwargs)()


@functools.singledispatch
def _as_pointer(array: DistributedArray) -> int:
    raise NotImplementedError


@_as_pointer.register
def _(array: MultiArray):
    return array._data.ptr_rw


@_as_pointer.register
def _(array: PetscMat):
    return array.petscmat.handle


def fix_intents(tunit, accesses):
    """

    The local kernel has underspecified accessors (is_input, is_output).
    Here coerce them to match the access descriptors provided.

    This should arguably be done properly in TSFC.

    Note that even if this isn't done in TSFC we need to guard against this properly
    as the default error is very unclear.

    """
    kernel = tunit.default_entrypoint
    new_args = []
    for arg, access in checked_zip(kernel.args, accesses):
        assert access in {READ, WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}
        is_input = access in {READ, RW, INC, MIN_RW, MAX_RW}
        is_output = access in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_WRITE, MAX_RW}
        new_args.append(arg.copy(is_input=is_input, is_output=is_output))
    return tunit.with_kernel(kernel.copy(args=new_args))

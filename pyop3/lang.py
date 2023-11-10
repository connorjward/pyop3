from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import operator
from functools import cached_property, partial
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
from pyop3.distarray import Dat, MultiArray, PetscMat
from pyop3.distarray.multiarray import IndexExpressionReplacer, substitute_layouts
from pyop3.dtypes import IntType, dtype_limits
from pyop3.indices.tree import (
    IndexedAxisTree,
    _compose_bits,
    _index_axes,
    as_index_forest,
    partition_iterset,
)
from pyop3.utils import as_tuple, checked_zip, merge_dicts


# TODO I don't think that this belongs in this file, it belongs to the function?
# create a function.py file?
# class Intent?
class Access(enum.Enum):
    # developer note, MIN_RW and MIN_WRITE are distinct (unlike PyOP2) to avoid
    # passing "requires_zeroed_output_arguments" around, yuck

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
RW = Access.RW
INC = Access.INC
MIN_RW = Access.MIN_RW
MIN_WRITE = Access.MIN_WRITE
MAX_RW = Access.MAX_RW
MAX_WRITE = Access.MAX_WRITE


class IntentMismatchError(Exception):
    pass


class LoopExpr(pytools.ImmutableRecord, abc.ABC):
    fields = set()

    @property
    @abc.abstractmethod
    def datamap(self):
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

    @cached_property
    def datamap(self):
        return self.index.datamap | merge_dicts(
            stmt.datamap for stmt in self.statements
        )

    def __str__(self):
        return f"for {self.index} ∊ {self.index.point_set}"

    def __call__(self, **kwargs):
        if self.is_parallel:
            # interleave computation and communication
            # FIXME self.datamap.values is probably many more arrays than we care about
            icore, inoncore = partition_iterset(self.index, self.datamap.values)

            finalizers = self.prepare_arrays_begin()

            # core
            # replace the parallel axis subset with one for the specific indices here
            core_kwargs = merge_dicts(
                [kwargs, {"iparallel": icore, "psize": len(icore)}]
            )
            self._call(**core_kwargs)

            self.prepare_arrays_end(finalizers)

            # noncore
            noncore_kwargs = merge_dicts(
                [kwargs, {"iparallel": inoncore, "psize": len(icore)}]
            )
            self._call(**noncore_kwargs)

            # need to set last write op
            # also may need to eagerly assemble Mats, or be clever?
        else:
            self._call(**kwargs)

    def _call(self, **kwargs):
        """kwargs overwrite arrays stored in datamaps."""
        args = [
            _as_pointer(kwargs.get(arg.name, self.datamap[arg.name]))
            for arg in self.loopy_code.default_entrypoint.args
        ]
        self.executable(*args)

    @cached_property
    def is_parallel(self):
        return len(self._distarray_args) > 0

    def prepare_arrays_begin(self):
        # NOTE: It is safe to include reductions in the finalizers because
        # core entities (in the iterset) are defined as being those that do
        # not overlap with any points in the star forest.
        finalizers = []
        for array, intent in self._distarray_args:
            touches_ghost_points = _has_nontrivial_stencil(array)

            if intent in {READ, RW}:
                if touches_ghost_points:
                    if not array._roots_valid:
                        array.reduce_then_broadcast_begin()
                        finalizers.append(array.reduce_then_broadcast_end)
                    else:
                        array.broadcast_roots_to_leaves_begin()
                        finalizers.append(array.broadcast_roots_to_leaves_end)
                    assert array._roots_valid and array._leaves_valid
                else:
                    if not array._roots_valid:
                        array.reduce_leaves_to_roots_begin()
                        finalizers.append(array.reduce_leaves_to_roots_end)
                    assert array._roots_valid

            elif intent == WRITE:
                # Assumes that all points are written to (i.e. not a subset). If
                # this is not the case then a manual reduction is needed.
                array._leaves_valid = False
                array._pending_reduction = None

            elif intent in {INC, MIN_WRITE, MIN_RW, MAX_WRITE, MAX_RW}:  # reductions
                # We don't need to update roots if performing the same reduction
                # again. For example we can increment into an array as many times
                # as we want. The reduction only needs to be done when the
                # data is read.
                if array._roots_valid or intent == array._pending_reduction:
                    pass
                else:
                    # We assume that all points are visited, and therefore that
                    # WRITE accesses do not need to update roots. If only a subset
                    # of entities are written to then a manual reduction is required.
                    # This is the same assumption that we make for data_wo and is
                    # explained in the documentation.
                    # TODO Add this to the documentation
                    if intent in {INC, MIN_RW, MAX_RW}:
                        assert array._pending_reduction is not None
                        array.reduce_leaves_to_roots_begin()
                        finalizers.append(array.reduce_leaves_to_roots_end)

                # If ghost points are not modified then no future reduction is required
                if not touches_ghost_points:
                    array._leaves_valid = False
                    array._pending_reduction = None
                else:
                    array._leaves_valid = False
                    array._pending_reduction = intent

                    # set leaves to appropriate nil value
                    if intent == INC:
                        array._data[array.axes.sf.ileaf] = 0
                    elif intent in {MIN_WRITE, MIN_RW}:
                        array._data[array.axes.sf.ileaf] = dtype_limits(array.dtype).max
                    elif intent in {MAX_WRITE, MAX_RW}:
                        array._data[array.axes.sf.ileaf] = dtype_limits(array.dtype).min
                    else:
                        raise AssertionError

            else:
                raise AssertionError

        return tuple(finalizers)

    def prepare_arrays_end(self, finalizers):
        for f in finalizers:
            f()

    @cached_property
    def all_function_arguments(self):
        func_args = {}
        for stmt in self.statements:
            for arg, intent in stmt.all_function_arguments:
                if arg in func_args:
                    if func_args[arg] != intent:
                        # I think that it does not make sense to access arrays with
                        # different intents in the same kernel but it is always OK
                        # if the same intent is used
                        raise IntentMismatchError
                else:
                    func_args[arg] = intent
        # now sort
        return tuple(
            (arg, func_args[arg])
            for arg in sorted(func_args.keys(), key=lambda a: a.name)
        )

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

    @cached_property
    def _distarray_args(self):
        return tuple(
            (arg.array, intent)
            for arg, intent in self.all_function_arguments
            if isinstance(arg, Dat) and arg.array.is_distributed
        )


def _has_nontrivial_stencil(array):
    """

    This is a proxy for 'this array touches halo points'.

    """
    # FIXME This is WRONG, there are cases (e.g. support(extfacet)) where
    # the halo might be touched but the size (i.e. map arity) is 1. I need
    # to look at index_exprs probably.
    return array.axes.size > 1


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
    def datamap(self):
        return merge_dicts([arg.datamap for arg in self.arguments])

    @property
    def name(self):
        return self.function.name

    @property
    def argspec(self):
        return self.function.argspec

    @property
    def all_function_arguments(self):
        return tuple(
            sorted(
                [
                    (arg, intent)
                    for arg, intent in checked_zip(
                        self.arguments, self.function._access_descrs
                    )
                ],
                key=lambda a: a[0].name,
            )
        )


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
def _as_pointer(array) -> int:
    raise NotImplementedError


# bad name now, "as_kernel_arg"?
@_as_pointer.register
def _(array: int):
    return array


@_as_pointer.register
def _(array: np.ndarray):
    return array.ctypes.data


@_as_pointer.register
def _(array: MultiArray):
    return array.data.ctypes.data


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

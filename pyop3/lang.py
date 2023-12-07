from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import operator
from collections import defaultdict
from functools import cached_property, partial
from typing import Iterable, Sequence, Tuple
from weakref import WeakValueDictionary

import loopy as lp
import numpy as np
import pymbolic as pym
import pytools
from pyrsistent import freeze, pmap

from pyop3.axtree import as_axis_tree
from pyop3.axtree.tree import ContextFree, ContextSensitive, MultiArrayCollector
from pyop3.config import config
from pyop3.distarray import DistributedArray
from pyop3.dtypes import IntType, dtype_limits
from pyop3.extras.debug import print_with_rank
from pyop3.itree.tree import (
    IndexExpressionReplacer,
    _compose_bits,
    _index_axes,
    as_index_forest,
    partition_iterset,
)
from pyop3.tensor import Dat, MultiArray, PetscMat
from pyop3.tensor.dat import ContextSensitiveMultiArray
from pyop3.utils import as_tuple, checked_zip, just_one, merge_dicts, unique


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

    def __call__(self, **kwargs):
        from pyop3.ir.lower import compile

        if self.is_parallel:
            # interleave computation and communication
            new_index, (icore, iroot, ileaf) = partition_iterset(
                self.index, [a for a, _ in self.all_function_arguments]
            )

            assert self.index.id == new_index.id

            # substitute subsets into loopexpr, should maybe be done in partition_iterset
            parallel_loop = self.copy(index=new_index)
            code = compile(parallel_loop)

            # interleave communication and computation
            initializers, finalizerss = self._array_updates()

            for init in initializers:
                init()

            # replace the parallel axis subset with one for the specific indices here
            extent = just_one(icore.axes.root.components).count
            core_kwargs = merge_dicts(
                [kwargs, {icore.name: icore, extent.name: extent}]
            )
            code(**core_kwargs)

            # await reductions
            for fin in finalizerss[0]:
                fin()

            # roots
            # replace the parallel axis subset with one for the specific indices here
            root_extent = just_one(iroot.axes.root.components).count
            root_kwargs = merge_dicts(
                [kwargs, {icore.name: iroot, extent.name: root_extent}]
            )
            code(**root_kwargs)

            # await broadcasts
            for fin in finalizerss[1]:
                fin()

            # leaves
            leaf_extent = just_one(ileaf.axes.root.components).count
            leaf_kwargs = merge_dicts(
                [kwargs, {icore.name: ileaf, extent.name: leaf_extent}]
            )
            code(**leaf_kwargs)

            # also may need to eagerly assemble Mats, or be clever?
        else:
            compile(self)(**kwargs)

    @cached_property
    def loopy_code(self):
        from pyop3.ir.lower import compile

        return compile(self)

    @cached_property
    def is_parallel(self):
        return len(self._distarray_args) > 0

    @cached_property
    def all_function_arguments(self):
        # TODO overly verbose
        func_args = {}
        for stmt in self.statements:
            for arg, intent in stmt.all_function_arguments:
                if arg not in func_args:
                    func_args[arg] = intent
        # now sort
        return tuple(
            (arg, func_args[arg])
            for arg in sorted(func_args.keys(), key=lambda a: a.name)
        )

    @cached_property
    def _distarray_args(self):
        arrays = {}
        for arg, intent in self.all_function_arguments:
            if (
                not isinstance(arg.array, DistributedArray)
                or not arg.array.is_distributed
            ):
                continue
            if arg.array not in arrays:
                arrays[arg.array] = (intent, _has_nontrivial_stencil(arg))
            else:
                if arrays[arg.array][0] != intent:
                    # I think that it does not make sense to access arrays with
                    # different intents in the same kernel but that it is
                    # always OK if the same intent is used.
                    raise IntentMismatchError

                # We need to know if *any* uses of a particular array touch ghost points
                if not arrays[arg.array][1] and _has_nontrivial_stencil(arg):
                    arrays[arg.array] = (intent, True)

        # now sort
        return tuple(
            (arr, *arrays[arr]) for arr in sorted(arrays.keys(), key=lambda a: a.name)
        )

    def _array_updates(self):
        """Collect appropriate callables for updating shared values in the right order.

        Returns
        -------
        (initializers, (finalizers0, finalizers1))
            Collections of callables to be executed at the right times.

        """
        initializers = []
        finalizerss = ([], [])
        for array, intent, touches_ghost_points in self._distarray_args:
            if intent in {READ, RW}:
                if touches_ghost_points:
                    if not array._roots_valid:
                        initializers.append(array._reduce_leaves_to_roots_begin)
                        finalizerss[0].extend(
                            [
                                array._reduce_leaves_to_roots_end,
                                array._broadcast_roots_to_leaves_begin,
                            ]
                        )
                        finalizerss[1].append(array._broadcast_roots_to_leaves_end)
                    else:
                        initializers.append(array._broadcast_roots_to_leaves_begin)
                        finalizerss[1].append(array._broadcast_roots_to_leaves_end)
                else:
                    if not array._roots_valid:
                        initializers.append(array._reduce_leaves_to_roots_begin)
                        finalizerss[0].append(array._reduce_leaves_to_roots_end)

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
                    if intent in {INC, MIN_RW, MAX_RW}:
                        assert array._pending_reduction is not None
                        initializers.append(array._reduce_leaves_to_roots_begin)
                        finalizerss[0].append(array._reduce_leaves_to_roots_end)

                # We are modifying owned values so the leaves must now be wrong
                array._leaves_valid = False

                # If ghost points are not modified then no future reduction is required
                if not touches_ghost_points:
                    array._pending_reduction = None
                else:
                    array._pending_reduction = intent

                    # set leaves to appropriate nil value
                    if intent == INC:
                        array._data[array.sf.ileaf] = 0
                    elif intent in {MIN_WRITE, MIN_RW}:
                        array._data[array.sf.ileaf] = dtype_limits(array.dtype).max
                    elif intent in {MAX_WRITE, MAX_RW}:
                        array._data[array.sf.ileaf] = dtype_limits(array.dtype).min
                    else:
                        raise AssertionError

            else:
                raise AssertionError

        return initializers, finalizerss


# TODO singledispatch
def _has_nontrivial_stencil(array):
    """

    This is a proxy for 'this array touches halo points'.

    """
    # FIXME This is WRONG, there are cases (e.g. support(extfacet)) where
    # the halo might be touched but the size (i.e. map arity) is 1. I need
    # to look at index_exprs probably.
    if isinstance(array, Dat):
        return array.axes.size > 1
    elif isinstance(array, ContextSensitiveMultiArray):
        return any(_has_nontrivial_stencil(d) for d in array.context_map.values())
    else:
        raise TypeError


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

    # FIXME NEXT: Expand ContextSensitive things here
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

# TODO Rename this file insn.py - the pyop3 language is everything, not just this

from __future__ import annotations

import abc
import dataclasses
import enum
import functools
import numbers
from functools import cached_property
from typing import Iterable, Tuple

import loopy as lp
import numpy as np
import pytools
from petsc4py import PETSc

from pyop3.axtree import Axis
from pyop3.axtree.tree import ContextFree, ContextSensitive, MultiArrayCollector
from pyop3.dtypes import dtype_limits
from pyop3.utils import (
    UniqueRecord,
    OrderedSet,
    as_tuple,
    auto,
    checked_zip,
    just_one,
    merge_dicts,
    single_valued,
)


# TODO I don't think that this belongs in this file, it belongs to the function?
# create a function.py file?
class Intent(enum.Enum):
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
    NA = "na"  # TODO prefer NONE


# old alias
Access = Intent


READ = Intent.READ
WRITE = Intent.WRITE
RW = Intent.RW
INC = Intent.INC
MIN_RW = Intent.MIN_RW
MIN_WRITE = Intent.MIN_WRITE
MAX_RW = Intent.MAX_RW
MAX_WRITE = Intent.MAX_WRITE
NA = Intent.NA


# TODO: This exception is not actually ever raised. We should check the
# intents of the kernel arguments and complain if something illegal is
# happening.
class IntentMismatchError(Exception):
    pass


class KernelArgument(abc.ABC):
    """Abstract class for types that may be passed as arguments to kernels.

    Note that some types that can be passed to *functions* are not in fact
    kernel arguments. This is because they either wrap actual kernel arguments
    (e.g. `HierarchicalArray`), or because no argument is actually passed
    (e.g. a temporary).

    """

    @property
    @abc.abstractmethod
    def kernel_dtype(self):
        pass


# this is an expression, like passing an array through to a kernel
# but it is transformed first.
class Pack(KernelArgument, ContextFree):
    def __init__(self, big, small):
        self.big = big
        self.small = small

    @property
    def kernel_dtype(self):
        try:
            return single_valued([self.big.dtype, self.small.dtype])
        except ValueError:
            raise ValueError("dtypes must match")


class Instruction(UniqueRecord, abc.ABC):
    pass


class ContextAwareInstruction(Instruction):
    @property
    @abc.abstractmethod
    def datamap(self):
        """Map from names to arrays."""

    # @property
    # @abc.abstractmethod
    # def kernel_arguments(self):
    #     pass


class Loop(Instruction):
    fields = Instruction.fields | {"index", "statements"}

    # doubt that I need an ID here
    id_generator = pytools.UniqueNameGenerator()

    def __init__(
        self,
        index: LoopIndex,
        statements: Iterable[Instruction],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index = index
        self.statements = as_tuple(statements)

    def __call__(self, **kwargs):
        # TODO just parse into ContextAwareLoop and call that
        from pyop3.ir.lower import compile
        from pyop3.itree.tree import partition_iterset

        if self.is_parallel:
            # FIXME: The partitioning code does not seem to always run properly
            # so for now do all the transfers in advance.
            # interleave computation and communication
            # new_index, (icore, iroot, ileaf) = partition_iterset(
            #     self.index, [a for a, _ in self.function_arguments]
            # )
            #
            # assert self.index.id == new_index.id
            #
            # # substitute subsets into loopexpr, should maybe be done in partition_iterset
            # parallel_loop = self.copy(index=new_index)
            # code = compile(parallel_loop)
            code = compile(self)

            # interleave communication and computation
            initializers, reductions, broadcasts = self._array_updates()

            for init in initializers:
                init()

            # replace the parallel axis subset with one for the specific indices here
            # extent = just_one(icore.axes.root.components).count
            # core_kwargs = merge_dicts(
            #     [kwargs, {icore.name: icore, extent.name: extent}]
            # )
            # code(**core_kwargs)

            # await reductions
            for red in reductions:
                red()

            # roots
            # replace the parallel axis subset with one for the specific indices here
            # root_extent = just_one(iroot.axes.root.components).count
            # root_kwargs = merge_dicts(
            #     [kwargs, {icore.name: iroot, extent.name: root_extent}]
            # )
            # code(**root_kwargs)

            # await broadcasts
            for broadcast in broadcasts:
                broadcast()

            # leaves
            # leaf_extent = just_one(ileaf.axes.root.components).count
            # leaf_kwargs = merge_dicts(
            #     [kwargs, {icore.name: ileaf, extent.name: leaf_extent}]
            # )
            # code(**leaf_kwargs)

            code(**kwargs)

            # also may need to eagerly assemble Mats, or be clever and spike the accessors?
        else:
            compile(self)(**kwargs)

    @cached_property
    def loopy_code(self):
        from pyop3.ir.lower import compile

        return compile(self)

    @cached_property
    def is_parallel(self):
        from pyop3.buffer import DistributedBuffer

        for arg in self.kernel_arguments:
            if isinstance(arg, DistributedBuffer):
                if arg.is_distributed:
                    return True
            else:
                assert isinstance(arg, PETSc.Mat)
                for local_size, global_size in arg.getSizes():
                    if local_size != global_size:
                        return True
        return False
        return len(self._distarray_args) > 0

    @cached_property
    def function_arguments(self) -> tuple:
        args = {}  # ordered
        for stmt in self.statements:
            for arg, intent in stmt.function_arguments:
                args[arg] = intent
        return tuple((arg, intent) for arg, intent in args.items())

    @cached_property
    def kernel_arguments(self):
        args = OrderedSet()
        for stmt in self.statements:
            for arg in stmt.kernel_arguments:
                args.add(arg)
        return tuple(args)

    def _array_updates(self):
        """Collect appropriate callables for updating shared values in the right order.

        Returns
        -------
        (initializers, (finalizers0, finalizers1))
            Collections of callables to be executed at the right times.

        """
        from pyop3 import DistributedBuffer, HierarchicalArray, Mat
        from pyop3.array.harray import ContextSensitiveDat
        from pyop3.array.petsc import Sparsity

        initializers = []
        reductions = []
        broadcasts = []
        for arg, intent in self.function_arguments:
            if isinstance(arg, HierarchicalArray):
                buffer = arg.buffer
                if isinstance(buffer, DistributedBuffer) and buffer.is_distributed:
                    # for now assume the most conservative case
                    touches_ghost_points = True

                    inits, reds, bcasts = self._buffer_exchanges(
                        buffer, intent, touches_ghost_points=touches_ghost_points
                    )
                    initializers.extend(inits)
                    reductions.extend(reds)
                    broadcasts.extend(bcasts)
            elif isinstance(arg, ContextSensitiveDat):
                # assumed to not be distributed
                pass
            else:
                assert isinstance(arg, (Mat, Sparsity))
                # just in case
                broadcasts.append(arg.assemble)

        return initializers, reductions, broadcasts

    @staticmethod
    def _buffer_exchanges(buffer, intent, *, touches_ghost_points):
        initializers, reductions, broadcasts = [], [], []

        if intent in {READ, RW}:
            if touches_ghost_points:
                if not buffer._roots_valid:
                    initializers.append(buffer._reduce_leaves_to_roots_begin)
                    reductions.extend([
                        buffer._reduce_leaves_to_roots_end,
                        buffer._broadcast_roots_to_leaves_begin,
                    ])
                    broadcasts.append(buffer._broadcast_roots_to_leaves_end)
                else:
                    initializers.append(buffer._broadcast_roots_to_leaves_begin)
                    broadcasts.append(buffer._broadcast_roots_to_leaves_end)
            else:
                if not buffer._roots_valid:
                    initializers.append(buffer._reduce_leaves_to_roots_begin)
                    reductions.append(buffer._reduce_leaves_to_roots_end)

        elif intent == WRITE:
            # Assumes that all points are written to (i.e. not a subset). If
            # this is not the case then a manual reduction is needed.
            buffer._leaves_valid = False
            buffer._pending_reduction = None

        else:
            # reductions
            assert intent in {INC, MIN_WRITE, MIN_RW, MAX_WRITE, MAX_RW}
            # We don't need to update roots if performing the same reduction
            # again. For example we can increment into a buffer as many times
            # as we want. The reduction only needs to be done when the
            # data is read.
            if buffer._roots_valid or intent == buffer._pending_reduction:
                pass
            else:
                # We assume that all points are visited, and therefore that
                # WRITE accesses do not need to update roots. If only a subset
                # of entities are written to then a manual reduction is required.
                # This is the same assumption that we make for data_wo.
                if intent in {INC, MIN_RW, MAX_RW}:
                    assert buffer._pending_reduction is not None
                    initializers.append(buffer._reduce_leaves_to_roots_begin)
                    reductions.append(buffer._reduce_leaves_to_roots_end)

                # set leaves to appropriate nil value
                if intent == INC:
                    nil = 0
                elif intent in {MIN_WRITE, MIN_RW}:
                    nil = dtype_limits(buffer.dtype).max
                else:
                    assert intent in {MAX_WRITE, MAX_RW}
                    nil = dtype_limits(buffer.dtype).min

                def _init_nil():
                    buffer._data[buffer.sf.ileaf] = nil

                reductions.append(_init_nil)

            # We are modifying owned values so the leaves must now be wrong
            buffer._leaves_valid = False

            # If ghost points are not modified then no future reduction is required
            if not touches_ghost_points:
                buffer._pending_reduction = None
            else:
                buffer._pending_reduction = intent

        return tuple(initializers), tuple(reductions), tuple(broadcasts)


class ContextAwareLoop(ContextAwareInstruction):
    fields = Instruction.fields | {"index", "statements"}

    def __init__(self, index, statements, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.statements = statements

    @cached_property
    def datamap(self):
        return self.index.datamap | merge_dicts(
            stmt.datamap for stmts in self.statements.values() for stmt in stmts
        )

    @cached_property
    def loopy_code(self):
        from pyop3.ir.lower import compile

        return compile(self)


# TODO singledispatch
# TODO perhaps this is simply "has non unit stride"?
def _has_nontrivial_stencil(array):
    """

    This is a proxy for 'this array touches halo points'.

    """
    # FIXME This is WRONG, there are cases (e.g. support(extfacet)) where
    # the halo might be touched but the size (i.e. map arity) is 1. I need
    # to look at index_exprs probably.
    from pyop3.array import HierarchicalArray

    if isinstance(array, HierarchicalArray):
        return _has_nontrivial_stencil(array)
    else:
        raise TypeError


class Terminal(Instruction, abc.ABC):
    @cached_property
    def datamap(self):
        return merge_dicts(a.datamap for a, _ in self.function_arguments)

    @property
    @abc.abstractmethod
    def argument_shapes(self):
        pass

    @abc.abstractmethod
    def with_arguments(self, arguments: Iterable[KernelArgument]):
        pass


@dataclasses.dataclass(frozen=True)
class ArgumentSpec:
    access: Intent
    dtype: np.dtype
    space: Tuple[int]


class FunctionArgument(abc.ABC):
    """Abstract class for types that may be passed to functions."""


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
        if not all(isinstance(a, FunctionArgument) for a in args):
            raise TypeError("invalid kernel argument type")
        if len(args) != len(self.argspec):
            raise ValueError(
                f"Wrong number of arguments provided, expected {len(self.argspec)} "
                f"but received {len(args)}"
            )
        # if any(
        #     spec.dtype.numpy_dtype != arg.kernel_dtype
        #     for spec, arg in checked_zip(self.argspec, args)
        #     if arg.kernel_dtype is not auto
        # ):
        #     raise ValueError("Arguments to the kernel have the wrong dtype")
        return CalledFunction(self, args)

    @property
    def argspec(self):
        spec = []
        for access, arg in checked_zip(
            self._access_descrs, self.code.default_entrypoint.args
        ):
            shape = arg.shape if not isinstance(arg, lp.ValueArg) else ()
            spec.append(ArgumentSpec(access, arg.dtype, shape))
        return tuple(spec)

    @property
    def name(self):
        return self.code.default_entrypoint.name


class CalledFunction(Terminal):
    fields = Terminal.fields | {"function", "arguments"}

    def __init__(
        self, function: Function, arguments: Iterable[FunctionArgument], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.function = function
        self.arguments = arguments

    @property
    def name(self):
        return self.function.name

    @property
    def argspec(self):
        return self.function.argspec

    @cached_property
    def function_arguments(self):
        args = {}  # ordered
        for arg, spec in checked_zip(self.arguments, self.argspec):
            args[arg] = spec.access
        return tuple((arg, intent) for arg, intent in args.items())

    @cached_property
    def kernel_arguments(self):
        kargs = OrderedSet()
        for func_arg in self.arguments:
            for karg in _collect_kernel_arguments(func_arg):
                kargs.add(karg)
        return tuple(kargs)

    @property
    def argument_shapes(self):
        return tuple(
            arg.shape if not isinstance(arg, lp.ValueArg) else ()
            for arg in self.function.code.default_entrypoint.args
        )

    def with_arguments(self, arguments):
        return self.copy(arguments=arguments)


class Assignment(Terminal, abc.ABC):
    fields = Terminal.fields | {"assignee", "expression"}

    def __init__(self, assignee, expression, **kwargs):
        if not isinstance(expression, numbers.Number):
            assert assignee.dtype == expression.dtype

        super().__init__(**kwargs)
        self.assignee = assignee
        self.expression = expression

    def __call__(self):
        do_loop(Axis(1).index(), self)

    @property
    def arguments(self):
        # FIXME Not sure this is right for complicated expressions
        return (self.assignee, self.expression)

    @property
    def arrays(self):
        from pyop3.array import HierarchicalArray

        arrays_ = [self.assignee]
        if isinstance(self.expression, HierarchicalArray):
            arrays_.append(self.expression)
        else:
            if not isinstance(self.expression, numbers.Number):
                raise NotImplementedError
        return tuple(arrays_)

    @property
    def argument_shapes(self):
        return (None,) * len(self.kernel_arguments)

    def with_arguments(self, arguments):
        if len(arguments) != 2:
            raise ValueError("Must provide 2 arguments")

        assignee, expression = arguments
        return self.copy(assignee=assignee, expression=expression)

    @property
    def _expression_kernel_arguments(self):
        from pyop3.array import HierarchicalArray

        if isinstance(self.expression, HierarchicalArray):
            return ((self.expression, READ),)
        elif isinstance(self.expression, numbers.Number):
            return ()
        else:
            raise NotImplementedError("Complicated rvalues not yet supported")

    @property
    def kernel_arguments(self):
        from pyop3.array.harray import HierarchicalArray
        from pyop3.array.petsc import Mat

        args = OrderedSet()
        for array, _ in self.function_arguments:
            if isinstance(array, HierarchicalArray):
                args.add(array.buffer)
            elif isinstance(array, Mat):
                args.add(array.mat)
        return tuple(args)


class ReplaceAssignment(Assignment):
    """Like PETSC_INSERT_VALUES."""

    @cached_property
    def function_arguments(self):
        return ((self.assignee, WRITE),) + self._expression_kernel_arguments


class AddAssignment(Assignment):
    """Like PETSC_ADD_VALUES."""

    @cached_property
    def function_arguments(self):
        return ((self.assignee, INC),) + self._expression_kernel_arguments


# inherit from Assignment?
class PetscMatInstruction(Instruction):
    def __init__(self, mat_arg, array_arg):
        assert mat_arg.dtype == array_arg.dtype

        self.mat_arg = mat_arg
        self.array_arg = array_arg

    @property
    def kernel_arguments(self):
        args = (self.mat_arg.mat,)
        if isinstance(self.array_arg, ContextSensitive):
            args += tuple(dat.buffer for dat in self.array_arg.context_map.values())
        else:
            args += (self.array_arg.buffer,)
        return args

    @property
    def datamap(self):
        return self.mat_arg.datamap | self.array_arg.datamap


class PetscMatLoad(PetscMatInstruction):
    @cached_property
    def function_arguments(self):
        return ((self.mat_arg, READ), (self.array_arg, WRITE))


class PetscMatStore(PetscMatInstruction):
    @cached_property
    def function_arguments(self):
        return ((self.mat_arg, WRITE), (self.array_arg, READ))


# potentially confusing name
class PetscMatAdd(PetscMatInstruction):
    @cached_property
    def function_arguments(self):
        return ((self.mat_arg, INC), (self.array_arg, READ))


class OpaqueKernelArgument(KernelArgument, ContextFree):
    def __init__(self, dtype=auto):
        self._dtype = dtype

    @property
    def kernel_dtype(self):
        return self._dtype


class DummyKernelArgument(OpaqueKernelArgument):
    """Placeholder kernel argument.

    This class is useful when one simply wants to generate code from a loop
    expression and not execute it.

    ### dtypes not required here as sniffed from local kernel/context?

    """


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
        assert isinstance(access, Intent)
        is_input = access in {READ, RW, INC, MIN_RW, MAX_RW, NA}
        is_output = access in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_WRITE, MAX_RW}
        new_args.append(arg.copy(is_input=is_input, is_output=is_output))
    return tunit.with_kernel(kernel.copy(args=new_args))


@functools.singledispatch
def _collect_kernel_arguments(func_arg: FunctionArgument) -> tuple:
    from pyop3 import HierarchicalArray, Mat  # cyclic import
    from pyop3.buffer import DistributedBuffer, NullBuffer

    if isinstance(func_arg, HierarchicalArray):
        return _collect_kernel_arguments(func_arg.buffer)
    elif isinstance(func_arg, Mat):
        return _collect_kernel_arguments(func_arg.mat)
    elif isinstance(func_arg, DistributedBuffer):
        return (func_arg,)
    elif isinstance(func_arg, NullBuffer):
        return ()
    else:
        raise TypeError(f"No handler defined for {type(func_arg).__name__}")


@_collect_kernel_arguments.register
def _(mat: PETSc.Mat) -> tuple:
    return (mat,)

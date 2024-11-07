# TODO: rename this file to insn_visitors.py? Consistent with expr_visitors

from __future__ import annotations

import abc
import collections
import functools
import numbers
import operator
from os import access
from typing import Any, Union

import numpy as np
from petsc4py import PETSc
from pyrsistent import pmap, PMap
from immutabledict import ImmutableOrderedDict

from pyop3.array import Dat, Array, Mat, _ConcretizedDat
from pyop3.axtree import Axis, AxisTree, ContextFree, ContextSensitive, ContextMismatchException, ContextAware
from pyop3.axtree.tree import Operator, AxisVar, IndexedAxisTree
from pyop3.buffer import DistributedBuffer, NullBuffer, PackedBuffer
from pyop3.itree import Map, TabulatedMapComponent, collect_loop_contexts
from pyop3.itree.tree import LoopIndexVar
from pyop3.itree.parse import _as_context_free_indices
from pyop3.expr_visitors import (
    collect_loops as expr_collect_loops,
    extract_axes,
    materialize,
    restrict_to_context as restrict_expression_to_context,
    compress_indirection_maps as compress_expression_indirection_maps,
    concretize_arrays as concretize_expression_layouts,
)
from pyop3.lang import (
    INC,
    NA,
    READ,
    RW,
    WRITE,
    Assignment,
    AssignmentType,
    CalledFunction,
    DummyKernelArgument,
    Instruction,
    Loop,
    InstructionList,
    PetscMatAssign,
    ArrayAccessType,
)
from pyop3.utils import UniqueNameGenerator, checked_zip, just_one, single_valued, OrderedSet, merge_dicts, expand_collection_of_iterables


# NOTE: A sensible pattern is to have a public and private (rec) implementations of a
# transformation. Then the outer one can also drop extra instruction lists.


# GET RID OF THIS
# TODO Is this generic for other parsers/transformers? Esp. lower.py
class Transformer(abc.ABC):
    @abc.abstractmethod
    def apply(self, expr):
        pass


# NOTE: This is a bad name for this transformation. 'expand_multi_component_loops'?
def expand_loop_contexts(insn: Instruction, /) -> InstructionList:
    """
    This function also drops zero-sized loops.
    """
    return _expand_loop_contexts_rec(insn, loop_context_acc=pmap())


@functools.singledispatch
def _expand_loop_contexts_rec(obj: Any, /, *, loop_context_acc) -> InstructionList:
    raise TypeError


@_expand_loop_contexts_rec.register(InstructionList)
def _(insn_list: InstructionList, /, **kwargs) -> InstructionList:
    return InstructionList([_expand_loop_contexts_rec(insn, **kwargs) for insn in insn_list])


@_expand_loop_contexts_rec.register(Loop)
def _(loop: Loop, /, *, loop_context_acc) -> InstructionList:
    expanded_loops = []
    for axis, component_label in loop.index.iterset.leaves:
        # NOTE: I think that this should always just be the axis tree!? indexed bits
        # can be discarded by this point
        path = loop.index.iterset.source_path[axis.id, component_label]
        loop_context = {loop.index.id: path}

        restricted_loop_index = just_one(_as_context_free_indices(loop.index, loop_context))

        # skip empty loops
        if restricted_loop_index.iterset.size == 0:
            continue

        loop_context_acc_ = loop_context_acc | loop_context
        expanded_loop = type(loop)(
            restricted_loop_index,
            [
                _expand_loop_contexts_rec(stmt, loop_context_acc=loop_context_acc_)
                for stmt in loop.statements
            ]
        )
        expanded_loops.append(expanded_loop)
    return InstructionList(expanded_loops)


@_expand_loop_contexts_rec.register(CalledFunction)
def _(func: CalledFunction, /, *, loop_context_acc) -> CalledFunction:
    return CalledFunction(
        func.function,
        [arg.with_context(loop_context_acc) for arg in func.arguments],
    )


@_expand_loop_contexts_rec.register(Assignment)
def _(assignment: Assignment, /, *, loop_context_acc) -> Assignment:
    return Assignment(
        restrict_expression_to_context(assignment.assignee, loop_context_acc),
        restrict_expression_to_context(assignment.expression, loop_context_acc),
        assignment.assignment_type,
    )



# @functools.singledispatch
# def _restrict_instruction_to_loop_context(obj: Any, /, loop_context) -> Instruction:
#     raise TypeError(f"No handler defined for {type(obj).__name__}")
#
#
# @_restrict_instruction_to_loop_context.register(InstructionList)
# def _(insn_list: InstructionList, /, loop_context) -> InstructionList:
#     return InstructionList([
#         _restrict_instruction_to_loop_context(insn, loop_context)
#         for insn in insn_list
#     ])
#
#
# @_restrict_instruction_to_loop_context.register(Loop)
# def _(loop: Loop, /, loop_context) -> Loop:
#     cf_loop_index = just_one(_as_context_free_indices(loop.index, loop_context))
#     return Loop(
#         cf_loop_index,
#         [
#             _restrict_instruction_to_loop_context(stmt, loop_context)
#             for stmt in loop.statements
#         ],
#     )


# @_restrict_instruction_to_loop_context.register(CalledFunction)
# def _(func: CalledFunction, /, loop_context) -> CalledFunction:
#     return CalledFunction(
#         func.function,
#         [arg.with_context(loop_context) for arg in func.arguments],
#     )
#
#
# @_restrict_instruction_to_loop_context.register(Assignment)
# def _(assignment: Assignment, /, loop_context) -> Assignment:
#     return Assignment(
#         restrict_expression_to_context(assignment.assignee, loop_context),
#         restrict_expression_to_context(assignment.expression, loop_context),
#         assignment.assignment_type,
#     )


def context_product(contexts, acc=pmap()):
    contexts = tuple(contexts)

    if not contexts:
        return acc

    ctx, *subctxs = contexts
    index, pathss = ctx
    for paths in pathss:
        acc_ = acc | {index: paths}
        if subctxs:
            yield from context_product(subctxs, acc_)
        else:
            yield acc_


class ImplicitPackUnpackExpander(Transformer):
    def __init__(self):
        self._name_generator = UniqueNameGenerator()

    def apply(self, expr):
        return self._apply(expr)

    @functools.singledispatchmethod
    def _apply(self, expr: Any):
        raise NotImplementedError(f"No handler provided for {type(expr).__name__}")

    # TODO Can I provide a generic "operands" thing? Put in the parent class?
    @_apply.register
    def _(self, loop: Loop):
        new_statements = [s for stmt in loop.statements for s in self._apply(stmt)]
        return loop.copy(statements=new_statements)

    @_apply.register
    def _(self, insn_list: InstructionList):
        return insn_list.copy(instructions=[self._apply(insn) for insn in insn_list])

    # # TODO: Should be the same as Assignment
    # @_apply.register
    # def _(self, assignment: PetscMatInstruction):
    #     # FIXME: Probably will not work for things like mat[x, y].assign(dat[z])
    #     # where the expression is indexed.
    #     return (assignment,)

    @_apply.register
    def _(self, assignment: Assignment):
        # I think this is fine...
        return InstructionList([assignment])

        # same as for CalledFunction
        gathers = []
        # NOTE: scatters are executed in LIFO order
        scatters = []
        arguments = []

        # lazy coding, tidy up
        if isinstance(assignment, ReplaceAssignment):
            access = WRITE
        else:
            assert isinstance(assignment, AddAssignment)
            access = INC
        for arg, intent in [
            (assignment.assignee, access),
            (assignment.expression, READ),
        ]:
            if isinstance(arg, numbers.Number):
                arguments.append(arg)
                continue

            # emit function calls for PetscMat
            if isinstance(arg, Mat):
                axes = AxisTree(arg.axes.node_map)
                new_arg = Dat(
                    axes,
                    data=NullBuffer(arg.dtype),  # does this need a size?
                    prefix="t",
                )

                if intent == READ:
                    gathers.append(PetscMatLoad(arg, new_arg))
                elif intent == WRITE:
                    scatters.insert(0, PetscMatStore(arg, new_arg))
                elif intent == RW:
                    gathers.append(PetscMatLoad(arg, new_arg))
                    scatters.insert(0, PetscMatStore(arg, new_arg))
                else:
                    assert intent == INC
                    scatters.insert(0, PetscMatAdd(arg, new_arg))

                arguments.append(new_arg)
            else:
                arguments.append(arg)

        return InstructionList([*gathers, assignment.with_arguments(arguments), *scatters])

    @_apply.register
    def _(self, terminal: CalledFunction):
        gathers = []
        # NOTE: scatters are executed in LIFO order
        scatters = []
        arguments = []
        for (arg, intent), shape in checked_zip(
            terminal.function_arguments, terminal.argument_shapes
        ):
            if isinstance(arg, DummyKernelArgument):
                arguments.append(arg)
                continue

            # unpick pack/unpack instructions
            if intent != NA and _requires_pack_unpack(arg):
                # TODO: Make generic across Array types
                if isinstance(arg, Dat):
                    axes = AxisTree(arg.axes.node_map)
                    temporary = Dat(
                        # arg.axes.materialize(),  # TODO
                        axes,
                        data=NullBuffer(arg.dtype),  # does this need a size?
                        prefix="t",
                    )
                else:
                    assert isinstance(arg, Mat)
                    raxes = AxisTree(arg.raxes.node_map)
                    caxes = AxisTree(arg.caxes.node_map)
                    temporary = Mat(
                        raxes,
                        caxes,
                        mat=NullBuffer(arg.dtype),
                        prefix="t",
                    )

                if intent == READ:
                    gathers.append(Assignment(temporary, arg, "write"))
                elif intent == WRITE:
                    # This is currently necessary because some local kernels
                    # (interpolation) actually increment values instead of setting
                    # them directly. This should ideally be addressed.
                    gathers.append(Assignment(temporary, 0, "write"))
                    scatters.insert(0, Assignment(arg, temporary, "write"))
                elif intent == RW:
                    gathers.append(Assignment(temporary, arg, "write"))
                    scatters.insert(0, Assignment(arg, temporary, "write"))
                else:
                    assert intent == INC
                    gathers.append(Assignment(temporary, 0, "write"))
                    scatters.insert(0, Assignment(arg, temporary, "inc"))

                arguments.append(temporary)

            else:
                arguments.append(arg)

        return InstructionList([*gathers, terminal.with_arguments(arguments), *scatters])


# class ExprMarker


# TODO check this docstring renders correctly
def expand_implicit_pack_unpack(expr: Instruction):
    """Expand implicit pack and unpack operations.

    An implicit pack/unpack is something of the form

    .. code::
        kernel(dat[f(p)])

    In order for this to work the ``dat[f(p)]`` needs to be packed
    into a temporary. Assuming that its intent in ``kernel`` is
    `pyop3.WRITE`, we would expand this function into

    .. code::
        tmp <- [0, 0, ...]
        kernel(tmp)
        dat[f(p)] <- tmp

    Notes
    -----
    For this routine to work, any context-sensitive loops must have
    been expanded already (with `expand_loop_contexts`). This is
    because context-sensitive arrays may be packed into temporaries
    in some contexts but not others.

    """
    return ImplicitPackUnpackExpander().apply(expr)


def _requires_pack_unpack(arg):
    # TODO in theory packing isn't required for arrays that are contiguous,
    # but this is hard to determine
    # FIXME, we inefficiently copy matrix temporaries here because this
    # doesn't identify requiring pack/unpack properly. To demonstrate
    #   kernel(mat[p, q])
    # gets turned into
    #   t0 <- mat[p, q]
    #   kernel(t0)
    # However, the array mat[p, q] is actually retrieved from MatGetValues
    # so we really have something like
    #   MatGetValues(mat, ..., t0)
    #   t1 <- t0
    #   kernel(t1)
    # and the same for unpacking

    # if subst_layouts and layouts are the same I *think* it is safe to avoid a pack/unpack
    # however, it is overly restrictive since we could pass something like dat[i0, :] directly
    # to a local kernel
    # return isinstance(arg, Dat) and arg.subst_layouts != arg.layouts
    return isinstance(arg, (Dat, Mat))


@functools.singledispatch
def expand_assignments(obj: Any, /) -> InstructionList:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@expand_assignments.register(InstructionList)
def _(insn_list: InstructionList, /) -> InstructionList:
    return InstructionList([expand_assignments(insn) for insn in insn_list])


@expand_assignments.register(Loop)
def _(loop: Loop, /) -> InstructionList:
    return InstructionList([
        Loop(loop.index, [
            expand_assignments(stmt) for stmt in loop.statements
        ])
    ])


@expand_assignments.register(CalledFunction)
def _(func: CalledFunction, /) -> InstructionList:
    # Assume that this isn't a problem here, think about this
    return InstructionList([func])


@expand_assignments.register(Assignment)
def _(assignment: Assignment, /) -> InstructionList:
    # NOTE: This is incorrect, we only include this because if we have a 'basic' matrix assignment
    # like
    #
    #     mat[f(p), f(p)] <- t0
    #
    # we don't want to expand it into
    #
    #     t1 <- t0
    #     mat[f(p), f(p)] <- t1
    # if assignment.is_mat_access:
    #     raise NotImplementedError("think")
    #     return InstructionList([assignment])

    bare_expression, extra_input_insns = _expand_reshapes(
        assignment.expression, ArrayAccessType.READ
    )

    if assignment.assignment_type == AssignmentType.WRITE:
        assignee_access_type = ArrayAccessType.WRITE
    else:
        assert assignment.assignment_type == AssignmentType.INC
        assignee_access_type = ArrayAccessType.INC

    bare_assignee, extra_output_insns = _expand_reshapes(
        assignment.assignee, assignee_access_type
    )

    if bare_assignee == assignment.assignee:
        bare_assignment = Assignment(bare_assignee, bare_expression, assignment.assignment_type)
    else:
        bare_assignment = Assignment(bare_assignee, bare_expression, "write")

    return InstructionList([*extra_input_insns, bare_assignment, *extra_output_insns])


# TODO: better word than "mode"? And use an enum.
@functools.singledispatch
def _expand_reshapes(expr: Any, /, mode):
    raise TypeError(f"No handler provided for {type(expr).__name__}")


@_expand_reshapes.register(Operator)
def _(op: Operator, /, access_type):
    bare_a, a_insns = _expand_reshapes(op.a, access_type)
    bare_b, b_insns = _expand_reshapes(op.b, access_type)
    return (type(op)(bare_a, bare_b), a_insns + b_insns)


@_expand_reshapes.register(numbers.Number)
@_expand_reshapes.register(AxisVar)
@_expand_reshapes.register(LoopIndexVar)
def _(var, /, access_type):
    return (var, ())


@_expand_reshapes.register(Array)
def _(array: Array, /, access_type):
    if array.parent:
        # .materialize?
        if isinstance(array, Dat):
            temp_initial = Dat(
                AxisTree(array.parent.axes.node_map),
                data=NullBuffer(array.dtype),
                prefix="t"
            )
            temp_reshaped = temp_initial.with_axes(array.axes)
        else:
            assert isinstance(array, Mat)
            temp_initial = Mat(
                AxisTree(array.parent.raxes.node_map),
                AxisTree(array.parent.caxes.node_map),
                mat=NullBuffer(array.dtype),
                prefix="t"
            )
            temp_reshaped = temp_initial.with_axes(array.raxes, array.caxes)

        transformed_dat, extra_insns = _expand_reshapes(array.parent, access_type)

        if extra_insns:
            raise NotImplementedError("Pretty sure this doesn't work as is")

        if access_type == ArrayAccessType.READ:
            assignment = Assignment(temp_initial, transformed_dat, "write")
        elif access_type == ArrayAccessType.WRITE:
            assignment = Assignment(transformed_dat, temp_initial, "write")
        else:
            assert access_type == ArrayAccessType.INC
            assignment = Assignment(transformed_dat, temp_initial, "inc")

        return (temp_reshaped, extra_insns + (assignment,))
    else:
        return (array, ())


@functools.singledispatch
def prepare_petsc_calls(obj: Any, /) -> InstructionList:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@prepare_petsc_calls.register(InstructionList)
def _(insn_list: InstructionList, /) -> InstructionList:
    return InstructionList([prepare_petsc_calls(insn) for insn in insn_list])


@prepare_petsc_calls.register(Loop)
def _(loop: Loop, /) -> InstructionList:
    return InstructionList([
        Loop(loop.index, [
            prepare_petsc_calls(stmt) for stmt in loop.statements
        ])
    ])


@prepare_petsc_calls.register(CalledFunction)
def _(func: CalledFunction, /) -> InstructionList:
    return InstructionList([func])


# NOTE: At present we assume that matrices are never part of the expression, only
# the assignee. Ideally we should traverse the expression and emit extra READ instructions.
@prepare_petsc_calls.register(Assignment)
def _(assignment: Assignment, /) -> InstructionList:
    if isinstance(assignment.assignee.buffer, PETSc.Mat):
        mat = assignment.assignee

        # If we have an expression like
        #
        #     mat[f(p), f(p)] <- 666
        #
        # then we have to convert `666` into an appropriately sized temporary
        # for MatSetValues to work.
        if isinstance(assignment.expression, numbers.Number):
            expression = Mat(mat.raxes, mat.caxes, mat=np.full(mat.alloc_size, assignment.expression, dtype=mat.dtype), prefix="t", constant=True)
        else:
            assert (
                isinstance(assignment.expression, Mat)
                and isinstance(assignment.expression.buffer, NullBuffer)
            )
            expression = assignment.expression

        if assignment.assignment_type == AssignmentType.WRITE:
            access_type = ArrayAccessType.WRITE
        else:
            assert assignment.assignment_type == AssignmentType.INC
            access_type = ArrayAccessType.INC

        assignment = PetscMatAssign(mat, expression, access_type)

    return InstructionList([assignment])


# NOTE: I think this is a bit redundant - should do this much earlier!
@functools.singledispatch
def concretize_array_accesses(obj: Any, /) -> Instruction:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@concretize_array_accesses.register(InstructionList)
def _(insn_list: InstructionList, /) -> InstructionList:
    return InstructionList([concretize_array_accesses(insn) for insn in insn_list])


@concretize_array_accesses.register(Loop)
def _(loop: Loop, /) -> Loop:
    return loop.copy(statements=[concretize_array_accesses(stmt) for stmt in loop.statements])


@concretize_array_accesses.register(Assignment)
def _(assignment: Assignment, /) -> Assignment:
    return Assignment(
        concretize_expression_layouts(assignment.assignee),
        concretize_expression_layouts(assignment.expression),
        assignment.assignment_type,
    )


@concretize_array_accesses.register(CalledFunction)
def _(func: CalledFunction, /):
    return func.copy(arguments=[concretize_expression_layouts(arg) for arg in func.arguments])


def compress_indirection_maps(insn: Instruction) -> Instruction:
    return _compress_indirection_maps_rec(insn, loop_axes_acc=pmap())


@functools.singledispatch
def _compress_indirection_maps_rec(obj: Any, /, *, loop_axes_acc) -> Instruction:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_compress_indirection_maps_rec.register(InstructionList)
def _(insn_list: InstructionList, /, **kwargs) -> InstructionList:
    return InstructionList([_compress_indirection_maps_rec(insn, **kwargs) for insn in insn_list])


@_compress_indirection_maps_rec.register(Loop)
def _(loop: Loop, /, *, loop_axes_acc) -> Loop:
    # NOTE: I think that this should always just be the axis tree!?
    loop_axes_acc_ = loop_axes_acc | {loop.index.id: loop.index.iterset}
    return loop.copy(
        statements=[_compress_indirection_maps_rec(stmt, loop_axes_acc=loop_axes_acc_) for stmt in loop.statements]
    )


@_compress_indirection_maps_rec.register(Assignment)
def _(assignment: Assignment, /, *, loop_axes_acc) -> Assignment:
    return Assignment(
        _compress_array_indirection_maps(assignment.assignee, loop_axes_acc),
        _compress_array_indirection_maps(assignment.expression, loop_axes_acc),
        assignment.assignment_type,
    )


@_compress_indirection_maps_rec.register(CalledFunction)
def _(func: CalledFunction, /, *, loop_axes_acc):
    return func.copy(arguments=[_compress_array_indirection_maps(arg, loop_axes_acc) for arg in func.arguments])


# NOTE: This is technically suboptimal. We want to optimise the access patterns
# across all arrays, not just on a per-array basis.
def _compress_array_indirection_maps(dat, loop_axes):
    if not isinstance(dat, Array):
        return dat

    if not isinstance(dat, Dat):
        raise NotImplementedError

    layouts = {}
    for leaf_path, orig_layout in dat.axes.leaf_subst_layouts.items():
        visited_axes = dat.axes.path_with_nodes(dat.axes._node_from_path(leaf_path), and_components=True)

        if extract_axes(orig_layout, visited_axes, loop_axes).size == 0:
            chosen_layout = -1
        else:
            candidate_layouts = compress_expression_indirection_maps(orig_layout, visited_axes, loop_axes)

            # Now choose the candidate layout with the lowest cost, breaking ties
            # by choosing the left-most entry with a given cost.
            chosen_layout = min(candidate_layouts, key=lambda item: item[1])[0]

            # evaluate CompositeDats (bad name?)
            # do as part of compress_expression_indirection_maps?
            chosen_layout = materialize(chosen_layout, visited_axes, loop_axes)

            # TODO: Check for affine layouts and penalise...

        layouts[leaf_path] = chosen_layout

    return _ConcretizedDat(dat, layouts)

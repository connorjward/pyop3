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

from pyop3.array import Dat, Array, Mat, Sparsity, _ConcretizedDat, _ConcretizedMat, _ExpressionDat, AbstractMat
from pyop3.axtree import Axis, AxisTree, ContextFree, ContextSensitive, ContextMismatchException, ContextAware
from pyop3.axtree.tree import Operator, AxisVar, IndexedAxisTree
from pyop3.buffer import DistributedBuffer, NullBuffer, PackedBuffer
from pyop3.dtypes import IntType
from pyop3.itree import Map, TabulatedMapComponent, collect_loop_contexts
from pyop3.itree.tree import LoopIndexVar
from pyop3.itree.parse import _as_context_free_indices
from pyop3.expr_visitors import (
    replace as replace_expression,
    replace_terminals,
    _CompositeDat,
    collect_loops as expr_collect_loops,
    concretize_arrays as expr_concretize_arrays,
    extract_axes,
    materialize,
    restrict_to_context as restrict_expression_to_context,
    collect_candidate_indirections as collect_expression_candidate_indirections,
    compute_indirection_cost as compute_expression_indirection_cost,
    concretize_arrays as concretize_expression_layouts,
)
from pyop3.lang import (
    INC,
    NA,
    READ,
    RW,
    WRITE,
    AbstractAssignment,
    NullInstruction,
    Assignment,
    AssignmentType,
    CalledFunction,
    DummyKernelArgument,
    Instruction,
    Loop,
    InstructionList,
    PetscMatAssign,
    ArrayAccessType,
    enlist,
    maybe_enlist,
)
from pyop3.utils import UniqueNameGenerator, checked_zip, just_one, single_valued, OrderedSet, merge_dicts, expand_collection_of_iterables, strictly_all


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
    return maybe_enlist([_expand_loop_contexts_rec(insn, **kwargs) for insn in insn_list])


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
        # if restricted_loop_index.iterset.size == 0:
        #     continue

        loop_context_acc_ = loop_context_acc | loop_context
        expanded_loop = type(loop)(
            restricted_loop_index,
            [
                _expand_loop_contexts_rec(stmt, loop_context_acc=loop_context_acc_)
                for stmt in loop.statements
            ]
        )
        expanded_loops.append(expanded_loop)
    return maybe_enlist(expanded_loops)


@_expand_loop_contexts_rec.register(CalledFunction)
def _(func: CalledFunction, /, *, loop_context_acc) -> CalledFunction:
    return CalledFunction(
        func.function,
        [arg.with_context(loop_context_acc) for arg in func.arguments],
    )


@_expand_loop_contexts_rec.register(Assignment)
def _(assignment: Assignment, /, *, loop_context_acc) -> Assignment:
    assignee = restrict_expression_to_context(assignment.assignee, loop_context_acc)
    expression = restrict_expression_to_context(assignment.expression, loop_context_acc)

    try:
        size = assignee.size
    except AttributeError:
        size = expression.size
        # still might fail?

    if size == 0:
        return NullInstruction()
    else:
        return Assignment(assignee, expression, assignment.assignment_type)


class ImplicitPackUnpackExpander(Transformer):
    def __init__(self):
        self._name_generator = UniqueNameGenerator()

    def apply(self, expr):
        return self._apply(expr)

    @functools.singledispatchmethod
    def _apply(self, expr: Any):
        raise NotImplementedError(f"No handler provided for {type(expr).__name__}")

    @_apply.register(NullInstruction)
    def _(self, /, expr):
        return NullInstruction()

    # TODO Can I provide a generic "operands" thing? Put in the parent class?
    @_apply.register(Loop)
    def _(self, loop: Loop):
        new_statements = [s for stmt in loop.statements for s in enlist(self._apply(stmt))]
        return loop.copy(statements=new_statements)

    @_apply.register
    def _(self, insn_list: InstructionList):
        return type(insn_list)([insn_ for insn in insn_list for insn_ in enlist(self._apply(insn))])

    # # TODO: Should be the same as Assignment
    # @_apply.register
    # def _(self, assignment: PetscMatInstruction):
    #     # FIXME: Probably will not work for things like mat[x, y].assign(dat[z])
    #     # where the expression is indexed.
    #     return (assignment,)

    @_apply.register
    def _(self, assignment: Assignment):
        # I think this is fine...
        return assignment

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

        return maybe_enlist((*gathers, assignment.with_arguments(arguments), *scatters))

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

        return maybe_enlist((*gathers, terminal.with_arguments(arguments), *scatters))


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
    return maybe_enlist((expand_assignments(insn) for insn in insn_list))


@expand_assignments.register(Loop)
def _(loop: Loop, /) -> InstructionList:
    return Loop(loop.index, [
            expand_assignments(stmt) for stmt in loop.statements
    ])


@expand_assignments.register(CalledFunction)
def _(func: CalledFunction, /) -> InstructionList:
    # Assume that this isn't a problem here, think about this
    return func


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

    return maybe_enlist((*extra_input_insns, bare_assignment, *extra_output_insns))


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
    return type(insn_list)((prepare_petsc_calls(insn) for insn in insn_list))


@prepare_petsc_calls.register(Loop)
def _(loop: Loop, /) -> Loop:
    return Loop(loop.index, [
        prepare_petsc_calls(stmt) for stmt in loop.statements
    ])


@prepare_petsc_calls.register(CalledFunction)
def _(func: CalledFunction, /) -> InstructionList:
    return func


# NOTE: At present we assume that matrices are never part of the expression, only
# the assignee. Ideally we should traverse the expression and emit extra READ instructions.
@prepare_petsc_calls.register(Assignment)
def _(assignment: Assignment, /) -> InstructionList:
    if isinstance(assignment.assignee.buffer, PETSc.Mat):
        mat = assignment.assignee

        if isinstance(mat, _ConcretizedMat):
            mat = mat.mat

        # If we have an expression like
        #
        #     mat[f(p), f(p)] <- 666
        #
        # then we have to convert `666` into an appropriately sized temporary
        # for MatSetValues to work.
        if isinstance(assignment.expression, numbers.Number):
            # TODO: There must be a more elegant way of doing this
            expression = Dat(AxisTree(mat.axes.node_map), data=np.full(mat.alloc_size, assignment.expression, dtype=mat.dtype), prefix="t", constant=True)
        else:
            assert (
                isinstance(assignment.expression, (Mat, _ConcretizedMat))
                and isinstance(assignment.expression.buffer, NullBuffer)
            )
            expression = assignment.expression

        if assignment.assignment_type == AssignmentType.WRITE:
            access_type = ArrayAccessType.WRITE
        else:
            assert assignment.assignment_type == AssignmentType.INC
            access_type = ArrayAccessType.INC

        assignment = PetscMatAssign(mat, expression, access_type)

    # If we are doing a non-PETSc matrix assignment then we cast the buffer to a 'Dat'
    elif isinstance(assignment.assignee, (Sparsity, Mat)):
        assert isinstance(assignment.assignee.buffer, NullBuffer), "Must be a temporary"
        mat = assignment.assignee
        dat = Dat(mat.axes, data=mat.buffer, name=mat.name)
        assignment = Assignment(dat, assignment.expression, assignment.assignment_type)

    return assignment


# NOTE: Should perhaps take a different input type to ensure that always called at root?
# E.g. PreprocessedInstruction?
@PETSc.Log.EventDecorator()
def compress_indirection_maps(insn: Instruction) -> Instruction:

    # try setting a 'global' cache here
    mycache = {}

    # cache this?
    arg_candidatess = _collect_candidate_indirections(insn, loop_axes_acc=pmap())

    # Start by combining the best per-arg candidates into the initial overall best candidate
    best_candidate = {}
    max_cost = 0
    for arg_id, arg_candidates in arg_candidatess.items():
        expr, expr_cost = min(arg_candidates, key=lambda item: item[1])
        best_candidate[arg_id] = (expr, expr_cost)
        max_cost += expr_cost

    # Optimise by dropping any immediately bad candidates. We do this by dropping
    # any candidates whose cost (per-arg) is greater than the current best candidate.
    arg_candidatess = {
        arg_id: tuple(
            (arg_candidate, cost)
            for arg_candidate, cost in arg_candidates
            if cost <= max_cost
        )
        for arg_id, arg_candidates in arg_candidatess.items()
    }

    # Now select the combination with the lowest combined cost. We can make savings here
    # by sharing indirection maps between different arguments. For example, if we have
    #
    #     dat1[mapA[mapB[mapC[i]]]]
    #     dat2[mapB[mapC[i]]]
    #
    # then we can (sometimes) minimise the data cost by having
    #     dat1[mapA[mapBC[i]]]
    #     dat2[mapBC[i]]
    #
    # instead of
    #
    #     dat1[mapABC[i]]
    #     dat2[mapBC[i]]
    min_cost = max_cost
    for shared_candidate in expand_collection_of_iterables(arg_candidatess):
        cost = 0
        seen_exprs = set()
        for expr, expr_cost in shared_candidate.values():
            if expr not in seen_exprs:
                cost += expr_cost
                seen_exprs.add(expr)

        if cost < min_cost:
            best_candidate = shared_candidate
            min_cost = cost

    # Now materialise any symbolic (composite) dats and propagate the
    # decision back to the tree.
    composite_dats = frozenset.union(
        *(_collect_composite_dats(expr) for (expr, _) in best_candidate.values())
    )
    replace_map = {
        comp_dat: materialize_composite_dat(comp_dat)
        for comp_dat in composite_dats
    }

    # now apply to best layout candidate
    best_layouts = {
        key: replace_expression(expr, replace_map)
        for key, (expr, _) in best_candidate.items()
    }

    # now traverse the instruction tree and replace the layouts.
    optimised_insn = _replace_with_real_dats(insn, best_layouts)

    return optimised_insn


@functools.singledispatch
def _collect_candidate_indirections(obj: Any, /, *, loop_axes_acc) -> ImmutableOrderedDict:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_collect_candidate_indirections.register(InstructionList)
def _(insn_list: InstructionList, /, **kwargs) -> ImmutableOrderedDict:
    return merge_dicts(
        [_collect_candidate_indirections(insn, **kwargs) for insn in insn_list],
        ordered=True,
    )


@_collect_candidate_indirections.register(Loop)
def _(loop: Loop, /, *, loop_axes_acc) -> ImmutableOrderedDict:
    loop_axes_acc_ = loop_axes_acc | {loop.index.id: loop.index.iterset}
    return merge_dicts(
        [
            _collect_candidate_indirections(stmt, loop_axes_acc=loop_axes_acc_)
            for stmt in loop.statements
        ],
        ordered=True,
    )


@_collect_candidate_indirections.register(AbstractAssignment)
def _(assignment: Assignment, /, *, loop_axes_acc) -> ImmutableOrderedDict:
    candidates = {}
    for i, arg in enumerate([assignment.assignee, assignment.expression]):
        for key, value in _collect_array_candidate_indirections(arg, loop_axes_acc).items():
            candidates[assignment.id, i, key] = value
    return ImmutableOrderedDict(candidates)


@_collect_candidate_indirections.register(CalledFunction)
def _(func: CalledFunction, /, *, loop_axes_acc) -> ImmutableOrderedDict:
    candidates = {}
    for i, arg in enumerate(func.arguments):
        for key, value in _collect_array_candidate_indirections(arg, loop_axes_acc).items():
            candidates[func.id, i, key] = value
    return ImmutableOrderedDict(candidates)


@PETSc.Log.EventDecorator()
def _compute_indirection_cost(insn: Instruction, arg_layouts, *, cache) -> int:
    seen_exprs_mut = set()
    return _compute_indirection_cost_rec(insn, arg_layouts, seen_exprs_mut, loop_axes_acc=pmap(), cache=cache)


@functools.singledispatch
def _compute_indirection_cost_rec(obj: Any, /, arg_layouts, seen_exprs_mut, *, loop_axes_acc, cache) -> int:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_compute_indirection_cost_rec.register(InstructionList)
def _(insn_list: InstructionList, /, *args, **kwargs) -> int:
    return sum(_compute_indirection_cost_rec(insn, *args, **kwargs) for insn in insn_list)


@_compute_indirection_cost_rec.register(Loop)
def _(loop: Loop, /, arg_layouts, seen_exprs_mut, *, loop_axes_acc, **kwargs) -> int:
    loop_axes_acc_ = loop_axes_acc | {loop.index.id: loop.index.iterset}
    return sum(
        _compute_indirection_cost_rec(stmt, arg_layouts, seen_exprs_mut, loop_axes_acc=loop_axes_acc_, **kwargs)
        for stmt in loop.statements
    )


@_compute_indirection_cost_rec.register(Assignment)
def _(assignment: Assignment, /, arg_layouts, seen_exprs_mut, *, loop_axes_acc, cache) -> int:
    return sum(
        _compute_array_indirection_cost(arg, arg_layouts, seen_exprs_mut, loop_axes_acc, (assignment.id, i), cache)
        for i, arg in enumerate([assignment.assignee, assignment.expression])
    )


@_compute_indirection_cost_rec.register(CalledFunction)
def _(func: CalledFunction, /, arg_layouts, seen_exprs_mut, *, loop_axes_acc, cache) -> int:
    return sum(
        _compute_array_indirection_cost(arg, arg_layouts, seen_exprs_mut, loop_axes_acc, (func.id, i), cache)
        for i, arg in enumerate(func.arguments)
    )


@functools.singledispatch
def _collect_array_candidate_indirections(dat, loop_axes) -> ImmutableOrderedDict:
    if not isinstance(dat, Array):
        return ImmutableOrderedDict()

    raise NotImplementedError


@_collect_array_candidate_indirections.register(Dat)
def _(dat: Dat, loop_axes):
    return dat.candidate_layouts(loop_axes)


@_collect_array_candidate_indirections.register(AbstractMat)
def _(mat: Mat, loop_axes):
    return mat.candidate_layouts(loop_axes)


@functools.singledispatch
def _collect_composite_dats(obj: Any) -> frozenset:
    raise TypeError


@_collect_composite_dats.register(Operator)
def _(op, /) -> frozenset:
    return _collect_composite_dats(op.a) | _collect_composite_dats(op.b)


@_collect_composite_dats.register(AxisVar)
@_collect_composite_dats.register(LoopIndexVar)
@_collect_composite_dats.register(numbers.Number)
def _(op, /) -> frozenset:
    return frozenset()


@_collect_composite_dats.register(_ExpressionDat)
def _(dat, /) -> frozenset:
    return _collect_composite_dats(dat.layout)


@_collect_composite_dats.register(_CompositeDat)
def _(dat, /) -> frozenset:
    return frozenset({dat})


# NOTE: Think this lives in expr_visitors or something
def materialize_composite_dat(dat: _CompositeDat) -> _ExpressionDat:
    axes = extract_axes(dat, dat.visited_axes, dat.loop_axes, {})

    if axes.size == 0:
        return None

    # FIXME: This is almost certainly wrong in general
    result = Dat(axes, dtype=IntType)

    # replace LoopIndexVars in the expression with AxisVars
    loop_index_replace_map = {}
    for loop_id, iterset in dat.loop_axes.items():
        for axis in iterset.nodes:
            loop_index_replace_map[(loop_id, axis.label)] = AxisVar(f"{axis.label}_{loop_id}")
    expr = replace_terminals(dat.expr, loop_index_replace_map)

    result.assign(expr, eager=True)

    # now put the loop indices back
    inv_map = {axis_var.axis_label: LoopIndexVar(loop_id, axis_label) for (loop_id, axis_label), axis_var in loop_index_replace_map.items()}
    layout = just_one(result.axes.leaf_subst_layouts.values())
    newlayout = replace_terminals(layout, inv_map)

    return _ExpressionDat(result, newlayout)


@functools.singledispatch
def _replace_with_real_dats(obj, layouts) -> Instruction:
    raise TypeError


@_replace_with_real_dats.register(InstructionList)
def _(insn_list: InstructionList, /, layouts) -> InstructionList:
    return maybe_enlist((_replace_with_real_dats(insn, layouts) for insn in insn_list))


@_replace_with_real_dats.register(Loop)
def _(loop: Loop, /, layouts) -> Loop:
    return loop.copy(statements=[_replace_with_real_dats(stmt, layouts) for stmt in loop.statements])


@_replace_with_real_dats.register(Assignment)
def _(assignment: Assignment, /, layouts) -> Assignment:
    return Assignment(
        _compress_array_indirection_maps(assignment.assignee, layouts, (assignment.id, 0)),
        _compress_array_indirection_maps(assignment.expression, layouts, (assignment.id, 1)),
        assignment.assignment_type,
    )


@_replace_with_real_dats.register(PetscMatAssign)
def _(assignment: PetscMatAssign, /, layouts) -> PetscMatAssign:
    return PetscMatAssign(
        _compress_array_indirection_maps(assignment.mat, layouts, (assignment.id, 0)),
        _compress_array_indirection_maps(assignment.values, layouts, (assignment.id, 1)),
        assignment.access_type,
    )


@_replace_with_real_dats.register(CalledFunction)
def _(func: CalledFunction, /, layouts) -> CalledFunction:
    return func.copy(arguments=[_compress_array_indirection_maps(arg, layouts, (func.id, i))
                                for i, arg in enumerate(func.arguments)])


@functools.singledispatch
def _compress_array_indirection_maps(dat, layouts, outer_key):
    if not isinstance(dat, Array):
        return dat

    assert False


@_compress_array_indirection_maps.register(Dat)
def _(dat: Dat, layouts, outer_key):
    newlayouts = {}
    for leaf_path in dat.axes.leaf_paths:
        try:
            chosen_layout = layouts[outer_key + ((dat, leaf_path),)]
        except KeyError:
            # zero-sized axis, no layout needed
            chosen_layout = -1
        newlayouts[leaf_path] = chosen_layout

    return _ConcretizedDat(dat, newlayouts)


@_compress_array_indirection_maps.register(Mat)
@_compress_array_indirection_maps.register(Sparsity)
def _(mat, layouts, outer_key):
    def collect(axes, newlayouts, counter):
        for leaf_path in axes.leaf_paths:
            try:
                chosen_layout = layouts[outer_key + ((mat, leaf_path, counter),)]
            except KeyError:
                # zero-sized axis, no layout needed
                chosen_layout = -1
            newlayouts[leaf_path] = chosen_layout

    row_layouts = {}
    col_layouts = {}

    collect(mat.raxes, row_layouts, 0)
    collect(mat.caxes, col_layouts, 1)

    return _ConcretizedMat(mat, row_layouts, col_layouts)


def concretize_arrays(insn: Instruction, /) -> Instruction:
    return _concretize_arrays_rec(insn, pmap())


@functools.singledispatch
def _concretize_arrays_rec(insn:Instruction, /, loop_axes_acc) -> Instruction:
    raise TypeError


@_concretize_arrays_rec.register(InstructionList)
def _(insn_list: InstructionList, /, loop_axes_acc):
    return InstructionList([_concretize_arrays_rec(insn, loop_axes_acc) for insn in insn_list])


@_concretize_arrays_rec.register(Loop)
def _(loop: Loop, /, loop_axes_acc) -> Loop:
    loop_axes_acc_ = loop_axes_acc | {loop.index.id: loop.index.iterset}
    return loop.copy(statements=[_concretize_arrays_rec(stmt, loop_axes_acc_) for stmt in loop.statements])


@_concretize_arrays_rec.register(Assignment)
def _(assignment: Assignment, /, loop_axes_acc) -> Assignment:
    return Assignment(
        expr_concretize_arrays(assignment.assignee, loop_axes_acc),
        expr_concretize_arrays(assignment.expression, loop_axes_acc),
        assignment.assignment_type,
    )


@_concretize_arrays_rec.register(PetscMatAssign)
def _(assignment: PetscMatAssign, /, loop_axes_acc) -> PetscMatAssign:
    return PetscMatAssign(
        expr_concretize_arrays(assignment.mat, loop_axes_acc),
        expr_concretize_arrays(assignment.values, loop_axes_acc),
        assignment.access_type,
    )


@_concretize_arrays_rec.register(CalledFunction)
def _(func: CalledFunction, /, loop_axes_acc) -> CalledFunction:
    return func.copy(arguments=[expr_concretize_arrays(arg, loop_axes_acc) for arg in func.arguments])

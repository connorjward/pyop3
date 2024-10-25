# TODO: rename this file to insn_visitors.py? Consistent with expr_visitors

from __future__ import annotations

import abc
import collections
import functools
import numbers
import operator
from typing import Any, Union

from pyrsistent import pmap, PMap
from immutabledict import ImmutableOrderedDict

from pyop3.array import Dat, AbstractMat
from pyop3.array.transforms import DatReshape
from pyop3.array.petsc import AbstractMat
from pyop3.axtree import Axis, AxisTree, ContextFree, ContextSensitive, ContextMismatchException, ContextAware
from pyop3.axtree.tree import Operator, AxisVar, IndexedAxisTree
from pyop3.buffer import DistributedBuffer, NullBuffer, PackedBuffer
from pyop3.itree import Map, TabulatedMapComponent, collect_loop_contexts
from pyop3.itree.tree import LoopIndexVar
from pyop3.itree.parse import _as_context_free_indices
from pyop3.expr_visitors import collect_loops as expr_collect_loops, collect_datamap as collect_expr_datamap, restrict_to_context as restrict_expression_to_context
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
    PetscMatAccess,
)
from pyop3.utils import UniqueNameGenerator, checked_zip, just_one, single_valued, OrderedSet, merge_dicts, expand_collection_of_iterables


@functools.singledispatch
def collect_loop_indices(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@collect_loop_indices.register(InstructionList)
def _(insn_list: InstructionList, /) -> OrderedSet:
    loop_indices = OrderedSet()
    for insn in insn_list:
        loop_indices |= collect_loop_indices(insn)
    return loop_indices


@collect_loop_indices.register(Loop)
def _(loop: Loop, /) -> OrderedSet:
    # NOTE: Need to look at loop index in more detail to extract other indices
    loop_indices = OrderedSet([loop.index])
    for stmt in loop.statements:
        loop_indices |= collect_loop_indices(stmt)
    return loop_indices


@collect_loop_indices.register(Assignment)
def _(assignment: Assignment, /) -> OrderedSet:
    return expr_collect_loops(assignment.assignee) | expr_collect_loops(assignment.expression)


@collect_loop_indices.register(PetscMatAccess)
def _(mat_access: PetscMatAccess, /) -> OrderedSet:
    return expr_collect_loops(mat_access.mat_arg) | expr_collect_loops(mat_access.array_arg)


@collect_loop_indices.register(CalledFunction)
def _(func: CalledFunction, /) -> OrderedSet:
    loop_indices = OrderedSet()
    for arg in func.arguments:
        loop_indices |= expr_collect_loops(arg)
    return loop_indices


# TODO Is this generic for other parsers/transformers? Esp. lower.py
class Transformer(abc.ABC):
    @abc.abstractmethod
    def apply(self, expr):
        pass


"""
TODO
We sometimes want to pass loop indices to functions even without an external loop.
This is particularly useful when we only want to generate code. We should (?) unpick
this so that there is an outer set of loop contexts that applies at the highest level.

Alternatively, we enforce that this loop exists. But I don't think that that's feasible
right now.
"""


# class LoopContextExpander(Transformer):

    # @_apply.register
    # def _(self, terminal: Assignment, *, context):
    #     # FIXME for now we assume an outer context of {}. In other words anything
    #     # context sensitive in the assignment is completely handled by the existing
    #     # outer loops.
    #     # This is meaningful if the kernel accepts a loop index as an argument.
    #
    #     cf_args = []
    #     for arg in terminal.arguments:
    #         if isinstance(arg, ContextAware):
    #             try:
    #                 cf_args.append(arg.with_context(context))
    #             except ContextMismatchException:
    #                 # assignment is not valid in this context, do nothing
    #                 return ((pmap(), None),)
    #         else:
    #             cf_args.append(arg)
    #     return ((pmap(), terminal.with_arguments(cf_args)),)

    # # TODO: this is just an assignment, fix inheritance
    # @_apply.register
    # def _(self, terminal: PetscMatInstruction, *, context):
    #     try:
    #         mat = terminal.mat_arg.with_context(context)
    #         array = terminal.array_arg.with_context(context)
    #         return ((pmap(), terminal.copy(mat_arg=mat, array_arg=array)),)
    #     except ContextMismatchException:
    #         return ((pmap(), None),)


def expand_loop_contexts(insn: Instruction, /) -> InstructionList:
    insns = []

    loop_indices = collect_loop_indices(insn)
    compressed_loop_contexts = collect_loop_contexts(loop_indices)
    # Pass `pmap` as the mapping type because we do not care about the ordering
    # of `loop_context` (though we *do* care about the order of iteration).
    for loop_context in expand_collection_of_iterables(compressed_loop_contexts, mapping_type=pmap):
        cf_insn = _restrict_instruction_to_loop_context(insn, loop_context)
        insns.append(cf_insn)

    return InstructionList(insns)


@functools.singledispatch
def _restrict_instruction_to_loop_context(obj: Any, /, loop_context) -> Instruction:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_restrict_instruction_to_loop_context.register(InstructionList)
def _(insn_list: InstructionList, /, loop_context) -> InstructionList:
    return InstructionList([
        _restrict_instruction_to_loop_context(insn, loop_context)
        for insn in insn_list
    ])


@_restrict_instruction_to_loop_context.register(Loop)
def _(loop: Loop, /, loop_context) -> Loop:
    cf_loop_index = just_one(_as_context_free_indices(loop.index, loop_context))
    return Loop(
        cf_loop_index,
        [
            _restrict_instruction_to_loop_context(stmt, loop_context)
            for stmt in loop.statements
        ],
    )


@_restrict_instruction_to_loop_context.register(CalledFunction)
def _(func: CalledFunction, /, loop_context) -> CalledFunction:
    return CalledFunction(
        func.function,
        [arg.with_context(loop_context) for arg in func.arguments],
    )


@_restrict_instruction_to_loop_context.register(Assignment)
def _(assignment: Assignment, /, loop_context) -> Assignment:
    return Assignment(
        restrict_expression_to_context(assignment.assignee, loop_context),
        restrict_expression_to_context(assignment.expression, loop_context),
        assignment.assignment_type,
    )


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
            if isinstance(arg, AbstractMat):
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
            # assert isinstance(
            #     arg, ContextFree
            # ), "Loop contexts should already be expanded"

            if isinstance(arg, DummyKernelArgument):
                arguments.append(arg)
                continue

            # unpick pack/unpack instructions
            if intent != NA and _requires_pack_unpack(arg):
                axes = AxisTree(arg.axes.node_map)
                temporary = Dat(
                    # arg.axes.materialize(),  # TODO
                    axes,
                    data=NullBuffer(arg.dtype),  # does this need a size?
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
    return isinstance(arg, (Dat, AbstractMat))


@functools.singledispatch
def expand_array_transformations(insn: Any, /):
    raise TypeError(f"No handler defined for {type(insn).__name__}")


@expand_array_transformations.register(InstructionList)
def _(insn_list: InstructionList, /):
    return insn_list.copy(
        instructions=[
            insn_ for insn in insn_list for insn_ in expand_array_transformations(insn)
        ],
    )


# NOTE: in theory loop could be over something transformed
@expand_array_transformations.register(Loop)
def _(loop: Loop, /):
    return InstructionList([
        loop.copy(
            statements=[
                insn for stmt in loop.statements for insn in expand_array_transformations(stmt)
            ],
        )
    ])


@expand_array_transformations.register(Assignment)
def _(assignment: Assignment, /) -> InstructionList:
    # NOTE: I think perhaps we need a new type for this...
    if any(isinstance(x, AbstractMat) for x in {assignment.assignee, assignment.expression}):
        mat_arg = just_one(arg for arg in {assignment.assignee, assignment.expression} if isinstance(arg, AbstractMat))
        # if mat_arg.transform:
        #     raise NotImplementedError
        # else:
        #     return InstructionList([assignment])
        return InstructionList([assignment])

    bare_expression, input_insns = _generate_array_transformations(assignment.expression, "in")
    bare_assignee, output_insns = _generate_array_transformations(assignment.assignee, "out")

    bare_assignment = Assignment(bare_assignee, bare_expression, assignment.assignment_type)

    return InstructionList([*input_insns, bare_assignment, *output_insns])


@expand_array_transformations.register(CalledFunction)
def _(func: CalledFunction, /) -> InstructionList:
    breakpoint()
    if any(a.transform for a in func.arguments):
        raise NotImplementedError
    else:
        return InstructionList([func])


# @expand_array_transformations.register(PetscMatInstruction)
# def _(mat_insn, /) -> InstructionList:
#     # if mat_insn.mat_arg.transform or mat_insn.array_arg.transform:
#     if mat_insn.array_arg.transform:
#         raise NotImplementedError
#     else:
#         return InstructionList([mat_insn])


# TODO: better word than "mode"? And use an enum.
@functools.singledispatch
def _generate_array_transformations(expr: Any, /, mode):
    raise TypeError(f"No handler provided for {type(expr).__name__}")


@_generate_array_transformations.register(Operator)
def _(op: Operator, /, mode):
    bare_a, a_insns = _generate_array_transformations(op.a, mode)
    bare_b, b_insns = _generate_array_transformations(op.b, mode)
    # reconstruct?
    return (type(op)(bare_a, bare_b), a_insns + b_insns)


@_generate_array_transformations.register(numbers.Number)
@_generate_array_transformations.register(AxisVar)
@_generate_array_transformations.register(LoopIndexVar)
def _(var, /, mode):
    return (var, ())


@_generate_array_transformations.register(Dat)
def _(dat: Dat, /, mode):
    # NOTE: In this function we assume that subst_layouts cannot contain
    # any transformations. This is probably never going to occur but for
    # things to work properly we should be tracking the path so as not
    # to emit unnecessary assignments.
    if dat.transform:
        bare_dat, transform_insns = _generate_array_transformations2(dat.transform, dat, mode)
        return (bare_dat, transform_insns)
    else:
        return (dat, ())


# TODO: Need a good name for this.
@functools.singledispatch
def _generate_array_transformations2(expr: Any, /, mode):
    raise TypeError(f"No handler provided for {type(expr).__name__}")


@_generate_array_transformations2.register(DatReshape)
def _(reshape: DatReshape, /, dat, mode):
    breakpoint()
    temp_initial_axes = AxisTree(reshape.initial.axes.node_map)
    temp_initial = Dat(temp_initial_axes, data=NullBuffer(dat.dtype), prefix="t")

    temp_axes_reshaped = AxisTree(dat.axes.node_map)
    temp_reshaped = temp_initial.with_axes(temp_axes_reshaped)

    transformed_dat, transform_insns = _generate_array_transformations(reshape.initial, mode)

    if mode == "in":
        assignment = ReplaceAssignment(temp_initial, transformed_dat)
    else:
        assert mode == "out"
        assignment = ReplaceAssignment(transformed_dat, temp_initial)

    return (temp_reshaped, transform_insns + (assignment,))


# NOTE: Not used any more
@functools.singledispatch
def collect_datamap(insn: Any, /) -> PMap:
    raise TypeError(f"No handler defined for {type(insn).__name__}")


@collect_datamap.register(InstructionList)
def _(insn_list: InstructionList, /) -> PMap:
    return merge_dicts(collect_datamap(insn) for insn in insn_list)


@collect_datamap.register(Loop)
def _(loop: Loop, /) -> PMap:
    # NOTE: I don't think that this will work if we have something ragged as
    # we will miss the extent variable. I think in most other cases this is fine however.
    return merge_dicts(collect_datamap(stmt) for stmt in loop.statements)


@collect_datamap.register(Assignment)
def _(assignment: Assignment, /) -> PMap:
    return collect_expr_datamap(assignment.assignee) | collect_expr_datamap(assignment.expression)


@collect_datamap.register(PetscMatAccess)
def _(mat_insn: PetscMatAccess, /) -> PMap:
    return collect_expr_datamap(mat_insn.mat_arg) | collect_expr_datamap(mat_insn.array_arg)


@collect_datamap.register(CalledFunction)
def _(func: CalledFunction, /) -> PMap:
    return merge_dicts(collect_expr_datamap(arg) for arg in func.arguments)


@functools.singledispatch
def expand_petsc_mat_accesses(insn: Any, /) -> InstructionList:
    raise TypeError()


@expand_petsc_mat_accesses.register(InstructionList)
def _(insn_list: InstructionList, /) -> InstructionList:
    return InstructionList([expand_petsc_mat_accesses(insn) for insn in insn_list])


@expand_petsc_mat_accesses.register(Loop)
def _(loop: Loop, /) -> InstructionList:
    return InstructionList([
        Loop(loop.index, [
            expand_petsc_mat_accesses(stmt) for stmt in loop.statements
        ])
    ])


@expand_petsc_mat_accesses.register(CalledFunction)
def _(func: CalledFunction, /) -> InstructionList:
    # Assume that this isn't a problem here, think about this
    return InstructionList([func])


@expand_petsc_mat_accesses.register(Assignment)
def _(assignment: Assignment, /) -> InstructionList:
    # NOTE: Can imagine more complex scenarios where an indexed matrix is
    # part of an expression or similar. For now we just deal with the
    # core Firedrake use case of
    #
    #   mat[f(i0, i1), f(i0, i1)] <- t0[i1, i1]
    array = assignment.assignee
    if isinstance(array, AbstractMat) and any(isinstance(axes, IndexedAxisTree) for axes in {array.raxes, array.caxes}):
        if assignment.assignment_type == AssignmentType.WRITE:
            assignment = PetscMatAccess(array, assignment.expression, "write")
        else:
            assert assignment.assignment_type == AssignmentType.INC
            assignment = PetscMatAccess(array, assignment.expression, "inc")

    return InstructionList([assignment])



# *below is old untested code*
#
# def compress(iterset, map_func, *, uniquify=False):
#     # TODO Ultimately we should be able to generate code for this set of
#     # loops. We would need to have a construct to describe "unique packing"
#     # with hash sets like we do in the Python version. PETSc have PetscHSetI
#     # which I think would be suitable.
#
#     if not uniquify:
#         raise NotImplementedError("TODO")
#
#     iterset = iterset.as_tree()
#
#     # prepare size arrays, we want an array per target path per iterset path
#     sizess = {}
#     for leaf_axis, leaf_clabel in iterset.leaves:
#         iterset_path = iterset.path(leaf_axis, leaf_clabel)
#
#         # bit unpleasant to have to create a loop index for this
#         sizes = {}
#         index = iterset.index()
#         cf_map = map_func(index).with_context({index.id: iterset_path})
#         for target_path in cf_map.leaf_target_paths:
#             if iterset.depth != 1:
#                 # TODO For now we assume iterset to have depth 1
#                 raise NotImplementedError
#             # The axes of the size array correspond only to the specific
#             # components selected from iterset by iterset_path.
#             clabels = (just_one(iterset_path.values()),)
#             subiterset = iterset[clabels]
#
#             # subiterset is an axis tree with depth 1, we only want the axis
#             assert subiterset.depth == 1
#             subiterset = subiterset.root
#
#             sizes[target_path] = Dat(
#                 subiterset, dtype=IntType, prefix="nnz"
#             )
#         sizess[iterset_path] = sizes
#     sizess = freeze(sizess)
#
#     # count sizes
#     for p in iterset.iter():
#         entries = collections.defaultdict(set)
#         for q in map_func(p.index).iter({p}):
#             # we expect maps to only output a single target index
#             q_value = just_one(q.target_exprs.values())
#             entries[q.target_path].add(q_value)
#
#         for target_path, points in entries.items():
#             npoints = len(points)
#             nnz = sizess[p.source_path][target_path]
#             nnz.set_value(p.source_path, p.source_exprs, npoints)
#
#     # prepare map arrays
#     flat_mapss = {}
#     for iterset_path, sizes in sizess.items():
#         flat_maps = {}
#         for target_path, nnz in sizes.items():
#             subiterset = nnz.axes.root
#             map_axes = AxisTree.from_nest({subiterset: Axis(nnz)})
#             flat_maps[target_path] = Dat(
#                 map_axes, dtype=IntType, prefix="map"
#             )
#         flat_mapss[iterset_path] = flat_maps
#     flat_mapss = freeze(flat_mapss)
#
#     # populate compressed maps
#     for p in iterset.iter():
#         entries = collections.defaultdict(set)
#         for q in map_func(p.index).iter({p}):
#             # we expect maps to only output a single target index
#             q_value = just_one(q.target_exprs.values())
#             entries[q.target_path].add(q_value)
#
#         for target_path, points in entries.items():
#             flat_map = flat_mapss[p.source_path][target_path]
#             leaf_axis, leaf_clabel = flat_map.axes.leaf
#             for i, pt in enumerate(sorted(points)):
#                 path = p.source_path | {leaf_axis.label: leaf_clabel}
#                 indices = p.source_exprs | {leaf_axis.label: i}
#                 flat_map.set_value(path, indices, pt)
#
#     # build the actual map
#     connectivity = {}
#     for iterset_path, flat_maps in flat_mapss.items():
#         map_components = []
#         for target_path, flat_map in flat_maps.items():
#             # since maps only target a single axis, component pair
#             target_axlabel, target_clabel = just_one(target_path.items())
#             map_component = TabulatedMapComponent(
#                 target_axlabel, target_clabel, flat_map
#             )
#             map_components.append(map_component)
#         connectivity[iterset_path] = map_components
#     return Map(connectivity)
#
#
# def split_loop(loop: Loop, path, tile_size: int) -> Loop:
#     orig_loop_index = loop.index
#
#     # I think I need to transform the index expressions of the iterset?
#     # or get a new iterset? let's try that
#     # It will not work because then the target path would change and the
#     # data structures would not know what to do.
#
#     orig_index_exprs = orig_loop_index.index_exprs
#     breakpoint()
#     # new_index_exprs
#
#     new_loop_index = orig_loop_index.copy(index_exprs=new_index_exprs)
#     return loop.copy(index=new_loop_index)

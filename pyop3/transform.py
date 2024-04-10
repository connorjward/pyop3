from __future__ import annotations

import abc
import collections
import functools
import numbers

from pyrsistent import freeze, pmap

from pyop3.array import ContextSensitiveMultiArray, HierarchicalArray
from pyop3.array.petsc import AbstractMat
from pyop3.axtree import Axis, AxisTree, ContextFree, ContextSensitive
from pyop3.buffer import DistributedBuffer, NullBuffer, PackedBuffer
from pyop3.itree import Map, TabulatedMapComponent
from pyop3.lang import (
    INC,
    NA,
    READ,
    RW,
    WRITE,
    AddAssignment,
    Assignment,
    CalledFunction,
    ContextAwareLoop,
    DummyKernelArgument,
    Instruction,
    Loop,
    Pack,
    PetscMatAdd,
    PetscMatInstruction,
    PetscMatLoad,
    PetscMatStore,
    ReplaceAssignment,
    Terminal,
)
from pyop3.utils import UniqueNameGenerator, checked_zip, just_one


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


class LoopContextExpander(Transformer):
    # TODO prefer __call__ instead
    def apply(self, expr: Instruction):
        return self._apply(expr, context=pmap())

    @functools.singledispatchmethod
    def _apply(self, expr: Instruction, **kwargs):
        raise TypeError(f"No handler provided for {type(expr).__name__}")

    @_apply.register
    def _(self, loop: Loop, *, context):
        # this is very similar to what happens in PetscMat.__getitem__
        outer_context = collections.defaultdict(dict)  # ordered set per index
        if isinstance(loop.index.iterset, ContextSensitive):
            for ctx in loop.index.iterset.context_map.keys():
                for index, paths in ctx.items():
                    if index in context:
                        # assert paths == context[index]
                        continue
                    else:
                        outer_context[index][paths] = None
        # convert ordered set to a list
        outer_context = {k: tuple(v.keys()) for k, v in outer_context.items()}

        # convert to a product-like structure of [{index: paths, ...}, {index: paths}, ...]
        outer_context_ = tuple(context_product(outer_context.items()))

        if not outer_context_:
            outer_context_ = (pmap(),)

        loops = []
        for octx in outer_context_:
            cf_iterset = loop.index.iterset.with_context(context | octx)
            source_paths = cf_iterset.leaf_paths
            target_paths = cf_iterset.leaf_target_paths
            assert len(source_paths) == len(target_paths)

            if len(source_paths) == 1:
                # single component iterset, no branching required
                source_path = just_one(source_paths)
                target_path = just_one(target_paths)

                context_ = context | {loop.index.id: (source_path, target_path)}

                statements = collections.defaultdict(list)
                for stmt in loop.statements:
                    for myctx, mystmt in self._apply(stmt, context=context_ | octx):
                        if myctx:
                            raise NotImplementedError(
                                "need to think about how to wrap inner instructions "
                                "that need outer loops"
                            )
                        statements[source_path].append(mystmt)
            else:
                assert len(source_paths) > 1
                statements = {}
                for source_path, target_path in checked_zip(source_paths, target_paths):
                    context_ = context | {loop.index.id: (source_path, target_path)}

                    statements[source_path] = []

                    for stmt in loop.statements:
                        for myctx, mystmt in self._apply(stmt, context=context_ | octx):
                            if myctx:
                                raise NotImplementedError(
                                    "need to think about how to wrap inner instructions "
                                    "that need outer loops"
                                )
                            if mystmt is None:
                                continue
                            statements[source_path].append(mystmt)

            # FIXME this does not propagate inner outer contexts
            loop = ContextAwareLoop(
                loop.index.copy(iterset=cf_iterset),
                statements,
            )
            loops.append((octx, loop))
        return tuple(loops)

    @_apply.register
    def _(self, terminal: CalledFunction, *, context):
        # this is very similar to what happens in PetscMat.__getitem__
        outer_context = collections.defaultdict(dict)  # ordered set per index
        for arg in terminal.arguments:
            if not isinstance(arg, ContextSensitive):
                continue

            for ctx in arg.context_map.keys():
                for index, paths in ctx.items():
                    if index in context:
                        assert paths == context[index]
                    else:
                        outer_context[index][paths] = None
        # convert ordered set to a list
        outer_context = {k: tuple(v.keys()) for k, v in outer_context.items()}

        # convert to a product-like structure of [{index: paths, ...}, {index: paths}, ...]
        outer_context_ = tuple(context_product(outer_context.items()))

        if not outer_context_:
            outer_context_ = (pmap(),)

        for arg in terminal.arguments:
            if isinstance(arg, ContextSensitive):
                outer_context.update(
                    {
                        index: paths
                        for ctx in arg.context_map.keys()
                        for index, paths in ctx.items()
                        if index not in context
                    }
                )

        retval = []
        for octx in outer_context_:
            cf_args = [a.with_context(octx | context) for a in terminal.arguments]
            retval.append((octx, terminal.with_arguments(cf_args)))
        return retval

    @_apply.register
    def _(self, terminal: Assignment, *, context):
        # FIXME for now we assume an outer context of {}. In other words anything
        # context sensitive in the assignment is completely handled by the existing
        # outer loops.

        valid = True
        cf_args = []
        for arg in terminal.arguments:
            try:
                cf_arg = (
                    arg.with_context(context)
                    if isinstance(arg, ContextSensitive)
                    else arg
                )
            # FIXME We will hit issues here when we are missing outer context I think
            except KeyError:
                # assignment is not valid in this context, do nothing
                valid = False
                break
            cf_args.append(cf_arg)

        if valid:
            return ((pmap(), terminal.with_arguments(cf_args)),)
        else:
            return ((pmap(), None),)

    # TODO: this is just an assignment, fix inheritance
    @_apply.register
    def _(self, terminal: PetscMatInstruction, *, context):
        if any(
            isinstance(a, ContextSensitive)
            for a in {terminal.mat_arg, terminal.array_arg}
        ):
            raise NotImplementedError

        return ((pmap(), terminal),)


def expand_loop_contexts(expr: Instruction):
    return LoopContextExpander().apply(expr)


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
    def _(self, loop: ContextAwareLoop):
        return (
            loop.copy(
                statements={
                    ctx: [stmt_ for stmt in stmts for stmt_ in self._apply(stmt)]
                    for ctx, stmts in loop.statements.items()
                }
            ),
        )

    # TODO: Should be the same as Assignment
    @_apply.register
    def _(self, assignment: PetscMatInstruction):
        # FIXME: Probably will not work for things like mat[x, y].assign(dat[z])
        # where the expression is indexed.
        return (assignment,)

    @_apply.register
    def _(self, assignment: Assignment):
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
                axes = AxisTree(arg.axes.parent_to_children)
                new_arg = HierarchicalArray(
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

        return (*gathers, assignment.with_arguments(arguments), *scatters)

    @_apply.register
    def _(self, terminal: CalledFunction):
        gathers = []
        # NOTE: scatters are executed in LIFO order
        scatters = []
        arguments = []
        for (arg, intent), shape in checked_zip(
            terminal.kernel_arguments, terminal.argument_shapes
        ):
            assert isinstance(
                arg, ContextFree
            ), "Loop contexts should already be expanded"

            if isinstance(arg, DummyKernelArgument):
                arguments.append(arg)
                continue

            # TODO: old code, delete
            # emit function calls for PetscMat
            # this is a separate stage to the assignment operations because one
            # can index a packed mat. E.g. mat[p, q][::2] would decompose into
            # two calls, one to pack t0 <- mat[p, q] and another to pack t1 <- t0[::2]
            # if (
            #     isinstance(arg, Pack)
            #     and isinstance(arg.big.buffer, PackedBuffer)
            #     or not isinstance(arg, Pack)
            #     and isinstance(arg.buffer, PackedBuffer)
            # ):
            #     if isinstance(arg, Pack):
            #         myarg = arg.big
            #     else:
            #         myarg = arg
            #
            #     # TODO add PackedPetscMat as a subclass of buffer?
            #     if not isinstance(myarg.buffer.array, PetscMat):
            #         raise NotImplementedError("Only handle Mat at the moment")
            #
            #     axes = AxisTree(myarg.axes.parent_to_children)
            #     new_arg = HierarchicalArray(
            #         axes,
            #         data=NullBuffer(myarg.dtype),  # does this need a size?
            #         prefix="t",
            #     )
            #
            #     if intent == READ:
            #         gathers.append(PetscMatLoad(myarg, new_arg))
            #     elif intent == WRITE:
            #         scatters.insert(0, PetscMatStore(myarg, new_arg))
            #     elif intent == RW:
            #         gathers.append(PetscMatLoad(myarg, new_arg))
            #         scatters.insert(0, PetscMatStore(myarg, new_arg))
            #     else:
            #         assert intent == INC
            #         gathers.append(ReplaceAssignment(new_arg, 0))
            #         scatters.insert(0, PetscMatAdd(myarg, new_arg))
            #
            #     # the rest of the packing code is now dealing with the result of this
            #     # function call
            #     arg = new_arg

            # unpick pack/unpack instructions
            if intent != NA and _requires_pack_unpack(arg):
                is_petsc_mat = isinstance(arg, AbstractMat)

                axes = AxisTree(arg.axes.parent_to_children)
                temporary = HierarchicalArray(
                    axes,
                    data=NullBuffer(arg.dtype),  # does this need a size?
                    prefix="t",
                )

                if intent == READ:
                    if is_petsc_mat:
                        gathers.append(PetscMatLoad(arg, temporary))
                    else:
                        gathers.append(ReplaceAssignment(temporary, arg))
                elif intent == WRITE:
                    # This is currently necessary because some local kernels
                    # (interpolation) actually increment values instead of setting
                    # them directly. This should ideally be addressed.
                    gathers.append(ReplaceAssignment(temporary, 0))
                    if is_petsc_mat:
                        scatters.insert(0, PetscMatStore(arg, temporary))
                    else:
                        scatters.insert(0, ReplaceAssignment(arg, temporary))
                elif intent == RW:
                    if is_petsc_mat:
                        gathers.append(PetscMatLoad(arg, temporary))
                        scatters.insert(0, PetscMatStore(arg, temporary))
                    else:
                        gathers.append(ReplaceAssignment(temporary, arg))
                        scatters.insert(0, ReplaceAssignment(arg, temporary))
                else:
                    assert intent == INC
                    gathers.append(ReplaceAssignment(temporary, 0))
                    if is_petsc_mat:
                        scatters.insert(0, PetscMatAdd(arg, temporary))
                    else:
                        scatters.insert(0, AddAssignment(arg, temporary))

                arguments.append(temporary)

            else:
                arguments.append(arg)

        return (*gathers, terminal.with_arguments(arguments), *scatters)


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
    # return isinstance(arg, HierarchicalArray) and arg.subst_layouts != arg.layouts
    return isinstance(arg, (HierarchicalArray, AbstractMat))


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
#             sizes[target_path] = HierarchicalArray(
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
#             flat_maps[target_path] = HierarchicalArray(
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

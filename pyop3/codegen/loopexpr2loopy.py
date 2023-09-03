from __future__ import annotations

import abc
import collections
import contextlib
import copy
import dataclasses
import enum
import functools
import itertools
import numbers
import operator
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import loopy as lp
import loopy.symbolic
import numpy as np
import pymbolic as pym
import pytools
from pyrsistent import pmap

from pyop3 import tlang, utils
from pyop3.axis import Axis, AxisComponent, AxisTree, AxisVariable, CalledAxisTree
from pyop3.distarray import IndexedMultiArray, MultiArray
from pyop3.dtypes import IntType
from pyop3.index import (
    AffineMapComponent,
    AffineSliceComponent,
    CalledMap,
    GlobalLoopIndex,
    Index,
    IndexedArray,
    IndexedAxisTree,
    IndexTree,
    LocalLoopIndex,
    LoopIndex,
    Map,
    MapVariable,
    Slice,
    Subset,
    TabulatedMapComponent,
)
from pyop3.log import logger
from pyop3.loopexpr import (
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    WRITE,
    FunctionCall,
    Loop,
)
from pyop3.utils import (
    PrettyTuple,
    checked_zip,
    just_one,
    merge_dicts,
    single_valued,
    strictly_all,
)

# FIXME this needs to be synchronised with TSFC, tricky
# shared base package? or both set by Firedrake - better solution
LOOPY_TARGET = lp.CWithGNULibcTarget()
LOOPY_LANG_VERSION = (2018, 2)


def _noop(leaf, preorder_ctx, **kwargs):
    return preorder_ctx


def _none(*args, **kwargs):
    pass


def visit_indices(
    indices,
    pre_ctx,
    *,
    index_callback=_none,  # FIXME bad default, will break
    pre_callback=_noop,
    post_callback_nonterminal=_noop,
    post_callback_terminal=_none,
    final_callback=_none,
    **kwargs,
):
    return _visit_indices_rec(
        indices,
        pre_ctx,
        index_callback=index_callback,
        pre_callback=pre_callback,
        post_callback_nonterminal=post_callback_nonterminal,
        post_callback_terminal=post_callback_terminal,
        final_callback=final_callback,
        current_index=indices.root,
        **kwargs,
    )


def _visit_indices_rec(
    indices,
    preorder_ctx,
    *,
    index_callback,
    pre_callback,
    post_callback_nonterminal,
    post_callback_terminal,
    final_callback,
    current_index,
    **kwargs,
):
    leaves, index_data = index_callback(current_index, preorder_ctx, **kwargs)

    leafdata = {}
    for i, (leafkey, leaf) in enumerate(leaves.items()):
        preorder_ctx_ = pre_callback(leaf, preorder_ctx, **kwargs)

        if current_index.id in indices.parent_to_children:
            for subindex in indices.parent_to_children[current_index.id]:
                retval = _visit_indices_rec(
                    indices,
                    preorder_ctx_,
                    index_callback=index_callback,
                    pre_callback=pre_callback,
                    post_callback_nonterminal=post_callback_nonterminal,
                    post_callback_terminal=post_callback_terminal,
                    final_callback=final_callback,
                    current_index=subindex,
                    **kwargs,
                )
                # this is now a no-op
                leafdata[leafkey] = post_callback_nonterminal(
                    retval, leaf, preorder_ctx_, **kwargs
                )
        else:
            leafdata[leafkey] = post_callback_terminal(
                leafkey, leaf, preorder_ctx_, **kwargs
            )

    return final_callback(index_data, leafdata)


class CodegenContext(abc.ABC):
    pass


class LoopyCodegenContext(CodegenContext):
    def __init__(self):
        self._domains = []
        self._insns = []
        self._args = []
        self._subkernels = []

        self._within_inames_mut = set()
        self._last_insn_id = None

        self._name_generator = pytools.UniqueNameGenerator()

    @property
    def domains(self):
        return tuple(self._domains)

    @property
    def instructions(self):
        return tuple(self._insns)

    @property
    def arguments(self):
        # TODO should renumber things here
        return tuple(self._args)

    @property
    def subkernels(self):
        return tuple(self._subkernels)

    def add_domain(self, iname, *args):
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1]
        self._domains.append(f"{{ [{iname}]: {start} <= {iname} < {stop} }}")

    def add_assignment(self, assignee, expression, prefix="insn"):
        insn = lp.Assignment(
            assignee,
            expression,
            id=self._name_generator(prefix),
            within_inames=frozenset(self._within_inames),
            within_inames_is_final=True,
            depends_on=self._depends_on,
            depends_on_is_final=True,
        )
        self._add_instruction(insn)

    def add_function_call(self, assignees, expression, prefix="insn"):
        insn = lp.CallInstruction(
            assignees,
            expression,
            id=self._name_generator(prefix),
            within_inames=self._within_inames,
            within_inames_is_final=True,
            depends_on=self._depends_on,
            depends_on_is_final=True,
        )
        self._add_instruction(insn)

    def add_argument(self, name, dtype):
        # FIXME if self._args is a set then we can add duplicates here provided
        # that we canonically renumber at a later point
        if name in [a.name for a in self._args]:
            logger.debug(
                f"Skipping adding {name} to the codegen context as it is already present"
            )
            return
        arg = lp.GlobalArg(name, dtype=dtype, shape=None)
        self._args.append(arg)

    def add_temporary(self, name, dtype=IntType, shape=()):
        temp = lp.TemporaryVariable(name, dtype=dtype, shape=shape)
        self._args.append(temp)

    def add_subkernel(self, subkernel):
        self._subkernels.append(subkernel)

    # I am not sure that this belongs here, I generate names separately from adding domains etc
    def unique_name(self, prefix):
        # add prefix to the generator so names are generated starting with
        # "prefix_0" instead of "prefix"
        self._name_generator.add_name(prefix, conflicting_ok=True)
        return self._name_generator(prefix)

    # def add_iname(self, iname: str) -> None:
    #     self._within_inames |= {iname}
    #
    # def save_within_inames(self) -> None:
    #     self._saved_within_inames.append(self._within_inames)
    #
    # def restore_within_inames(self) -> None:
    #     self._within_inames = self._saved_within_inames.pop(-1)

    @contextlib.contextmanager
    def within_inames(self, inames) -> None:
        self._within_inames_mut |= inames
        yield
        for iname in inames:
            self._within_inames_mut.remove(iname)

    @property
    def _within_inames(self):
        return frozenset(self._within_inames_mut)

    @property
    def _depends_on(self):
        return frozenset({self._last_insn_id}) - {None}

    def _add_instruction(self, insn):
        self._insns.append(insn)
        self._last_insn_id = insn.id


def compile(expr: LoopExpr, name="mykernel"):
    ctx = LoopyCodegenContext()
    _compile(expr, pmap(), ctx)

    # add a no-op instruction touching all of the kernel arguments so they are
    # not silently dropped
    noop = lp.CInstruction(
        (),
        "",
        read_variables=frozenset({a.name for a in ctx.arguments}),
        within_inames=frozenset(),
        within_inames_is_final=True,
    )
    ctx._insns.append(noop)

    translation_unit = lp.make_kernel(
        ctx.domains,
        ctx.instructions,
        ctx.arguments,
        name=name,
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
        # options=lp.Options(check_dep_resolution=False),
    )
    tu = lp.merge((translation_unit, *ctx.subkernels))
    # breakpoint()
    return tu.with_entrypoints("mykernel")


@functools.singledispatch
def _compile(expr: Any, ctx: LoopyCodegenContext) -> None:
    raise TypeError


@_compile.register
def _(
    loop: Loop,
    loop_indices,
    ctx: LoopyCodegenContext,
) -> None:
    _parse_loop(
        loop,
        loop_indices,
        ctx,
    )


# jnames refer to things above but within the axis hierarchy
# loop_indices are matched on the *specific* index component and come from elsewher
def _parse_loop(
    loop: Loop,
    loop_indices,
    codegen_ctx: LoopyCodegenContext,
):
    """This is for something like map(p).index() or map0(map1(p)).index().

    We 'expand' the provided index into something with shape and then
    traverse the leaves.
    """
    """It should be possible to loop over things such as axes[::2][1:].index().

    For such a case we need to traverse the indexed axes in turn until we end up
    with a final set of axes that give the shape of the iteration. As above we then
    need to traverse the leaves in turn.
    """
    """
    11/08

    It is not sufficient to just use _expand_index here. We could have a whole index
    tree here. We need to use visit_indices instead.

    14/08

    Ah but we always have a loop index here, which is itself part of an index tree.
    We need to use the iterset here instead of the index. The iterset must be either
    an index tree, axis tree or indexed axis tree.
    """
    loop_context = {}
    for loop_index, (path, _) in loop_indices.items():
        loop_context[loop_index] = pmap(path)
    loop_context = pmap(loop_context)

    _finalize_parse_loop(
        loop,
        loop.index.iterset.with_context(loop_context),
        codegen_ctx,
        loop_context,
        loop_indices,
    )


def _finalize_parse_loop(
    loop,
    axes,
    codegen_ctx,
    loop_context,
    loop_indices,
):
    _finalize_parse_loop_rec(
        loop,
        axes,
        codegen_ctx,
        loop_context,
        loop_indices,
        current_axis=axes.root,
        current_path=(),
        current_jnames=pmap(),
    )


# this is similar to what we do for assignments but we do something different at the bottom
def _finalize_parse_loop_rec(
    loop,
    axes,
    codegen_ctx,
    loop_context,
    loop_indices,
    *,
    current_axis,
    current_path,
    current_jnames,
):
    # very very similar to _parse_assignment_final
    axis_labels_to_jnames = {}
    for loop_index in loop_context.keys():
        # we don't do anything with src_jnames currently. I should probably just register
        # it as a separate loop index
        _, target_jnames, src_jnames = loop_indices[loop_index]
        for axis_label, jname in target_jnames.items():
            axis_labels_to_jnames[axis_label] = pym.var(jname)

    array_path = pmap(
        {ax: cpt for path in loop_context.values() for ax, cpt in path.items()}
    )

    parse_loop_final_rec(
        loop,
        axes,
        axis_labels_to_jnames,
        array_path,
        codegen_ctx,
        loop_indices,
        current_axis=current_axis,
    )


def parse_loop_final_rec(
    loop,
    axes,
    current_jnames,
    current_path,  # is this needed?
    codegen_ctx,
    loop_indices,
    target_path=pmap(),
    mini_path=(),
    jname_expr_per_axis_label=None,
    *,
    current_axis,
):
    if not jname_expr_per_axis_label:
        jname_expr_per_axis_label = {}
    for axcpt in current_axis.components:
        size = register_extent(axcpt.count, current_path, current_jnames, codegen_ctx)
        iname = codegen_ctx.unique_name("i")
        codegen_ctx.add_domain(iname, size)

        new_jname_expr_per_target_axis_label = jname_expr_per_axis_label.copy()

        new_path = current_path | {current_axis.label: axcpt.label}
        current_jname = iname
        new_jnames = current_jnames | {current_axis.label: pym.var(current_jname)}

        new_target_path = target_path
        new_mini_path = mini_path + ((current_axis, axcpt),)
        if new_mini_path in axes.target_paths:
            mytargetpath = axes.target_paths[new_mini_path]
            new_target_path = target_path | mytargetpath

            # for now, can in theory target multiple target axes
            target_axis = just_one(mytargetpath.keys())

            # breakpoint()
            # just in case
            found = False
            for myaxis, mycomponent in new_mini_path:
                if (myaxis.id, mycomponent.label) in axes.index_exprs:
                    assert not found
                    index_expr = axes.index_exprs[myaxis.id, mycomponent.label]
                    jname_expr = JnameSubstitutor(new_jnames, codegen_ctx)(index_expr)
                    new_jname_expr_per_target_axis_label[target_axis] = jname_expr
                    found = True

            new_mini_path = ()

        with codegen_ctx.within_inames({iname}):
            if subaxis := axes.child(current_axis, axcpt):
                parse_loop_final_rec(
                    loop,
                    axes,
                    new_jnames,
                    new_path,
                    codegen_ctx,
                    loop_indices,
                    new_target_path,
                    new_mini_path,
                    new_jname_expr_per_target_axis_label,
                    current_axis=subaxis,
                )
            else:
                assert not new_mini_path

                for stmt in loop.statements:
                    _compile(
                        stmt,
                        loop_indices
                        | {
                            loop.index: (
                                new_target_path,
                                pmap(new_jname_expr_per_target_axis_label),
                                pmap(
                                    new_jnames
                                ),  # "local index", should probably also be a variable
                            )
                        },
                        codegen_ctx,
                    )


@_compile.register
def _(call: FunctionCall, loop_indices, ctx: LoopyCodegenContext) -> None:
    """
    Turn an exprs.FunctionCall into a series of assignment instructions etc.
    Handles packing/accessor logic.
    """

    temporaries = []
    subarrayrefs = {}
    extents = {}

    # loopy args can contain ragged params too
    loopy_args = call.function.code.default_entrypoint.args[: len(call.arguments)]
    for loopy_arg, arg, spec in checked_zip(loopy_args, call.arguments, call.argspec):
        # create an appropriate temporary
        # we need the indices here because the temporary shape needs to be indexed
        # by the same indices as the original array
        # is this definitely true??? think so. because it gives us the right loops
        # but we only really need it to determine "within" or not...
        # if not isinstance(arg, MultiArray):
        #     # think PetscMat etc
        #     raise NotImplementedError(
        #         "Need to handle indices to create temp shape differently"
        #     )

        loop_context = {}
        for loop_index, (path, _, _) in loop_indices.items():
            loop_context[loop_index] = pmap(path)
        loop_context = pmap(loop_context)

        axes = arg.axes.with_context(loop_context).copy(
            index_exprs=None, layout_exprs=None
        )
        temporary = MultiArray(
            axes,
            name=ctx.unique_name("t"),
            dtype=arg.dtype,
        )
        indexed_temp = temporary

        if loopy_arg.shape is None:
            shape = (temporary.alloc_size,)
        else:
            if np.prod(loopy_arg.shape, dtype=int) != temporary.alloc_size:
                raise RuntimeError("Shape mismatch between inner and outer kernels")
            shape = loopy_arg.shape

        temporaries.append((arg, indexed_temp, spec.access, shape))

        # Register data
        if not isinstance(arg, CalledAxisTree):
            ctx.add_argument(arg.name, arg.dtype)

        ctx.add_temporary(temporary.name, temporary.dtype, shape)

        # subarrayref nonsense/magic
        indices = []
        for s in shape:
            iname = ctx.unique_name("i")
            ctx.add_domain(iname, s)
            indices.append(pym.var(iname))
        indices = tuple(indices)

        subarrayrefs[arg.name] = lp.symbolic.SubArrayRef(
            indices, pym.subscript(pym.var(temporary.name), indices)
        )

        # we need to pass sizes through if they are only known at runtime (ragged)
        # NOTE: If we register an extent to pass through loopy will complain
        # unless we register it as an assumption of the local kernel (e.g. "n <= 3")

        # FIXME ragged is broken since I commented this out! determining shape of
        # ragged things requires thought!
        # for cidx in range(indexed_temp.index.root.degree):
        #     extents |= self.collect_extents(
        #         indexed_temp.index,
        #         indexed_temp.index.root,
        #         cidx,
        #         within_indices,
        #         within_inames,
        #         depends_on,
        #     )

    # TODO this is pretty much the same as what I do in fix_intents in loopexpr.py
    # probably best to combine them - could add a sensible check there too.
    assignees = tuple(
        subarrayrefs[arg.name]
        for arg, spec in checked_zip(call.arguments, call.argspec)
        if spec.access in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}
    )
    expression = pym.primitives.Call(
        pym.var(call.function.code.default_entrypoint.name),
        tuple(
            subarrayrefs[arg.name]
            for arg, spec in checked_zip(call.arguments, call.argspec)
            if spec.access in {READ, RW, INC, MIN_RW, MAX_RW}
        )
        + tuple(extents.values()),
    )

    # TODO get rid of tlang entirely
    # gathers
    for arg, temp, access, shape in temporaries:
        if access in {READ, RW, MIN_RW, MAX_RW}:
            gather = tlang.Read(arg, temp, shape)
        else:
            assert access in {WRITE, INC, MIN_WRITE, MAX_WRITE}
            gather = tlang.Zero(arg, temp, shape)
        build_assignment(gather, loop_indices, ctx)

    ctx.add_function_call(assignees, expression)
    ctx.add_subkernel(call.function.code)

    # scatters
    for arg, temp, access, shape in temporaries:
        if access == READ:
            continue
        elif access in {WRITE, RW, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}:
            scatter = tlang.Write(arg, temp, shape)
        else:
            assert access == INC
            scatter = tlang.Increment(arg, temp, shape)
        build_assignment(scatter, loop_indices, ctx)


# FIXME this is practically identical to what we do in build_loop
# parse_assignment?
def build_assignment(
    assignment,
    loop_indices,
    codegen_ctx,
):
    # each application of an index tree takes an input axis tree and the
    # jnames that apply to each axis component and then filters/transforms the
    # tree and determines instructions that generate these jnames. The resulting
    # axis tree also has unspecified jnames. These are parsed in a final step into
    # actual loops.
    # The first step is therefore to generate these initial jnames, and the last
    # is to emit the loops for the final tree.
    # jnames_per_cpt, array_expr_per_leaf, insns_per_leaf = _prepare_assignment(
    #     assignment, codegen_ctx
    # )

    """
    The difference between iterating over map0(map1(p)).index() and axes.index()
    is that the former may emit multiple loops but only a single jname is produced. For
    the latter multiple jnames may result.

    (This is not quite true. We produce multiple jnames but only a single "jname expr" that
    gets used to index the "prior" thing)??? I suppose the distinction is between whether we
    are indexing the thing (in which case we want the jnames), or using it to index something
    else, where we would want the "jname expr". Maybe this can be thought of as a function from
    jnames -> "jname expr" and we want to go backwards.

    In both cases though the pattern is "loop over this object as if it were a tree".
    I want to generalise this to both of these.

    This seems like a natural thing to do. In the rest of this we maintain the concept of
    "prior" things and transform between indexed axes. In these cases we do not have to. It
    is equivalent to a single step of this mapping. Sort of.
    """

    # get the right index tree given the loop context
    loop_context = {}
    for loop_index, (path, _, _) in loop_indices.items():
        loop_context[loop_index] = pmap(path)
    loop_context = pmap(loop_context)

    _parse_assignment_final(
        assignment,
        assignment.array.axes.with_context(loop_context),
        loop_context,
        loop_indices,
        # I think that these can be skipped and just traverse the final expression thing...
        # jnames_per_cpt,
        # array_expr_per_leaf,
        # insns_per_leaf,
        codegen_ctx,
    )


@dataclasses.dataclass(frozen=True)
class ParseAssignmentPreorderContext:
    target_paths: pmap = pmap()
    index_expr_per_target: pmap = pmap()
    layout_expr_per_target: pmap = pmap()


def _parse_assignment_final(
    assignment,
    axes,
    loop_context,
    loop_indices,
    ctx: LoopyCodegenContext,
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    # target_path = {}
    array_axis_labels_to_jnames = {}
    for loop_index in loop_context.keys():
        # we don't do anything with src_jnames currently. I should probably just register
        # it as a separate loop index
        tpath, target_jnames, src_jnames = loop_indices[loop_index]
        # target_path |= tpath
        # for axis_label, jname in src_jnames.items():
        #     array_axis_labels_to_jnames[axis_label] = jname

        # is this right??? I think this must be wrong
        for axis_label, jname in target_jnames.items():
            assert axis_label not in array_axis_labels_to_jnames
            array_axis_labels_to_jnames[axis_label] = jname
    # target_path = pmap(target_path)
    target_path = axes.target_paths.get((), pmap())

    # array_axis_labels_to_jnames = ???
    # breakpoint()

    # loop indices aren't included in the temporary
    temp_axis_labels_to_jnames = {}

    # breakpoint()
    # array_path = pmap(
    #     {ax: cpt for path in loop_context.values() for ax, cpt in path.items()}
    # )
    array_path = pmap()
    temp_path = pmap()

    if not axes.root:  # catch empty axes here
        jname_expr_per_target_axis_label = {}

        new_target_path = target_path
        target_path_with_axes = axes.orig_axes.path_with_nodes(
            *axes.orig_axes._node_from_path(new_target_path), ordered=True
        )
        for target_axis, target_component in target_path_with_axes:
            index_expr = axes.index_exprs[target_axis.id, target_component]
            jname_expr = JnameSubstitutor(array_axis_labels_to_jnames, ctx)(index_expr)
            jname_expr_per_target_axis_label[target_axis.label] = jname_expr
        jname_expr_per_target_axis_label = pmap(jname_expr_per_target_axis_label)

        # now use this as the replace map to get the right layout expression
        layout_fn = IndexExpressionReplacer(jname_expr_per_target_axis_label)(
            axes.orig_layout_fn[new_target_path]
        )

        array_insns, array_expr = _assignment_array_insn(
            assignment,
            layout_fn,
            array_path,
            array_axis_labels_to_jnames,
            ctx,
        )
        temp_insns, temp_expr = _assignment_temp_insn(
            assignment, temp_path, temp_axis_labels_to_jnames, ctx
        )
        for insn in array_insns:
            ctx.add_assignment(*insn)
        for insn in temp_insns:
            ctx.add_assignment(*insn)
        _shared_assignment_insn(assignment, array_expr, temp_expr, ctx)

    else:
        _parse_assignment_final_rec(
            assignment,
            axes,
            array_axis_labels_to_jnames,
            array_path,
            temp_axis_labels_to_jnames,
            temp_path,
            ctx,
            target_path,
            current_axis=axes.root,
        )


def _parse_assignment_final_rec(
    assignment,
    axes,
    array_jnames,
    array_path,
    temp_jnames,
    temp_path,
    ctx,
    target_path=pmap(),
    mini_path=(),
    jname_expr_per_target_axis_label=None,
    *,
    current_axis,
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if not jname_expr_per_target_axis_label:
        jname_expr_per_target_axis_label = {}

    for axcpt in current_axis.components:
        size = register_extent(axcpt.count, array_path, array_jnames, ctx)
        iname = ctx.unique_name("i")
        ctx.add_domain(iname, size)

        with ctx.within_inames({iname}):
            # definitely don't need to track both of these
            current_jname = iname
            new_array_jnames = array_jnames | {
                current_axis.label: pym.var(current_jname)
            }
            new_temp_jnames = temp_jnames | {current_axis.label: pym.var(current_jname)}
            new_array_path = array_path | {current_axis.label: axcpt.label}
            new_temp_path = temp_path | {current_axis.label: axcpt.label}

            new_target_path = target_path
            new_mini_path = mini_path + ((current_axis, axcpt),)
            if new_mini_path in axes.target_paths:
                mytargetpath = axes.target_paths[new_mini_path]
                new_target_path = target_path | mytargetpath

                # for now, can in theory target multiple target axes
                target_axis = just_one(mytargetpath.keys())

                # breakpoint()
                # just in case
                found = False
                for myaxis, mycomponent in new_mini_path:
                    if (myaxis.id, mycomponent.label) in axes.index_exprs:
                        assert not found
                        index_expr = axes.index_exprs[myaxis.id, mycomponent.label]
                        jname_expr = JnameSubstitutor(new_array_jnames, ctx)(index_expr)
                        jname_expr_per_target_axis_label[target_axis] = jname_expr
                        found = True
                # jname_expr_per_target_axis_label = pmap(
                #     jname_expr_per_target_axis_label
                # )

                new_mini_path = ()

            if subaxis := axes.child(current_axis, axcpt):
                _parse_assignment_final_rec(
                    assignment,
                    axes,
                    new_array_jnames,
                    new_array_path,
                    new_temp_jnames,
                    new_temp_path,
                    ctx,
                    new_target_path,
                    new_mini_path,
                    jname_expr_per_target_axis_label,
                    current_axis=subaxis,
                )
            else:
                assert not new_mini_path
                # if isinstance(assignment.array, CalledAxisTree):
                #     temp_insns, temp_expr = _assignment_temp_insn(
                #         assignment, pmap(), pmap(), ctx
                #     )
                #     for insn in temp_insns:
                #         ctx.add_assignment(*insn)

                # now use this as the replace map to get the right layout expression
                layout_fn = IndexExpressionReplacer(jname_expr_per_target_axis_label)(
                    axes.orig_layout_fn[new_target_path]
                )

                array_insns, array_expr = _assignment_array_insn(
                    assignment,
                    layout_fn,
                    new_array_path,
                    new_array_jnames,
                    ctx,
                )
                temp_insns, temp_expr = _assignment_temp_insn(
                    assignment, new_temp_path, new_temp_jnames, ctx
                )
                for insn in array_insns:
                    ctx.add_assignment(*insn)
                for insn in temp_insns:
                    ctx.add_assignment(*insn)
                _shared_assignment_insn(assignment, array_expr, temp_expr, ctx)


def _assignment_array_insn(assignment, layouts, path, jnames, ctx):
    """

    Return a list of (assignee, expression) tuples and the array expr used
    in the assignment.

    """
    offset_insns, array_offset = emit_assignment_insn(
        layouts,
        path,
        jnames,
        ctx,
    )
    array = assignment.array
    array_expr = pym.subscript(pym.var(array.name), pym.var(array_offset))

    return offset_insns, array_expr


def _assignment_temp_insn(assignment, path, jnames, ctx):
    """

    Return a list of (assignee, expression) tuples and the temp expr used
    in the assignment.

    """
    offset_insns, temp_offset = emit_assignment_insn(
        assignment.temporary.axes.layouts[path],
        path,
        jnames,
        ctx,
    )

    temporary = assignment.temporary

    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    extra_indices = (0,) * (len(assignment.shape) - 1)
    temp_expr = pym.subscript(
        pym.var(temporary.name), extra_indices + (pym.var(temp_offset),)
    )
    return offset_insns, temp_expr


def _shared_assignment_insn(assignment, array_expr, temp_expr, ctx):
    if isinstance(assignment, tlang.Read):
        lexpr = temp_expr
        rexpr = array_expr
    elif isinstance(assignment, tlang.Write):
        lexpr = array_expr
        rexpr = temp_expr
    elif isinstance(assignment, tlang.Increment):
        lexpr = array_expr
        rexpr = array_expr + temp_expr
    elif isinstance(assignment, tlang.Zero):
        lexpr = temp_expr
        rexpr = 0
    else:
        raise NotImplementedError

    ctx.add_assignment(lexpr, rexpr)


def emit_assignment_insn(
    layouts,
    path,
    labels_to_jnames,
    ctx,
):
    offset = ctx.unique_name("off")
    ctx.add_temporary(offset, IntType)
    # ctx.add_assignment(pym.var(offset), 0)

    return (
        emit_layout_insns(
            layouts,
            offset,
            labels_to_jnames,
            ctx,
        ),
        offset,
    )


class JnameSubstitutor(pym.mapper.IdentityMapper):
    # def __init__(self, path, jnames, codegen_context):
    def __init__(self, replace_map, codegen_context):
        # self._path = path
        self._labels_to_jnames = replace_map
        self._codegen_context = codegen_context

    def map_axis_variable(self, expr):
        return self._labels_to_jnames[expr.axis_label]

    # I don't think that this should be required.
    # def map_subscript(self, subscript):
    #     index = self.rec(subscript.index)
    #
    #     trimmed_path = {}
    #     trimmed_jnames = {}
    #     axes = subscript.aggregate.axes
    #     axis = axes.root
    #     while axis:
    #         trimmed_path[axis.label] = self._path[axis.label]
    #         trimmed_jnames[axis.label] = self._labels_to_jnames[axis.label]
    #         cpt = just_one(axis.components)
    #         axis = axes.child(axis, cpt)
    #     trimmed_path = pmap(trimmed_path)
    #     trimmed_jnames = pmap(trimmed_jnames)
    #
    #     insns, varname = _scalar_assignment(
    #         subscript.aggregate,
    #         trimmed_path,
    #         trimmed_jnames,
    #         self._codegen_context,
    #     )
    #     for insn in insns:
    #         self._codegen_context.add_assignment(*insn)
    #     return varname

    # this is cleaner if I do it as a single line expression
    # rather than register assignments for things.
    def map_multi_array(self, array):
        # must be single-component here
        path = array.axes.path(*array.axes.leaf)

        trimmed_jnames = {}
        axes = array.axes
        axis = axes.root
        while axis:
            trimmed_jnames[axis.label] = self._labels_to_jnames[axis.label]
            cpt = just_one(axis.components)
            axis = axes.child(axis, cpt)
        trimmed_jnames = pmap(trimmed_jnames)

        insns, varname = _scalar_assignment(
            array,
            path,
            trimmed_jnames,
            self._codegen_context,
        )
        for insn in insns:
            self._codegen_context.add_assignment(*insn)
        return varname

    def map_called_map(self, expr):
        if not isinstance(expr.function.map_component.array, MultiArray):
            raise NotImplementedError("Affine map stuff not supported yet")

        inner_expr = [self.rec(param) for param in expr.parameters]
        map_array = expr.function.map_component.array

        # handle [map0(p)][map1(p)] where map0 does not have an associated loop
        try:
            jname = self._labels_to_jnames[expr.function.full_map.name]
        except KeyError:
            jname = self._codegen_context.unique_name("j")
            self._codegen_context.add_temporary(jname)
            jname = pym.var(jname)

        # ? = map[j0, j1]
        # where j0 comes from the from_index and j1 is advertised as the shape
        # of the resulting axis (jname_per_cpt)
        # j0 is now fixed but j1 can still be changed
        rootaxis = map_array.axes.root
        inner_axis, inner_cpt = map_array.axes.leaf
        insns, jname_expr = _scalar_assignment(
            map_array,
            pmap({rootaxis.label: just_one(rootaxis.components).label})
            | pmap({inner_axis.label: inner_cpt.label}),
            {rootaxis.label: inner_expr[0]} | {inner_axis.label: inner_expr[1]},
            self._codegen_context,
        )
        for insn in insns:
            self._codegen_context.add_assignment(*insn)
        return jname_expr


def emit_layout_insns(
    layouts,
    offset_var,
    labels_to_jnames,
    ctx,
):
    expr = JnameSubstitutor(labels_to_jnames, ctx)(layouts)

    if expr == ():
        expr = 0

    return ((pym.var(offset_var), expr),)


def register_extent(extent, path, jnames, ctx):
    if isinstance(extent, numbers.Integral):
        return extent

    # actually a pymbolic expression

    path = dict(path)

    replace_map = {}
    for array in collect_arrays(extent):
        # trim path and labels so only existing axes are used
        trimmed_path = {}
        trimmed_jnames = {}
        laxes = array.axes
        laxis = laxes.root
        while laxis:
            trimmed_path[laxis.label] = path[laxis.label]
            trimmed_jnames[laxis.label] = jnames[laxis.label]
            lcpt = just_one(laxis.components)
            laxis = laxes.child(laxis, lcpt)
        trimmed_path = pmap(trimmed_path)
        trimmed_jnames = pmap(trimmed_jnames)

        insns, varname = _scalar_assignment(array, trimmed_path, trimmed_jnames, ctx)
        for lhs, rhs in insns:
            ctx.add_assignment(lhs, rhs)
        replace_map[array.name] = varname

    varname = ctx.unique_name("p")
    ctx.add_temporary(varname)
    ctx.add_assignment(pym.var(varname), replace_variables(extent, replace_map))
    return varname


class MultiArrayCollector(pym.mapper.Collector):
    def map_multi_array(self, expr):
        return {expr}


class VariableReplacer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        self._replace_map = replace_map

    def map_variable(self, expr):
        return self._replace_map.get(expr.name, expr)


def collect_arrays(expr: pym.primitives.Expr):
    collector = MultiArrayCollector()
    return collector(expr)


def replace_variables(
    expr: pym.primitives.Expr, replace_map: dict[str, pym.primitives.Variable]
):
    return VariableReplacer(replace_map)(expr)


def _scalar_assignment(
    array,
    path,
    array_labels_to_jnames,
    ctx,
):
    # Register data
    ctx.add_argument(array.name, array.dtype)

    offset = ctx.unique_name("off")
    ctx.add_temporary(offset, IntType)

    layout_insns = emit_layout_insns(
        array.axes.layouts[path],
        offset,
        array_labels_to_jnames,
        ctx,
    )
    rexpr = pym.subscript(pym.var(array.name), pym.var(offset))
    return layout_insns, rexpr


def find_axis(axes, path, target, current_axis=None):
    """Return the axis matching ``target`` along ``path``.

    ``path`` is a mapping between axis labels and the selected component indices.
    """
    current_axis = current_axis or axes.root

    if current_axis.label == target:
        return current_axis
    else:
        subaxis = axes.child(current_axis, path[current_axis.label])
        if not subaxis:
            assert False, "oops"
        return find_axis(axes, path, target, subaxis)


@functools.singledispatch
def collect_shape_index_callback(index, *args, **kwargs):
    raise TypeError(f"No handler provided for {type(index)}")


@collect_shape_index_callback.register
def _(loop_index: LoopIndex, preorder_ctx, *, loop_indices, **kwargs):
    # global_index = loop_index.loop_index
    # if isinstance(global_index, LocalLoopIndex):
    #     global_index = global_index.global_index
    path = loop_indices[loop_index]

    if isinstance(loop_index.iterset, IndexedAxisTree):
        iterset = just_one(loop_index.iterset.values.values())
    else:
        iterset = loop_index.iterset

    myleaf = iterset.orig_axes._node_from_path(path)
    visited_nodes = iterset.orig_axes.path_with_nodes(*myleaf, ordered=True)

    target_path_per_component = pmap(
        {(): pmap({node.label: cpt_label for node, cpt_label in visited_nodes})}
    )

    # make LoopIndex property?
    index_expr_per_component = pmap(
        {
            (): pmap(
                {
                    # node.label: iterset.index_exprs[node.id, cpt_label]
                    node.label: AxisVariable(node.label)
                    for node, cpt_label in visited_nodes
                }
            )
        }
    )

    layout_expr_per_component = pmap()  # not allowed I believe, or zero?
    return {None: ()}, (
        AxisTree(),
        target_path_per_component,
        index_expr_per_component,
        layout_expr_per_component,
    )


@collect_shape_index_callback.register
def _(local_index: LocalLoopIndex, *, loop_indices, **kwargs):
    raise NotImplementedError
    return collect_shape_index_callback(
        local_index.loop_index, loop_indices=loop_indices, **kwargs
    )


@collect_shape_index_callback.register
def _(called_map: CalledMap, preorder_ctx, **kwargs):
    leaves, index_data = collect_shape_index_callback(
        called_map.from_index, preorder_ctx, **kwargs
    )
    axes, from_target_path_per_cpt, from_index_expr_per_cpt, _ = index_data

    leaf_keys = []
    target_path_per_component = {}
    index_expr_per_component = {}
    layout_expr_per_component = {}

    for from_leaf_key, leaf in leaves.items():
        from_path = from_target_path_per_cpt.get((), pmap())
        from_index_expr = from_index_expr_per_cpt.get((), pmap())
        if not axes.is_empty:
            # FIXME I think that this will break if we have something with multiple axes
            mypath = axes.path_with_nodes(
                *from_leaf_key, ordered=True, and_components=True
            )
            from_path |= from_target_path_per_cpt[mypath]
            from_index_expr |= from_index_expr_per_cpt[mypath]
        else:
            mypath = ()

        # breakpoint()

        # clean this up, we know some of this at an earlier point (loop context)
        components = []
        index_exprs = []
        layout_exprs = []

        bits = called_map.map.bits[pmap(from_path)]
        for map_component in bits:  # each one of these is a new "leaf"
            cpt = AxisComponent(map_component.arity, label=map_component.label)
            components.append(cpt)

            map_var = MapVariable(called_map, map_component)
            axisvar = AxisVariable(called_map.name)

            index_exprs.append(
                # not super happy about this use of values. The called variable doesn't now
                # necessarily know the right axis labels
                pmap(
                    {
                        map_component.target_axis: map_var(
                            *from_index_expr.values(), axisvar
                        )
                    }
                )
            )

            # don't think that this is possible for maps
            layout_exprs.append(pmap())

        axis = Axis(components, label=called_map.name)
        if axes.root:
            # breakpoint()
            axes = axes.add_subaxis(axis, *from_leaf_key)
        else:
            axes = AxisTree(axis)

        for i, (cpt, mapcpt) in enumerate(checked_zip(components, bits)):
            leaf_keys.append((axis.id, cpt.label))

            otherkey = mypath + ((axis, cpt),)
            target_path_per_component[otherkey] = pmap(
                {mapcpt.target_axis: mapcpt.target_component}
            )
            index_expr_per_component[otherkey] = index_exprs[i]
            layout_expr_per_component[otherkey] = layout_exprs[i]

    leaves = {leaf_key: () for leaf_key in leaf_keys}
    return leaves, (
        axes,
        target_path_per_component,
        index_expr_per_component,
        layout_expr_per_component,
    )


@collect_shape_index_callback.register
def _(slice_: Slice, preorder_ctx, *, prev_axes, **kwargs):
    components = []
    target_path_per_axis_tuple = {}
    index_expr_per_target = []
    layout_expr_per_target = {}

    for subslice in slice_.slices:
        # we are assuming that axes with the same label *must* be identical. They are
        # only allowed to differ in that they have different IDs.
        target_axis, target_cpt = prev_axes.find_component(
            slice_.axis, subslice.component, also_node=True
        )
        if isinstance(subslice, AffineSliceComponent):
            # FIXME should be ceiling
            if subslice.stop is None:
                stop = target_cpt.count
            else:
                stop = subslice.stop
            size = (stop - subslice.start) // subslice.step
        else:
            assert isinstance(subslice, Subset)
            size = subslice.array.axes.leaf_component.count
        cpt = AxisComponent(size, label=subslice.label)
        components.append(cpt)

        newvar = AxisVariable(slice_.label)
        if isinstance(subslice, AffineSliceComponent):
            target = target_axis.id, target_cpt.label
            index_expr_per_target.append(
                pmap({slice_.axis: newvar * subslice.step + subslice.start})
            )
            layout_expr_per_target[slice_.axis] = (
                newvar - subslice.start
            ) // subslice.step
        else:
            raise NotImplementedError
            index_expr_per_leaf.append(pmap({slice_.axis: subslice.array}))
            layout_expr_per_leaf.append(pmap({slice_.axis: "inverse search"}))

    # breakpoint()

    axes = AxisTree(Axis(components, label=slice_.label))

    leaves = {}
    target_path_per_cpt = {}
    index_expr_per_cpt = {}
    layout_expr_per_cpt = {}
    for (
        cpt,
        subslice,
        iexpr,
    ) in checked_zip(components, slice_.slices, index_expr_per_target):
        target_path_per_cpt[((axes.root, cpt),)] = pmap(
            {slice_.axis: subslice.component}
        )
        index_expr_per_cpt[((axes.root, cpt),)] = iexpr
        leaves[axes.root.id, cpt.label] = ()

    return leaves, (axes, target_path_per_cpt, index_expr_per_cpt, layout_expr_per_cpt)


def collect_shape_pre_callback(leaf, preorder_ctx, **kwargs):
    return preorder_ctx
    target_path_per_axis_tuple, index_exprs, layout_exprs = leaf

    return ParseAssignmentPreorderContext(
        preorder_ctx.target_paths | target_path_per_axis_tuple,
        preorder_ctx.index_expr_per_target | index_exprs,
        preorder_ctx.layout_expr_per_target | layout_exprs,
    )


def collect_shape_post_callback_terminal(
    leafkey, leaf, preorder_ctx, *, prev_axes, **kwargs
):
    # leaf is a 2-tuple of path and index_exprs. We don't need the path any more
    # return axis and index exprs
    # if preorder_ctx.path:
    #     leaf_axis, leaf_cpt = prev_axes._node_from_path(preorder_ctx.path)
    #     axis_path = prev_axes.path_with_nodes(leaf_axis, leaf_cpt)
    # else:
    #     axis_path = pmap()
    #
    # target_path_per_leaf = {leafkey: preorder_ctx.path}
    # expr_per_leaf = {leafkey: preorder_ctx.jname_exprs}
    # layoutexpr_per_leaf = {leafkey: preorder_ctx.layout_exprs}
    # return None, target_path_per_leaf, expr_per_leaf, layoutexpr_per_leaf
    return None
    return (
        None,
        preorder_ctx.target_paths,
        preorder_ctx.index_expr_per_target,
        preorder_ctx.layout_expr_per_target,
    )


def collect_shape_post_callback_nonterminal(retval, *args, **kwargs):
    """Accumulate results

    We just return an axis tree from below. No special treatment is needed here.

    """
    return retval


# leafdata is return values here
def collect_shape_final_callback(index_data, leafdata):
    if all(d is None for d in leafdata.values()):
        return index_data

    axes, target_paths, index_exprs, layout_exprs = index_data
    for k, (subax, target_paths_, index_expr, layout_expr) in leafdata.items():
        if subax is not None:
            if axes.root:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = subax

        target_paths |= target_paths_
        index_exprs |= index_expr
        layout_exprs |= layout_expr
    return (
        axes,
        pmap(target_paths),
        pmap(index_exprs),
        pmap(layout_exprs),
    )


def collect_target_paths_pre_callback(leaf, preorder_ctx, **kwargs):
    return preorder_ctx


def _collect_target_paths_post_callback_terminal(
    leafkey,
    leaf,
    preorder_ctx,
    **kwargs,
):
    (target_path_for_this_leaf,) = leaf
    return None, target_path_for_this_leaf


def collect_target_paths_final_callback(index_data, leafdata):
    axes = index_data["axes"]
    target_paths_per_leaf = {}
    for k, (subax, target_path) in leafdata.items():
        if subax is not None:
            if axes.root:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = subax
        target_paths_per_leaf[k] = target_path

    return axes, target_paths_per_leaf


# FIXME doesn't belong here
def index_axes(axes: AxisTree, indices: IndexTree, loop_context):
    # offsets are always scalar
    # if isinstance(indexed, CalledAxisTree):
    #     raise NotImplementedError
    # return AxisTree()

    (
        indexed_axes,
        tpaths,
        index_expr_per_target,
        layout_expr_per_target,
    ) = visit_indices(
        indices,
        ParseAssignmentPreorderContext(),
        index_callback=collect_shape_index_callback,
        pre_callback=collect_shape_pre_callback,
        post_callback_terminal=collect_shape_post_callback_terminal,
        post_callback_nonterminal=collect_shape_post_callback_nonterminal,
        final_callback=collect_shape_final_callback,
        loop_indices=loop_context,
        prev_axes=axes,
    )

    if indexed_axes is None:
        indexed_axes = AxisTree()

    # if axes is not None:
    #     return axes
    # else:
    #     return AxisTree()

    # return the new axes plus the new index expressions per leaf
    return indexed_axes, tpaths, index_expr_per_target, layout_expr_per_target

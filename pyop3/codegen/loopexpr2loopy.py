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
from pyop3.axis import (
    AffineLayout,
    Axis,
    AxisComponent,
    AxisTree,
    CalledAxisTree,
    TabulatedLayout,
)
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
    leaves, index_data = index_callback(current_index, **kwargs)

    leafdata = {}
    # loop has size matching the degree of the current_index
    for i, (leafkey, leaf) in enumerate(leaves.items()):
        preorder_ctx_ = pre_callback(leaf, preorder_ctx, **kwargs)

        subindex = indices.parent_to_children[current_index.id][i]
        if subindex is not None:
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
    iterset = loop.index.iterset
    if isinstance(iterset, AxisTree):
        # e.g. axes.index()
        axes = iterset
        itrees = []
    elif isinstance(iterset, IndexedAxisTree):
        # e.g. axes[::2].index()
        # very much like what we do for assignment

        # unroll the index trees and axes
        axes = iterset
        itrees = []
        while isinstance(axes, IndexedAxisTree):
            itrees.insert(0, axes.indices)
            axes = axes.axes
    elif isinstance(iterset, IndexTree):
        raise NotImplementedError("Building the initial axes requires some thought")
        # something like map(p).index()
        axes = None
        itrees = [iterset]
    else:
        raise TypeError

    # initialise some bits
    myleaves, jnames_per_cpt = _parse_index_axis_tree_rec(
        axes,
        codegen_ctx,
        current_axis=axes.root,
        current_path=pmap(),
        current_target_jnames=pmap(),
    )

    insns_per_leaf = {}
    array_expr_per_leaf = {}
    for leaf_key, leafdata in myleaves.items():
        insns_per_leaf[leaf_key] = ()
        array_expr_per_leaf[leaf_key] = leafdata[1]

    for indices in itrees:
        # get the right index tree given the loop context
        loop_context = {}
        for loop_index, (path, _) in loop_indices.items():
            loop_context[loop_index] = pmap(path)
        loop_context = pmap(loop_context)
        index_tree = indices[loop_context]

        # this is exactly the same as what we do for assignment (hence the currently bad name)
        # the only difference between these is that at the end we do something different
        # for loops
        (
            axes,
            jnames_per_cpt,
            array_expr_per_leaf,
            insns_per_leaf,
        ) = _parse_assignment(
            index_tree,
            ParseAssignmentPreorderContext(),
            codegen_ctx=codegen_ctx,
            loop_indices=loop_indices,
            prev_axes=axes,
            prev_jnames_per_cpt=jnames_per_cpt,
            prev_array_expr_per_leaf=array_expr_per_leaf,
            prev_insns_per_leaf=insns_per_leaf,
        )

    ###

    _finalize_parse_loop(
        loop,
        axes,
        jnames_per_cpt,
        array_expr_per_leaf,
        insns_per_leaf,
        codegen_ctx,
        loop_indices,
    )


def _finalize_parse_loop(
    loop,
    axes,
    jnames_per_axcpt,
    array_expr_per_leaf,
    insns_per_leaf,
    codegen_ctx,
    loop_indices,
):
    _finalize_parse_loop_rec(
        loop,
        axes,
        jnames_per_axcpt,
        array_expr_per_leaf,
        insns_per_leaf,
        codegen_ctx,
        loop_indices,
        current_axis=axes.root,
        current_path=(),
        current_jnames=pmap(),
    )


# this is similar to what we do for assignments but we do something different at the bottom
def _finalize_parse_loop_rec(
    loop,
    axes,
    jnames_per_axcpt,
    array_expr_per_leaf,
    insns_per_leaf,
    codegen_ctx,
    loop_indices,
    *,
    current_axis,
    current_path,
    current_jnames,
):
    for axcpt in current_axis.components:
        size = register_extent(axcpt.count, current_path, current_jnames, codegen_ctx)
        iname = codegen_ctx.unique_name("i")
        codegen_ctx.add_domain(iname, size)

        new_path = current_path + ((current_axis.label, axcpt.label),)
        current_jname = jnames_per_axcpt[current_axis.id, axcpt.label]
        new_jnames = current_jnames | {current_axis.label: current_jname}

        with codegen_ctx.within_inames({iname}):
            codegen_ctx.add_assignment(pym.var(current_jname), pym.var(iname))

            if subaxis := axes.child(current_axis, axcpt):
                _finalize_parse_loop_rec(
                    loop,
                    axes,
                    jnames_per_axcpt,
                    array_expr_per_leaf,
                    insns_per_leaf,
                    codegen_ctx,
                    current_axis=subaxis,
                    current_path=new_path,
                    current_jnames=new_jnames,
                    loop_indices=loop_indices,
                )
            else:
                for insn in insns_per_leaf[current_axis.id, axcpt.label]:
                    codegen_ctx.add_assignment(*insn)

                for stmt in loop.statements:
                    _compile(
                        stmt,
                        loop_indices
                        | {
                            loop.index: (
                                new_path,
                                array_expr_per_leaf[current_axis.id, axcpt.label],
                                new_jnames,  # "local index"
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

        # axes = _indexed_axes(arg, loop_indices)
        loop_context = pmap(
            {
                loop_index: pmap(path)
                for loop_index, (path, _, _) in loop_indices.items()
            }
        )
        axes = arg.axis_trees[loop_context]
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

    # each index tree transforms an axis tree into another and produces
    # one index instruction (jname) per index component

    # retrieve the original axes
    # array = assignment.array
    # while isinstance(array, Indexed):
    #     array = array.obj
    # axes = array.axes
    #
    # # unroll the index trees, this should be tidied up
    # array = assignment.array
    # if isinstance(array, CalledAxisTree):
    #     itrees = [array.indices]
    # else:
    #     itrees = []
    #     while isinstance(array, Indexed):
    #         itrees.insert(0, array.itree)
    #         array = array.obj
    #
    # for indices in itrees:
    #     # get the right index tree given the loop context
    #     loop_context = {}
    #     for loop_index, (path, _, _) in loop_indices.items():
    #         loop_context[loop_index] = pmap(path)
    #     loop_context = pmap(loop_context)
    #     index_tree = indices[loop_context]
    #
    #     (
    #         axes,
    #         jnames_per_cpt,
    #         array_expr_per_leaf,
    #         insns_per_leaf,
    #     ) = _parse_assignment(
    #         index_tree,
    #         ParseAssignmentPreorderContext(),
    #         codegen_ctx=codegen_ctx,
    #         loop_indices=loop_indices,
    #         prev_axes=axes,
    #         prev_jnames_per_cpt=jnames_per_cpt,
    #         prev_array_expr_per_leaf=array_expr_per_leaf,
    #         prev_insns_per_leaf=insns_per_leaf,
    #     )

    # lastly generate loops for the tree structure at the end, also generate
    # the intermediate index instructions
    # This will traverse the final axis tree, collecting jnames. At the bottom the
    # leaf insns will be emitted and the temp_expr will be assigned to the array one.
    loop_context = {}
    for loop_index, (path, _, _) in loop_indices.items():
        loop_context[loop_index] = pmap(path)
    loop_context = pmap(loop_context)

    _parse_assignment_final(
        assignment,
        assignment.array.axis_trees[loop_context],
        assignment.array.layout_exprs[loop_context],
        loop_context,
        loop_indices,
        # I think that these can be skipped and just traverse the final expression thing...
        # jnames_per_cpt,
        # array_expr_per_leaf,
        # insns_per_leaf,
        codegen_ctx,
    )


def _prepare_assignment(assignment, ctx: LoopyCodegenContext) -> tuple[pmap, pmap]:
    array = assignment.array
    while isinstance(array, Indexed):
        array = array.obj

    return _prepare_assignment_rec(
        assignment,
        array.axes,
        array.axes.root,
        pmap(),
        pmap(),
        ctx,
    )


# I think that this is pretty much exactly the same as what _expand_index does for
# an axis tree
def _prepare_assignment_rec(
    assignment,
    axes: AxisTree,
    axis: Axis,
    path: pmap,
    jnames: pmap,
    ctx: LoopyCodegenContext,
) -> tuple[pmap, pmap]:
    # catch empty axis trees
    # if axes.is_empty:
    #     return pmap(), pmap({None: 0}), pmap({None: ()})

    jnames_per_axcpt = {}
    insns_per_leaf = {}
    array_expr_per_leaf = {}
    for axcpt in axis.components:
        jname = ctx.unique_name("j")
        ctx.add_temporary(jname)
        new_jnames = jnames | {axis.label: jname}
        # FIXME should only add once per axis label, component label combination
        jnames_per_axcpt[axis.id, axcpt.label] = jname
        new_path = path | {axis.label: axcpt.label}

        if subaxis := axes.child(axis, axcpt):
            (
                subjnames_per_axcpt,
                subarray_expr_per_leaf,
                subinsns_per_leaf,
            ) = _prepare_assignment_rec(
                assignment, axes, subaxis, new_path, new_jnames, ctx
            )
            jnames_per_axcpt |= subjnames_per_axcpt
            insns_per_leaf |= subinsns_per_leaf
            array_expr_per_leaf |= subarray_expr_per_leaf
        else:
            if isinstance(assignment.array, CalledAxisTree):
                # just the offset instructions here, no subscript
                insns, array_expr = emit_assignment_insn(
                    axes, new_path, new_jnames, ctx
                )
            else:
                insns, array_expr = _assignment_array_insn(
                    assignment, axes, new_path, new_jnames, ctx
                )
            insns_per_leaf[axis.id, axcpt.label] = insns
            array_expr_per_leaf[axis.id, axcpt.label] = array_expr

    return pmap(jnames_per_axcpt), pmap(array_expr_per_leaf), pmap(insns_per_leaf)


@dataclasses.dataclass(frozen=True)
class ParseAssignmentPreorderContext:
    path: pmap = pmap()
    insns: tuple = ()
    jname_exprs: pmap = pmap()


def _parse_assignment_pre_callback(
    leaf, preorder_ctx, *, prev_axes, prev_jnames_per_cpt, **kwargs
):
    # unpack leaf
    iaxes_axis_path, iaxes_jname_exprs, iaxes_insns = leaf

    # FIXME iaxes_axis_path should really be unordered
    new_axis_path = preorder_ctx.path | dict(iaxes_axis_path)
    new_insns = preorder_ctx.insns + iaxes_insns
    new_jname_exprs = preorder_ctx.jname_exprs | iaxes_jname_exprs

    # iaxes_axis_path is the path targeted by the handled index
    # for maps this means the target axis.
    # for loop indices this can have multiple entries. We assume them ordered here
    # and successively add them to new_axis_path

    """
    I sort of want to connect the previous jnames to the new exprs. But these may not
    be ordered.
    Let's try looking at the ancestors of the new path.

    I need to do this at the bottom of the tree. Indices do not have the full picture
    at this point.

    Don't pass instructions down. Instead?
    """

    # if preorder_ctx.path:
    #     start_leaf_axis = prev_axes._node_from_path(preorder_ctx.path)
    #     start_path = prev_axes.path_with_nodes(*start_leaf_axis)
    # else:
    #     start_path = set()
    #
    #
    # end_leaf_axis = prev_axes._node_from_path(new_axis_path)
    # end_path = prev_axes.path_with_nodes(*end_leaf_axis)
    #
    # new_axes = {k: v for k, v in end_path.items() if k not in start_path}
    #
    # for prev_axis, prev_cpt_label in new_axes.items():
    #     prev_jname = prev_jnames_per_cpt[prev_axis.id, prev_cpt_label]
    #     jname_expr = iaxes_jname_exprs[prev_axis.label]  # not sure about this
    #     new_insns += ((pym.var(prev_jname), jname_expr),)

    return ParseAssignmentPreorderContext(new_axis_path, new_insns, new_jname_exprs)


def _parse_assignment_post_callback_nonterminal(
    retval,
    leaf,
    preorder_ctx,
    **kwargs,
):
    # this is all just "leaf data"
    return retval


def _parse_assignment_post_callback_terminal(
    leafkey,
    leaf,
    preorder_ctx,
    *,
    prev_axes,
    prev_array_expr_per_leaf,
    prev_insns_per_leaf,
    prev_jnames_per_cpt,
    **kwargs,
):
    new_insns = preorder_ctx.insns

    if preorder_ctx.path:
        leaf_axis, leaf_cpt = prev_axes._node_from_path(preorder_ctx.path)
        axis_path = prev_axes.path_with_nodes(leaf_axis, leaf_cpt)
    else:
        axis_path = {}

    for prev_axis, prev_cpt_label in axis_path.items():
        prev_jname = prev_jnames_per_cpt[prev_axis.id, prev_cpt_label]
        jname_expr = preorder_ctx.jname_exprs[prev_axis.label]
        new_insns += ((pym.var(prev_jname), jname_expr),)

    prev_leaf_axis, prev_leaf_cpt = prev_axes._node_from_path(preorder_ctx.path)
    prev_leaf_key = (prev_leaf_axis.id, prev_leaf_cpt.label)

    array_expr_per_leaf = {leafkey: prev_array_expr_per_leaf[prev_leaf_key]}
    insns_per_leaf = {leafkey: new_insns + prev_insns_per_leaf[prev_leaf_key]}
    return None, pmap(), array_expr_per_leaf, insns_per_leaf


def _parse_assignment_final_callback(
    index_data,
    leafdata,
):
    axes = index_data["axes"]
    jnames_per_cpt = index_data["jnames"]
    array_expr_per_leaf = {}
    insns_per_leaf = {}
    for leafkey, (
        subaxes,
        subjnames,
        subarray_expr_per_leaf,
        subinsns_per_leaf,
    ) in leafdata.items():
        if subaxes:
            if axes.root:
                axes = axes.add_subtree(subaxes, *leafkey)
            else:
                axes = subaxes

        jnames_per_cpt |= subjnames
        array_expr_per_leaf |= subarray_expr_per_leaf
        insns_per_leaf |= subinsns_per_leaf

    return (
        axes,
        jnames_per_cpt,
        array_expr_per_leaf,
        insns_per_leaf,
    )


def _parse_assignment_final(
    assignment,
    axes,
    layout_expr,
    loop_context,
    loop_indices,
    # jnames_per_axcpt,
    # array_expr_per_leaf,
    # insns_per_leaf,
    ctx: LoopyCodegenContext,
):
    if not axes.root:  # catch empty axes here
        # produce the map of axis labels to jnames

        # for the case where shape comes from the temporary do the same thing
        axis_labels_to_jnames = {}
        # FIXME i don't think this is used for loop indices unless we have local indices
        src_axis_labels_to_jnames = {}
        for loop_index, (_, target_jnames, src_jnames) in loop_indices.items():
            for axis_label, jname in target_jnames.items():
                axis_labels_to_jnames[axis_label] = jname
            for axis_label, jname in src_jnames.items():
                src_axis_labels_to_jnames[axis_label] = jname

        path = pmap(
            {ax: cpt for path in loop_context.values() for ax, cpt in path.items()}
        )

        # for insn in insns_per_leaf[None]:
        #     ctx.add_assignment(*insn)
        # array_expr = array_expr_per_leaf[None]
        array_insns, array_expr = _assignment_array_insn(
            assignment, layout_expr[None], path, axis_labels_to_jnames, ctx
        )
        temp_insns, temp_expr = _assignment_temp_insn(assignment, pmap(), pmap(), ctx)
        for insn in array_insns:
            ctx.add_assignment(*insn)
        for insn in temp_insns:
            ctx.add_assignment(*insn)
        _shared_assignment_insn(assignment, array_expr, temp_expr, ctx)

    else:
        _parse_assignment_final_rec(
            assignment,
            axes,
            axes.root,
            # jnames_per_axcpt,
            # array_expr_per_leaf,
            # insns_per_leaf,
            pmap(),
            pmap(),
            ctx,
        )


def _parse_assignment_final_rec(
    assignment,
    axes,
    axis,
    jnames_per_axcpt,
    array_expr_per_leaf,
    insns_per_leaf,
    path: pmap,
    jnames: pmap,
    ctx,
):
    for axcpt in axis.components:
        size = register_extent(axcpt.count, path, jnames, ctx)
        iname = ctx.unique_name("i")
        ctx.add_domain(iname, size)

        current_jname = jnames_per_axcpt[axis.id, axcpt.label]
        new_jnames = jnames | {axis.label: current_jname}
        new_path = path | {axis.label: axcpt.label}

        with ctx.within_inames({iname}):
            ctx.add_assignment(pym.var(current_jname), pym.var(iname))

            if subaxis := axes.child(axis, axcpt):
                _parse_assignment_final_rec(
                    assignment,
                    axes,
                    subaxis,
                    jnames_per_axcpt,
                    array_expr_per_leaf,
                    insns_per_leaf,
                    new_path,
                    new_jnames,
                    ctx,
                )
            else:
                for insn in insns_per_leaf[axis.id, axcpt.label]:
                    ctx.add_assignment(*insn)
                array_expr = array_expr_per_leaf[axis.id, axcpt.label]
                if isinstance(assignment.array, CalledAxisTree):
                    temp_insns, temp_expr = _assignment_temp_insn(
                        assignment, pmap(), pmap(), ctx
                    )
                    for insn in temp_insns:
                        ctx.add_assignment(*insn)
                else:
                    temp_insns, temp_expr = _assignment_temp_insn(
                        assignment, new_path, new_jnames, ctx
                    )
                    for insn in temp_insns:
                        ctx.add_assignment(*insn)
                _shared_assignment_insn(assignment, array_expr, temp_expr, ctx)


@functools.singledispatch
def _expand_index(index, *, loop_indices, codegen_ctx, **kwargs):
    """
    Return an axis tree and jnames corresponding to unfolding the index.

    Note that the # of jnames and path length is often longer than the size
    of the resultant axes. This is because loop indices add jnames but no shape.
    """
    raise TypeError(f"No handler provided for {type(index).__name__}")


@_expand_index.register
def _(index: LoopIndex, *, loop_indices, codegen_ctx, **kwargs):
    global_index = index.loop_index
    if isinstance(global_index, GlobalLoopIndex):
        path, jname_exprs, _ = loop_indices[global_index]
    else:
        assert isinstance(global_index, LocalLoopIndex)
        global_index = global_index.global_index
        path, _, jname_exprs = loop_indices[global_index]
    insns = ()
    # TODO namedtuple anyone?
    return {None: (path, jname_exprs, insns)}, {"axes": AxisTree(), "jnames": pmap()}


@_expand_index.register
def _(index: CalledMap, *, loop_indices, codegen_ctx, **kwargs):
    # old alias
    ctx = codegen_ctx

    leaves, index_data = _expand_index(
        index.from_index, loop_indices=loop_indices, codegen_ctx=ctx, **kwargs
    )
    axes = index_data.get("axes")
    from_jnames = index_data.get("jnames", {})

    jnames_per_cpt = dict(from_jnames)
    leaf_keys = []
    insns_per_leaf = []
    jname_expr_per_leaf = []
    path_per_leaf = []

    for from_leaf_key, leaf in leaves.items():
        from_path, from_jname_exprs, from_insns = leaf

        components = []
        jnames = []
        insns = []
        jname_exprs = []

        bits = index.bits[pmap(from_path)]
        for map_cpt in bits:  # each one of these is a new "leaf"
            myinsns = []

            # map composition does sort of rely on emitting the prior loops. Only the final
            # loop can be sliced? Not really, the whole resulting tree can be...
            myjnames = {}
            for myaxislabel, _ in from_path:
                myjname = ctx.unique_name("j")
                ctx.add_temporary(myjname)
                myexpr = from_jname_exprs[myaxislabel]
                myinsns.append((pym.var(myjname), myexpr))
                myjnames[myaxislabel] = myjname
            myjnames = pmap(myjnames)

            if isinstance(map_cpt, TabulatedMapComponent):
                cpt = AxisComponent(map_cpt.arity, label=map_cpt.label)
                components.append(cpt)

                jname = ctx.unique_name("j")
                ctx.add_temporary(jname)
                jnames.append(jname)

                # ? = map[j0, j1]
                # where j0 comes from the from_index and j1 is advertised as the shape
                # of the resulting axis (jname_per_cpt)
                # j0 is now fixed but j1 can still be changed
                inner_axis, inner_cpt = map_cpt.array.axes.leaf
                insns_, jname_expr = _scalar_assignment(
                    map_cpt.array,
                    pmap(from_path) | pmap({inner_axis.label: inner_cpt.label}),
                    myjnames | {inner_axis.label: jname},
                    ctx,
                )
                myinsns.extend(insns_)

            else:
                raise NotImplementedError

            insns.append(myinsns)
            jname_exprs.append({map_cpt.target_axis: jname_expr})
            path_per_leaf.append(((map_cpt.target_axis, map_cpt.target_component),))

        axis = Axis(components, label=index.name)
        if from_leaf_key:
            axes = axes.add_subaxis(axis, *from_leaf_key)
        else:
            axes = AxisTree(axis)

        for i, cpt in enumerate(components):
            jnames_per_cpt[axis.id, cpt.label] = jnames[i]
            leaf_keys.append((axis.id, cpt.label))
            insns_per_leaf.append(from_insns + tuple(insns[i]))
            jname_expr_per_leaf.append(from_jname_exprs | jname_exprs[i])

    leaves = {
        leaf_key: (path, jname_exprs, insns)
        for (leaf_key, path, jname_exprs, insns) in checked_zip(
            leaf_keys,
            path_per_leaf,
            jname_expr_per_leaf,
            insns_per_leaf,
        )
    }
    return leaves, {"axes": axes, "jnames": jnames_per_cpt}


@_expand_index.register
def _(slice_: Slice, *, codegen_ctx, prev_axes, **kwargs):
    # alias, fix
    ctx = codegen_ctx

    jnames_per_cpt = {}
    leaf_keys = []
    insns_per_leaf = []
    jname_expr_per_leaf = []
    path_per_leaf = []

    components = []
    jnames = []
    jname_exprs = []

    # each one of these is a new "leaf"
    for subslice in slice_.slices:
        prev_cpt = prev_axes.find_component(slice_.axis, subslice.component)
        if isinstance(subslice, AffineSliceComponent):
            # FIXME should be ceiling
            if subslice.stop is None:
                stop = prev_cpt.count
            else:
                stop = subslice.stop
            size = (stop - subslice.start) // subslice.step
        else:
            assert isinstance(subslice, Subset)
            size = subslice.array.axes.size
        cpt = AxisComponent(size, label=prev_cpt.label)
        components.append(cpt)

        jname = ctx.unique_name("j")
        ctx.add_temporary(jname)
        jnames.append(jname)

        if isinstance(subslice, AffineSliceComponent):
            jname_expr = pym.var(jname) * subslice.step + subslice.start
            insns_per_leaf.append(())
        else:
            assert isinstance(subslice, Subset)
            # subsets must be 1D, single component arrays
            subset = subslice.array
            assert subset.axes.depth == 1
            subset_axis = subset.axes.root
            subset_component = just_one(subset_axis.components)
            insns_, jname_expr = _scalar_assignment(
                subset,
                pmap({subset_axis.label: subset_component.label}),
                pmap({subset_axis.label: jname}),
                codegen_ctx,
            )
            insns_per_leaf.append(insns_)

        jname_exprs.append({slice_.axis: jname_expr})

    axis = Axis(components, label=slice_.axis)
    axes = AxisTree(axis)

    for i, (cpt, subslice) in enumerate(checked_zip(components, slice_.slices)):
        jnames_per_cpt[axis.id, cpt.label] = jnames[i]
        leaf_keys.append((axis.id, cpt.label))
        jname_expr_per_leaf.append(jname_exprs[i])
        path_per_leaf.append(((slice_.axis, subslice.component),))

    leaves = {
        leaf_key: (path, jname_exprs, insns)
        for (leaf_key, path, jname_exprs, insns) in checked_zip(
            leaf_keys,
            path_per_leaf,
            jname_expr_per_leaf,
            insns_per_leaf,
        )
    }
    return leaves, {"axes": axes, "jnames": jnames_per_cpt}


@_expand_index.register
def _(axes: AxisTree, *, loop_indices, codegen_ctx, **kwargs):
    leaves, jnames_per_axcpt = _parse_index_axis_tree_rec(
        axes,
        codegen_ctx,
        current_axis=axes.root,
        current_path=pmap(),
        current_target_jnames=pmap(),
    )

    # don't need to construct any axes here since they already exist
    index_data = {"axes": axes, "jnames": jnames_per_axcpt}
    return leaves, index_data


def _parse_index_axis_tree_rec(
    axes: AxisTree, codegen_ctx, *, current_axis, current_path, current_target_jnames
):
    leaves = {}
    jnames_per_cpt = {}
    for axcpt in current_axis.components:
        # for axis trees the src and target jnames are the same
        jname = codegen_ctx.unique_name("j")
        codegen_ctx.add_temporary(jname)
        jnames_per_cpt[current_axis.id, axcpt.label] = jname
        new_target_jnames = current_target_jnames | {current_axis.label: jname}

        new_path = current_path | {current_axis.label: axcpt.label}
        new_insns = ()  # don't generate instructions for axis trees

        if subaxis := axes.child(current_axis, axcpt):
            subleaves, subsrc_jnames = _parse_index_axis_tree_rec(
                axes,
                codegen_ctx,
                current_axis=subaxis,
                current_path=new_path,
                current_target_jnames=new_target_jnames,
            )
            leaves |= subleaves
            jnames_per_cpt |= subsrc_jnames
        else:
            leaves[current_axis.id, axcpt.label] = (
                new_path,
                new_target_jnames,
                new_insns,
            )

    return pmap(leaves), pmap(jnames_per_cpt)


"""This function effectively traverses an index tree and returns a new set of axes
plus things like src and target jnames as well as extra instructions.

This is generic across assignments and loops.

I want to generalise this to work for anything that can be used as an index.
"""
_parse_assignment = functools.partial(
    visit_indices,
    index_callback=_expand_index,
    pre_callback=_parse_assignment_pre_callback,
    post_callback_nonterminal=_parse_assignment_post_callback_nonterminal,
    post_callback_terminal=_parse_assignment_post_callback_terminal,
    final_callback=_parse_assignment_final_callback,
)


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
            path,
        ),
        offset,
    )


class JnameSubstitutor(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        self._replace_map = replace_map

    def map_axis_variable(self, expr):
        return pym.var(self._replace_map[expr.axis_label])


def emit_layout_insns(
    layouts,
    offset_var,
    labels_to_jnames,
    ctx,
    path,
):
    """
    TODO
    """
    # breakpoint()
    insns = []

    expr = JnameSubstitutor(labels_to_jnames)(layouts)

    if expr == ():
        expr = 0

    return ((pym.var(offset_var), expr),)

    # old code
    # expr = 0  # pym.var(offset_var)
    # for layout_fn in layouts:
    #     # TODO singledispatch!
    #     if isinstance(layout_fn, TabulatedLayout):
    #         # trim path and labels so only existing axes are used
    #         trimmed_path = {}
    #         trimmed_jnames = {}
    #         laxes = layout_fn.data.axes
    #         laxis = laxes.root
    #         while laxis:
    #             trimmed_path[laxis.label] = path[laxis.label]
    #             trimmed_jnames[laxis.label] = labels_to_jnames[laxis.label]
    #             lcpt = just_one(laxis.components)
    #             laxis = laxes.child(laxis, lcpt)
    #         trimmed_path = pmap(trimmed_path)
    #         trimmed_jnames = pmap(trimmed_jnames)
    #
    #         varname = ctx.unique_name("p")
    #         new_insns, varname = _scalar_assignment(
    #             layout_fn.data,
    #             trimmed_path,
    #             trimmed_jnames,
    #             ctx,
    #         )
    #         insns += new_insns
    #         expr += varname
    #     elif isinstance(layout_fn, AffineLayout):
    #         start = layout_fn.start
    #         step = layout_fn.step
    #         jname = pym.var(labels_to_jnames[layout_fn.axis])
    #         expr += jname * step + start
    #     else:
    #         raise NotImplementedError
    #
    # ret = tuple(insns) + ((pym.var(offset_var), expr),)
    # return ret


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
    # I don't think that I have to zero it since it all gets added together
    # ctx.add_assignment(pym.var(offset), 0)

    layout_insns = emit_layout_insns(
        array.axes,
        offset,
        array_labels_to_jnames,
        ctx,
        path,
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
def _(loop_index: LoopIndex, *, loop_indices, **kwargs):
    # global_index = loop_index.loop_index
    # if isinstance(global_index, LocalLoopIndex):
    #     global_index = global_index.global_index
    path = loop_indices[loop_index]

    # hacky, path or leaf ID?
    leaf_axis, leaf_cpt = loop_index.iterset._node_from_path(path)
    index_exprs = loop_index.iterset.index_exprs[leaf_axis.id, leaf_cpt.label]
    return {None: (path, index_exprs)}, {"axes": AxisTree()}


@collect_shape_index_callback.register
def _(local_index: LocalLoopIndex, *, loop_indices, **kwargs):
    return collect_shape_index_callback(
        local_index.loop_index, loop_indices=loop_indices, **kwargs
    )


# TODO this could be done with callbacks so we share code with when
# we also want to emit instructions
@collect_shape_index_callback.register
def _(called_map: CalledMap, **kwargs):
    leaves, index_data = collect_shape_index_callback(called_map.from_index, **kwargs)
    axes = index_data["axes"]

    leaf_keys = []
    path_per_leaf = []
    for from_leaf_key, leaf in leaves.items():
        components = []
        (from_path,) = leaf

        # clean this up, we know some of this at an earlier point (loop context)
        bits = called_map.map.map.bits[pmap(from_path)]
        for map_component in bits:  # each one of these is a new "leaf"
            cpt = AxisComponent(map_component.arity, label=map_component.label)
            components.append(cpt)
            path_per_leaf.append(
                pmap({map_component.target_axis: map_component.target_component})
            )

        axis = Axis(components, label=called_map.name)
        if axes.root:
            axes = axes.add_subaxis(axis, *from_leaf_key)
        else:
            axes = AxisTree(axis)

        for i, cpt in enumerate(components):
            leaf_keys.append((axis.id, cpt.label))

    leaves = {
        leaf_key: (path,)
        for (leaf_key, path) in checked_zip(
            leaf_keys,
            path_per_leaf,
        )
    }
    return leaves, {"axes": axes}


@collect_shape_index_callback.register
def _(slice_: Slice, *, prev_axes, **kwargs):
    components = []
    # I think that axis_label should probably be the same for all bits of the slice
    for subslice in slice_.slices:
        prev_cpt = prev_axes.find_component(slice_.axis, subslice.component)
        if isinstance(subslice, AffineSliceComponent):
            # FIXME should be ceiling
            if subslice.stop is None:
                stop = prev_cpt.count
            else:
                stop = subslice.stop
            size = (stop - subslice.start) // subslice.step
        else:
            assert isinstance(subslice, Subset)
            size = subslice.array.axes.size
        cpt = AxisComponent(size, label=prev_cpt.label)
        components.append(cpt)

    axes = AxisTree(Axis(components, label=slice_.axis))
    leaves = {}
    for cpt in axes.root.components:
        path = pmap({axes.root.label: cpt.label})
        leaves[axes.root.id, cpt.label] = (path,)

    return leaves, {"axes": axes}


def collect_shape_pre_callback(leaf, preorder_ctx, **kwargs):
    return preorder_ctx


def collect_shape_post_callback_terminal(leafkey, leaf, preorder_ctx, **kwargs):
    # leaf is a 2-tuple of path and index_exprs. We don't need the path any more
    # return axis and index exprs
    return None, *leaf


def collect_shape_post_callback_nonterminal(retval, **kwargs):
    """Accumulate results

    We just return an axis tree from below. No special treatment is needed here.

    """
    return retval


# leafdata is return values here
def collect_shape_final_callback(index_data, leafdata):
    axes = index_data["axes"]
    expr_per_leaf = {}
    target_path_per_leaf = {}
    for k, (subax, path, expr) in leafdata.items():
        if subax is not None:
            if axes.root:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = subax
        expr_per_leaf[k] = expr
        target_path_per_leaf[k] = path
    return axes, pmap(expr_per_leaf), pmap(target_path_per_leaf)


def _indexed_axes(indexed, loop_indices):
    """Construct an axis tree corresponding to indexed shape.

    Parameters
    ----------
    indicess
        Iterable of index trees corresponding to something like ``dat[x][y]``.


    Notes
    -----
    This function is very similar to other traversals of indexed things. The
    core difference here is that we only care about the shape of the final
    axis tree. No instructions need be emitted.

    """
    # offsets are always scalar
    if isinstance(indexed, CalledAxisTree):
        return AxisTree()

    # get the right index tree given the loop context
    loop_context = {}
    for loop_index, (path, _, _) in loop_indices.items():
        loop_context[loop_index] = pmap(path)
    loop_context = pmap(loop_context)

    # nasty hack, pass a tuple of axis tree and index tree sometimes
    if isinstance(indexed, tuple):
        orig_axes, split_index_tree = indexed
        index_tree = split_index_tree[loop_context]
    else:
        # handle 'indexed of indexed' things
        if isinstance(indexed.obj, Indexed):
            orig_axes = _indexed_axes(indexed.obj, loop_indices)
        else:
            assert isinstance(indexed.obj, MultiArray)
            orig_axes = indexed.obj.axes
        index_tree = indexed.indices[loop_context]

    axes = visit_indices(
        index_tree,
        None,
        index_callback=collect_shape_index_callback,
        pre_callback=collect_shape_pre_callback,
        post_callback_terminal=collect_shape_post_callback_terminal,
        post_callback_nonterminal=collect_shape_post_callback_nonterminal,
        final_callback=collect_shape_final_callback,
        loop_indices=loop_indices,
        prev_axes=orig_axes,
    )

    if axes is not None:
        return axes
    else:
        return AxisTree()


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


def collect_target_paths(indexed, loop_indices):
    # handle 'indexed of indexed' things
    if isinstance(indexed, Indexed):
        if isinstance(indexed.obj, Indexed):
            orig_axes, _ = collect_target_paths(indexed.obj, loop_indices)
        else:
            assert isinstance(indexed.obj, MultiArray)
            orig_axes = indexed.obj.axes
    else:
        assert isinstance(indexed, IndexedAxisTree)
        if isinstance(indexed.axes, IndexedAxisTree):
            orig_axes, _ = collect_target_paths(indexed.axes, loop_indices)
        else:
            assert isinstance(indexed.axes, AxisTree)
            orig_axes = indexed.axes

    # get the right index tree given the loop context
    loop_context = {}
    for loop_index, (path, _) in loop_indices.items():
        loop_context[loop_index] = pmap(path)
    loop_context = pmap(loop_context)
    index_tree = indexed.indices[loop_context]

    axes, target_paths = visit_indices(
        index_tree,
        None,
        index_callback=collect_shape_index_callback,
        pre_callback=collect_target_paths_pre_callback,
        post_callback_terminal=_collect_target_paths_post_callback_terminal,
        post_callback_nonterminal=collect_shape_post_callback_nonterminal,
        final_callback=collect_target_paths_final_callback,
        loop_indices=loop_indices,
        prev_axes=orig_axes,
    )

    if axes is not None:
        return axes, target_paths
    else:
        return AxisTree(), target_paths


# FIXME doesn't belong here
def index_axes(axes: AxisTree, indices: IndexTree, loop_context):
    # offsets are always scalar
    # if isinstance(indexed, CalledAxisTree):
    #     raise NotImplementedError
    # return AxisTree()

    indexed_axes, expr_per_leaf, target_path_per_leaf = visit_indices(
        indices,
        None,
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
    return indexed_axes, expr_per_leaf, target_path_per_leaf

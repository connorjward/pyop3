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

from pyop3 import utils
from pyop3.axes import Axis, AxisComponent, AxisTree, AxisVariable
from pyop3.distarray import MultiArray
from pyop3.dtypes import IntType
from pyop3.indices import (
    AffineMapComponent,
    AffineSliceComponent,
    CalledMap,
    Index,
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
from pyop3.lang import (
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    WRITE,
    Assignment,
    CalledFunction,
    Increment,
    Loop,
    Offset,
    Read,
    Write,
    Zero,
)
from pyop3.log import logger
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


class CodegenContext(abc.ABC):
    pass


class LoopyCodegenContext(CodegenContext):
    def __init__(self):
        self._domains = []
        self._insns = []
        self._args = []
        self._subkernels = []

        self._within_inames = frozenset()
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
        domain_str = f"{{ [{iname}]: {start} <= {iname} < {stop} }}"
        self._domains.append(domain_str)

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

    @contextlib.contextmanager
    def within_inames(self, inames) -> None:
        orig_within_inames = self._within_inames
        self._within_inames |= inames
        yield
        self._within_inames = orig_within_inames

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
        depends_on=ctx._depends_on,
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
    codegen_context: LoopyCodegenContext,
) -> None:
    loop_context = {}
    for loop_index, (source_path, target_path, _, _) in loop_indices.items():
        loop_context[loop_index] = source_path, target_path
    loop_context = pmap(loop_context)

    iterset = loop.index.iterset.with_context(loop_context)
    parse_loop_properly_this_time(loop, iterset, loop_indices, codegen_context)


def parse_loop_properly_this_time(
    loop,
    axes,
    loop_indices,
    codegen_context,
    *,
    axis=None,
    source_path=pmap(),
    target_path=pmap(),
    iname_replace_map=pmap(),
    jname_replace_map=pmap(),
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if axes.is_empty:
        raise NotImplementedError("does this even make sense?")

    axis = axis or axes.root

    domain_insns = []
    leaf_data = []

    for component in axis.components:
        iname = codegen_context.unique_name("i")
        extent_var = register_extent(
            component.count, iname_replace_map | jname_replace_map, codegen_context
        )
        codegen_context.add_domain(iname, extent_var)

        new_source_path = source_path | {axis.label: component.label}
        new_target_path = target_path | axes.target_path_per_component.get(
            (axis.id, component.label), {}
        )
        new_iname_replace_map = iname_replace_map | {axis.label: pym.var(iname)}

        # these aren't jnames!
        my_index_exprs = axes.index_exprs_per_component.get(
            (axis.id, component.label), {}
        )
        jname_extras = {}
        for axis_label, index_expr in my_index_exprs.items():
            jname_expr = JnameSubstitutor(
                new_iname_replace_map | jname_replace_map, codegen_context
            )(index_expr)
            jname_extras[axis_label] = jname_expr

        new_jname_replace_map = jname_replace_map | jname_extras

        with codegen_context.within_inames({iname}):
            if subaxis := axes.child(axis, component):
                parse_loop_properly_this_time(
                    loop,
                    axes,
                    loop_indices,
                    codegen_context,
                    axis=subaxis,
                    source_path=new_source_path,
                    target_path=new_target_path,
                    iname_replace_map=new_iname_replace_map,
                    jname_replace_map=new_jname_replace_map,
                )
            else:
                for stmt in loop.statements:
                    _compile(
                        stmt,
                        loop_indices
                        | {
                            loop.index: (
                                new_source_path,
                                new_target_path,
                                new_jname_replace_map,
                                new_iname_replace_map,
                            )
                        },
                        codegen_context,
                    )


@_compile.register
def _(call: CalledFunction, loop_indices, ctx: LoopyCodegenContext) -> None:
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
        for loop_index, (source_path, target_path, _, _) in loop_indices.items():
            loop_context[loop_index] = source_path, target_path
        loop_context = pmap(loop_context)

        if isinstance(arg, MultiArray):
            axes = arg.axes.with_context(loop_context).copy(
                index_exprs=None,
            )
        else:
            assert isinstance(arg, Offset)
            axes = AxisTree()
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
        # TODO more generic check
        if not isinstance(arg, Offset):
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

    # gathers
    for arg, temp, access, shape in temporaries:
        if access in {READ, RW, MIN_RW, MAX_RW}:
            gather = Read(arg, temp, shape)
        else:
            assert access in {WRITE, INC, MIN_WRITE, MAX_WRITE}
            gather = Zero(arg, temp, shape)
        build_assignment(gather, loop_indices, ctx)

    ctx.add_function_call(assignees, expression)
    ctx.add_subkernel(call.function.code)

    # scatters
    for arg, temp, access, shape in temporaries:
        if access == READ:
            continue
        elif access in {WRITE, RW, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}:
            scatter = Write(arg, temp, shape)
        else:
            assert access == INC
            scatter = Increment(arg, temp, shape)
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
    for loop_index, (source_path, target_path, _, _) in loop_indices.items():
        loop_context[loop_index] = source_path, target_path
    loop_context = pmap(loop_context)

    parse_assignment_properly_this_time(
        assignment,
        assignment.array.axes.with_context(loop_context),
        loop_indices,
        codegen_ctx,
    )


def parse_assignment_properly_this_time(
    assignment,
    axes,
    loop_indices,
    codegen_context,
    *,
    axis=None,
    source_path=pmap(),
    target_path=None,
    iname_replace_map=pmap(),
    jname_replace_map=None,
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if axis is None:
        axis = axes.root
        target_path = axes.target_path_per_component.get(None, pmap())
        iname_replace_map = pmap(
            {
                axis_label: iname_var
                for _, _, _, replace_map in loop_indices.values()
                for axis_label, iname_var in replace_map.items()
            }
        )
        jname_replace_map = pmap(
            {
                axis_label: iname_var
                for _, _, replace_map, _ in loop_indices.values()
                for axis_label, iname_var in replace_map.items()
            }
        )

        my_index_exprs = axes.index_exprs_per_component.get(None, pmap())
        jname_extras = {}
        for axis_label, index_expr in my_index_exprs.items():
            jname_expr = JnameSubstitutor(
                iname_replace_map | jname_replace_map, codegen_context
            )(index_expr)
            jname_extras[axis_label] = jname_expr
        jname_replace_map = jname_replace_map | jname_extras

    if axes.is_empty:
        add_leaf_assignment(
            assignment,
            axes,
            source_path,
            target_path,
            iname_replace_map,
            jname_replace_map,
            codegen_context,
        )
        return

    for component in axis.components:
        iname = codegen_context.unique_name("i")
        extent_var = register_extent(
            component.count, iname_replace_map | jname_replace_map, codegen_context
        )
        codegen_context.add_domain(iname, extent_var)

        new_source_path = source_path | {axis.label: component.label}  # not used
        new_target_path = target_path | axes.target_path_per_component.get(
            (axis.id, component.label), {}
        )

        new_iname_replace_map = iname_replace_map | {axis.label: pym.var(iname)}

        # I don't like that I need to do this here and also when I emit the layout
        # instructions.
        my_index_exprs = axes.index_exprs_per_component.get(
            (axis.id, component.label), {}
        )
        jname_extras = {}
        for axis_label, index_expr in my_index_exprs.items():
            jname_expr = JnameSubstitutor(
                new_iname_replace_map | jname_replace_map, codegen_context
            )(index_expr)
            jname_extras[axis_label] = jname_expr
        breakpoint()
        new_jname_replace_map = jname_replace_map | jname_extras

        with codegen_context.within_inames({iname}):
            if subaxis := axes.child(axis, component):
                parse_assignment_properly_this_time(
                    assignment,
                    axes,
                    loop_indices,
                    codegen_context,
                    axis=subaxis,
                    source_path=new_source_path,
                    target_path=new_target_path,
                    iname_replace_map=new_iname_replace_map,
                    jname_replace_map=new_jname_replace_map,
                )

            else:
                add_leaf_assignment(
                    assignment,
                    axes,
                    new_source_path,
                    new_target_path,
                    new_iname_replace_map,
                    new_jname_replace_map,
                    codegen_context,
                )


# TODO I should disable emitting instructions for things like zero where we
# don't want insns for the array
def add_leaf_assignment(
    assignment,
    axes,
    source_path,
    target_path,
    iname_replace_map,
    jname_replace_map,
    codegen_context,
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if isinstance(assignment.array, MultiArray):
        array_expr = make_array_expr(
            assignment,
            axes.orig_layout_fn[target_path],
            target_path,
            iname_replace_map | jname_replace_map,
            codegen_context,
        )
    else:
        assert isinstance(assignment.array, Offset)
        array_expr = make_offset_expr(
            axes.orig_layout_fn[target_path],
            iname_replace_map | jname_replace_map,
            codegen_context,
        )
    temp_expr = make_temp_expr(
        assignment, source_path, iname_replace_map, codegen_context
    )
    _shared_assignment_insn(assignment, array_expr, temp_expr, codegen_context)


def make_array_expr(assignment, layouts, path, jnames, ctx):
    """

    Return a list of (assignee, expression) tuples and the array expr used
    in the assignment.

    """
    array_offset = make_offset_expr(
        layouts,
        jnames,
        ctx,
    )
    array = assignment.array
    array_expr = pym.subscript(pym.var(array.name), array_offset)

    return array_expr


def make_temp_expr(assignment, path, jnames, ctx):
    """

    Return a list of (assignee, expression) tuples and the temp expr used
    in the assignment.

    """
    layout = assignment.temporary.axes.layouts[path]
    temp_offset = make_offset_expr(
        layout,
        jnames,
        ctx,
    )

    temporary = assignment.temporary

    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    extra_indices = (0,) * (len(assignment.shape) - 1)
    # also has to be a scalar, not an expression
    temp_offset_var = ctx.unique_name("off")
    ctx.add_temporary(temp_offset_var)
    ctx.add_assignment(temp_offset_var, temp_offset)
    temp_offset_var = pym.var(temp_offset_var)
    temp_expr = pym.subscript(
        pym.var(temporary.name), extra_indices + (temp_offset_var,)
    )
    return temp_expr


def _shared_assignment_insn(assignment, array_expr, temp_expr, ctx):
    if isinstance(assignment, Read):
        lexpr = temp_expr
        rexpr = array_expr
    elif isinstance(assignment, Write):
        lexpr = array_expr
        rexpr = temp_expr
    elif isinstance(assignment, Increment):
        lexpr = array_expr
        rexpr = array_expr + temp_expr
    elif isinstance(assignment, Zero):
        lexpr = temp_expr
        rexpr = 0
    else:
        raise NotImplementedError

    ctx.add_assignment(lexpr, rexpr)


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

        varname = _scalar_assignment(
            array,
            path,
            trimmed_jnames,
            self._codegen_context,
        )
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
        jname_expr = _scalar_assignment(
            map_array,
            pmap({rootaxis.label: just_one(rootaxis.components).label})
            | pmap({inner_axis.label: inner_cpt.label}),
            {rootaxis.label: inner_expr[0], inner_axis.label: inner_expr[1]},
            self._codegen_context,
        )
        return jname_expr


def make_offset_expr(
    layouts,
    jname_replace_map,
    codegen_context,
):
    expr = 0
    for axis_label, layout_expr in layouts.items():
        jname_expr = JnameSubstitutor(jname_replace_map, codegen_context)(layout_expr)
        expr += jname_expr

    # if expr == ():
    #     expr = 0

    return expr


def register_extent(extent, jnames, ctx):
    if isinstance(extent, numbers.Integral):
        return extent

    # actually a pymbolic expression
    if not isinstance(extent, MultiArray):
        raise NotImplementedError("need to tidy up assignment logic")

    path = extent.axes.path(*extent.axes.leaf)
    expr = _scalar_assignment(extent, path, jnames, ctx)

    varname = ctx.unique_name("p")
    ctx.add_temporary(varname)
    ctx.add_assignment(pym.var(varname), expr)
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

    offset_expr = make_offset_expr(
        array.axes.layouts[path],
        array_labels_to_jnames,
        ctx,
    )
    rexpr = pym.subscript(pym.var(array.name), offset_expr)
    return rexpr


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

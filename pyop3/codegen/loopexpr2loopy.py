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
from pyop3.axis import AffineLayout, Axis, AxisComponent, AxisTree, TabulatedLayout
from pyop3.distarray import IndexedMultiArray, MultiArray
from pyop3.dtypes import IntType
from pyop3.index import (
    AffineMap,
    IdentityMap,
    Index,
    IndexTree,
    Map,
    Slice,
    TabulatedMap,
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
    StrictlyUniqueSet,
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
    breakpoint()
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
    _parse_loop(loop, loop_indices, ctx, loop.index.root, pmap(), pmap(), pmap(), ())


# jnames refer to things above but within the axis hierarchy
# loop_indices are matched on the *specific* index component and come from elsewher
def _parse_loop(
    loop: Loop,
    loop_indices,
    ctx: LoopyCodegenContext,
    index,
    path,
    loop_sizes,
    jnames,
    ipath,
):
    for icpt in index.components:
        # TODO I hate that these are mutable
        new_loop_indices = dict(loop_indices)
        # is this needed?
        new_path = dict(path)
        new_path.pop(icpt.from_axis, None)
        new_path |= {icpt.to_axis: icpt.to_cpt}

        new_loop_sizes = dict(loop_sizes)
        new_jnames = dict(jnames)
        new_ipath = ipath + ((index, icpt),)

        # don't overwrite loop sizes, the only valid occasion where we can
        # target an existing domain is if the input and output axes of a
        # map match
        if icpt.to_axis in loop_sizes and icpt.from_axis != icpt.to_axis:
            raise ValueError

        # I am not sure if I need to track the sizes like this anymore, just
        # emit the loops if they don't exist
        # if icpt in new_loop_indices:
        #     # basically do nothing
        #     assert icpt.to_axis not in new_jnames
        #     new_loop_sizes[icpt.to_axis] = 1
        #
        # else:

        # might need to emit a loop over the original domain (fencepost thing)
        new_inames = set()
        if icpt.from_axis not in new_loop_sizes:
            if isinstance(icpt, Slice):
                # the stop is either provided by the index, already registered, or, lastly, the axis size
                if icpt.stop:
                    stop = icpt.stop
                else:
                    # TODO is this still required?
                    axis = find_axis(loop.axes, path, icpt.to_axis)
                    cpt_index = axis.component_index(icpt.to_cpt)
                    stop = axis.components[cpt_index].count

                # TODO add a remainder?
                size = (stop - icpt.start) // icpt.step

            else:
                assert isinstance(cpt, Map)
                # maps map a "from" axis to a "to" axis, "from" must already exist
                # so emit a loop for "to" (i.e. arity)
                axis = find_axis(loop.axes, path, icpt.from_axis)
                cpt_index = axis.component_index(icpt.from_cpt)
                size = axis.components[cpt_index].count

            sizename = register_extent(
                size,
                new_jnames,
                ctx,
            )
            # should these be to_axis instead?
            new_iname = ctx.unique_name("i")
            ctx.add_domain(new_iname, sizename)
            new_jnames[icpt.from_axis] = new_iname
            new_inames.add(new_iname)
            # do I even need this?
            new_loop_sizes[icpt.from_axis] = (
                pym.var(sizename) if isinstance(sizename, str) else sizename
            )

        # now emit a loop for the target axis of the slice/map
        with ctx.within_inames(new_inames):
            new_inames_ = set()
            if isinstance(icpt, Map) and icpt.arity != 1:
                # FIXME
                sizename = register_extent(
                    icpt.arity,
                    new_jnames,
                    ctx,
                )
                new_iname = ctx.unique_name("i")
                ctx.add_domain(new_iname, sizename)
                new_inames_.add(new_iname)
                new_jnames[icpt.to_axis] = new_iname

            with ctx.within_inames(new_inames_):
                # maps transform jnames
                jname = new_jnames.pop(icpt.from_axis)
                new_jname = myinnerfunc(
                    jname,
                    index,
                    icpt,
                    new_jnames,
                    loop_indices,
                    ctx,
                )
                new_jnames[icpt.to_axis] = new_jname
                new_loop_indices[icpt] = new_jname

                if subidx := loop.index.child(index, icpt):
                    # do I need new_path??
                    _parse_loop(
                        loop,
                        loop_indices,
                        ctx,
                        subidx,
                        pmap(new_path),
                        pmap(new_loop_sizes),
                        pmap(new_jnames),
                        new_ipath,
                    )

                # at the bottom, handle remaining sizes
                else:
                    for stmt in loop.statements:
                        _compile(stmt, pmap(new_loop_indices), ctx)


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
        if not isinstance(arg, MultiArray):
            # think PetscMat etc
            raise NotImplementedError(
                "Need to handle indices to create temp shape differently"
            )

        axes = temporary_axes(arg.axes, arg.index, loop_indices)
        temporary = MultiArray(
            axes,
            name=ctx.unique_name("t"),
            dtype=arg.dtype,
        )
        indexed_temp = temporary[...]

        if loopy_arg.shape is None:
            shape = (temporary.alloc_size,)
        else:
            if np.prod(loopy_arg.shape, dtype=int) != temporary.alloc_size:
                raise RuntimeError("Shape mismatch between inner and outer kernels")
            shape = loopy_arg.shape

        temporaries.append((arg, indexed_temp, spec.access, shape))

        # Register data
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
def build_assignment(
    assignment,
    loop_indices,
    ctx,
):
    _parse_assignment_rec(
        assignment,
        loop_indices,
        ctx,
        assignment.array.index.root,
        pmap(),
        pmap(),
        pmap(),
        (),
        pmap(),
        (),
        assignment.temporary.axes.root,
    )


def _parse_assignment_rec(
    assignment,
    loop_indices,
    ctx,
    index,
    path,
    loop_sizes,
    jnames,
    ipath,
    temporary_jnames,
    temp_path,
    taxis,
):
    for icpt, tcpt in checked_zip(index.components, taxis.components):
        new_loop_indices = dict(loop_indices)
        new_loop_sizes = dict(loop_sizes)
        new_ipath = ipath + ((index, icpt),)
        new_jnames = dict(jnames)

        new_temp_path = temp_path + ((taxis.label, tcpt.label),)
        new_temporary_jnames = dict(temporary_jnames)

        # is this needed?
        new_path = dict(path)
        new_path.pop(icpt.from_axis, None)
        new_path.update({icpt.to_axis: icpt.to_cpt})

        ### emit a loop if required

        inames = set()
        if icpt in loop_indices:
            assert icpt.to_axis not in new_jnames
            new_loop_sizes[icpt.to_axis] = 1
            new_jnames[icpt.to_axis] = loop_indices[icpt]

            # hack since temporaries generate layout functions for scalar axes
            iname = ctx.unique_name("i")
            ctx.add_domain(iname, 1)
            inames.add(iname)
            new_temporary_jnames[taxis.label] = iname

        else:
            if isinstance(icpt, Slice):
                # the stop is either provided by the index, already registered, or, lastly, the axis size
                if icpt.stop:
                    stop = icpt.stop
                elif icpt.from_axis in new_loop_sizes:
                    # TODO always pop
                    stop = new_loop_sizes.pop(icpt.from_axis)
                else:
                    # TODO is this still required?
                    axis = find_axis(assignment.array.axes, new_path, icpt.to_axis)
                    cpt_index = axis.component_index(icpt.to_cpt)
                    stop = axis.components[cpt_index].count
                # TODO add a remainder?
                extent = (stop - icpt.start) // icpt.step

            else:
                assert isinstance(cpt, TabulatedMap)
                # FIXME
                extent = icpt.arity

            new_loop_sizes[icpt.to_axis] = extent
            extent_varname = register_extent(
                extent,
                new_jnames,
                ctx,
            )
            new_iname = ctx.unique_name("i")
            ctx.add_domain(new_iname, extent_varname)
            new_jnames[icpt.to_axis] = new_iname
            inames.add(new_iname)
            new_temporary_jnames[taxis.label] = new_iname

        ###

        with ctx.within_inames(inames):
            if subidx := assignment.array.index.child(index, icpt):
                subtaxis = assignment.temporary.axes.child(taxis, tcpt)
                _parse_assignment_rec(
                    assignment,
                    loop_indices,
                    ctx,
                    subidx,
                    pmap(new_path),
                    pmap(new_loop_sizes),
                    pmap(new_jnames),
                    new_ipath,
                    pmap(new_temporary_jnames),
                    new_temp_path,
                    subtaxis,
                )

            else:
                array_offset = emit_assignment_insn(
                    assignment.array.name,
                    assignment.array.axes,
                    pmap(new_path),
                    pmap(new_jnames),
                    ctx,
                )
                temp_offset = emit_assignment_insn(
                    assignment.temporary.name,
                    assignment.temporary.axes,
                    pmap(new_temp_path),
                    pmap(new_temporary_jnames),
                    ctx,
                )

                array = assignment.array
                temporary = assignment.temporary

                # hack to handle the fact that temporaries can have shape but we want to
                # linearly index it here
                extra_indices = (0,) * (len(assignment.shape) - 1)
                temp_expr = pym.subscript(
                    pym.var(temporary.name), extra_indices + (pym.var(temp_offset),)
                )
                array_expr = pym.subscript(pym.var(array.name), pym.var(array_offset))

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


# loop indices and jnames are very similar...
def myinnerfunc(iname, multi_index, index, jnames, loop_indices, ctx):
    if index in loop_indices:
        return loop_indices[index]
    elif isinstance(index, Slice):
        jname = ctx.unique_name("j")
        ctx.add_temporary(jname, IntType)
        ctx.add_assignment(pym.var(jname), pym.var(iname) * index.step + index.start)
        return jname

    elif isinstance(index, IdentityMap):
        index_insns = []
        new_labels = existing_labels
        new_jnames = existing_jnames
        jnames = ()
        new_within = {}

    elif isinstance(index, AffineMap):
        raise NotImplementedError
        jname = self._namer.next("j")
        self._temp_kernel_data.append(
            lp.TemporaryVariable(jname, shape=(), dtype=np.uintp)
        )

        subst_rules = {
            var: pym.var(j)
            for var, j in checked_zip(
                index.expr[0][:-1],
                existing_jnames[-len(index.from_labels) :],
            )
        }
        subst_rules |= {index.expr[0][-1]: pym.var(iname)}

        expr = pym.substitute(index.expr[1], subst_rules)

        index_insn = lp.Assignment(
            pym.var(jname),
            expr,
            id=self._namer.next("myid"),
            within_inames=within_inames | {iname},
            within_inames_is_final=True,
            depends_on=depends_on,
        )
        index_insns = [index_insn]

        temp_labels = list(existing_labels)
        temp_jnames = list(existing_jnames)
        assert len(index.from_labels) == 1
        assert len(index.to_labels) == 1
        for label in index.from_labels:
            assert temp_labels.pop() == label
            temp_jnames.pop()

        (to_label,) = index.to_labels
        new_labels = PrettyTuple(temp_labels) | to_label
        new_jnames = PrettyTuple(temp_jnames) | jname
        jnames = (jname,)
        new_within = {multi_index.label: ((to_label,), (jname,))}

    elif isinstance(index, TabulatedMap):
        jname = ctx.unique_name("j")
        ctx.add_temporary(jname, IntType)

        varname = register_scalar_assignment(
            index.data,
            jnames | {index.from_tuple: iname},
            ctx,
        )
        ctx.add_assignment(pym.var(jname), pym.var(varname))
        return jname
    else:
        raise AssertionError

    assert index_insns is not None
    assert new_labels is not None
    assert new_jnames is not None
    assert jnames is not None
    assert new_within is not None
    self.instructions.extend(index_insns)
    return new_labels, new_jnames, new_within, new_deps


def emit_assignment_insn(
    array_name,
    array_axes,  # can be None
    path,
    labels_to_jnames,
    ctx,
    scalar=False,
):
    offset = ctx.unique_name("off")
    ctx.add_temporary(offset, IntType)
    ctx.add_assignment(pym.var(offset), 0)

    if not scalar:
        axes = array_axes
        axis = axes.root
        emit_layout_insns(
            axes,
            offset,
            labels_to_jnames,
            ctx,
            path,
        )
    return offset


def emit_layout_insns(
    axes,
    offset_var,
    labels_to_jnames,
    ctx,
    path,
):
    """
    TODO
    """
    # breakpoint()
    expr = pym.var(offset_var)
    for layout_fn in axes.layouts[path]:
        # TODO singledispatch!
        if isinstance(layout_fn, TabulatedLayout):
            layout_var = register_scalar_assignment(
                layout_fn.data,
                labels_to_jnames,
                ctx,
            )
            expr += pym.var(layout_var)
        elif isinstance(layout_fn, AffineLayout):
            start = layout_fn.start
            step = layout_fn.step
            jname = pym.var(labels_to_jnames[layout_fn.axis])
            expr += jname * step + start
        else:
            raise NotImplementedError
    ctx.add_assignment(offset_var, expr)


def register_extent(extent, jnames, ctx):
    if isinstance(extent, numbers.Integral):
        return extent

    # actually a pymbolic expression

    # TODO
    # * Traverse the pymbolic expression and generate a replace map for the multi-arrays

    replace_map = {}
    for array in collect_arrays(extent):
        varname = register_scalar_assignment(array, jnames, ctx)
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


def register_scalar_assignment(
    array,
    array_labels_to_jnames,
    ctx,
):
    # Register data
    ctx.add_argument(array.name, array.dtype)
    varname = ctx.unique_name("p")
    ctx.add_temporary(varname)

    array_offset = emit_assignment_insn(
        array.name,
        array.axes,
        array_labels_to_jnames,
        ctx,
    )
    # TODO: Does this function even do anything when I use a scalar?
    # temp_offset = emit_assignment_insn(temp_name, None, {}, ctx, scalar=True)

    rexpr = pym.subscript(pym.var(array.name), pym.var(array_offset))
    ctx.add_assignment(pym.var(varname), rexpr)
    return varname


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


def temporary_axes(axes, indices, loop_indices):
    return _temporary_axes_rec(
        axes, indices, loop_indices, indices.root, pmap(), PrettyTuple(), pmap()
    )


def _temporary_axes_rec(
    axes,
    indices,
    loop_indices,
    index,
    axis_path,
    index_path,
    sizes,
):
    component_sizes = []
    subtrees = []
    for icpt in index.components:
        new_axis_path = axis_path
        if icpt.from_axis in new_axis_path:
            assert new_axis_path[icpt.from_axis] == icpt.from_cpt
            new_axis_path = new_axis_path.discard(icpt.from_axis)
        new_axis_path |= {icpt.to_axis: icpt.to_cpt}
        new_index_path = index_path | (index, icpt)
        new_sizes = sizes

        """
        not sure quite what to do here, if I emit two loops what does this mean
        for the shape of the temporary? it doesn't really make sense
        the idea is that we emit a loop for each target axis here and this is the
        source of the shape. what about fencepost stuff then?

        is there a situation where this would occur for a temporary?
        it happens every time we have a fresh full slice.

        we could enforce that only slices "begin" fresh loops. This effectively
        means that maps would emit shape for their "output" axes whilst slices
        would emit shape for their "input" axes (their output arity being 1).

        maybe this makes some sense. Maps have natural semantics of "from" and "to"
        whereas a slice expects a source axis instead.
        """
        if icpt in loop_indices:
            assert icpt.to_axis not in new_sizes
            component_sizes.append(1)
            new_sizes |= {icpt.to_axis: 1}
        else:
            if isinstance(icpt, Slice):
                if icpt.stop:
                    stop = icpt.stop
                else:
                    # TODO is this still required?
                    axis = find_axis(axes, axis_path, icpt.to_axis)
                    cpt_index = axis.component_index(icpt.to_cpt)
                    stop = axis.components[cpt_index].count

                # TODO add a remainder?
                size = (stop - icpt.start) // icpt.step
                component_sizes.append(size)

            else:
                assert isinstance(icpt, Map)
                axis = find_axis(axes, axis_path, icpt.from_axis)
                cpt_index = axis.component_index(icpt.from_cpt)
                size = axis.components[cpt_index].count
                component_sizes.append(size)

        if subidx := indices.child(index, icpt):
            subtree = _temporary_axes_rec(
                axes,
                indices,
                loop_indices,
                subidx,
                pmap(new_axis_path),
                new_index_path,
                new_sizes,
            )
            subtrees.append(subtree)
        else:
            subtree = AxisTree()
            subtrees.append(subtree)

    # convert the subtrees to a full one
    root = Axis(component_sizes)
    parent_to_children = {
        root.id: [subtree.root] for subtree in subtrees
    } | merge_dicts([subtree.parent_to_children for subtree in subtrees])
    return AxisTree(root, parent_to_children)

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

        self._within_inames = frozenset()
        self._saved_within_inames = []
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
            within_inames=self._within_inames,
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

    def add_iname(self, iname: str) -> None:
        self._within_inames |= {iname}

    def save_within_inames(self) -> None:
        self._saved_within_inames.append(self._within_inames)

    def restore_within_inames(self) -> None:
        self._within_inames = self._saved_within_inames.pop(-1)

    @contextlib.contextmanager
    def saved_within_inames(self) -> None:
        self.save_within_inames()
        yield
        self.restore_within_inames()

    @property
    def _depends_on(self):
        return frozenset({self._last_insn_id}) - {None}

    def _add_instruction(self, insn):
        self._insns.append(insn)
        self._last_insn_id = insn.id


def compile(expr: LoopExpr, name="mykernel"):
    ctx = LoopyCodegenContext()
    _compile(expr, {}, ctx)

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
def _(loop: Loop, loop_indices, ctx: LoopyCodegenContext) -> None:
    # this is a map from axis labels to extents, if we hit an index over a new thing
    # then we get the size from the array, else we can modify the sizes provided
    # how do maps impact this? just like a slice I think, "consume" the prior index
    # and yield a new one with size arity? no. it wouldn't modify the input set, a slice does

    # I need to think about how I traverse the iterset since I only want one path at a time,
    # but some loops are reused!

    # we should definitely do some of the this at the bottom of the traversal, that way we
    # know which axes are getting looped over? tricky if we have duplicate axis labels on
    # different paths. collect axes and components as we go down and then find the right ones
    # to reconstruct the path at the bottom.

    # NOTE: currently maps only map between components of the same axis.
    # I don't know if this is a strict limitation or just untested.

    for leaf_index, leaf_cpt in loop.index.leaves:
        index_path = loop.index.path(leaf_index, leaf_cpt)

        axis_path = {}
        for index, cpt in index_path:
            if cpt.from_axis in axis_path:
                axis_path.pop(cpt.from_axis)
            axis_path |= {cpt.to_axis: cpt.to_cpt}

        # register loops. this must be done at the bottom of the tree because we
        # only know now how big the emitted loops need to be. For example repeated
        # slices of the same axis will result in a single, smaller, loop.

        # map from axes to sizes, components? maps always target the same axis
        # so should be fine.
        extents = collect_extents(loop.index.axes, axis_path, index_path, loop_indices)

        # now generate loops for each of these extents, keep the same mapping from
        # axis labels to, now, inames
        with ctx.saved_within_inames():
            new_inames = set()
            jnames = {}
            for axis_label, extent in extents.items():
                extent = register_extent(
                    extent,
                    jnames,
                    ctx,
                )
                iname = ctx.unique_name("i")
                ctx.add_domain(iname, extent)
                ctx.add_iname(iname)
                jnames[axis_label] = iname
                new_inames.add(iname)

            # now traverse the slices in reverse, transforming the inames to jnames and
            # the right index expressions
            new_loop_indices = {}
            for index, icpt in reversed(loop.index.path(leaf_index, leaf_cpt)):
                jname = jnames.pop(icpt.to_tuple)
                new_jname = myinnerfunc(
                    jname,
                    index,
                    icpt,
                    jnames,
                    new_loop_indices,
                    ctx,
                )
                assert icpt.from_tuple not in jnames
                jnames[icpt.from_tuple] = new_jname
                new_loop_indices |= {icpt: new_jname}

            # The loop indices have been registered, now handle the loop statements
            for stmt in loop.statements:
                _compile(stmt, new_loop_indices, ctx)


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
def build_assignment(assignment, loop_indices, ctx):
    for (lleaf, lcidx), (rleaf, rcidx) in checked_zip(
        assignment.lhs.index.leaves, assignment.rhs.index.leaves
    ):
        # copied from build_loop
        lindex_path = assignment.lhs.index.path(lleaf, lcidx)
        laxis_path = {}
        for _, cpt in lindex_path:
            if cpt.from_axis in laxis_path:
                laxis_path.pop(cpt.from_axis)
            laxis_path |= {cpt.to_axis: cpt.to_cpt}

        rindex_path = assignment.rhs.index.path(rleaf, rcidx)
        raxis_path = {}
        for _, cpt in rindex_path:
            if cpt.from_axis in raxis_path:
                raxis_path.pop(cpt.from_axis)
            raxis_path |= {cpt.to_axis: cpt.to_cpt}

        # at the bottom, now emit the loops and assignment

        # register loops. this must be done at the bottom of the tree because we
        # only know now how big the emitted loops need to be. For example repeated
        # slices of the same axis will result in a single, smaller, loop.

        # map from axes to sizes, components? maps always target the same axis
        # so should be fine.
        lextents = collect_extents(
            assignment.lhs.axes,
            laxis_path,
            lindex_path,
            loop_indices,
        )
        rextents = collect_extents(
            assignment.rhs.axes,
            raxis_path,
            rindex_path,
            loop_indices,
        )

        # breakpoint()

        liter = iter(lextents.items())
        riter = iter(rextents.items())

        ljnames = {}
        rjnames = {}
        with ctx.saved_within_inames():
            try:
                while True:
                    lnext = next(liter)
                    while lnext[1] == 1:
                        iname = ctx.unique_name("i")
                        ctx.add_domain(iname, 1)
                        ljnames[lnext[0]] = iname
                        ctx.add_iname(iname)
                        lnext = next(liter)
                    rnext = next(riter)
                    while rnext[1] == 1:
                        iname = ctx.unique_name("i")
                        ctx.add_domain(iname, 1)
                        rjnames[rnext[0]] = iname
                        ctx.add_iname(iname)
                        rnext = next(riter)

                    extent = register_extent(
                        single_valued([lnext[1], rnext[1]]),
                        loop_indices,
                        ctx,
                    )
                    iname = ctx.unique_name("i")
                    ctx.add_domain(iname, extent)
                    ljnames[lnext[0]] = iname
                    rjnames[rnext[0]] = iname
                    ctx.add_iname(iname)
            except StopIteration:
                try:
                    # FIXME what if rhs throws the exception instead of lhs?
                    rnext = next(riter)
                    while rnext[1] == 1:
                        iname = ctx.unique_name("i")
                        ctx.add_domain(iname, 1)
                        rjnames[rnext[0]] = iname
                        ctx.add_iname(iname)
                        rnext = next(riter)
                    raise AssertionError("iterator should also be consumed")
                except StopIteration:
                    pass

            # now traverse the slices in reverse, transforming the inames to jnames and
            # the right index expressions
            # LHS
            for multi_index, index in reversed(assignment.lhs.index.path(lleaf, lcidx)):
                jname = ljnames.pop(index.to_tuple)
                new_jname = myinnerfunc(
                    jname,
                    multi_index,
                    index,
                    ljnames,
                    loop_indices,
                    ctx,
                )
                assert index.from_tuple not in ljnames
                ljnames[index.from_tuple] = new_jname

            # RHS
            for multi_index, index in reversed(assignment.rhs.index.path(rleaf, rcidx)):
                jname = rjnames.pop(index.to_tuple)
                new_jname = myinnerfunc(
                    jname,
                    multi_index,
                    index,
                    rjnames,
                    loop_indices,
                    ctx,
                )
                assert index.from_tuple not in rjnames
                rjnames[index.from_tuple] = new_jname

            lhs_labels_to_jnames = ljnames
            rhs_labels_to_jnames = rjnames

            if assignment.lhs is assignment.array:
                array_labels_to_jnames = lhs_labels_to_jnames
                temp_labels_to_jnames = rhs_labels_to_jnames
            else:
                temp_labels_to_jnames = lhs_labels_to_jnames
                array_labels_to_jnames = rhs_labels_to_jnames

            ###

            array_offset = emit_assignment_insn(
                assignment.array.name,
                assignment.array.axes,
                array_labels_to_jnames,
                ctx,
            )
            temp_offset = emit_assignment_insn(
                assignment.temporary.name,
                assignment.temporary.axes,
                temp_labels_to_jnames,
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
        path = pmap()
        while axis:
            cpt = just_one(
                c for c in axis.components if (axis.label, c.label) in labels_to_jnames
            )
            path |= {axis.label: cpt.label}
            axis = axes.child(axis, cpt)

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
            jname = pym.var(labels_to_jnames[(layout_fn.axis, layout_fn.cpt)])
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
        return find_axis(axes, path, target, subaxis)


# FIXME I think that I may have to emit instructions here if I have maps
# (esp. map composition). Also I should probably "consume" extents when I emit maps
def collect_extents(axes, path, index_path, loop_indices):
    extents = {}
    for index, cpt in index_path:
        if cpt in loop_indices:
            extents[cpt.to_tuple] = 1
        else:
            # TODO singledispatch
            if isinstance(cpt, Slice):
                # the stop is either provided by the index, already registered, or, lastly, the axis size
                if cpt.stop:
                    stop = cpt.stop
                elif cpt.from_tuple in extents:
                    stop = extents.pop(cpt.from_tuple)
                else:
                    # TODO is this still required?
                    axis = find_axis(axes, path, cpt.from_axis)
                    cpt_index = axis.component_index(cpt.from_cpt)
                    stop = axis.components[cpt_index].count
                new_extent = (stop - cpt.start) // cpt.step
                extents[cpt.to_tuple] = new_extent
            else:
                assert isinstance(cpt, TabulatedMap)
                # maps do not declare a "stop" so we can only find the extent
                # from prior calculations or from looking at the axes
                # we don't actually care what this size is though I don't think
                # because we always emit arity-many
                if cpt.from_tuple in extents:
                    extents.pop(cpt.from_tuple)
                extents[cpt.to_tuple] = cpt.arity

    return extents


def temporary_axes(
    axes,
    indices,
    loop_indices,
    index=None,
    axis_path=pmap(),
    index_path=PrettyTuple(),
):
    index = index or indices.root
    # TODO copied from build_loop, refactor into a tree traversal
    # also copied from build_assignment - convergence!
    # also the same as fill_shape
    # I can just loop over the leaves at the base? sorta, need to build the tree
    # If I track the leaves I can still reconstruct at the base
    # FIXME need to handle within_indices
    subtrees = []
    for index_cpt in index.components:
        new_axis_path = axis_path
        new_index_path = index_path

        if index_cpt.from_axis in new_axis_path:
            assert new_axis_path[index_cpt.from_axis] == index_cpt.from_cpt
            new_axis_path = new_axis_path.discard(index_cpt.from_axis)
        new_axis_path |= {index_cpt.to_axis: index_cpt.to_cpt}

        new_index_path |= (index, index_cpt)

        if subindex := indices.child(index, index_cpt):
            subtree = temporary_axes(
                axes,
                indices,
                loop_indices,
                subindex,
                new_axis_path,
                new_index_path,
            )
            subtrees.append(subtree)
        else:
            # at the bottom, build the axes
            extents = collect_extents(axes, new_axis_path, new_index_path, loop_indices)

            root = None
            parent_to_children = {}
            for axis_label, extent in extents.items():
                new_axis = Axis(extent, axis_label)
                if root is None:
                    root = new_axis
                else:
                    parent_to_children[prev_axis.id] = (new_axis,)
                prev_axis = new_axis
            subtree = AxisTree(root, parent_to_children)
            subtrees.append(subtree)

    # convert the subtrees to a full one
    root = Axis([1] * index.degree, index.label)
    parent_to_children = {
        root.id: [subtree.root] for subtree in subtrees
    } | merge_dicts([subtree.parent_to_children for subtree in subtrees])
    return AxisTree(root, parent_to_children)

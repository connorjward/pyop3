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
    CalledMap,
    IdentityMap,
    Index,
    IndexTree,
    LoopIndex,
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
    if not isinstance(loop.index.iterset, AxisTree):
        raise NotImplementedError("Could use _expand index to get the right thing?")
    _parse_loop(
        loop,
        loop_indices,
        ctx,
        loop.index.iterset,
        loop.index.iterset.root,
        pmap(),
        pmap(),
        pmap(),
        (),
    )


# jnames refer to things above but within the axis hierarchy
# loop_indices are matched on the *specific* index component and come from elsewher
def _parse_loop(
    loop: Loop,
    loop_indices,
    ctx: LoopyCodegenContext,
    axes,
    axis,
    path,
    loop_sizes,
    jnames,
    ipath,
):
    # I want to do a traversal like I do for assignment. Then I can loop over indexed
    # things. Firstly collect jnames and instructions then traverse...

    # do a BIG hack for now, just for one test
    iname = ctx.unique_name("i0")
    ctx.add_domain(iname, 5)

    # this is awful
    loop_indices = {loop.index: (pmap({"ax0": "cpt0"}), pmap({"ax0": iname}))}

    with ctx.within_inames({iname}):
        for stmt in loop.statements:
            _compile(stmt, loop_indices, ctx)
    return

    ### old code

    for axcpt in axis.components:
        assert isinstance(icpt, (LoopIndex, Map)), "slices not allowed for loops"

        # TODO I hate that these are mutable
        new_loop_indices = loop_indices

        # is this needed?
        new_path = path
        if isinstance(icpt, LoopIndex):
            new_path |= {icpt.to_axis: icpt.to_cpt}
        else:
            assert isinstance(icpt, Map)
            del new_path[icpt.from_axis]
            new_path |= {icpt.to_axis: icpt.to_cpt}

        new_loop_sizes = dict(loop_sizes)
        new_jnames = dict(jnames)
        new_ipath = ipath + ((index, icpt),)

        # don't overwrite loop sizes, the only valid occasion where we can
        # target an existing domain is if the input and output axes of a
        # map match
        # if icpt.to_axis in loop_sizes and icpt.from_axis != icpt.to_axis:
        #     raise ValueError

        new_inames = set()
        if icpt in loop_indices:
            assert icpt.to_axis not in new_jnames
            new_loop_sizes[icpt.to_axis] = 1
            new_jnames[icpt.to_axis] = loop_indices[icpt]

        else:
            # need to emit a loop
            if isinstance(icpt, LoopIndex):
                size = icpt.cpt.count
            else:
                assert isinstance(icpt, Map)
                size = icpt.arity

            sizename = register_extent(
                size,
                new_path,
                new_jnames,
                ctx,
            )
            new_iname = ctx.unique_name("i")
            new_inames.add(new_iname)
            ctx.add_domain(new_iname, sizename)

            # do I even need this?
            new_loop_sizes[icpt.to_axis] = (
                pym.var(sizename) if isinstance(sizename, str) else sizename
            )

        with ctx.within_inames(new_inames):
            # maps transform jnames
            if isinstance(icpt, Map):
                jname = new_iname  # ???
                new_jname = myinnerfunc(
                    jname,
                    index,
                    icpt,
                    new_jnames,
                    loop_indices,
                    ctx,
                )
                new_jnames[icpt.to_axis] = new_jname
                new_loop_indices |= {icpt: new_jname}
            else:
                assert isinstance(icpt, LoopIndex)
                new_jnames[icpt.to_axis] = new_iname
                new_loop_indices |= {icpt: new_iname}

            if subaxis := axes.child(axis, axcpt):
                _parse_loop(
                    loop,
                    new_loop_indices,
                    ctx,
                    subidx,
                    new_path,
                    pmap(new_loop_sizes),
                    pmap(new_jnames),
                    new_ipath,
                )

            else:
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

        # !!!!!!!!!!!!!!!!!!!!
        # FIXME hack for testing
        # axes = temporary_axes(arg.axes, indices, loop_indices)
        axes = AxisTree(
            Axis([AxisComponent(2, "a")], "map0", id="root"),
            {"root": [Axis([AxisComponent(3, "a")], "map1")]},
        )
        temporary = MultiArray(
            axes,
            name=ctx.unique_name("t"),
            dtype=arg.dtype,
        )
        # indexed_temp = temporary[...]
        indexed_temp = temporary

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
    # each application of an index tree takes an input axis tree and the
    # jnames that apply to each axis component and then filters/transforms the
    # tree and determines instructions that generate these jnames. The resulting
    # axis tree also has unspecified jnames. These are parsed in a final step into
    # actual loops.
    # The first step is therefore to generate these initial jnames, and the last
    # is to emit the loops for the final tree.
    jnames_per_cpt, insns_per_leaf, array_expr_per_leaf = _prepare_assignment(
        assignment, ctx
    )

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
    axes = assignment.array.axes
    for indices in assignment.array.indicess:
        (
            axes,
            jnames_per_cpt,
            insns_per_leaf,
            array_expr_per_leaf,
        ) = _parse_assignment_rec(
            assignment,
            loop_indices,
            ctx,
            axes,
            indices,
            jnames_per_cpt,
            insns_per_leaf,
            array_expr_per_leaf,
            pmap(),
            pmap(),
            (),
        )

    # lastly generate loops for the tree structure at the end, also generate
    # the intermediate index instructions
    # This will traverse the final axis tree, collecting jnames. At the bottom the
    # leaf insns will be emitted and the temp_expr will be assigned to the array one.
    _parse_assignment_final(
        assignment, axes, jnames_per_cpt, insns_per_leaf, array_expr_per_leaf, ctx
    )


# bad name now
def _parse_assignment_rec(
    assignment,
    loop_indices,
    ctx,
    prior_axes: AxisTree,
    indices: tuple,
    prior_jnames_per_axcpt: pmap,
    prior_insns_per_leaf: pmap,
    prior_array_expr_per_leaf: pmap,
    path: pmap,  # path through prior axes
    jnames: pmap,
    extra_insns: tuple,
):
    """Multi-component index trees are bad because we could then combine a map
    with a slice side-by-side and this doesn't make any sense. What would the new set
    of axes look like? Maps themselves yield
    multiple axis components. We could similarly have a "multi-slice" that could do the
    same? I don't like that very much.
    For now just permit single slices. Since full slices are the default we shouldn't (?)
    lose shape.

    Note: since indices can only act on the current axis of the array the the number of
    options available to slices is very limited. I can slice some or all of the available
    axes! So a slice can be a single entity. Avoid multi-component.

    Slices are basically maps from the current axis, so index.from_axis is undefined and
    we can't do the sort of recursion we can with maps.
    """
    # indices are some sort of conditional tree, only work for specific axis labels
    index, *subindices = indices

    axis = prior_axes._node_from_path(path)

    # loop over the leaves of the axis, jname combo from the maps?
    # indices are therefore a list of things that can turn into trees
    # This function turns an index into an axis tree that is added to the new shape.
    (
        iaxes,
        ijnames_per_cpt,
        iinsns_per_leaf,
        ijname_expr_per_leaf,  # actually multiple of these, one per bit of the path
        ipath_per_leaf,
    ) = _expand_index(index, axis, loop_indices, ctx)

    insns_per_leaf = {}
    array_expr_per_leaf = {}
    jnames_per_cpt = ijnames_per_cpt

    if iaxes:
        # maybe this should actually be turned into the key that I want
        leaves = iaxes.leaves
        new_axes = iaxes
    else:
        leaves = [None]
        new_axes = None

    for ileaf in leaves:
        if ileaf:
            iaxis, icpt = ileaf
            ileaf_key = (iaxis.id, icpt.label)
        else:
            ileaf_key = None

        iinsns = iinsns_per_leaf[ileaf_key]
        ijname_expr = ijname_expr_per_leaf[ileaf_key]
        ipath = ipath_per_leaf[ileaf_key]

        # I believe that this is always the case. Each index in the tree will
        # always target one axis in the input axes. Things like map composition
        # do not matter here (ie map1(map0(p)) means that we care what the output
        # of map1 is, map0 is just the input).
        # Even if we have complicated maps they do not matter here as they are
        # complicated for the input to the map, not the target.
        new_path = path | ipath
        new_extra_insns = list(extra_insns) + list(iinsns)

        # loop over these? Will I hit duplicates? no I wont since ipath is the *new*
        # set of axes that get touched by the index
        mypath = path
        for myaxislabel, mycpt in ipath.items():
            axis = prior_axes._node_from_path(mypath)
            existing_jname = prior_jnames_per_axcpt[axis.id, mycpt]
            myexpr = ijname_expr[myaxislabel]
            new_extra_insns.append((pym.var(existing_jname), myexpr))
            mypath |= {myaxislabel: mycpt}

        # I think that indices also need to form a tree, how could I otherwise
        # index a mixed real space thing?
        if subindices:
            (
                subaxes,
                subjnames_per_axcpt,
                subinsns_per_leaf,
                subarray_expr_per_leaf,
            ) = _parse_assignment_rec(
                assignment,
                loop_indices,
                ctx,
                axes,
                subindices,
                prior_jnames_per_axcpt,
                prior_insns_per_leaf,
                prior_array_expr_per_leaf,
                new_path,
                new_jnames,
                new_extra_insns,
            )
            new_axes = new_axes.add_subtree(subaxes, *ileaf)
            jnames_per_axcpt |= subjnames_per_axcpt
            insns_per_leaf |= subinsns_per_leaf
            array_expr_per_leaf |= prior_array_expr_per_leaf
        else:
            # prepend new_insns_per_leaf
            # awful hack, will only work if indices are depth 1 (as mycpt is output from earlier loop)
            insns_per_leaf[ileaf_key] = (
                tuple(new_extra_insns) + prior_insns_per_leaf[axis.id, mycpt]
            )

            # transfer per leaf things to the new tree (since the leaves change)
            array_expr_per_leaf[ileaf_key] = prior_array_expr_per_leaf[axis.id, mycpt]

    return (
        new_axes,
        pmap(jnames_per_cpt),
        pmap(insns_per_leaf),
        pmap(array_expr_per_leaf),
    )

    ###

    assert False, "old impl"

    # if isinstance(icpt, Slice):
    #     # the stop is either provided by the index, already registered, or, lastly, the axis size
    #     if icpt.stop:
    #         stop = icpt.stop
    #     elif icpt.from_axis in new_loop_sizes:
    #         # TODO always pop
    #         stop = new_loop_sizes.pop(icpt.from_axis)
    #     else:
    #         # TODO is this still required?
    #         axis = find_axis(assignment.array.axes, new_path, icpt.to_axis)
    #         cpt_index = axis.component_index(icpt.to_cpt)
    #         stop = axis.components[cpt_index].count
    #     # TODO add a remainder?
    #     extent = (stop - icpt.start) // icpt.step


def _assignment_array_insn(assignment, path, jnames, ctx):
    """

    Return a list of (assignee, expression) tuples and the array expr used
    in the assignment.

    """
    offset_insns, array_offset = emit_assignment_insn(
        assignment.array.name,
        assignment.array.axes,
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
        assignment.temporary.name,
        assignment.temporary.axes,
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


@functools.singledispatch
def _expand_index(index, axis, loop_indices, ctx):
    """
    Return an axis tree and jnames corresponding to unfolding the index.

    Note that the # of jnames and path length is often longer than the size
    of the resultant axes. This is because loop indices add jnames but no shape.
    """
    raise TypeError


@_expand_index.register
def _(index: LoopIndex, axis, loop_indices, ctx):
    # what do here?
    # I don't have leaves since those are handled outside
    """
    _expand_index either returns a tree and leaf data or it returns None and a single set of leaf data
    """
    path, jname_exprs = loop_indices[index]
    return None, {}, {None: ()}, {None: jname_exprs}, {None: path}


@_expand_index.register
def _(index: CalledMap, axis, loop_indices, ctx):
    (
        from_axes,
        from_jnames_per_cpt,
        from_insns_per_leaf,
        from_jname_expr_per_leaf,
        from_path_per_leaf,
    ) = _expand_index(index.from_index, axis, loop_indices, ctx)

    jnames_per_cpt = dict(from_jnames_per_cpt)
    insns_per_leaf = {}
    jname_expr_per_leaf = {}
    path_per_leaf = {}

    if from_axes:
        leaf_keys = [(a.id, c.label) for a, c in from_axes.leaves]
        axes = from_axes
    else:
        # scalar index, from_axes is None
        leaf_keys = [None]
        axes = None

    for from_leaf_key in leaf_keys:
        from_path = from_path_per_leaf[from_leaf_key]
        myexprs = from_jname_expr_per_leaf[from_leaf_key]

        components = []
        jnames = []
        insns = []
        jname_exprs = []

        # this is a mapping from (from_axis, from_cpt) to an iterable of
        # (map_func, arity, to_axis, to_cpt)
        bits = index.bits[from_path]
        for (
            mycptlabel,
            map_func,
            arity,
            to_axis,
            to_cpt,
        ) in bits:  # each one of these is a new "leaf"
            myinsns = []

            # materialise the jname_expr. When we are indexing arrays this jname
            # is provided externally, but not for index composition
            # could I do this earlier to mirror what I do elsewhere?
            # what jnames are required to index this axis?
            # extra loop indices? generated first?
            # need to convert index into jnames and path - materialise?

            # map composition does sort of rely on emitting the prior loops. Only the final
            # loop can be sliced? Not really, the whole resulting tree can be...
            myjnames = {}
            for myaxislabel in from_path:
                myjname = ctx.unique_name("j")
                ctx.add_temporary(myjname)
                myexpr = myexprs[myaxislabel]
                myinsns.append((pym.var(myjname), myexpr))
                myjnames[myaxislabel] = myjname
            myjnames = pmap(myjnames)

            if isinstance(map_func, MultiArray):  # is this the right class?
                cpt = AxisComponent(arity, label=mycptlabel)
                components.append(cpt)

                jname = ctx.unique_name("j")
                ctx.add_temporary(jname)
                jnames.append(jname)

                # ? = map[j0, j1]
                # where j0 comes from the from_index and j1 is advertised as the shape
                # of the resulting axis (jname_per_cpt)
                # j0 is now fixed but j1 can still be changed
                # no. j0 can *still be modified*
                inner_axis, inner_cpt = map_func.axes.leaf
                insns_, jname_expr = _scalar_assignment(
                    map_func,
                    from_path | pmap({inner_axis.label: inner_cpt.label}),
                    myjnames | {inner_axis.label: jname},
                    ctx,
                )
                myinsns.extend(insns_)

            else:
                raise NotImplementedError

            insns.append(myinsns)
            jname_exprs.append({to_axis: jname_expr})

        axis = Axis(components, label=index.name)
        if axes:
            axes = axes.add_subaxis(axis, *from_leaf_key)
        else:
            axes = AxisTree(axis)

        for i, cpt in enumerate(components):
            # hmm...
            leaf_key = (axis.id, cpt.label)
            jnames_per_cpt[leaf_key] = jnames[i]
            insns_per_leaf[leaf_key] = from_insns_per_leaf[from_leaf_key] + tuple(
                insns[i]
            )
            jname_expr_per_leaf[leaf_key] = (
                from_jname_expr_per_leaf[from_leaf_key] | jname_exprs[i]
            )
            path_per_leaf[leaf_key] = pmap({to_axis: to_cpt})

    return (
        axes,
        pmap(jnames_per_cpt),
        pmap(insns_per_leaf),
        pmap(jname_expr_per_leaf),
        pmap(path_per_leaf),
    )


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
    axes,
    path,
    labels_to_jnames,
    ctx,
):
    offset = ctx.unique_name("off")
    ctx.add_temporary(offset, IntType)
    # ctx.add_assignment(pym.var(offset), 0)

    return (
        emit_layout_insns(
            axes,
            offset,
            labels_to_jnames,
            ctx,
            path,
        ),
        offset,
    )


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
    insns = []

    expr = 0  # pym.var(offset_var)
    for layout_fn in axes.layouts[path]:
        # TODO singledispatch!
        if isinstance(layout_fn, TabulatedLayout):
            # trim path and labels so only existing axes are used
            trimmed_path = {}
            trimmed_jnames = {}
            laxes = layout_fn.data.axes
            laxis = laxes.root
            while laxis:
                trimmed_path[laxis.label] = path[laxis.label]
                trimmed_jnames[laxis.label] = labels_to_jnames[laxis.label]
                lcpt = just_one(laxis.components)
                laxis = laxes.child(laxis, lcpt)
            trimmed_path = pmap(trimmed_path)
            trimmed_jnames = pmap(trimmed_jnames)

            varname = ctx.unique_name("p")
            insns += register_scalar_assignment(
                layout_fn.data,
                varname,
                trimmed_path,
                trimmed_jnames,
                ctx,
            )
            expr += pym.var(varname)
        elif isinstance(layout_fn, AffineLayout):
            start = layout_fn.start
            step = layout_fn.step
            jname = pym.var(labels_to_jnames[layout_fn.axis])
            expr += jname * step + start
        else:
            raise NotImplementedError

    ret = tuple(insns) + ((pym.var(offset_var), expr),)
    return ret


def register_extent(extent, path, jnames, ctx):
    if isinstance(extent, numbers.Integral):
        return extent

    # actually a pymbolic expression

    # TODO
    # * Traverse the pymbolic expression and generate a replace map for the multi-arrays

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

        varname = register_scalar_assignment(array, trimmed_path, trimmed_jnames, ctx)
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


def temporary_axes(axes, indices, loop_indices):
    # TODO think about "prior_indicess" (affects sizes but not ordering)
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
        new_index_path = index_path | (index, icpt)
        new_sizes = sizes

        if icpt in loop_indices:
            assert icpt.to_axis not in new_sizes
            component_sizes.append(1)
            new_sizes |= {icpt.to_axis: 1}

            # but what about loop(map[p][:2], ...)???
            assert not isinstance(icpt, Slice)

            if isinstance(icpt, Map):
                ...
            new_axis_path |= {icpt.to_axis: icpt.to_cpt}
        else:
            if isinstance(icpt, ScalarIndex):
                raise NotImplementedError("TODO")
            elif isinstance(icpt, Slice):
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

                new_axis_path |= {icpt.to_axis: icpt.to_cpt}

            else:
                assert isinstance(icpt, Map)
                axis = find_axis(axes, axis_path, icpt.from_axis)
                cpt_index = axis.component_index(icpt.from_cpt)
                size = axis.components[cpt_index].count
                component_sizes.append(size)

                new_axis_path = new_axis_path.discard(icpt.from_axis)
                new_axis_path |= {icpt.to_axis: icpt.to_cpt}

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


def _prepare_assignment(assignment, ctx: LoopyCodegenContext) -> tuple[pmap, pmap]:
    return _prepare_assignment_rec(
        assignment,
        assignment.array.axes,
        assignment.array.axes.root,
        pmap(),
        pmap(),
        ctx,
    )


def _prepare_assignment_rec(
    assignment,
    axes: AxisTree,
    axis: Axis,
    path: pmap,
    jnames: pmap,
    ctx: LoopyCodegenContext,
) -> tuple[pmap, pmap]:
    jnames_per_axcpt = {}
    insns_per_leaf = {}
    array_expr_per_leaf = {}
    for axcpt in axis.components:
        jname = ctx.unique_name("j")
        ctx.add_temporary(jname)
        new_jnames = jnames | {axis.label: jname}
        jnames_per_axcpt[axis.id, axcpt.label] = jname
        new_path = path | {axis.label: axcpt.label}

        if subaxis := axes.child(axis, axcpt):
            (
                subjnames_per_axcpt,
                subinsns_per_leaf,
                subarray_expr_per_leaf,
            ) = _parse_assignment_rec(axes, subaxis, new_path, new_jnames, ctx)
            jnames_per_axcpt |= subjnames_per_axcpt
            insns_per_leaf |= subinsns_per_leaf
            array_expr_per_leaf |= subarray_expr_per_leaf
        else:
            insns, array_expr = _assignment_array_insn(
                assignment, new_path, new_jnames, ctx
            )
            insns_per_leaf[axis.id, axcpt.label] = insns
            array_expr_per_leaf[axis.id, axcpt.label] = array_expr

    return pmap(jnames_per_axcpt), pmap(insns_per_leaf), pmap(array_expr_per_leaf)


def _parse_assignment_final(
    assignment,
    axes,
    jnames_per_axcpt,
    insns_per_leaf,
    array_expr_per_leaf,
    ctx: LoopyCodegenContext,
):
    _parse_assignment_final_rec(
        assignment,
        axes,
        axes.root,
        jnames_per_axcpt,
        insns_per_leaf,
        array_expr_per_leaf,
        pmap(),
        pmap(),
        ctx,
    )


def _parse_assignment_final_rec(
    assignment,
    axes,
    axis,
    jnames_per_axcpt,
    insns_per_leaf,
    array_expr_per_leaf,
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
                    insns_per_leaf,
                    array_expr_per_leaf,
                    new_path,
                    new_jnames,
                    ctx,
                )
            else:
                for insn in insns_per_leaf[axis.id, axcpt.label]:
                    ctx.add_assignment(*insn)
                array_expr = array_expr_per_leaf[axis.id, axcpt.label]
                temp_insns, temp_expr = _assignment_temp_insn(
                    assignment, new_path, new_jnames, ctx
                )
                for insn in temp_insns:
                    ctx.add_assignment(*insn)
                _shared_assignment_insn(assignment, array_expr, temp_expr, ctx)


def visit_indices(indices, pre, post_term, post_nonterm, *args, **kwargs):
    return _visit_indices_rec(indices, pre, post_nonterm, post_term)


# TODO This traversal is not quite right. I am often iterating over indexed things
# that look like axes. At least I think so. Do I get arbitrary temp shapes with my
# algorithm or not?
# If so then I am not using subaxes here, I am traversing the index tree.
"""
I just have ambiguity in the trees if one axis comes before another in a multi-part thing
and I want to index them in reverse. I think I can safely assume that the labels require
consistency even if they have different IDs.

Therefore things should work? Yes. I claim this to be the case.
"""


def _visit_indices_rec(indices, pre, post_nonterm, post_term, *args):
    index, *subindices = indices

    # loop over the leaves of the expanded index
    retvals = []
    for x in leaves:
        args, kwargs = pre(*args, **kwargs)

        if subindices:
            retval = _visit_indices_rec(subindices, *args, **kwargs)
            retval = post_nonterm(retval)
        else:
            retval = post_term(*args, **kwargs)
        retvals.append(retval)
    return last_func(retvals)

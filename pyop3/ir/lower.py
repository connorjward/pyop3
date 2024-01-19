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
import textwrap
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple, Union

import loopy as lp
import loopy.symbolic
import numpy as np
import pymbolic as pym
import pytools
from petsc4py import PETSc
from pyrsistent import freeze, pmap

from pyop3.array import HierarchicalArray, PetscMatAIJ
from pyop3.array.harray import CalledMapVariable, ContextSensitiveMultiArray
from pyop3.array.petsc import PetscMat, PetscObject
from pyop3.axtree import Axis, AxisComponent, AxisTree, AxisVariable
from pyop3.axtree.tree import ContextSensitiveAxisTree
from pyop3.buffer import DistributedBuffer, PackedBuffer
from pyop3.dtypes import IntType, PointerType
from pyop3.itree import (
    AffineSliceComponent,
    CalledMap,
    Index,
    IndexTree,
    LocalLoopIndex,
    LoopIndex,
    Map,
    Slice,
    Subset,
    TabulatedMapComponent,
)
from pyop3.itree.tree import (
    IndexExpressionReplacer,
    LoopIndexVariable,
    collect_shape_index_callback,
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
    Loop,
)
from pyop3.log import logger
from pyop3.utils import (
    PrettyTuple,
    UniqueNameGenerator,
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


class OpaqueType(lp.types.OpaqueType):
    def __repr__(self) -> str:
        return f"OpaqueType('{self.name}')"


class AssignmentType(enum.Enum):
    READ = enum.auto()
    WRITE = enum.auto()
    INC = enum.auto()
    ZERO = enum.auto()


class Renamer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        super().__init__()
        self._replace_map = replace_map

    def map_variable(self, var):
        try:
            return pym.var(self._replace_map[var.name])
        except KeyError:
            return var


class CodegenContext(abc.ABC):
    pass


class LoopyCodegenContext(CodegenContext):
    def __init__(self):
        self._domains = []
        self._insns = []
        self._args = []
        self._subkernels = []

        self.actual_to_kernel_rename_map = {}

        self._within_inames = frozenset()
        self._last_insn_id = None

        self._name_generator = UniqueNameGenerator()

    @property
    def domains(self):
        return tuple(self._domains)

    @property
    def instructions(self):
        return tuple(self._insns)

    @property
    def arguments(self):
        return tuple(self._args)

    @property
    def subkernels(self):
        return tuple(self._subkernels)

    @property
    def kernel_to_actual_rename_map(self):
        return {
            kernel: actual
            for actual, kernel in self.actual_to_kernel_rename_map.items()
        }

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
        renamer = Renamer(self.actual_to_kernel_rename_map)
        assignee = renamer(assignee)
        expression = renamer(expression)

        insn = lp.Assignment(
            assignee,
            expression,
            id=self._name_generator(prefix),
            within_inames=frozenset(self._within_inames),
            depends_on=self._depends_on,
        )
        self._add_instruction(insn)

    def add_cinstruction(self, insn_str, read_variables=frozenset()):
        cinsn = lp.CInstruction(
            (),
            insn_str,
            read_variables=frozenset(read_variables),
            id=self.unique_name("insn"),
            within_inames=self._within_inames,
            within_inames_is_final=True,
            depends_on=self._depends_on,
        )
        self._add_instruction(cinsn)

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

    def add_argument(self, array):
        if array.name in self.actual_to_kernel_rename_map:
            return

        arg_name = self.actual_to_kernel_rename_map.setdefault(
            array.name, self.unique_name("arg")
        )

        if isinstance(array.buffer, PackedBuffer):
            arg = lp.ValueArg(arg_name, dtype=self._dtype(array))
        else:
            assert isinstance(array.buffer, DistributedBuffer)
            arg = lp.GlobalArg(arg_name, dtype=self._dtype(array), shape=None)
        self._args.append(arg)

    def add_temporary(self, name, dtype=IntType, shape=()):
        temp = lp.TemporaryVariable(name, dtype=dtype, shape=shape)
        self._args.append(temp)

    def add_subkernel(self, subkernel):
        self._subkernels.append(subkernel)

    # I am not sure that this belongs here, I generate names separately from adding domains etc
    def unique_name(self, prefix):
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

    # TODO Perhaps this should be made more public so external users can register
    # arguments. I don't want to make it a property to attach to the objects since
    # that would "tie us in" to loopy more than I would like.
    @functools.singledispatchmethod
    def _dtype(self, array):
        """Return the dtype corresponding to a given kernel argument.

        This function is required because we need to distinguish between the
        dtype of the data stored in the array and that of the array itself. For
        basic arrays loopy can figure this out but for complex types like `PetscMat`
        it would otherwise get this wrong.

        """
        raise TypeError(f"No handler provided for {type(array).__name__}")

    @_dtype.register
    def _(self, array: ContextSensitiveMultiArray):
        return single_valued(self._dtype(a) for a in array.context_map.values())

    @_dtype.register(HierarchicalArray)
    def _(self, array):
        return self._dtype(array.buffer)

    @_dtype.register(DistributedBuffer)
    def _(self, array):
        return array.dtype

    @_dtype.register
    def _(self, array: PackedBuffer):
        return self._dtype(array.array)

    @_dtype.register
    def _(self, array: PetscMat):
        return OpaqueType("Mat")

    def _add_instruction(self, insn):
        self._insns.append(insn)
        self._last_insn_id = insn.id


class CodegenResult:
    def __init__(self, expr, ir, arg_replace_map):
        self.expr = expr
        self.ir = ir
        self.arg_replace_map = arg_replace_map

    def __call__(self, **kwargs):
        from pyop3.target import compile_loopy

        data_args = []
        for kernel_arg in self.ir.default_entrypoint.args:
            actual_arg_name = self.arg_replace_map.get(kernel_arg.name, kernel_arg.name)
            array = kwargs.get(actual_arg_name, self.expr.datamap[actual_arg_name])
            data_arg = _as_pointer(array)
            data_args.append(data_arg)
        compile_loopy(self.ir)(*data_args)

    def target_code(self, target):
        raise NotImplementedError("TODO")


class BinarySearchCallable(lp.ScalarCallable):
    def __init__(self, name="bsearch", **kwargs):
        super().__init__(name, **kwargs)

    def with_types(self, arg_id_to_dtype, callables_table):
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        new_arg_id_to_dtype[-1] = int
        return (
            self.copy(name_in_target="bsearch", arg_id_to_dtype=new_arg_id_to_dtype),
            callables_table,
        )

    def with_descrs(self, arg_id_to_descr, callables_table):
        return self.copy(arg_id_to_descr=arg_id_to_descr), callables_table

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        assert False
        from pymbolic import var

        mat_descr = self.arg_id_to_descr[0]
        m, n = mat_descr.shape
        ecm = expression_to_code_mapper
        mat, vec = insn.expression.parameters
        (result,) = insn.assignees

        c_parameters = [
            var("CblasRowMajor"),
            var("CblasNoTrans"),
            m,
            n,
            1,
            ecm(mat).expr,
            1,
            ecm(vec).expr,
            1,
            ecm(result).expr,
            1,
        ]
        return (
            var(self.name_in_target)(*c_parameters),
            False,  # cblas_gemv does not return anything
        )

    def generate_preambles(self, target):
        assert isinstance(target, lp.CTarget)
        yield ("20_stdlib", "#include <stdlib.h>")
        return


# prefer generate_code?
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

    preambles = [
        ("20_debug", "#include <stdio.h>"),  # dont always inject
        ("30_petsc", "#include <petsc.h>"),  # perhaps only if petsc callable used?
        (
            "30_bsearch",
            textwrap.dedent(
                """
                #include <stdlib.h>
                
                
                int32_t cmpfunc(const void * a, const void * b) {
                   return ( *(int32_t*)a - *(int32_t*)b );
                }
            """
            ),
        ),
    ]

    translation_unit = lp.make_kernel(
        ctx.domains,
        ctx.instructions,
        ctx.arguments,
        name=name,
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
        preambles=preambles,
        # options=lp.Options(check_dep_resolution=False),
    )
    tu = lp.merge((translation_unit, *ctx.subkernels))

    # add callables
    tu = lp.register_callable(tu, "bsearch", BinarySearchCallable())

    tu = tu.with_entrypoints("mykernel")

    # breakpoint()
    return CodegenResult(expr, tu, ctx.kernel_to_actual_rename_map)


@functools.singledispatch
def _compile(expr: Any, ctx: LoopyCodegenContext) -> None:
    raise TypeError


@_compile.register
def _(
    loop: Loop,
    loop_indices,
    codegen_context: LoopyCodegenContext,
) -> None:
    loop_context = context_from_indices(loop_indices)
    iterset = loop.index.iterset.with_context(loop_context)

    loop_index_replace_map = {}
    for _, replace_map in loop_indices.values():
        loop_index_replace_map.update(replace_map)
    loop_index_replace_map = pmap(loop_index_replace_map)

    parse_loop_properly_this_time(
        loop,
        iterset,
        loop_indices,
        codegen_context,
    )


def parse_loop_properly_this_time(
    loop,
    axes,
    loop_indices,
    codegen_context,
    *,
    axis=None,
    source_path=pmap(),
    iname_replace_map=pmap(),
    target_path=None,
    index_exprs=None,
):
    if axes.is_empty:
        raise NotImplementedError("does this even make sense?")

    # need to pick bits out of this, could be neater
    outer_replace_map = {}
    for _, replace_map in loop_indices.values():
        outer_replace_map.update(replace_map)
    outer_replace_map = freeze(outer_replace_map)

    if axis is None:
        target_path = freeze(axes.target_paths.get(None, {}))

        # again, repeated this pattern all over the place
        # target_replace_map = {}
        index_exprs = freeze(axes.index_exprs.get(None, {}))
        # replacer = JnameSubstitutor(outer_replace_map, codegen_context)
        # for axis_label, index_expr in index_exprs.items():
        #     target_replace_map[axis_label] = replacer(index_expr)
        # target_replace_map = freeze(target_replace_map)

        axis = axes.root

    for component in axis.components:
        axis_index_exprs = axes.index_exprs.get((axis.id, component.label), {})
        index_exprs_ = index_exprs | axis_index_exprs

        # Maps "know" about indices that aren't otherwise available. Eg map(p)
        # knows about p and this isn't accessible to axes.index_exprs except via
        # the index expression
        domain_index_exprs = axes.domain_index_exprs.get(
            (axis.id, component.label), pmap()
        )

        iname = codegen_context.unique_name("i")
        extent_var = register_extent(
            component.count,
            index_exprs | domain_index_exprs,
            # TODO just put these in the default replace map
            iname_replace_map | outer_replace_map,
            codegen_context,
        )
        codegen_context.add_domain(iname, extent_var)

        axis_replace_map = {axis.label: pym.var(iname)}

        source_path_ = source_path | {axis.label: component.label}
        iname_replace_map_ = iname_replace_map | axis_replace_map

        target_path_ = target_path | axes.target_paths.get(
            (axis.id, component.label), {}
        )

        with codegen_context.within_inames({iname}):
            subaxis = axes.child(axis, component)
            if subaxis:
                parse_loop_properly_this_time(
                    loop,
                    axes,
                    loop_indices,
                    codegen_context,
                    axis=subaxis,
                    source_path=source_path_,
                    iname_replace_map=iname_replace_map_,
                    target_path=target_path_,
                    index_exprs=index_exprs_,
                )
            else:
                target_replace_map = {}
                replacer = JnameSubstitutor(
                    outer_replace_map | iname_replace_map_, codegen_context
                )
                for axis_label, index_expr in index_exprs_.items():
                    target_replace_map[axis_label] = replacer(index_expr)

                index_replace_map = pmap(
                    {
                        (loop.index.id, ax): iexpr
                        for ax, iexpr in target_replace_map.items()
                    }
                )
                local_index_replace_map = freeze(
                    {
                        (loop.index.local_index.id, ax): iexpr
                        for ax, iexpr in iname_replace_map_.items()
                    }
                )
                for stmt in loop.statements:
                    _compile(
                        stmt,
                        loop_indices
                        | {
                            loop.index: (
                                target_path_,
                                index_replace_map,
                            ),
                            loop.index.local_index: (
                                source_path_,
                                local_index_replace_map,
                            ),
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
        loop_context = context_from_indices(loop_indices)

        # do we need the original arg any more?
        cf_arg = arg.with_context(loop_context)

        if isinstance(arg, (HierarchicalArray, ContextSensitiveMultiArray)):
            # FIXME materialize is a bad name here, it implies actually packing the values
            # into the temporary.
            temporary = cf_arg.materialize()
        else:
            assert isinstance(arg, LoopIndex)

            temporary = HierarchicalArray(
                cf_arg.axes,
                dtype=arg.dtype,
                target_paths=cf_arg.target_paths,
                index_exprs=cf_arg.index_exprs,
                domain_index_exprs=cf_arg.domain_index_exprs,
                name=ctx.unique_name("t"),
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
        if isinstance(arg, (HierarchicalArray, ContextSensitiveMultiArray)):
            ctx.add_argument(arg)

        ctx.add_temporary(temporary.name, temporary.dtype, shape)

        # subarrayref nonsense/magic
        indices = []
        for s in shape:
            iname = ctx.unique_name("i")
            ctx.add_domain(iname, s)
            indices.append(pym.var(iname))
        indices = tuple(indices)

        subarrayrefs[arg] = lp.symbolic.SubArrayRef(
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
        subarrayrefs[arg]
        for arg, spec in checked_zip(call.arguments, call.argspec)
        if spec.access in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}
    )
    expression = pym.primitives.Call(
        pym.var(call.function.code.default_entrypoint.name),
        tuple(
            subarrayrefs[arg]
            for arg, spec in checked_zip(call.arguments, call.argspec)
            if spec.access in {READ, RW, INC, MIN_RW, MAX_RW}
        )
        + tuple(extents.values()),
    )

    # gathers
    for arg, temp, access, shape in temporaries:
        if access in {READ, RW, MIN_RW, MAX_RW}:
            op = AssignmentType.READ
        else:
            assert access in {WRITE, INC, MIN_WRITE, MAX_WRITE}
            op = AssignmentType.ZERO
        parse_assignment(arg, temp, shape, op, loop_indices, ctx)

    ctx.add_function_call(assignees, expression)
    ctx.add_subkernel(call.function.code)

    # scatters
    for arg, temp, access, shape in temporaries:
        if access == READ:
            continue
        elif access in {WRITE, RW, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}:
            op = AssignmentType.WRITE
        else:
            assert access == INC
            op = AssignmentType.INC
        parse_assignment(arg, temp, shape, op, loop_indices, ctx)


# FIXME this is practically identical to what we do in build_loop
def parse_assignment(
    array,
    temp,
    shape,
    op,
    loop_indices,
    codegen_ctx,
):
    # TODO singledispatch
    loop_context = context_from_indices(loop_indices)

    if isinstance(array, (HierarchicalArray, ContextSensitiveMultiArray)):
        if (
            isinstance(array.with_context(loop_context).buffer, PackedBuffer)
            and op != AssignmentType.ZERO
        ):
            if not isinstance(
                array.with_context(loop_context).buffer.array, PetscMatAIJ
            ):
                raise NotImplementedError("TODO")
            parse_assignment_petscmat(
                array.with_context(loop_context),
                temp,
                shape,
                op,
                loop_indices,
                codegen_ctx,
            )
            return
        else:
            # assert isinstance(
            #     array.with_context(loop_context).buffer, DistributedBuffer
            # )
            pass
    else:
        assert isinstance(array, LoopIndex)

    # get the right index tree given the loop context

    # TODO Is this right to remove? Can it be handled further down?
    axes = array.with_context(loop_context).axes
    # minimal_context = array.filter_context(loop_context)
    #
    # target_path = {}
    # # for _, jnames in new_indices.values():
    # for loop_index, (path, iname_expr) in loop_indices.items():
    #     if loop_index in minimal_context:
    #         # assert all(k not in jname_replace_map for k in iname_expr)
    #         # jname_replace_map.update(iname_expr)
    #         target_path.update(path)
    # # jname_replace_map = freeze(jname_replace_map)
    # target_path = freeze(target_path)
    target_path = pmap()

    jname_replace_map = merge_dicts(mymap for _, mymap in loop_indices.values())

    parse_assignment_properly_this_time(
        array,
        temp,
        shape,
        op,
        axes,
        loop_indices,
        codegen_ctx,
        iname_replace_map=jname_replace_map,
        # jname_replace_map=jname_replace_map,
        # probably wrong
        index_exprs=pmap(),
        target_path=target_path,
    )


def parse_assignment_petscmat(array, temp, shape, op, loop_indices, codegen_context):
    from pyop3.array.harray import MultiArrayVariable

    # now emit the right line of code, this should properly be a lp.ScalarCallable
    # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/
    # PetscErrorCode MatGetValuesLocal(Mat mat, PetscInt nrow, const PetscInt irow[], PetscInt ncol, const PetscInt icol[], PetscScalar y[])
    # nrow = rexpr.array.axes.leaf_component.count
    # ncol = cexpr.array.axes.leaf_component.count
    # TODO check this? could compare matches temp (flat) size
    nrow, ncol = shape

    mat = array.buffer.mat

    # rename things
    mat_name = codegen_context.actual_to_kernel_rename_map[mat.name]
    # renamer = Renamer(codegen_context.actual_to_kernel_rename_map)
    # irow = renamer(array.buffer.rmap)
    # icol = renamer(array.buffer.cmap)
    rmap = array.buffer.rmap
    cmap = array.buffer.cmap
    rloop_index = array.buffer.rindex
    cloop_index = array.buffer.cindex
    riname = just_one(loop_indices[rloop_index][1].values())
    ciname = just_one(loop_indices[cloop_index][1].values())

    context = context_from_indices(loop_indices)
    rsize = rmap[rloop_index].with_context(context).size
    csize = cmap[cloop_index].with_context(context).size

    # breakpoint()

    codegen_context.add_argument(rmap)
    codegen_context.add_argument(cmap)
    irow = f"{codegen_context.actual_to_kernel_rename_map[rmap.name]}[{riname}*{rsize}]"
    icol = f"{codegen_context.actual_to_kernel_rename_map[cmap.name]}[{ciname}*{csize}]"

    # can only use GetValuesLocal when lgmaps are set (which I don't yet do)
    if op == AssignmentType.READ:
        call_str = f"MatGetValues({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({temp.name}[0]));"
    elif op == AssignmentType.WRITE:
        call_str = f"MatSetValues({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({temp.name}[0]), INSERT_VALUES);"
    elif op == AssignmentType.INC:
        call_str = f"MatSetValues({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({temp.name}[0]), ADD_VALUES);"
    else:
        raise NotImplementedError
    codegen_context.add_cinstruction(call_str)
    return

    ### old code below ###

    # we should flatten the array before this point as an earlier pass
    # if array.axes.depth != 2:
    #     raise ValueError

    # TODO We currently emit separate calls to MatSetValues if we have
    # multi-component arrays. This is naturally quite inefficient and we
    # could do things in a single call if we could "compress" the data
    # correctly beforehand. This is an optimisation I want to implement
    # generically though.
    for leaf_axis, leaf_cpt in array.axes.leaves:
        # This is wrong - we now have shape to deal with...
        (iraxis, ircpt), (icaxis, iccpt) = array.axes.path_with_nodes(
            leaf_axis, leaf_cpt, ordered=True
        )
        rkey = (iraxis.id, ircpt)
        ckey = (icaxis.id, iccpt)

        rexpr = array.index_exprs[rkey][just_one(array.target_paths[rkey])]
        cexpr = array.index_exprs[ckey][just_one(array.target_paths[ckey])]

        mat = array.buffer.array

        # need to generate code like map0[i0] instead of the usual map0[i0, i1]
        # this is because we are passing the full map through to the function call

        # similarly we also need to be careful to interrupt this function early
        # we don't want to emit loops for things!

        # I believe that this is probably the right place to be flattening the map
        # expressions. We want to have already done any clever substitution for arity 1
        # objects.

        # rexpr = self._flatten(rexpr)
        # cexpr = self._flatten(cexpr)

        assert temp.axes.depth == 2
        # sniff the right labels from the temporary, they tell us what jnames to substitute
        rlabel = temp.axes.root.label
        clabel = temp.axes.leaf_axis.label

        iname_expr_replace_map = {}
        for _, replace_map in loop_indices.values():
            iname_expr_replace_map.update(replace_map)

        # for now assume that we pass exactly the right map through, do no composition
        if not isinstance(rexpr, MultiArrayVariable):
            raise NotImplementedError

        # substitute a zero for the inner axis, we want to avoid this inner loop
        new_rexpr = JnameSubstitutor(
            iname_expr_replace_map | {rlabel: 0}, codegen_context
        )(rexpr)

        if not isinstance(cexpr, MultiArrayVariable):
            raise NotImplementedError

        # substitute a zero for the inner axis, we want to avoid this inner loop
        new_cexpr = JnameSubstitutor(
            iname_expr_replace_map | {clabel: 0}, codegen_context
        )(cexpr)

        # now emit the right line of code, this should properly be a lp.ScalarCallable
        # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/
        # PetscErrorCode MatGetValuesLocal(Mat mat, PetscInt nrow, const PetscInt irow[], PetscInt ncol, const PetscInt icol[], PetscScalar y[])
        nrow = rexpr.array.axes.leaf_component.count
        ncol = cexpr.array.axes.leaf_component.count

        # rename things
        mat_name = codegen_context.actual_to_kernel_rename_map[mat.name]
        renamer = Renamer(codegen_context.actual_to_kernel_rename_map)
        irow = renamer(new_rexpr)
        icol = renamer(new_cexpr)

        # can only use GetValuesLocal when lgmaps are set (which I don't yet do)
        if op == AssignmentType.READ:
            call_str = f"MatGetValues({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({temp.name}[0]));"
        # elif op == AssignmentType.WRITE:
        else:
            raise NotImplementedError
        codegen_context.add_cinstruction(call_str)


# TODO now I attach a lot of info to the context-free array, do I need to pass axes around?
def parse_assignment_properly_this_time(
    array,
    temp,
    shape,
    op,
    axes,
    loop_indices,
    codegen_context,
    *,
    axis=None,
    iname_replace_map,
    target_path,
    index_exprs,
    source_path=pmap(),
):
    context = context_from_indices(loop_indices)
    ctx_free_array = array.with_context(context)

    if axis is None:
        axis = axes.root
        target_path = target_path | ctx_free_array.target_paths.get(None, pmap())
        index_exprs = ctx_free_array.index_exprs.get(None, pmap())

    if axes.is_empty:
        add_leaf_assignment(
            array,
            temp,
            shape,
            op,
            axes,
            source_path,
            target_path,
            index_exprs,
            iname_replace_map,
            codegen_context,
            loop_indices,
        )
        return

    for component in axis.components:
        iname = codegen_context.unique_name("i")

        # map magic
        domain_index_exprs = ctx_free_array.domain_index_exprs.get(
            (axis.id, component.label), pmap()
        )
        extent_var = register_extent(
            component.count,
            index_exprs | domain_index_exprs,
            iname_replace_map,
            codegen_context,
        )
        codegen_context.add_domain(iname, extent_var)

        new_source_path = source_path | {axis.label: component.label}  # not used
        new_target_path = target_path | ctx_free_array.target_paths.get(
            (axis.id, component.label), {}
        )

        new_iname_replace_map = iname_replace_map | {axis.label: pym.var(iname)}

        index_exprs_ = index_exprs | ctx_free_array.index_exprs.get(
            (axis.id, component.label), {}
        )

        with codegen_context.within_inames({iname}):
            if subaxis := axes.child(axis, component):
                parse_assignment_properly_this_time(
                    array,
                    temp,
                    shape,
                    op,
                    axes,
                    loop_indices,
                    codegen_context,
                    axis=subaxis,
                    source_path=new_source_path,
                    target_path=new_target_path,
                    iname_replace_map=new_iname_replace_map,
                    index_exprs=index_exprs_,
                )

            else:
                add_leaf_assignment(
                    array,
                    temp,
                    shape,
                    op,
                    axes,
                    new_source_path,
                    new_target_path,
                    index_exprs_,
                    new_iname_replace_map,
                    codegen_context,
                    loop_indices,
                )


def add_leaf_assignment(
    array,
    temporary,
    shape,
    op,
    axes,
    source_path,
    target_path,
    index_exprs,
    iname_replace_map,
    codegen_context,
    loop_indices,
):
    context = context_from_indices(loop_indices)

    if isinstance(array, (HierarchicalArray, ContextSensitiveMultiArray)):

        def array_expr():
            replace_map = {}
            replacer = JnameSubstitutor(iname_replace_map, codegen_context)
            for axis, index_expr in index_exprs.items():
                replace_map[axis] = replacer(index_expr)

            array_ = array.with_context(context)
            return make_array_expr(
                array,
                array_.layouts[target_path],
                target_path,
                replace_map,
                codegen_context,
            )

    else:
        assert isinstance(array, LoopIndex)

        array_ = array.with_context(context)

        if array_.axes.depth != 0:
            raise NotImplementedError("Tricky when dealing with vectors here")

        def array_expr():
            replace_map = {}
            replacer = JnameSubstitutor(iname_replace_map, codegen_context)
            for axis, index_expr in index_exprs.items():
                replace_map[axis] = replacer(index_expr)

            if len(replace_map) > 1:
                # use leaf_target_path to get the right bits from replace_map?
                raise NotImplementedError("Needs more thought")
            return just_one(replace_map.values())

    temp_expr = functools.partial(
        make_temp_expr,
        temporary,
        shape,
        source_path,
        iname_replace_map,
        codegen_context,
    )

    if op == AssignmentType.READ:
        lexpr = temp_expr()
        rexpr = array_expr()
    elif op == AssignmentType.WRITE:
        lexpr = array_expr()
        rexpr = temp_expr()
    elif op == AssignmentType.INC:
        lexpr = array_expr()
        rexpr = lexpr + temp_expr()
    elif op == AssignmentType.ZERO:
        lexpr = temp_expr()
        rexpr = 0
    else:
        raise AssertionError("Invalid assignment type")

    codegen_context.add_assignment(lexpr, rexpr)


def make_array_expr(array, layouts, path, jnames, ctx):
    array_offset = make_offset_expr(
        layouts,
        jnames,
        ctx,
    )
    return pym.subscript(pym.var(array.name), array_offset)


def make_temp_expr(temporary, shape, path, jnames, ctx):
    layout = temporary.axes.layouts[path]
    temp_offset = make_offset_expr(
        layout,
        jnames,
        ctx,
    )

    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    extra_indices = (0,) * (len(shape) - 1)
    # also has to be a scalar, not an expression
    temp_offset_name = ctx.unique_name("off")
    temp_offset_var = pym.var(temp_offset_name)
    ctx.add_temporary(temp_offset_name)
    ctx.add_assignment(temp_offset_var, temp_offset)
    return pym.subscript(pym.var(temporary.name), extra_indices + (temp_offset_var,))


class JnameSubstitutor(pym.mapper.IdentityMapper):
    def __init__(self, replace_map, codegen_context):
        self._labels_to_jnames = replace_map
        self._codegen_context = codegen_context

    def map_axis_variable(self, expr):
        return self._labels_to_jnames[expr.axis_label]

    # this is cleaner if I do it as a single line expression
    # rather than register assignments for things.
    def map_multi_array(self, expr):
        # Register data
        self._codegen_context.add_argument(expr.array)

        target_path = expr.target_path
        index_exprs = expr.index_exprs

        replace_map = {ax: self.rec(expr_) for ax, expr_ in index_exprs.items()}

        offset_expr = make_offset_expr(
            expr.array.layouts[target_path],
            replace_map,
            self._codegen_context,
        )
        rexpr = pym.subscript(pym.var(expr.array.name), offset_expr)
        return rexpr

    def map_called_map(self, expr):
        if not isinstance(expr.function.map_component.array, HierarchicalArray):
            raise NotImplementedError("Affine map stuff not supported yet")

        # TODO I think I can clean the indexing up a lot here
        inner_expr = {axis: self.rec(idx) for axis, idx in expr.parameters.items()}
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

        # the inner_expr tells us the right mapping for the temporary, however,
        # for maps that are arrays the innermost axis label does not always match
        # the label used by the temporary. Therefore we need to do a swap here.
        # I don't like this.
        replace_map = inner_expr.copy()
        replace_map[inner_axis.label] = replace_map.pop(expr.function.full_map.name)

        jname_expr = _scalar_assignment(
            map_array,
            pmap({rootaxis.label: just_one(rootaxis.components).label})
            | pmap({inner_axis.label: inner_cpt.label}),
            replace_map,
            self._codegen_context,
        )
        return jname_expr

    def map_loop_index(self, expr):
        return self._labels_to_jnames[expr.name, expr.axis]

    def map_call(self, expr):
        if expr.function.name == "mybsearch":
            return self._map_bsearch(expr)
        else:
            raise NotImplementedError("hmm")

    # def _flatten(self, expr):
    #     for

    def _map_bsearch(self, expr):
        indices_var, axis_var = expr.parameters
        indices = indices_var.array

        leaf_axis = indices.axes.leaf_axis
        leaf_component = indices.axes.leaf_component
        ctx = self._codegen_context

        # should do elsewhere?
        ctx.add_argument(indices)

        # for reference
        """
        void *bsearch(
            const void *key,
            const void *base,
            size_t nitems,
            size_t size,
            int (*compar)(const void *, const void *)
        )
        """
        # key
        key_varname = ctx.unique_name("key")
        ctx.add_temporary(key_varname)
        key_var = pym.var(key_varname)
        key_expr = self.rec(axis_var)
        ctx.add_assignment(key_var, key_expr)

        # base
        replace_map = {}
        for key, replace_expr in self._labels_to_jnames.items():
            # for (LoopIndex_id0, axis0)
            if isinstance(key, tuple):
                replace_map[key[1]] = replace_expr
            else:
                assert isinstance(key, str)
                replace_map[key] = replace_expr
        # and set start to zero
        start_replace_map = replace_map.copy()
        start_replace_map[leaf_axis.label] = 0

        start_expr = make_offset_expr(
            indices.layouts[indices.axes.path(leaf_axis, leaf_component)],
            start_replace_map,
            self._codegen_context,
        )
        base_varname = ctx.unique_name("base")

        # rename things
        indices_name = ctx.actual_to_kernel_rename_map[indices.name]
        renamer = Renamer(ctx.actual_to_kernel_rename_map)
        start_expr = renamer(start_expr)

        # breaks if unsigned
        ctx.add_cinstruction(
            f"int32_t* {base_varname} = {indices_name} + {start_expr};", {indices_name}
        )

        # nitems
        nitems_varname = ctx.unique_name("nitems")
        ctx.add_temporary(nitems_varname)

        myindexexprs = {}
        for ax, cpt in indices.axes.path_with_nodes(leaf_axis, leaf_component).items():
            myindexexprs.update(indices.index_exprs[ax.id, cpt])

        nitems_expr = register_extent(
            leaf_component.count, myindexexprs, replace_map, ctx
        )

        # result
        found_varname = ctx.unique_name("ptr")

        # call
        bsearch_str = f"int32_t* {found_varname} = (int32_t*) bsearch(&{key_var}, {base_varname}, {nitems_expr}, sizeof(int32_t), cmpfunc);"
        ctx.add_cinstruction(bsearch_str, {indices.name})

        # equivalent to offset_var = found_var - base_var (but pointer arithmetic is hard in loopy)
        offset_varname = ctx.unique_name("offset")
        ctx.add_temporary(offset_varname)
        offset_var = pym.var(offset_varname)
        offset_str = f"size_t {offset_varname} = {found_varname} - {base_varname};"
        ctx.add_cinstruction(offset_str, {indices.name})

        # This gives us a pointer to the right place in the array. To recover
        # the offset we need to subtract the initial offset (if nested) and also
        # the address of the array itself.
        return offset_var
        # return offset_var - start_expr


def make_offset_expr(
    layouts,
    jname_replace_map,
    codegen_context,
):
    return JnameSubstitutor(jname_replace_map, codegen_context)(layouts)


def register_extent(extent, index_exprs, iname_replace_map, ctx):
    if isinstance(extent, numbers.Integral):
        return extent

    # actually a pymbolic expression
    if not isinstance(extent, HierarchicalArray):
        raise NotImplementedError("need to tidy up assignment logic")

    if not extent.axes.is_empty:
        path = extent.axes.path(*extent.axes.leaf)
    else:
        path = pmap()

    expr = _scalar_assignment(extent, path, index_exprs, iname_replace_map, ctx)

    varname = ctx.unique_name("p")
    ctx.add_temporary(varname)
    ctx.add_assignment(pym.var(varname), expr)
    return varname


class VariableReplacer(pym.mapper.IdentityMapper):
    def __init__(self, replace_map):
        self._replace_map = replace_map

    def map_variable(self, expr):
        return self._replace_map.get(expr.name, expr)


def _scalar_assignment(
    array,
    source_path,
    index_exprs,
    iname_replace_map,
    ctx,
):
    # Register data
    ctx.add_argument(array)

    # can this all go?
    index_keys = [None] + [
        (axis.id, cpt.label)
        for axis, cpt in array.axes.detailed_path(source_path).items()
    ]
    target_path = merge_dicts(array.target_paths.get(key, {}) for key in index_keys)
    # index_exprs = merge_dicts(array.index_exprs.get(key, {}) for key in index_keys)

    jname_replace_map = {}
    replacer = JnameSubstitutor(iname_replace_map, ctx)
    for axlabel, index_expr in index_exprs.items():
        jname_replace_map[axlabel] = replacer(index_expr)

    offset_expr = make_offset_expr(
        array.layouts[target_path],
        jname_replace_map,
        ctx,
    )
    rexpr = pym.subscript(pym.var(array.name), offset_expr)
    return rexpr


def context_from_indices(loop_indices):
    loop_context = {}
    for loop_index, (path, _) in loop_indices.items():
        loop_context[loop_index.id] = path
    return freeze(loop_context)


# lives here??
@functools.singledispatch
def _as_pointer(array) -> int:
    raise NotImplementedError


# bad name now, "as_kernel_arg"?
@_as_pointer.register
def _(arg: int):
    return arg


@_as_pointer.register
def _(arg: np.ndarray):
    return arg.ctypes.data


@_as_pointer.register
def _(arg: HierarchicalArray):
    # TODO if we use the right accessor here we modify the state appropriately
    return _as_pointer(arg.buffer)


@_as_pointer.register
def _(arg: DistributedBuffer):
    # TODO if we use the right accessor here we modify the state appropriately
    # NOTE: Do not use .data_rw accessor here since this would trigger a halo exchange
    return _as_pointer(arg._data)


@_as_pointer.register
def _(arg: PackedBuffer):
    return _as_pointer(arg.array)


@_as_pointer.register
def _(array: PetscMat):
    return array.petscmat.handle

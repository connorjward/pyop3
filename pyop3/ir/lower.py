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
from functools import cached_property
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
from pyop3.axtree import Axis, AxisComponent, AxisTree, AxisVariable, ContextFree
from pyop3.axtree.tree import ContextSensitiveAxisTree
from pyop3.buffer import DistributedBuffer, NullBuffer, PackedBuffer
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
    ContextFreeLoopIndex,
    IndexExpressionReplacer,
    LocalLoopIndexVariable,
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
    AddAssignment,
    Assignment,
    CalledFunction,
    ContextAwareLoop,
    Loop,
    PetscMatAdd,
    PetscMatInstruction,
    PetscMatLoad,
    PetscMatStore,
    ReplaceAssignment,
)
from pyop3.log import logger
from pyop3.utils import (
    PrettyTuple,
    UniqueNameGenerator,
    as_tuple,
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
        if isinstance(array.buffer, NullBuffer):
            if array.name in self.actual_to_kernel_rename_map:
                return

            # Temporaries can have variable size, hence we allocate space for the
            # largest possible array
            shape = array._shape if array._shape is not None else (array.alloc_size,)

            # could rename array like the rest
            # TODO do i need to be clever about shapes?
            temp = lp.TemporaryVariable(array.name, dtype=array.dtype, shape=shape)
            self._args.append(temp)

            # hasty no-op, refactor
            arg_name = self.actual_to_kernel_rename_map.setdefault(
                array.name, array.name
            )
            return
        else:
            # we only set this property for temporaries
            assert array._shape is None

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

    # can this now go?
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
        self.expr = as_tuple(expr)
        self.ir = ir
        self.arg_replace_map = arg_replace_map

    @cached_property
    def datamap(self):
        return merge_dicts(e.datamap for e in self.expr)

    def __call__(self, **kwargs):
        from pyop3.target import compile_loopy

        data_args = []
        for kernel_arg in self.ir.default_entrypoint.args:
            actual_arg_name = self.arg_replace_map.get(kernel_arg.name, kernel_arg.name)
            array = kwargs.get(actual_arg_name, self.datamap[actual_arg_name])
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
def compile(expr: Instruction, name="mykernel"):
    # preprocess expr before lowering
    from pyop3.transform import expand_implicit_pack_unpack, expand_loop_contexts

    expr = expand_loop_contexts(expr)
    expr = expand_implicit_pack_unpack(expr)

    ctx = LoopyCodegenContext()

    # expr can be a tuple if we don't start with a loop
    for e in as_tuple(expr):
        _compile(e, pmap(), ctx)

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

    # done by attaching "shape" to HierarchicalArray
    # tu = match_caller_callee_dimensions(tu)

    # breakpoint()
    return CodegenResult(expr, tu, ctx.kernel_to_actual_rename_map)


@functools.singledispatch
def _compile(expr: Any, loop_indices, ctx: LoopyCodegenContext) -> None:
    raise TypeError(f"No handler defined for {type(expr).__name__}")


@_compile.register
def _(
    loop: ContextAwareLoop,
    loop_indices,
    codegen_context: LoopyCodegenContext,
) -> None:
    parse_loop_properly_this_time(
        loop,
        loop.index.iterset,
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

        iname = codegen_context.unique_name("i")
        # breakpoint()
        extent_var = register_extent(
            component.count,
            iname_replace_map | loop_indices,
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
                    # outer_replace_map | iname_replace_map_, codegen_context
                    iname_replace_map_ | loop_indices,
                    codegen_context,
                )
                for axis_label, index_expr in index_exprs_.items():
                    target_replace_map[axis_label] = replacer(index_expr)

                # index_replace_map = pmap(
                #     {
                #         (loop.index.id, ax): iexpr
                #         for ax, iexpr in target_replace_map.items()
                #     }
                # )
                # local_index_replace_map = freeze(
                #     {
                #         (loop.index.id, ax): iexpr
                #         for ax, iexpr in iname_replace_map_.items()
                #     }
                # )
                index_replace_map = target_replace_map
                local_index_replace_map = iname_replace_map_
                for stmt in loop.statements[source_path_]:
                    _compile(
                        stmt,
                        loop_indices
                        | {
                            loop.index.id: (
                                local_index_replace_map,
                                index_replace_map,
                            ),
                        },
                        codegen_context,
                    )


@_compile.register
def _(call: CalledFunction, loop_indices, ctx: LoopyCodegenContext) -> None:
    temporaries = []
    subarrayrefs = {}
    extents = {}

    # loopy args can contain ragged params too
    loopy_args = call.function.code.default_entrypoint.args[: len(call.arguments)]
    for loopy_arg, arg, spec in checked_zip(loopy_args, call.arguments, call.argspec):
        # this check fails because we currently assume that all arrays require packing
        # from pyop3.transform import _requires_pack_unpack
        # assert not _requires_pack_unpack(arg)
        # old names
        temporary = arg
        indexed_temp = arg

        if loopy_arg.shape is None:
            shape = (temporary.alloc_size,)
        else:
            if np.prod(loopy_arg.shape, dtype=int) != temporary.alloc_size:
                raise RuntimeError("Shape mismatch between inner and outer kernels")
            shape = loopy_arg.shape

        temporaries.append((arg, indexed_temp, spec.access, shape))

        # Register data
        # TODO This might be bad for temporaries
        if isinstance(arg, (HierarchicalArray, ContextSensitiveMultiArray)):
            ctx.add_argument(arg)

        # this should already be done in an assignment
        # ctx.add_temporary(temporary.name, temporary.dtype, shape)

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

    ctx.add_function_call(assignees, expression)
    ctx.add_subkernel(call.function.code)


# FIXME this is practically identical to what we do in build_loop
@_compile.register(Assignment)
def parse_assignment(
    assignment,
    loop_indices,
    codegen_ctx,
):
    # this seems wrong
    parse_assignment_properly_this_time(
        assignment,
        loop_indices,
        codegen_ctx,
    )


@_compile.register(PetscMatInstruction)
def _(assignment, loop_indices, codegen_context):
    # FIXME, need to track loop indices properly. I think that it should be
    # possible to index a matrix like
    #
    #     loop(p, loop(q, mat[[p, q], [p, q]].assign(666)))
    #
    # but the current class design does not keep track of loop indices. For
    # now we assume there is only a single outer loop and that this is used
    # to index the row and column maps.
    if len(loop_indices) != 1:
        raise NotImplementedError(
            "For simplicity we currently assume a single outer loop"
        )
    iname_replace_map, _ = just_one(loop_indices.values())
    iname = just_one(iname_replace_map.values())

    # now emit the right line of code, this should properly be a lp.ScalarCallable
    # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/

    mat = assignment.mat_arg.buffer.mat
    array = assignment.array_arg
    rmap = assignment.mat_arg.buffer.rmap
    cmap = assignment.mat_arg.buffer.cmap

    # TODO cleanup
    codegen_context.add_argument(assignment.mat_arg)
    codegen_context.add_argument(array)
    codegen_context.add_argument(rmap)
    codegen_context.add_argument(cmap)

    mat_name = codegen_context.actual_to_kernel_rename_map[mat.name]
    array_name = codegen_context.actual_to_kernel_rename_map[array.name]
    rmap_name = codegen_context.actual_to_kernel_rename_map[rmap.name]
    cmap_name = codegen_context.actual_to_kernel_rename_map[cmap.name]

    # these sizes can be expressions that need evaluating
    rsize, csize = assignment.mat_arg.buffer.shape

    # my_replace_map = {}
    # for mappings in loop_indices.values():
    #     global_map, _ = mappings
    #     for (_, k), v in global_map.items():
    #         my_replace_map[k] = v
    my_replace_map = loop_indices

    if not isinstance(rsize, numbers.Integral):
        rindex_exprs = merge_dicts(
            rsize.index_exprs.get((ax.id, clabel), {})
            for ax, clabel in rsize.axes.path_with_nodes(*rsize.axes.leaf).items()
        )
        rsize_var = register_extent(
            rsize, rindex_exprs, my_replace_map, codegen_context
        )
    else:
        rsize_var = rsize

    if not isinstance(csize, numbers.Integral):
        cindex_exprs = merge_dicts(
            csize.index_exprs.get((ax.id, clabel), {})
            for ax, clabel in csize.axes.path_with_nodes(*csize.axes.leaf).items()
        )
        csize_var = register_extent(
            csize, cindex_exprs, my_replace_map, codegen_context
        )
    else:
        csize_var = csize

    rlayouts = rmap.layouts[
        freeze({rmap.axes.root.label: rmap.axes.root.component.label})
    ]
    roffset = JnameSubstitutor(my_replace_map, codegen_context)(rlayouts)

    clayouts = cmap.layouts[
        freeze({cmap.axes.root.label: cmap.axes.root.component.label})
    ]
    coffset = JnameSubstitutor(my_replace_map, codegen_context)(clayouts)

    irow = f"{rmap_name}[{roffset}]"
    icol = f"{cmap_name}[{coffset}]"

    call_str = _petsc_mat_insn(
        assignment, mat_name, array_name, rsize_var, csize_var, irow, icol
    )
    codegen_context.add_cinstruction(call_str)


@functools.singledispatch
def _petsc_mat_insn(assignment, *args):
    raise TypeError(f"{assignment} not recognised")


# can only use GetValuesLocal when lgmaps are set (which I don't yet do)
@_petsc_mat_insn.register
def _(assignment: PetscMatLoad, mat_name, array_name, nrow, ncol, irow, icol):
    return f"MatGetValues({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]));"


@_petsc_mat_insn.register
def _(assignment: PetscMatStore, mat_name, array_name, nrow, ncol, irow, icol):
    return f"MatSetValues({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), INSERT_VALUES);"


@_petsc_mat_insn.register
def _(assignment: PetscMatAdd, mat_name, array_name, nrow, ncol, irow, icol):
    return f"MatSetValues({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), ADD_VALUES);"


# TODO now I attach a lot of info to the context-free array, do I need to pass axes around?
def parse_assignment_properly_this_time(
    assignment,
    loop_indices,
    codegen_context,
    *,
    iname_replace_map=pmap(),
    # TODO document these under "Other Parameters"
    axis=None,
    path=None,
):
    axes = assignment.assignee.axes

    if strictly_all(x is None for x in [axis, path]):
        for array in assignment.arrays:
            codegen_context.add_argument(array)

        axis = axes.root
        path = pmap()

    if axes.is_empty:
        add_leaf_assignment(
            assignment,
            path,
            iname_replace_map | loop_indices,
            codegen_context,
            loop_indices,
        )
        return

    for component in axis.components:
        iname = codegen_context.unique_name("i")

        extent_var = register_extent(
            component.count,
            iname_replace_map | loop_indices,
            codegen_context,
        )
        codegen_context.add_domain(iname, extent_var)

        path_ = path | {axis.label: component.label}
        new_iname_replace_map = iname_replace_map | {axis.label: pym.var(iname)}

        with codegen_context.within_inames({iname}):
            if subaxis := axes.child(axis, component):
                parse_assignment_properly_this_time(
                    assignment,
                    loop_indices,
                    codegen_context,
                    iname_replace_map=new_iname_replace_map,
                    axis=subaxis,
                    path=path_,
                )

            else:
                add_leaf_assignment(
                    assignment,
                    path_,
                    new_iname_replace_map | loop_indices,
                    codegen_context,
                    loop_indices,
                )


def add_leaf_assignment(
    assignment,
    path,
    iname_replace_map,
    codegen_context,
    loop_indices,
):
    larr = assignment.assignee
    rarr = assignment.expression

    if isinstance(rarr, HierarchicalArray):
        rexpr = make_array_expr(
            rarr,
            path,
            iname_replace_map,
            codegen_context,
            rarr._shape,
        )
    else:
        assert isinstance(rarr, numbers.Number)
        rexpr = rarr

    lexpr = make_array_expr(
        larr,
        path,
        iname_replace_map,
        codegen_context,
        larr._shape,
    )

    if isinstance(assignment, AddAssignment):
        rexpr = lexpr + rexpr
    else:
        assert isinstance(assignment, ReplaceAssignment)

    codegen_context.add_assignment(lexpr, rexpr)


def make_array_expr(array, path, inames, ctx, shape):
    array_offset = make_offset_expr(
        array.subst_layouts[path],
        inames,
        ctx,
    )
    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    if shape is not None:
        extra_indices = (0,) * (len(shape) - 1)
        # also has to be a scalar, not an expression
        temp_offset_name = ctx.unique_name("j")
        temp_offset_var = pym.var(temp_offset_name)
        ctx.add_temporary(temp_offset_name)
        ctx.add_assignment(temp_offset_var, array_offset)
        indices = extra_indices + (temp_offset_var,)
    else:
        indices = (array_offset,)

    return pym.subscript(pym.var(array.name), indices)


class JnameSubstitutor(pym.mapper.IdentityMapper):
    def __init__(self, replace_map, codegen_context):
        self._replace_map = replace_map
        self._codegen_context = codegen_context

    def map_axis_variable(self, expr):
        return self._replace_map[expr.axis_label]

    # this is cleaner if I do it as a single line expression
    # rather than register assignments for things.
    def map_multi_array(self, expr):
        # Register data
        # if STOP:
        #     breakpoint()
        self._codegen_context.add_argument(expr.array)
        new_name = self._codegen_context.actual_to_kernel_rename_map[expr.array.name]

        target_path = expr.target_path
        index_exprs = expr.index_exprs

        replace_map = {ax: self.rec(expr_) for ax, expr_ in index_exprs.items()}

        offset_expr = make_offset_expr(
            expr.array.layouts[target_path],
            replace_map,
            self._codegen_context,
        )
        rexpr = pym.subscript(pym.var(new_name), offset_expr)
        return rexpr

    def map_called_map(self, expr):
        if not isinstance(expr.function.map_component.array, HierarchicalArray):
            raise NotImplementedError("Affine map stuff not supported yet")

        # TODO I think I can clean the indexing up a lot here
        inner_expr = {axis: self.rec(idx) for axis, idx in expr.parameters.items()}
        map_array = expr.function.map_component.array

        # handle [map0(p)][map1(p)] where map0 does not have an associated loop
        try:
            jname = self._replace_map[expr.function.full_map.name]
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
        # FIXME pretty sure I have broken local loop index stuff
        if isinstance(expr, LocalLoopIndexVariable):
            return self._replace_map[expr.id][0][expr.axis]
        else:
            assert isinstance(expr, LoopIndexVariable)
            return self._replace_map[expr.id][1][expr.axis]

    def map_call(self, expr):
        if expr.function.name == "mybsearch":
            return self._map_bsearch(expr)
        else:
            raise NotImplementedError("hmm")

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
        for key, replace_expr in self._replace_map.items():
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

        nitems_expr = register_extent(leaf_component.count, replace_map, ctx)

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


def register_extent(extent, iname_replace_map, ctx):
    if isinstance(extent, numbers.Integral):
        return extent

    # actually a pymbolic expression
    if not isinstance(extent, HierarchicalArray):
        raise NotImplementedError("need to tidy up assignment logic")

    if not extent.axes.is_empty:
        path = extent.axes.path(*extent.axes.leaf)
    else:
        path = pmap()

    index_exprs = extent.index_exprs.get(None, {})
    # extent must be linear
    if not extent.axes.is_empty:
        for axis, cpt in extent.axes.path_with_nodes(*extent.axes.leaf).items():
            index_exprs.update(extent.index_exprs[axis.id, cpt])

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
    return array.mat.handle

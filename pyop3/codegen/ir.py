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
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple, Union

import loopy as lp
import loopy.symbolic
import numpy as np
import pymbolic as pym
import pytools
from pyrsistent import freeze, pmap

from pyop3 import utils
from pyop3.axes import Axis, AxisComponent, AxisTree, AxisVariable
from pyop3.axes.tree import ContextSensitiveAxisTree
from pyop3.device import CPUDevice, CUDADevice, OpenCLDevice, offloading_device
from pyop3.distarray import DistributedArray, MultiArray
from pyop3.distarray.multiarray import ContextSensitiveMultiArray
from pyop3.distarray.petsc import IndexedPetscMat, PetscMat, PetscObject
from pyop3.dtypes import IntType, PointerType
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
from pyop3.indices.tree import CalledMapVariable, LoopIndexVariable
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
    Offset,
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


def loopy_target():
    if isinstance(offloading_device, CPUDevice):
        return lp.CWithGNULibcTarget()
    elif isinstance(offloading_device, CUDADevice):
        return lp.CudaTarget()
    elif isinstance(offloading_device, OpenCLDevice):
        return lp.PyOpenCLTarget()
    else:
        raise AssertionError


def loopy_lang_version():
    return (2018, 2)


class OpaqueType(lp.types.OpaqueType):
    def __repr__(self) -> str:
        return f"OpaqueType('{self.name}')"


class CodegenContext(abc.ABC):
    pass


class AssignmentType(enum.Enum):
    READ = enum.auto()
    WRITE = enum.auto()
    INC = enum.auto()
    ZERO = enum.auto()


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
        # FIXME if self._args is a set then we can add duplicates here provided
        # that we canonically renumber at a later point
        if array.name in [a.name for a in self._args]:
            logger.debug(
                f"Skipping adding {array.name} to the codegen context as it is already present"
            )
            return

        if isinstance(array.orig_array, PetscMat):
            arg = lp.ValueArg(array.name, dtype=self._dtype(array))
        else:
            arg = lp.GlobalArg(array.name, dtype=self._dtype(array), shape=None)
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
        return self._dtype(array.orig_array)

    @_dtype.register
    def _(self, array: MultiArray):
        return array.dtype

    @_dtype.register
    def _(self, array: PetscMat):
        return OpaqueType("Mat")

    def _add_instruction(self, insn):
        self._insns.append(insn)
        self._last_insn_id = insn.id


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
            """
#include <stdlib.h>


int32_t cmpfunc(const void * a, const void * b) {
   return ( *(int32_t*)a - *(int32_t*)b );
}

            """,
        ),
    ]

    translation_unit = lp.make_kernel(
        ctx.domains,
        ctx.instructions,
        ctx.arguments,
        name=name,
        target=loopy_target(),
        lang_version=loopy_lang_version(),
        preambles=preambles,
        # options=lp.Options(check_dep_resolution=False),
    )
    tu = lp.merge((translation_unit, *ctx.subkernels))

    # add callables
    tu = lp.register_callable(tu, "bsearch", BinarySearchCallable())

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
    target_path=pmap(),
    iname_replace_map=pmap(),
    jname_replace_map=pmap(),
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    outer_replace_map = {}
    for _, replace_map in loop_indices.values():
        outer_replace_map.update(replace_map)
    outer_replace_map = freeze(outer_replace_map)

    if axes.is_empty:
        raise NotImplementedError("does this even make sense?")

    axis = axis or axes.root

    domain_insns = []
    leaf_data = []

    for component in axis.components:
        iname = codegen_context.unique_name("i")
        extent_var = register_extent(
            component.count,
            iname_replace_map | jname_replace_map | outer_replace_map,
            codegen_context,
        )
        codegen_context.add_domain(iname, extent_var)

        new_source_path = source_path | {axis.label: component.label}
        new_target_path = target_path | axes.target_paths.get(
            (axis.id, component.label), {}
        )
        new_iname_replace_map = iname_replace_map | {axis.label: pym.var(iname)}

        # these aren't jnames!
        my_index_exprs = axes.index_exprs.get((axis.id, component.label), {})
        jname_extras = {}
        for axis_label, index_expr in my_index_exprs.items():
            jname_expr = JnameSubstitutor(
                new_iname_replace_map | jname_replace_map | outer_replace_map,
                codegen_context,
            )(index_expr)
            # jname_extras[axis_label] = jname_expr
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
                new_iname_replace_map = pmap(
                    {
                        (loop.index.local_index.id, myaxislabel): jname_expr
                        for myaxislabel, jname_expr in new_iname_replace_map.items()
                    }
                )
                new_jname_replace_map = pmap(
                    {
                        (loop.index.id, myaxislabel): jname_expr
                        for myaxislabel, jname_expr in new_jname_replace_map.items()
                    }
                )
                # breakpoint()

                for stmt in loop.statements:
                    _compile(
                        stmt,
                        loop_indices
                        | {
                            loop.index: (
                                new_target_path,
                                new_jname_replace_map,
                            ),
                            loop.index.local_index: (
                                new_source_path,
                                new_iname_replace_map,
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

        loop_context = context_from_indices(loop_indices)

        if isinstance(arg, (MultiArray, ContextSensitiveMultiArray)):
            axes = arg.with_context(loop_context).temporary_axes.restore()
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
            ctx.add_argument(arg)

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

    if isinstance(array.with_context(loop_context), IndexedPetscMat):
        parse_assignment_petscmat(
            array.with_context(loop_context), temp, shape, op, loop_indices, codegen_ctx
        )
        return
    # else:
    # assert isinstance(assignment, MultiArrayAssignment

    # get the right index tree given the loop context

    # TODO cleanup
    if isinstance(array, Offset):
        axes = array.with_context(loop_context)
        minimal_context = array.filter_context(loop_context)
    else:
        axes = array.with_context(loop_context).axes
        minimal_context = array.filter_context(loop_context)

    target_path = {}
    # for _, jnames in new_indices.values():
    for loop_index, (path, iname_expr) in loop_indices.items():
        if loop_index in minimal_context:
            # assert all(k not in jname_replace_map for k in iname_expr)
            # jname_replace_map.update(iname_expr)
            target_path.update(path)
    # jname_replace_map = freeze(jname_replace_map)
    target_path = freeze(target_path)

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
        jname_replace_map=jname_replace_map,
        target_path=target_path,
    )


def parse_assignment_petscmat(array, temp, shape, op, loop_indices, codegen_context):
    ctx = codegen_context

    expr = array.getvalues
    matvar, rexpr, cexpr = expr.parameters

    mat = matvar.obj

    # need to generate code like map0[i0] instead of the usual map0[i0, i1]
    # this is because we are passing the full map through to the function call

    # similarly we also need to be careful to interrupt this function early
    # we don't want to emit loops for things!

    # I believe that this is probably the right place to be flattening the map
    # expressions. We want to have already done any clever substitution for arity 1
    # objects.

    # rexpr = self._flatten(rexpr)
    # cexpr = self._flatten(cexpr)

    iname_expr_replace_map = {}
    for _, replace_map in loop_indices.values():
        iname_expr_replace_map.update(replace_map)

    # for now assume that we pass exactly the right map through, do no composition
    if not isinstance(rexpr, CalledMapVariable) or len(rexpr.parameters) != 2:
        raise NotImplementedError
    rinner_axis_label = rexpr.parameters[1].axis
    # substitute a zero for the inner axis, we want to avoid this inner loop
    new_rexpr = JnameSubstitutor(
        iname_expr_replace_map | {rinner_axis_label: 0}, codegen_context
    )(rexpr)

    if not isinstance(cexpr, CalledMapVariable) or len(cexpr.parameters) != 2:
        raise NotImplementedError
    cinner_axis_label = cexpr.parameters[1].axis
    # substitute a zero for the inner axis, we want to avoid this inner loop
    new_cexpr = JnameSubstitutor(
        iname_expr_replace_map | {cinner_axis_label: 0}, codegen_context
    )(cexpr)

    # now emit the right line of code, this should properly be a lp.ScalarCallable
    # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/
    # PetscErrorCode MatGetValuesLocal(Mat mat, PetscInt nrow, const PetscInt irow[], PetscInt ncol, const PetscInt icol[], PetscScalar y[])
    nrow = rexpr.function.map_component.arity
    irow = new_rexpr
    ncol = cexpr.function.map_component.arity
    icol = new_cexpr

    # can only use GetValuesLocal when lgmaps are set (which I don't yet do)
    call_str = (
        # f"MatGetValuesLocal({mat.name}, {nrow}, &({irow}), {ncol}, &({icol}), &({temp.name}[0]));"
        f"MatGetValues({mat.name}, {nrow}, &({irow}), {ncol}, &({icol}), &({temp.name}[0]));"
    )
    codegen_context.add_cinstruction(call_str)

    # debug
    # codegen_context.add_cinstruction("PetscAttachDebugger();")


def parse_assignment_properly_this_time(
    array,
    temp,
    shape,
    op,
    axes,
    loop_indices,
    codegen_context,
    *,
    iname_replace_map,
    jname_replace_map,
    target_path,
    axis=None,
    source_path=pmap(),
):
    from pyop3.distarray.multiarray import IndexExpressionReplacer

    if axis is None:
        axis = axes.root
        my_index_exprs = axes.index_exprs.get(None, pmap())
        jname_extras = {}
        for axis_label, index_expr in my_index_exprs.items():
            jname_expr = JnameSubstitutor(
                iname_replace_map | jname_replace_map, codegen_context
            )(index_expr)
            jname_extras[axis_label] = jname_expr
        jname_replace_map = jname_replace_map | jname_extras

    if axes.is_empty:
        add_leaf_assignment(
            array,
            temp,
            shape,
            op,
            axes,
            source_path,
            target_path,
            iname_replace_map,
            jname_replace_map,
            codegen_context,
            loop_indices,
        )
        return

    for component in axis.components:
        iname = codegen_context.unique_name("i")
        extent_var = register_extent(
            component.count, iname_replace_map | jname_replace_map, codegen_context
        )
        codegen_context.add_domain(iname, extent_var)

        new_source_path = source_path | {axis.label: component.label}  # not used
        new_target_path = target_path | axes.target_paths.get(
            (axis.id, component.label), {}
        )

        new_iname_replace_map = iname_replace_map | {axis.label: pym.var(iname)}

        # I don't like that I need to do this here and also when I emit the layout
        # instructions.
        # Do I need the jnames on the way down? Think so for things like ragged...
        my_index_exprs = axes.index_exprs.get((axis.id, component.label), {})
        jname_extras = {}
        for axis_label, index_expr in my_index_exprs.items():
            jname_expr = JnameSubstitutor(
                new_iname_replace_map | jname_replace_map, codegen_context
            )(index_expr)
            jname_extras[axis_label] = jname_expr
        new_jname_replace_map = jname_replace_map | jname_extras
        # new_jname_replace_map = new_iname_replace_map

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
                    jname_replace_map=new_jname_replace_map,
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
                    new_iname_replace_map,
                    new_jname_replace_map,
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
    iname_replace_map,
    jname_replace_map,
    codegen_context,
    loop_indices,
):
    context = context_from_indices(loop_indices)

    if isinstance(array, (MultiArray, ContextSensitiveMultiArray)):
        array_expr = functools.partial(
            make_array_expr,
            array,
            array.with_context(context).layouts[source_path],
            target_path,
            iname_replace_map | jname_replace_map,
            codegen_context,
        )
    else:
        assert isinstance(array, Offset)
        array_expr = functools.partial(
            make_offset_expr,
            array.with_context(context).layouts[source_path],
            iname_replace_map | jname_replace_map,
            codegen_context,
        )
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
    """

    Return a list of (assignee, expression) tuples and the array expr used
    in the assignment.

    """
    array_offset = make_offset_expr(
        layouts,
        jnames,
        ctx,
    )
    return pym.subscript(pym.var(array.name), array_offset)


def make_temp_expr(temporary, shape, path, jnames, ctx):
    layout = temporary.layout_axes.layouts[path]
    temp_offset = make_offset_expr(
        layout,
        jnames,
        ctx,
    )

    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    extra_indices = (0,) * (len(shape) - 1)
    # also has to be a scalar, not an expression
    temp_offset_var = ctx.unique_name("off")
    ctx.add_temporary(temp_offset_var)
    ctx.add_assignment(temp_offset_var, temp_offset)
    temp_offset_var = pym.var(temp_offset_var)
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
        path = expr.array.axes.path(*expr.array.axes.leaf, ordered=True)
        replace_map = {
            axis: self.rec(index)
            for (axis, _), index in checked_zip(path, expr.index_tuple)
        }
        varname = _scalar_assignment(
            expr.array,
            pmap(path),
            replace_map,
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

    def map_loop_index(self, expr):
        return self._labels_to_jnames[expr.name, expr.axis]

    def map_call(self, expr):
        if expr.function.name == "mybsearch":
            return self._map_bsearch(expr)
        elif expr.function.name == "MatGetValues":
            assert False, "not here"
            return self._map_matgetvalues(expr)
        else:
            raise NotImplementedError("hmm")

    # def _flatten(self, expr):
    #     for

    def _map_bsearch(self, expr):
        indices_var, axis_var = expr.parameters
        indices = indices_var.array

        leaf_axis, leaf_component = indices.axes.leaf
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
        key_expr = self._labels_to_jnames[(axis_var.index.id, axis_var.axis)]
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
        # breaks if unsigned
        ctx.add_cinstruction(
            f"int32_t* {base_varname} = {indices.name} + {start_expr};", {indices.name}
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
    ctx.add_argument(array)

    offset_expr = make_offset_expr(
        array.layout_axes.layouts[path],
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


def context_from_indices(loop_indices):
    loop_context = {}
    for loop_index, (path, _) in loop_indices.items():
        loop_context[loop_index] = path
    return freeze(loop_context)

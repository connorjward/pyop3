import abc
import collections
import dataclasses
import numbers
import enum
import functools
import itertools
from typing import Dict, Any, Tuple, FrozenSet

import loopy as lp
import loopy.symbolic
import numpy as np
import pytools
import pymbolic as pym

import pyop3.exprs
from pyop3 import exprs, tlang
import pyop3.utils
from pyop3 import utils
from pyop3.utils import MultiNameGenerator, NameGenerator
from pyop3.utils import CustomTuple, checked_zip, NameGenerator, rzip
from pyop3.tensors import Tensor, Index, Map, Dim, UniformDim, MixedDim, NonAffineMap
from pyop3.tensors import Slice, indexed_shape, indexed_size_per_index_group, AffineMap, Stencil, StencilGroup
from pyop3.codegen.tlang import to_tlang

LOOPY_TARGET = lp.CTarget()
LOOPY_LANG_VERSION = (2018, 2)


class CodegenTarget(enum.Enum):

    LOOPY = enum.auto()
    C = enum.auto()


def compile(expr, *, target):
    if target == CodegenTarget.LOOPY:
        return to_loopy(expr)
    elif target == CodegenTarget.C:
        return to_c(expr)
    else:
        raise ValueError


def to_loopy(expr):
    return _make_loopy_kernel(to_tlang(expr))


def to_c(expr):
    program = to_loopy(expr)
    return lp.generate_code_v2(program).device_code()


class LoopyKernelBuilder:
    def __init__(self):
        self._namer = MultiNameGenerator()
        self.domains = []
        self.instructions = []
        self.kernel_data = []
        self.subkernels = []

    def build(self, tlang_expr):
        self._namer.reset()
        self._build(tlang_expr, {})

        # breakpoint()
        translation_unit = lp.make_kernel(
            utils.unique(self.domains),
            utils.unique(self.instructions),
            utils.unique(self.kernel_data),
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
            name="mykernel",
        )
        tu = lp.merge((translation_unit, *self.subkernels))
        # import pdb; pdb.set_trace()
        return tu.with_entrypoints("mykernel")

    @functools.singledispatchmethod
    def _build(self, expr, within_loops):
        raise TypeError

    @_build.register
    def _(self, expr: exprs.Loop, within_loops):
        # be assertive
        stencil, = expr.index
        indices, = stencil

        within_loops = {}
        for index in indices:
            if isinstance(index, Slice):
                iname = self._namer.next("i")
                within_loops[index] = iname

                if isinstance(index.stop, Tensor):
                    raise NotImplementedError
                    temporary_name = self._namer.next("p")
                    temporary = Tensor((), name=temporary_name, dtype=np.int32)
                    indexed = self._index_the_map(index.stop, within_loops)
                    mapassignment = tlang.Read(indexed, temporary)
                    self._make_instruction_context(mapassignment, within_loops)
                    size = temporary_name
                else:
                    size = (index.stop-index.start)//index.step
            elif isinstance(index, NonAffineMap):
                iname = self._namer.next("i")
                temporary = Tensor((), name=iname, dtype=np.int32)
                new_index = Slice(index.dim, 0, stop=index.arity)
                indexed = self._index_the_map(index.tensor, new_index)
                mapassignment = tlang.Read(indexed, temporary)
                within_loops[new_index] = iname
                self._make_instruction_context(mapassignment, within_loops)
                # ???
                del within_loops[new_index]
                within_loops[index] = iname
                size = index.arity
            else:
                raise AssertionError

            self.domains.append(self._make_domain(iname, 0, size, 1))

        for stmt in expr.statements:
            self._build(stmt, within_loops.copy())

    @_build.register
    def _(self, insn: tlang.Instruction, within_loops):
        self._make_instruction_context(insn, within_loops)


    @functools.singledispatchmethod
    def _make_instruction_context(self, instruction: tlang.Instruction, within_inames):
        raise TypeError


    @_make_instruction_context.register
    def _(self, call: tlang.FunctionCall, within_loops):
        def mysize(t):
            stencil, = t.stencils
            indices, = stencil
            s = 1
            for idx in indices:
                s *= idx.size
            return s

        # breakpoint()
        subarrayrefs = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            self.domains.append(self._make_domain(iname, 0, mysize(temp), 1))

        reads = tuple(subarrayrefs[var] for var in call.reads)
        writes = tuple(subarrayrefs[var] for var in call.writes)
        assignees = tuple(writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(reads),
        )

        within_inames = frozenset({iname for iname in within_loops.values()})

        kernel_data = [lp.TemporaryVariable(temp.name, shape=(mysize(temp),)) for temp in subarrayrefs]

        depends_on = frozenset({f"{insn}*" for insn in call.depends_on})
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=call.id,
            within_inames=within_inames,
            depends_on=depends_on
        )

        self.instructions.append(call_insn)
        self.kernel_data.extend(kernel_data)
        self.subkernels.append(call.function.code)


    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, within_loops):
        # breakpoint()
        assert len(assignment.lhs.stencils) == len(assignment.rhs.stencils)
        for lstencil, rstencil in zip(assignment.lhs.stencils, assignment.rhs.stencils):
            assert len(lstencil) == len(rstencil)
            local_offset = 0  # FIXME
            within_inames = set()
            for lidxs, ridxs in zip(lstencil, rstencil):
                linames, rinames = self.sync_inames(assignment.lhs, assignment.rhs, lidxs, ridxs, within_loops)
                lexpr = self.handle_assignment(assignment.lhs, lidxs, linames, within_loops)
                rexpr = self.handle_assignment(assignment.rhs, ridxs, rinames, within_loops)

                # import pdb; pdb.set_trace()

                lname = pym.var(assignment.lhs.name)
                # lhs = pym.subscript(lname, lexpr)
                if assignment.lhs.dim:
                    lhs = pym.subscript(lname, lexpr)
                else:
                    lhs = lname

                if isinstance(assignment, tlang.Zero):
                    rhs = 0
                else:
                    rname = pym.var(assignment.rhs.name)
                    # rhs = pym.subscript(rname, rexpr)
                    if assignment.rhs.dim:
                        rhs = pym.subscript(rname, rexpr)
                    else:
                        rhs = rname

                assign_insn = lp.Assignment(
                        lhs, rhs,
                        id=self._namer.next(f"{assignment.id}_"),
                        within_inames=frozenset(within_inames),
                        depends_on=frozenset({f"{dep}*" for dep in assignment.depends_on}))
                self.instructions.append(assign_insn)

        # register kernel arguments
        if assignment.temporary.dim:
            stencil, = assignment.temporary.stencils
            indices, = stencil
            temp_size = 1
            for index in indices:
                temp_size *= index.size
            temp_shape = (temp_size,)
        else:
            temp_shape =(1,)

        self.kernel_data += [
            lp.GlobalArg(assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None),

            lp.TemporaryVariable(assignment.temporary.name, shape=temp_shape)
        ]

    def handle_assignment(self, tensor, indices, inames, within_loops):
        if not tensor.dim:
            assert not indices and not inames
            return 0
        within_loops = within_loops.copy()
        current_dim = tensor.dim.root
        index_expr = 0
        for index, iname in zip(indices, inames):
            assert index.dim == current_dim

            if isinstance(index, Slice):
                dim_expr = pym.var(iname)*index.step + index.start
            elif isinstance(index, NonAffineMap):
                temporary = Tensor((), name=iname, dtype=np.int32)
                indexed = self._index_the_map(index.tensor, within_loops)
                mapassignment = tlang.Read(indexed, temporary)
                self._make_instruction_context(mapassignment, within_loops)
                dim_expr = pym.var(iname)
            else:
                raise NotImplementedError

            new_map_name = self._namer.next("sec")
            index_expr += pym.subscript(pym.var(new_map_name), dim_expr)
            # this allows for ragged+mixed
            index_expr -= current_dim.offset
            self.kernel_data.append(lp.GlobalArg(new_map_name, shape=None, dtype=np.int32))

        return index_expr

    def sync_inames(self, lhs, rhs, lidxs, ridxs, within_loops):
        linames = []
        rinames = []
        for lidx, ridx in rzip(lidxs, ridxs):
            assert lidx.dim == ridx.dim
            if lidx in within_loops:
                iname = within_loops[lidx]
                linames.append(iname)
                if ridx in within_loops:
                    assert lidx == ridx
                    rinames.append(iname)
                else:
                    assert ridx.size == 1
            else:
                if ridx in within_loops:
                    rinames.append(within_loops[ridx])
                else:
                    iname = self._namer.next("i")
                    size, = {lidx.size, ridx.size}
                    self.domains.append(self._make_domain(iname, 0, size, 1))
                    linames.append(iname)
                    rinames.append(iname)
        return linames, rinames

    def _index_the_map(self, tensor, new_index):
        """
            Traverse the tensor dim tree and create new indices/loops as required to get
            it down to a scalar.
        """
        stencil, = tensor.stencils
        indices, = stencil
        stencils = StencilGroup([Stencil([indices + (new_index,)])])
        return tensor.copy(stencils=stencils)

    def get_stratum(self, dim):
        ptr = 0
        for i, size in enumerate(dim.sizes):
            if dim.offset == ptr:
                return i
            ptr += size

    @staticmethod
    def _make_domain(iname, start, stop, step):
        if start is None:
            start = 0
        if step is None:
            step = 1

        if step != 1:
            raise NotImplementedError

        assert all(param is not None for param in [start, stop, step])

        return f"{{ [{iname}]: {start} <= {iname} < {stop} }}"


def _make_loopy_kernel(tlang_kernel):
    return LoopyKernelBuilder().build(tlang_kernel)


@functools.singledispatch
def _get_arguments_per_instruction(instruction):
    """Return a canonical collection of kernel arguments.
    This can be used by both codegen and execution to get args in the right order."""
    raise TypeError


@_get_arguments_per_instruction.register
def _(assignment: tlang.Assignment):
    raise NotImplementedError
    return data, maps, parameters




def as_subarrayref(temporary, iname):
    """Register an argument to a function."""
    index = (pym.var(iname),)
    return lp.symbolic.SubArrayRef(
        index, pym.subscript(pym.var(temporary.name), index)
    )



def resolve(instruction, *args):
    if isinstance(instruction, tlang.Read):
        resolver = ReadAssignmentResolver(instruction, *args)
    elif isinstance(instruction, tlang.Zero):
        resolver = ZeroAssignmentResolver(instruction, *args)
    elif isinstance(instruction, tlang.Write):
        resolver = WriteAssignmentResolver(instruction, *args)
    elif isinstance(instruction, tlang.Increment):
        resolver = IncAssignmentResolver(instruction, *args)
    else:
        raise AssertionError
    return resolver.assignee, resolver.expression


class AssignmentResolver:
    def __init__(self, instruction, global_idxs, local_idxs, local_offset):
        self.instruction = instruction
        self.global_idxs = global_idxs
        self.local_idxs = local_idxs
        self.local_offset = local_offset

    @property
    def global_expr(self):
        return pym.subscript(pym.var(self.instruction.tensor.name), self.global_idxs)

    @property
    def local_expr(self):
        name = pym.var(self.instruction.temporary.name)
        if self.instruction.temporary.dim:
            return pym.subscript(name, self.local_idxs + self.local_offset)
        else:
            return name



class ReadAssignmentResolver(AssignmentResolver):
    @property
    def assignee(self):
        return self.local_expr

    @property
    def expression(self):
        return self.global_expr


class ZeroAssignmentResolver(AssignmentResolver):
    @property
    def assignee(self):
        return self.local_expr

    @property
    def expression(self):
        return 0


class WriteAssignmentResolver(AssignmentResolver):
    @property
    def assignee(self):
        return self.global_expr

    @property
    def expression(self):
        return self.local_expr


class IncAssignmentResolver(AssignmentResolver):
    @property
    def assignee(self):
        return self.global_expr

    @property
    def expression(self):
        return self.global_expr + self.local_expr

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
from pyop3.utils import CustomTuple, checked_zip, NameGenerator
from pyop3.tensors import Tensor, Index, Map, Dim, UniformDim, MixedDim, NonAffineMap
from pyop3.tensors import Slice, indexed_shape, indexed_size_per_index_group, AffineMap, Stencil, StencilGroup, full_dim_size
from pyop3.codegen.tlang import to_tlang

LOOPY_TARGET = lp.ExecutableCTarget()
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
            if isinstance(index.dim, MixedDim):
                continue
            iname = self._namer.next("i")

            if isinstance(index, Slice) and not index.stop:
                stop = index.dim.sizes[index.stratum]
            elif isinstance(index.stop, Tensor):
                temporary_name = self._namer.next("p")
                temporary = Tensor((), name=temporary_name, dtype=np.int32)
                indexed = self._index_the_map(index.stop, within_loops)
                mapassignment = tlang.Read(indexed, temporary)
                self._make_instruction_context(mapassignment, within_loops)
                stop = temporary_name
            else:
                stop = index.stop
            self.domains.append(self._make_domain(iname, index.start, stop, index.step))
            within_loops[index] = iname

        for stmt in expr.statements:
            self._build(stmt, within_loops)

    @_build.register
    def _(self, insn: tlang.Instruction, within_loops):
        self._make_instruction_context(insn, within_loops)


    @functools.singledispatchmethod
    def _make_instruction_context(self, instruction: tlang.Instruction, within_inames):
        raise TypeError


    @_make_instruction_context.register
    def _(self, call: tlang.FunctionCall, within_loops):
        subarrayrefs = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            self.domains.append(self._make_domain(iname, 0, temp.dim.size, 1))

        reads = tuple(subarrayrefs[var] for var in call.reads)
        writes = tuple(subarrayrefs[var] for var in call.writes)
        assignees = tuple(writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(reads),
        )

        within_inames = frozenset({iname for iname in within_loops.values()})

        kernel_data = [lp.TemporaryVariable(temp.name, shape=(temp.dim.size,)) for temp in subarrayrefs]

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
        for stencil in assignment.tensor.stencils:
            local_offset = 0  # FIXME
            for indices in stencil:
                # breakpoint()
                within_inames = frozenset({iname for iname in within_loops.values()})
                local_offset = 0
                global_offset = 0
                current_dim = assignment.tensor.dim.root
                for i, index in enumerate(indices):
                    # sanity checking
                    assert index.dim == current_dim
                    try:
                        iname = within_loops[index]
                    except KeyError:
                        iname = self._namer.next("i")
                        if isinstance(index, NonAffineMap):
                            start = 0
                            stop = index.arity
                            step = 1
                        else:
                            start = index.start
                            stop = index.stop
                            step = index.step
                            self.domains.append(self._make_domain(iname, start, stop, step))


                    # only index temporaries by 'inner' loops
                    if index not in within_loops:
                        new_map_name = self._namer.next("map")
                        local_offset += pym.subscript(pym.var(new_map_name), pym.var(iname))
                        self.kernel_data.append(lp.GlobalArg(new_map_name, shape=None, dtype=np.int32))

                    # add a map for every index/dim that does index -> offset
                    if not isinstance(assignment, tlang.Zero):
                        new_map_name = self._namer.next("map")
                        global_offset += pym.subscript(pym.var(new_map_name), pym.var(iname))
                        # this allows for ragged+mixed
                        global_offset -= current_dim.offset
                        self.kernel_data.append(lp.GlobalArg(new_map_name, shape=None, dtype=np.int32))

                    if isinstance(index, NonAffineMap):
                        raise NotImplementedError
                        # breakpoint()
                        temporary_name = self._namer.next("t_map")
                        temporary = Tensor((), name=temporary_name)
                        indexed = self._index_the_map(index.map, within_loops)
                        mapassignment = tlang.Read(indexed, temporary)
                        self._make_instruction_context(mapassignment, within_loops)

                    within_inames |= {iname}

                    if children := assignment.tensor.dim.get_children(current_dim):
                        current_dim = children[index.stratum]
                    else:
                        # else we must be at the end
                        assert i == len(indices) - 1

                assignee, expression = resolve(assignment, global_offset, 0, local_offset)

                assign_insn = lp.Assignment(
                        assignee, expression,
                        id=self._namer.next(f"{assignment.id}_"),
                        within_inames=within_inames,
                        depends_on=frozenset({f"{dep}*" for dep in assignment.depends_on}))
                self.instructions.append(assign_insn)


        # breakpoint()
        if assignment.temporary.dim:
            temp_size = assignment.temporary.dim.size
            temp_shape = (temp_size,)
        else:
            temp_size = 1
            temp_shape =()

        # breakpoint()
        self.kernel_data += [
            lp.GlobalArg(assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None),

            lp.TemporaryVariable(assignment.temporary.name, shape=temp_shape)
        ]


    def _index_the_map(self, tensor, within_loops):
        dim2index = {index.dim: index for index in within_loops}
        indices = []
        dim = tensor.dim.root
        while dim:
            try:
                index = dim2index[dim]
            except KeyError:
                iname = self._namer.next("i")
                self.domains.append(self._make_domain(iname, 0, dim.size, 1))
                index = Slice(dim, 0, dim.size, 1)
            indices.append(index)
            dim = tensor.dim.get_child(dim)
        indices = tuple(indices)
        stencils = StencilGroup([Stencil([indices])])
        return tensor.copy(stencils=stencils)

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

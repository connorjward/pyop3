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
from pyop3.tensors import Tensor, Index, Map, Dim, IntIndex, LoopIndex, UniformDim, MixedDim, NonAffineMap
from pyop3.tensors import Range, Slice, indexed_shape, IndexTree, indexed_size_per_index_group, AffineMap, BasicIndex
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

    def build(self, tlang_kernel):
        self._namer.reset()

        # index: LoopContext
        within_loops = {}
        for insn in tlang_kernel.instructions:
            within_loops |= self._collect_within_loops_per_instruction(insn, within_loops, self._namer)

        domains = []
        instructions = []
        kernel_data = []
        subkernels = []
        for insn in tlang_kernel.instructions:
            ctx = _make_instruction_context(insn, within_loops, self._namer)

            domains += ctx.domains
            instructions += ctx.loopy_instructions
            kernel_data += ctx.kernel_data
            subkernels += ctx.subkernels

        for loop in within_loops.values():
            instructions.append(lp.Assignment(pym.var(loop.extent_temp), loop.extent_expr))
            kernel_data.append(lp.TemporaryVariable(loop.extent_temp, shape=(), dtype=np.int32))
            if loop.extent_tensor:
                kernel_data.append(lp.GlobalArg(loop.extent_tensor.name, shape=None, dtype=np.int32))

        breakpoint()
        translation_unit = lp.make_kernel(
            utils.unique(domains),
            utils.unique(instructions),
            utils.unique(kernel_data),
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )
        return lp.merge((translation_unit, *subkernels))

    @functools.singledispatchmethod
    def _collect_within_loops_per_instruction(self, instruction, within_loops, namer):
        return {}

    @_collect_within_loops_per_instruction.register
    def _(self, assignment: tlang.Assignment, existing_loops, namer):
        loops = existing_loops.copy()
        for stencil in assignment.tensor.stencils:
            for indices in stencil:
                for prev_index, index in zip((None,)+indices, indices):
                    while isinstance(index, Map):
                        index = index.from_index

                    if isinstance(index, LoopIndex) and index not in loops:
                        if prev_index in loops:
                            loops[index] = LoopContext.from_dim(index, loops[prev_index], namer)
                        else:
                            loops[index] = LoopContext.from_dim(index, None, namer)
        return loops


@dataclasses.dataclass(frozen=True)
class LoopContext:
    index: Index
    iname: str
    extent_temp: Any
    extent_expr: Any
    extent_tensor: Any

    # FIXME this needs to get set somehow
    global_offset: Any = 0

    @classmethod
    def from_dim(cls, index, prev_loop, namer):
        iname = namer.next("i")
        temp = namer.next("n")
        if isinstance(index, LoopIndex):
            index_ = index.domain
        else:
            index_ = index

        if isinstance(index_.size, Tensor):
            extent_expr = pym.subscript(pym.var(index_.size.name), pym.var(prev_loop.iname))
            extent_tensor = index_.size
        else:
            extent_expr = index_.size
            extent_tensor = None
        return cls(index, iname, temp, extent_expr, extent_tensor)


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


def _make_domains(indices, inames, bounds, prev_index=None):
    if not indices:
        return frozenset()

    index, *subindices = indices
    return frozenset({_make_domain(index, inames, bounds, prev_index)}) | _make_domains(subindices, inames, bounds, index)


def _make_domain(iname, size):
    return f"{{ [{iname}]: 0 <= {iname} < {size} }}"


def as_subarrayref(temporary, iname):
    """Register an argument to a function."""
    index = (pym.var(iname),)
    return lp.symbolic.SubArrayRef(
        index, pym.subscript(pym.var(temporary.name), index)
    )


class AssignmentContextBuilder:
    def __init__(self, namer):
        self._namer = namer

    def build(self):
        ...


@dataclasses.dataclass(frozen=True)
class InstructionContext:
    domains: Tuple
    loopy_instructions: Tuple
    kernel_data: Tuple
    subkernels: Tuple = ()


@functools.singledispatch
def _make_instruction_context(instruction: tlang.Instruction, within_inames, namer):
    raise TypeError


@_make_instruction_context.register
def _(call: tlang.FunctionCall, within_loops, namer):
    subarrayrefs = {}
    domains = []
    for temp in utils.unique(itertools.chain(call.reads, call.writes)):
        iname = namer.next("i")
        subarrayrefs[temp] = as_subarrayref(temp, iname)
        domains.append(_make_domain(iname, temp.dim.size))
    domains = frozenset(domains)

    reads = tuple(subarrayrefs[var] for var in call.reads)
    writes = tuple(subarrayrefs[var] for var in call.writes)
    assignees = tuple(writes)
    expression = pym.primitives.Call(
        pym.var(call.function.loopy_kernel.default_entrypoint.name),
        tuple(reads),
    )

    within_inames = frozenset({loop.iname for loop in within_loops.values()})

    kernel_data = [lp.TemporaryVariable(temp.name, shape=(temp.dim.size,)) for temp in subarrayrefs]

    depends_on = frozenset({f"{insn}*" for insn in call.depends_on})
    call_insn = lp.CallInstruction(
        assignees,
        expression,
        id=call.id,
        within_inames=within_inames,
        depends_on=depends_on
    )
    return InstructionContext(domains, [call_insn], kernel_data, {call.function.loopy_kernel})


@_make_instruction_context.register
def _(assignment: tlang.Assignment, within_loops, namer):
    all_insns = []
    domains = []
    kernel_data = []
    for stencil in assignment.tensor.stencils:
        local_offset = 0
        old_ids = set()  # this gives us deterministic codegen and avoids a warning
        for indices in stencil:
            loops = _collect_loops(indices, namer, within_loops)

            assign_id = namer.next(assignment.id)

            temp_active = namer.next("T")
            active = pym.var(temp_active)
            kernel_data.append(lp.TemporaryVariable(temp_active, shape=(), dtype=np.int32))

            local_temp = namer.next("T")
            local_active = pym.var(local_temp)
            kernel_data.append(lp.TemporaryVariable(local_temp, shape=(), dtype=np.int32))

            active_init_insns = (
                lp.Assignment(active, 0, id=assign_id+"active"),
                lp.Assignment(local_active, 0, id=assign_id+"local"),
            )

            init_insns, inc_insns, innermost_indices, kdata = _traverse_and_build(loops, assign_id, active, local_active, namer)

            global_idxs, local_idxs = innermost_indices

            main_within_inames = frozenset({loop.iname for loop in loops})
            main_insn = _make_loopy_instruction(assignment, assign_id, local_idxs, global_idxs, local_offset, main_within_inames, depends_on={insn.id for insn in active_init_insns} | old_ids)

            all_insns.extend(active_init_insns)
            all_insns.extend(init_insns)
            all_insns.extend(inc_insns)
            all_insns.append(main_insn)

            domains.extend([_make_domain(loop.iname, loop.extent_temp) for loop in loops])

            kernel_data.extend(kdata)

            for loop in loops:
                if isinstance(loop.index, NonAffineMap):
                    kernel_data.append(lp.GlobalArg(loop.index.name, dtype=np.int32, shape=None))

            old_ids.add(assign_id)
            local_offset += indexed_size_per_index_group(indices)

    kernel_data += [
        lp.GlobalArg(assignment.tensor.name, dtype=np.float64, shape=None),
        lp.TemporaryVariable(assignment.temporary.name, shape=(assignment.temporary.dim.size,))
    ]

    return InstructionContext(domains, all_insns, kernel_data)


def _traverse_and_build(loops, assign_id, active_index, local_active, namer, within_inames=frozenset()):
    loop, *other_loops = loops

    # this is needed to get a valid iname ordering
    within_inames = within_inames | {l.iname for l in loops if isinstance(l.index, LoopIndex)}

    # do these elsewhere (uniquify)
    if not isinstance(loop.index, LoopIndex):
        loop_bound_insn = (lp.Assignment(pym.var(loop.extent_temp), loop.extent_expr, within_inames=within_inames),)
        bound_kdata = (lp.TemporaryVariable(loop.extent_temp, shape=(), dtype=np.int32),)
        if loop.extent_tensor:
            bound_kdata += (lp.GlobalArg(loop.extent_tensor.name, shape=None, dtype=np.int32),)
    else:
        loop_bound_insn = ()
        bound_kdata = ()

    within_inames |= {loop.iname}

    if not other_loops:
        inc_insns = (
            lp.Assignment(local_active, local_active+1, depends_on=frozenset({assign_id}), within_inames=within_inames),
            lp.Assignment(active_index, active_index+1, depends_on=frozenset({assign_id}), within_inames=within_inames)
        )
    else:
        inc_insns = ()

    if isinstance(loop.index, NonAffineMap):
        # breakpoint()
        temp = namer.next("T")
        new_active_index = pym.var(temp)
        kdata = bound_kdata + (lp.TemporaryVariable(temp, shape=(), dtype=np.int32),)
        init_insn = (lp.Assignment(new_active_index, pym.subscript(pym.var(loop.index.name), active_index), within_inames=within_inames),)
        inc_insns += (lp.Assignment(active_index, active_index+1, depends_on=frozenset({assign_id}), within_inames=within_inames),)
        active_index = new_active_index
    else:
        init_insn = ()
        kdata = bound_kdata

    if other_loops:
        other_init, other_inc, innermost, other_kdata = _traverse_and_build(other_loops, assign_id, active_index, local_active, namer, within_inames)
        return loop_bound_insn + init_insn + other_init, inc_insns+other_inc, innermost, kdata + other_kdata
    else:
        return loop_bound_insn + init_insn, inc_insns, (active_index, local_active), kdata


def _make_loopy_instruction(instruction, id, local_idxs, global_idxs, local_offset, within_inames, depends_on=frozenset()):
    # wilcard this to catch subinsns
    depends_on = frozenset({f"{insn}*" for insn in instruction.depends_on}) | depends_on

    assignee, expression = resolve(instruction, global_idxs, local_idxs, local_offset)

    return lp.Assignment(assignee, expression, id=id,
            within_inames=within_inames, depends_on=depends_on)


def _collect_loops(indices, namer, within_loops, prev_loop=None):
    if not indices:
        return ()

    index, *subindices = indices

    loops = _collect_loops_per_index(index, prev_loop, namer, within_loops)
    return loops + _collect_loops(subindices, namer, within_loops, loops[-1])


@functools.singledispatch
def _collect_loops_per_index(index, prev_index, namer, within_loops):
    raise TypeError


@_collect_loops_per_index.register
def _(index: LoopIndex, prev_index, namer, within_loops):
    # already registered
    return (within_loops[index],)


@_collect_loops_per_index.register
def _(index: Slice, prev_loop, namer, within_loops):
    return (LoopContext.from_dim(index, prev_loop, namer),)


@_collect_loops_per_index.register
def _(index: Map, prev_loop, namer, within_loops):
    loops = _collect_loops_per_index(index.from_index, prev_loop, namer, within_loops)
    return loops + (LoopContext.from_dim(index, loops[-1], namer),)


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
        return pym.subscript(pym.var(self.instruction.temporary.name), self.local_idxs + self.local_offset)



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


"""
def _collect_indices(
        indices,
        loops,
        local_idxs=0, global_idxs=0,
        local_offset=0,
        within_inames=frozenset(),
        inames_dict={},
        within_indices=(),
):
    index, *subindices = indices

    within_indices += (index,)

    # inames, local_idxs_, global_idxs_ = handle_index(index, inames_dict)
    # do something with LoopContext
    loop = ...

    if global_idxs_:
        global_idxs += global_idxs_

    if local_idxs_:
        local_idxs += local_idxs_

    if inames:
        within_inames |= set(inames)

    if subindices:
        subindex, *_ = subindices
        global_idxs *= subdim_size(index, subindex, inames_dict)
        local_idxs *= subindex.size
        return _collect_indices(subindices, local_idxs, global_idxs, local_offset, within_inames, inames_dict, within_indices)

    else:
        return local_idxs, global_idxs, local_offset, within_inames


def subdim_size(index, subindex, inames):
    if isinstance(size := subindex.dim.size, Tensor):
        return pym.subscript(pym.var(size.name), pym.var(inames[index]))
    else:
        return size
"""

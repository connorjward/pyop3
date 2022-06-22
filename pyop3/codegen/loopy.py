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

        breakpoint()
        translation_unit = lp.make_kernel(
            utils.unique(self.domains),
            utils.unique(self.instructions),
            utils.unique(self.kernel_data),
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )
        return lp.merge((translation_unit, *self.subkernels))

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

            # start, stop, step = (self._register_domain_parameter(param, prev_iname) for param in [index.start, index.stop, index.step])
            if isinstance(index, Slice) and not index.stop:
                stop = index.dim.size
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
                for i, index in enumerate(indices):
                    if isinstance(index.dim, MixedDim):
                        continue

                    try:
                        iname = within_loops[index]
                    except KeyError:
                        iname = self._namer.next("i")

                    if isinstance(index, Slice) and not index.stop:
                        stop = index.dim.size
                    elif isinstance(index, NonAffineMap):
                        stop = index.dim.size
                    else:
                        stop = index.stop
                    self.domains.append(self._make_domain(iname, index.start, stop, index.step))

                    local_offset += pym.var("TODO")


                    # if iname == "i2":
                        # FIXME THIS IS BROKEN
                        # breakpoint()

                    if isinstance(index, NonAffineMap):
                        # breakpoint()
                        temporary_name = self._namer.next("t_map")
                        temporary = Tensor((), name=temporary_name)
                        indexed = self._index_the_map(index.map, within_loops)
                        mapassignment = tlang.Read(indexed, temporary)
                        self._make_instruction_context(mapassignment, within_loops)
                        goff = pym.var(temporary_name)
                        if child := _tensor.dim.get_child(index.dim):
                            goff *= full_dim_size(child, _tensor.dim)
                        global_offset += goff
                    elif isinstance(index, Slice):
                        _tensor = assignment.tensor
                        goff = pym.var(iname) * index.step + index.start
                        if child := _tensor.dim.get_child(index.dim):
                            goff *= full_dim_size(child, _tensor.dim)
                        global_offset += goff
                    else:
                        raise NotImplementedError

                    within_inames |= {iname}

                assign_insn = lp.Assignment(
                        pym.subscript(pym.var(assignment.temporary.name), local_offset),
                        pym.subscript(pym.var(assignment.tensor.name), global_offset),
                        id=self._namer.next(f"{assignment.id}_"),
                        within_inames=within_inames)
                self.instructions.append(assign_insn)


        # breakpoint()
        if assignment.temporary.dim:
            temp_size = assignment.temporary.dim.size
        else:
            temp_size = 1

        self.kernel_data += [
            lp.GlobalArg(assignment.tensor.name, dtype=np.float64, shape=None),

            lp.TemporaryVariable(assignment.temporary.name, shape=(temp_size,))
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

    def _register_domain_parameter(self, param, prev_iname):
        if isinstance(param, numbers.Integral):
            return param
        elif isinstance(param, Tensor):
            assert prev_iname is not None

            temp_name = self._namer.next("n")

            self.instructions.append(lp.Assignment(pym.var(temp_name), pym.subscript(pym.var(param.name), pym.var(prev_iname))))
            self.kernel_data.append(lp.TemporaryVariable(temp_name, shape=(), dtype=np.int32))

            return temp_name

    @staticmethod
    def _make_domain(iname, start, stop, step):
        if step != 1:
            raise NotImplementedError

        assert all(param is not None for param in [start, stop, step])

        return f"{{ [{iname}]: {start} <= {iname} < {stop} }}"

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

    def myfunc(self):
        domains, domain_temps = _collect_domains(loops, namer)
        breakpoint()

        # assign_id = namer.next(assignment.id)
        #
        # temp_active = namer.next("T")
        # active = pym.var(temp_active)
        # kernel_data.append(lp.TemporaryVariable(temp_active, shape=(), dtype=np.int32))
        #
        # local_temp = namer.next("T")
        # local_active = pym.var(local_temp)
        # kernel_data.append(lp.TemporaryVariable(local_temp, shape=(), dtype=np.int32))
        #
        # active_init_insns = (
        #     lp.Assignment(active, 0, id=assign_id+"active"),
        #     lp.Assignment(local_active, 0, id=assign_id+"local"),
        # )
        #
        # init_insns, inc_insns, innermost_indices, kdata = _traverse_and_build(loops, assign_id, active, local_active, namer)
        #
        # global_idxs, local_idxs = innermost_indices
        #
        # main_within_inames = frozenset({loop.iname for loop in loops})
        # main_insn = _make_loopy_instruction(assignment, assign_id, local_idxs, global_idxs, local_offset, main_within_inames, depends_on={insn.id for insn in active_init_insns} | old_ids)
        #
        # all_insns.extend(active_init_insns)
        # all_insns.extend(init_insns)
        # all_insns.extend(inc_insns)
        # all_insns.append(main_insn)
        #
        # kernel_data.extend(kdata)

        for loop in loops:
            if isinstance(loop.index, NonAffineMap):
                kernel_data.append(lp.GlobalArg(loop.index.name, dtype=np.int32, shape=None))

        old_ids.add(assign_id)
        local_offset += indexed_size_per_index_group(indices)

    def _collect_loops(indices, namer, within_loops):
        if not indices:
            return ()

        index, *subindices = indices

        if index.within:
            loop = within_loops[index]
        else:
            iname = namer.next("i")
            loop = Loop.from_index(iname, index)
        return (loop,) + _collect_loops(subindices, namer, within_loops)


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


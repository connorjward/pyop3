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
from pyop3.utils import CustomTuple, checked_zip, NameGenerator
from pyop3.tensors import Tensor, Index, Map, Dim, IntIndex, LoopIndex, UniformDim, MixedDim, NonAffineMap
from pyop3.tensors import Slice, indexed_shape, IndexTree, indexed_size_per_index_group, AffineMap
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


def _make_loopy_kernel(tlang_kernel):
    # index -> iname
    shared_inames = _make_shared_inames(tlang_kernel.instructions)

    domains = []
    instructions = []
    kernel_data_in = []
    subkernels = []
    for insn in tlang_kernel.instructions:
        ctx = _make_instruction_context(insn, shared_inames)

        if isinstance(insn, tlang.Assignment):
            domains += _make_domains(insn.tensor.dim, insn.tensor.stencils, ctx.inames)
        instructions += ctx.loopy_instructions
        kernel_data_in += ctx.kernel_data
        subkernels += ctx.subkernels

    domains = set(domains)

    # uniquify
    kernel_data = []
    for kd in kernel_data_in:
        if kd not in kernel_data:
            kernel_data.append(kd)

    breakpoint()
    translation_unit = lp.make_kernel(
        domains,
        instructions,
        kernel_data,
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
    )
    return lp.merge((translation_unit, *subkernels))


def _make_shared_inames(instructions):
    shared_idxs = {idx for insn in instructions for idx in insn.loop_indices}
    iname_generator = NameGenerator("i")
    return {idx.domain: iname_generator.next() for idx in shared_idxs}


def _make_domains(dim, stencils, inames):
    return [dom for stencil in stencils for indices in stencil for dom in _make_domains_per_indices(dim, indices, inames)]


def _make_domains_per_indices(dim, indices, inames, within_indices=()):
    if not dim:
        return []
    if not indices:
        raise AssertionError
    domains = []
    index, *subindices = indices

    if isinstance(index, LoopIndex):
        index = index.domain

    while isinstance(index, Map):
        if not isinstance(index, IntIndex):
            domain = _make_domain(index.to_dim, index, inames, within_indices=within_indices)
            domains.append(domain)
        index = index.from_index

    if not isinstance(index, (LoopIndex, IntIndex)):
        domain = _make_domain(dim, index, inames, within_indices=within_indices)
        domains.append(domain)

    if isinstance(dim, MixedDim):
        subdim = dim.subdims[index.value] if dim.subdims else None
    else:
        subdim = dim.subdim

    return domains + _make_domains_per_indices(subdim, subindices, inames, within_indices=within_indices + (index,))


def _make_domain(dim, index, inames, within_indices):
    if isinstance(dim.size, Tensor):
        _, myidxs, _, myoffset, _ = _collect_indices(dim.size.dim, within_indices, inames_dict=inames)
        size = pym.subscript(pym.var(dim.size.name), myidxs+myoffset)
    else:
        size = dim.size

    if isinstance(index, Slice):
        start = index.start or 0
        stop = index.stop or size
    elif isinstance(index, Map):
        start = 0
        stop = index.to_dim.size
    else:
        raise NotImplementedError
    iname = inames[index]
    return f"{{ [{iname}]: {start} <= {iname} < {stop} }}"


def as_subarrayref(temporary, iname):
    """Register an argument to a function."""
    index = (pym.var(iname),)
    return lp.symbolic.SubArrayRef(
        index, pym.subscript(pym.var(temporary.name), index)
    )

@functools.singledispatch
def _make_instruction_context(self, instruction: tlang.Instruction, within_indices_to_inames):
    raise ValueError

@_make_instruction_context.register
def _(call: tlang.FunctionCall, within_indices_to_inames):
    subarrayrefs = {}
    iname_namer = NameGenerator(prefix=f"{call.id}_i")
    inames = within_indices_to_inames.copy()
    for temp in itertools.chain(call.reads, call.writes):
        if temp in subarrayrefs:
            continue
        iname = iname_namer.next()
        inames[UniformDim(temp.dim.size)] = iname
        subarrayrefs[temp] = as_subarrayref(temp, iname)

    reads = tuple(subarrayrefs[var] for var in call.reads)
    writes = tuple(subarrayrefs[var] for var in call.writes)
    assignees = tuple(writes)
    expression = pym.primitives.Call(
        pym.var(call.function.loopy_kernel.default_entrypoint.name),
        tuple(reads),
    )

    within_inames = frozenset({within_indices_to_inames[index.domain] for index in call.loop_indices})

    kernel_data = [lp.TemporaryVariable(temp.name, shape=(temp.dim.size,)) for temp in subarrayrefs]

    depends_on = frozenset({f"{insn}*" for insn in call.depends_on})
    call_insn = lp.CallInstruction(
        assignees,
        expression,
        id=call.id,
        within_inames=within_inames,
        depends_on=depends_on
    )
    return InstructionContext([call_insn], inames, kernel_data, {call.function.loopy_kernel})


@dataclasses.dataclass(frozen=True)
class InstructionContext:
    loopy_instructions: Tuple
    inames: Dict
    kernel_data: Tuple
    subkernels: Tuple = ()


@_make_instruction_context.register
def _(assignment: tlang.Assignment, within_indices_to_inames):
    id_namer = NameGenerator(prefix=f"{assignment.id}_")
    stencils = frozenset({_complete_stencil(assignment.tensor.dim, stencil) for stencil in assignment.tensor.stencils})
    inames = _make_inames(stencils, assignment.id, within_indices_to_inames)
    loopy_instructions = []
    for stencil in stencils:
        # each entry into the temporary needs to be offset by the size of
        # the previous one
        initial_local_offset = 0
        for indices in stencil:
            local_idxs, global_idxs, local_offset, global_offset, within_inames = \
                _collect_indices(assignment.tensor.dim, indices, local_offset=initial_local_offset, inames_dict=inames)
            initial_local_offset += indexed_size_per_index_group(assignment.tensor.dim, indices)
            loopy_instructions.append(_make_loopy_instruction(assignment, id_namer.next(), local_idxs, global_idxs, local_offset, global_offset, within_inames))

    kernel_data = [
        lp.GlobalArg(assignment.tensor.name, dtype=np.float64, shape=None),
        lp.TemporaryVariable(assignment.temporary.name, shape=(assignment.temporary.dim.size,))
    ]

    kernel_data += [map_ for stencil in stencils for indices in stencil for map_ in _collect_maps(indices)]

    return InstructionContext(loopy_instructions, inames, kernel_data)


def _collect_maps(indices):
    if not indices:
        return ()

    index, *subindices = indices
    return _collect_maps_from_index(index) + _collect_maps(subindices)


@functools.singledispatch
def _collect_maps_from_index(index):
    return ()


@_collect_maps_from_index.register
def _(index: NonAffineMap):
    return _collect_maps_from_index(index.from_index) + (lp.GlobalArg(index.name, shape=None, dtype=np.int32),)


def _make_inames(stencils, instruction_id, within_inames):
    iname_generator = NameGenerator(prefix=f"{instruction_id}_i")
    return within_inames | {
        index: iname_generator.next()
        for stencil in stencils for idxs in stencil for idx in idxs
        for index in _expand_index(idx)
        if index not in within_inames
    }


@functools.singledispatch
def _expand_index(index):
    raise AssertionError

@_expand_index.register
def _(index: IntIndex):
    return ()


@_expand_index.register
def _(index: LoopIndex):
    return (index.domain,)


@_expand_index.register
def _(index: Slice):
    return (index,)


@_expand_index.register
def _(index: Map):
    return _expand_index(index.from_index) + (index,)


def _complete_stencil(dim, stencil):
    return tuple(_complete_indices(dim, indices) for indices in stencil)


def _complete_indices(dim, indices):
    if not dim:
        return ()

    if indices:
        index, *subindices = indices
    else:
        index, subindices = Slice(), ()

    if isinstance(dim, MixedDim):
        subdim = dim.subdims[index.value]
    else:
        subdim = dim.subdim
    return (index,) + _complete_indices(subdim, subindices)



def _make_loopy_instruction(instruction, id, local_idxs, global_idxs, local_offset, global_offset, within_inames):
    # wilcard this to catch subinsns
    depends_on = frozenset({f"{insn}*" for insn in instruction.depends_on})

    assignee, expression = resolve(instruction, global_idxs, local_idxs, global_offset, local_offset)

    return lp.Assignment(assignee, expression, id=id,
            within_inames=within_inames, depends_on=depends_on)


def _collect_indices(
        dim, indices,
        local_idxs=0, global_idxs=0,
        local_offset=0, global_offset=0,
        within_inames=frozenset(),
        inames_dict={},
        within_indices=(),
):
    index, *subindices = indices

    within_indices += (index,)

    if isinstance(dim, MixedDim):
        # The global offset is very complicated - we need to build up the offsets every time
        # we do not take the first subdim by adding the size of the prior subdims. Make sure to
        # not multiply this value.
        if dim.subdims:
            for d in dim.subdims[:index.value]:
                if isinstance(d.size, Tensor):
                    _, myidxs, _, myoffset, _ = _collect_indices(d.size.dim, within_indices, inames_dict=inames_dict)
                    size = pym.subscript(pym.var(d.size.name), myidxs+myoffset)
                else:
                    size = d.size
                global_offset += size
    else:
        ### tidy up
        inames, local_idxs_, global_idxs_ = handle_index(index, inames_dict)

        if global_idxs_:
            global_idxs += global_idxs_

        if local_idxs_:
            local_idxs += local_idxs_

        if inames:
            within_inames |= set(inames)
        ### !tidy up

    if isinstance(dim, MixedDim):
        if dim.subdims:
            subdim = dim.subdims[index.value]
        else:
            subdim = None
    else:
        subdim = dim.subdim


    if subdim:
        if isinstance(subdim.size, Tensor):
            _, myidxs, _, myoffset, _ = _collect_indices(subdim.size.dim, within_indices, inames_dict=inames_dict)
            size = pym.subscript(pym.var(subdim.size.name), myidxs+myoffset)
            global_idxs *= size
        else:
            global_idxs *= subdim.size
        local_idxs *= indexed_size_per_index_group(subdim, subindices)

    if not subdim:
        return local_idxs, global_idxs, local_offset, global_offset, within_inames
    else:
        return _collect_indices(subdim, subindices, local_idxs, global_idxs, local_offset, global_offset, within_inames, inames_dict, within_indices)


@functools.singledispatch
def handle_index(index, inames):
    raise AssertionError

@handle_index.register
def _(slice_: Slice, inames):
    iname = inames[slice_]
    return (iname,), pym.var(iname), pym.var(iname)

@handle_index.register
def _(map_: NonAffineMap, inames_dict):
    inames, temp_idxs, tensor_idxs = handle_index(map_.from_index, inames_dict)

    riname = inames_dict[map_]
    rinames = (riname,)
    rtemp_idxs = pym.var(riname)
    rtensor_idxs = pym.var(riname)

    if temp_idxs:
        temp_expr = temp_idxs * map_.to_dim.size + rtemp_idxs
    else:
        temp_expr = rtemp_idxs
    tensor_expr = tensor_idxs, rtensor_idxs

    return inames + rinames, temp_expr, pym.subscript(pym.var("map"), tensor_expr)


@handle_index.register
def _(map_: AffineMap, inames_dict):
    inames, temp_idxs, tensor_idxs = handle_index(map_.from_index, inames_dict)
    riname = inames_dict[map_]
    rinames = (riname,)
    rtemp_idxs = pym.var(riname)
    rtensor_idxs = pym.var(riname)

    """
    for i
        for j
            for d
                t[j, d] = dat[i+j, d]
    """

    if temp_idxs:
        temp_expr = temp_idxs * map_.to_dim.size + rtemp_idxs
    else:
        temp_expr = rtemp_idxs

    tensor_expr = tensor_idxs + rtensor_idxs

    return inames + rinames, temp_expr, tensor_expr

@handle_index.register
def _(index: LoopIndex, inames):
    iname = inames[index.domain]
    return (iname,), None, pym.var(iname)


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
    def __init__(self, instruction, global_idxs, local_idxs, global_offset, local_offset):
        self.instruction = instruction
        self.global_idxs = global_idxs
        self.local_idxs = local_idxs
        self.global_offset = global_offset
        self.local_offset = local_offset

    @property
    def global_expr(self):
        return pym.subscript(pym.var(self.instruction.tensor.name), self.global_idxs + self.global_offset)

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
def register_maps(self, map_):
    if isinstance(map_, Map) and map_ not in self.maps_dict:
        self.maps_dict[map_] = lp.GlobalArg(
            map_.name, dtype=np.int32, shape=(None, map_.size)
        )
        self.register_maps(map_.from_space)

def register_parameter(self, parameter, **kwargs):
    return self.parameter_dict.setdefault(parameter, lp.GlobalArg(**kwargs))

def register_domain_parameter(self, parameter, within_inames):
    if isinstance(parameter, Tensor):
        if parameter in self.parameters_dict:
            return self.parameters_dict[parameter]

        assert parameter.is_scalar

        # FIXME need to validate if already exists here...
        self.register_parameter(parameter,
            # TODO should really validate dtype
            name=parameter.name, dtype=np.int32, shape=(None,) * len(parameter.indices)
        )

        # if the tensor is an indexed expression we need to create a temporary for it
        if parameter.indices:
            temporary = self.register_temporary()
            assignee = pym.var(temporary.name)
            # TODO Put into utility function
            expression = pym.subscript(
                pym.var(parameter.name),
                tuple(pym.var(within_inames[index.space]) for index in parameter.indices)
            )
            self.register_assignment(
                assignee, expression, within_inames=within_inames
            )
            return self.parameters_dict.setdefault(parameter, pym.var(temporary.name))
        else:
            return self.parameters_dict.setdefault(parameter, pym.var(parameter.name))
    elif isinstance(parameter, str):
        self.register_parameter(parameter, name=parameter, dtype=np.int32, shape=())
        return self.parameters_dict.setdefault(parameter, pym.var(parameter))
    elif isinstance(parameter, numbers.Integral):
        return self.parameters_dict.setdefault(parameter, parameter)
    else:
        raise ValueError

def register_domain(self, domain, within_inames):
    if isinstance(domain, Map):
        start = 0
        stop = domain.size
    elif isinstance(domain, Slice):
        start = domain.start
        stop = domain.stop
    else:
        raise AssertionError

    iname = self.generate_iname()
    start, stop = (self.register_domain_parameter(param, within_inames)
                   for param in [start, stop])
    self.domains.append(DomainContext(iname, start, stop))
    return iname


def as_shape(self, domains, *, first_is_none=False):
    shape = []
    for i, domain in enumerate(domains):
        if first_is_none and i == 0:
            shape.append(None)
        else:
            if isinstance(domain, Map):
                start = 0
                stop = domain.size
            elif isinstance(domain, Slice):
                start = domain.start
                stop = domain.stop
            else:
                raise AssertionError

            assert start in self.parameters_dict
            assert stop in self.parameters_dict
            start = self.register_domain_parameter(start, None)
            stop = self.register_domain_parameter(stop, None)
            shape.append(stop-start)
    return tuple(shape)

def as_orig_shape(self, domains):
    if domains == ():
        return ()
    shape = [None]
    for domain in domains[1:]:
        stop = self.register_domain_parameter(domain, None)

        var = stop

        # hack for ragged (extruded)
        if hasattr(var, "name") and var.name.startswith("t"):
            shape.append(2)
        else:
            shape.append(stop)
    return tuple(shape)
"""

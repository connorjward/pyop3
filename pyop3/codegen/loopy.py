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
from pyop3.tensors import Tensor, Index, Map, Dim, IntIndex, LoopIndex, Range, UniformDim, MixedDim
from pyop3.tensors import Slice, indexed_shape, IndexTree, indexed_size_per_index_group
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
    return _LoopyKernelBuilder(to_tlang(expr)).build()


def to_c(expr):
    program = to_loopy(expr)
    return lp.generate_code_v2(program).device_code()


# @dataclasses.dataclass(frozen=True)
# class DomainContext:
#     iname: str
#     start: pym.Variable
#     stop: pym.Variable
#
#     def __str__(self):
#         return f"{{ [{self.iname}]: {self.start} <= {self.iname} < {self.stop} }}"
#
#     @property
#     def shape(self):
#         return self.stop - self.start
#


class _LoopyKernelBuilder:
    def __init__(self, expr):
        self.expr = expr

        self.existing_inames = {}
        self.tlang_instructions = collections.defaultdict(list)
        self.temp_to_itree = {}

        self.domains = []
        self.instructions = []
        self.arguments_dict = {}
        self.parameter_dict = {}
        self.parameters_dict = {}
        self.subkernels = []
        self.temporaries_dict = {}
        self.domain_parameters = {}
        self.maps_dict = {}
        self.function_temporaries = collections.defaultdict(dict)
        self.name_generator = pyop3.utils.UniqueNameGenerator()
        self.insn_within_indices = collections.defaultdict(list)

    @property
    def parameters(self):
        return tuple(self.parameter_dict.values())
    @property
    def arguments(self):
        return tuple(self.arguments_dict.values())
    @property
    def temporaries(self):
        return tuple(self.temporaries_dict.values())

    @property
    def maps(self):
        return tuple(self.maps_dict.values())

    @property
    def kernel_data(self):
        return tuple(self.arguments) + self.maps + tuple(self.temporaries) + self.parameters

    def build(self):
        # FIXME broken
        self.arguments_dict["map"] = lp.GlobalArg(
            "map",
            dtype=np.int32,  # TODO fix
            shape=None
        )

        shared_inames = self._make_shared_inames(self.expr.instructions)

        instruction_contexts = {
            _make_instruction_context(insn, shared_inames)
            for insn in self.expr.instructions
        }

        all_inames = shared_inames | {
            index: iname
            for handler in instruction_handlers
            for index, iname in handler.inames.items()
        }

        domains = {
            self._make_domain(index, iname) for index, iname in all_inames.items()
        }

        instructions = {
            insn for ctx in instruction_contexts for insn in ctx.loopy_instructions
        }

        translation_unit = lp.make_kernel(
            domains,
            instructions,
            self.kernel_data,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )
        return lp.merge((translation_unit, *self.subkernels))

    @staticmethod
    def _make_shared_inames(instructions):
        shared_idxs = {idx for insn in instructions for idx in insn.loop_indices}
        iname_generator = NameGenerator("i")
        return {idx.domain: iname_generator.next() for idx in shared_idxs}

    @staticmethod
    def _make_domain(index, iname):
        return f"{{ [{iname}]: {index.start} <= {iname} < {index.stop} }}"

    def register_temporary(self, temporary, dim, stencil):
        if temporary in self.temporaries_dict:
            return

        shape = indexed_shape(dim, stencil)
        self.temporaries_dict[temporary] = lp.TemporaryVariable(temporary.name, shape=shape)

    def register_tensor(self, tensor):
        if tensor.name in self.arguments_dict:
            return

        self.arguments_dict[tensor.name] = lp.GlobalArg(
            tensor.name,
            dtype=np.float64,  # TODO fix
            shape=None
        )

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


    def as_subarrayref(self, temporary):
        """Register an argument to a function."""
        iname = self.name_generator.next("i")
        self.domains.append(f"{{ [{iname}]: 0 <= {iname} < {temporary.dim.size} }}")
        index = (pym.var(iname),)
        return lp.symbolic.SubArrayRef(
            index, pym.subscript(pym.var(temporary.name), index)
        )

@functools.singledispatch
def _make_instruction_context(self, instruction: tlang.Instruction, within_indices_to_inames):
    raise ValueError

@_make_instruction_context.register
def _(assignment: tlang.FunctionCall, within_indices_to_inames):
    subarrayrefs = {}
    for temp in itertools.chain(instruction.reads, instruction.writes):
        if temp in subarrayrefs:
            continue
        subarrayrefs[temp] = self.as_subarrayref(temp)

    reads = tuple(subarrayrefs[var] for var in instruction.reads)
    writes = tuple(subarrayrefs[var] for var in instruction.writes)
    assignees = tuple(writes)
    expression = pym.primitives.Call(
        pym.var(instruction.function.loopy_kernel.default_entrypoint.name),
        tuple(reads),
    )

    within_inames = frozenset({within_indices[index] for index in instruction.within_indices})

    depends_on = frozenset({f"{insn}*" for insn in instruction.depends_on})
    call_insn = lp.CallInstruction(
        assignees,
        expression,
        id=instruction.id,
        within_inames=within_inames,
        depends_on=depends_on
    )
    self.instructions.append(call_insn)
    self.tlang_instructions[instruction].append(call_insn)
    self.subkernels.append(instruction.function.loopy_kernel)
    return call_insn


@dataclasses.dataclass
class InstructionContext:
    loopy_instructions: FrozenSet
    inames: Dict


@_make_instruction_context.register
def _(assignment: tlang.Assignment, within_indices_to_inames):
    stencils = frozenset({_complete_stencil(assignment.tensor.dim, stencil) for stencil in assignment.tensor.stencils})
    inames = _make_inames(stencils, assignment.id, within_indices_to_inames)
    loopy_instructions = set()
    for stencil in stencils:
        # each entry into the temporary needs to be offset by the size of
        # the previous one
        local_offset = 0
        for indices in stencil:
            local_idxs, global_idxs, within_inames = \
                _collect_indices(assignment.tensor.dim, indices, local_offset=local_offset, inames_dict=within_indices_to_inames|inames)
            local_offset += indexed_size_per_index_group(assignment.tensor.dim, indices)
            loopy_instructions.add(_make_loopy_instruction(within_inames))

    return InstructionContext(instructions)


def _make_inames(stencils, instruction_id, within_inames):
    iname_generator = NameGenerator(prefix=f"{instruction_id}_i")
    # return {
    #     idx: iname_generator.next()
    #     for stencil in stencils for idxs in stencil for idx in idxs
    #     if idx not in within_inames
    # }
    # TODO Make into a comprehension
    inames = {}
    for stencil in stencils:
        for indices in stencil:
            for index in indices:
                if index in within_inames:
                    continue
                if isinstance(index, LoopIndex):
                    index = index.domain
                inames[index] = iname_generator.next()
    return inames | within_inames


def _complete_stencil(dim, stencil):
    return frozenset({_complete_indices(dim, indices) for indices in stencil})


def _complete_indices(dim, indices):
    if not dim:
        return ()

    if indices:
        index, *subindices = indices
    else:
        index, subindices = Range(dim.size), ()
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
        inames_dict={}
):
    index, *subindices = indices

    if isinstance(dim, MixedDim):
        subdim = dim.subdims[index.value]
        # The global offset is very complicated - we need to build up the offsets every time
        # we do not take the first subdim by adding the size of the prior subdims. Make sure to
        # not multiply this value.
        for dim in dim.subdims[:index.value]:
            global_offset += dim.size
    else:
        subdim = dim.subdim

        ### tidy up
        inames, local_idxs_, global_idxs_ = handle_index(index, inames_dict)

        if global_idxs_:
            global_idxs += global_idxs_

        if local_idxs_:
            local_idxs += local_idxs_

        if inames:
            within_inames |= set(inames)
        ### !tidy up

    if subdim:
        global_idxs *= subdim.size
        local_idxs *= indexed_size_per_index_group(subdim, subindices)

    if not subdim:
        return local_idxs, global_idxs, local_offset, global_offset, within_inames
    else:
        return _collect_indices(subdim, subindices, local_idxs, global_idxs, local_offset, global_offset, within_inames, inames_dict)


@functools.singledispatch
def handle_index(index, inames):
    raise AssertionError

@handle_index.register
def _(range_: Range, inames):
    iname = inames[range_]
    return (iname,), pym.var(iname), pym.var(iname)

@handle_index.register
def _(index: IntIndex):
    raise NotImplementedError
    return None, 0, (index.value,)

@handle_index.register
def _(map_: Map, inames_dict):
    inames, temp_idxs, tensor_idxs = handle_index(map_.from_dim, inames_dict)
    rinames, rtemp_idxs, rtensor_idxs = handle_index(map_.to_dim, inames_dict)

    if temp_idxs:
        temp_expr = temp_idxs * map_.to_dim.size + rtemp_idxs
    else:
        temp_expr = rtemp_idxs
    tensor_expr = tensor_idxs, rtensor_idxs

    return inames + rinames, temp_expr, pym.subscript(pym.var("map"), tensor_expr)

@handle_index.register
def _(index: LoopIndex, inames):
    iname = inames[index.domain]
    return (iname,), None, pym.var(iname)


def resolve(self, *args):
    if isinstance(self.instruction, tlang.Read):
        resolver = ReadAssignmentResolver(*args)
    elif isinstance(self.instruction, tlang.Zero):
        resolver = ZeroAssignmentResolver(*args)
    elif isinstance(self.instruction, tlang.Write):
        resolver = WriteAssignmentResolver(*args)
    elif isinstance(self.instruction, tlang.Increment):
        resolver = IncAssignmentResolver(*args)
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
"""

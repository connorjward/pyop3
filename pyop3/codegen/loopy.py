import abc
import collections
import dataclasses
import numbers
import enum
import functools
import itertools
from typing import Dict, Any

import loopy as lp
import loopy.symbolic
import numpy as np
import pytools
import pymbolic as pym

import pyop3.exprs
from pyop3 import exprs, tlang
import pyop3.utils
from pyop3.utils import CustomTuple
from pyop3.tensors import Tensor, Index, Map, Dim, IntIndex, LoopIndex, Range
from pyop3.tensors import Slice
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

        for insn in self.expr.instructions:
            self.register_instruction(insn)

            if not isinstance(insn, tlang.FunctionCall):
                if insn.temporary not in self.temp_to_itree:
                    self.temp_to_itree[insn.temporary] = insn.tensor.indices

                self.register_temporary(insn.temporary, insn.tensor.indices, insn.tensor.dim)
                self.register_tensor(insn.tensor)

        breakpoint()
        translation_unit = lp.make_kernel(
            self.domains,
            self.instructions,
            self.kernel_data,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )

        return lp.merge((translation_unit, *self.subkernels))

    def register_temporary(self, name, itree, dim):
        if name in self.temporaries_dict:
            return

        self.temporaries_dict[name] = lp.TemporaryVariable(name, shape=(itree.compute_size(dim),))

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


    def as_subarrayref(self, temporary, itree, dim):
        """Register an argument to a function."""
        iname = self.name_generator.next("i")
        self.domains.append(f"{{ [{iname}]: 0 <= {iname} < {itree.compute_size(dim)} }}")
        index = (pym.var(iname),)
        return lp.symbolic.SubArrayRef(
            index, pym.subscript(pym.var(temporary), index)
        )

    def register_instruction(self, instruction: tlang.Instruction):
        if not isinstance(instruction, tlang.FunctionCall):
            self.traverse(instruction)
            return

        # FIXME dim itree nonsense
        return

        subarrayrefs = {}
        for temp in itertools.chain(instruction.reads, instruction.writes):
            if temp in subarrayrefs:
                continue
            # subarrayrefs[temp] = self.as_subarrayref(temp, self.temp_to_itree[temp], ???)

        reads = tuple(subarrayrefs[var] for var in instruction.reads)
        writes = tuple(subarrayrefs[var] for var in instruction.writes)
        assignees = tuple(writes)
        expression = pym.primitives.Call(
            pym.var(instruction.function.loopy_kernel.default_entrypoint.name),
            tuple(reads),
        )

        within_inames = frozenset.union(*(self.get_within_inames(index) for index in instruction.within_indices))

        depends_on = frozenset(itertools.chain(*(self.tlang_instructions[insn] for insn in instruction.depends_on)))
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=self.name_generator.next("func"),
            within_inames=within_inames,
            depends_on=depends_on
        )
        self.instructions.append(call_insn)
        self.tlang_instructions[instruction].append(call_insn)
        self.subkernels.append(instruction.function.loopy_kernel)
        return call_insn

    def get_within_inames(self, itree):
        if not itree:
            return frozenset()
        within_inames = set()
        for index, subtree in itree.indices.items():
            if not isinstance(index, IntIndex):
                within_inames.add(self.existing_inames[index])
            within_inames |= self.get_within_inames(subtree)
        return frozenset(within_inames)

    def register_maps(self, map_):
        if isinstance(map_, Map) and map_ not in self.maps_dict:
            self.maps_dict[map_] = lp.GlobalArg(
                map_.name, dtype=np.int32, shape=(None, map_.size)
            )
            self.register_maps(map_.from_space)

    def register_parameter(self, parameter, **kwargs):
        return self.parameter_dict.setdefault(parameter, lp.GlobalArg(**kwargs))

    def register_domain_parameter(self, parameter, within_inames):
        """Return a pymbolic expression"""
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

    @functools.singledispatchmethod
    def resolve(self, instruction, tensor_idxs, temp_idx):
        raise AssertionError

    @resolve.register
    def _(self, read: tlang.Read, tensor_indices, temp_index):
        assignee = pym.subscript(pym.var(read.temporary), temp_index)
        expression = pym.subscript(pym.var(read.tensor.name), tensor_indices)
        return assignee, expression

    @resolve.register
    def _(self, zero: tlang.Zero, tensor_indices, temp_indices):
        assignee = pym.subscript(pym.var(zero.temporary), temp_indices)
        expression = 0
        return assignee, expression

    @resolve.register
    def _(self, write: tlang.Write, tensor_indices, temp_indices):
        assignee = pym.subscript(pym.var(write.tensor.name), tensor_indices)
        expression = pym.subscript(pym.var(write.temporary), temp_indices)
        return assignee, expression

    @resolve.register
    def _(self, inc: tlang.Increment, tensor_indices, temp_indices):
        assignee = pym.subscript(pym.var(inc.tensor.name), tensor_indices)
        expression = assignee + pym.subscript(pym.var(inc.temporary), temp_indices)
        return assignee, expression

    def _register_instruction(self, instruction, assignee, expression, within_inames):
        id_ = self.name_generator.next(instruction.id)
        depends_on = frozenset(itertools.chain(*(self.tlang_instructions[insn] for insn in instruction.depends_on)))
        lp_instruction = lp.Assignment(assignee, expression, id=id_,
                within_inames=within_inames, depends_on=depends_on)
        self.instructions.append(lp_instruction)
        self.tlang_instructions[instruction].append(lp_instruction)


    def traverse(self, instruction):
        """Return an indexed expression of whatever the instruction is"""
        offset = 0
        for dim, (index, itree) in zip(instruction.tensor.dim.children, instruction.tensor.indices.indices.items()):
            self._traverse(instruction, instruction.tensor.dim, index, itree, offset)
            offset += itree.compute_size(dim)

    def _traverse(self, instruction, dim: Dim, index, itree, offset=0, tensor_expr_indices=(), temp_expr_index=0, within_inames=frozenset()):

        # handle the different indexing that needs to be done
        inames, temp_idx, tensor_index = self.handle_index(index)
        tensor_expr_indices += tensor_index

        if itree:
            temp_expr_index += temp_idx * itree.compute_size(dim)
        else:
            temp_expr_index += temp_idx
        temp_expr_index += offset

        if inames:
            within_inames |= set(inames)

        # if leaf
        if not dim.children:
            assert not itree
            assignee, expression = self.resolve(instruction, tensor_expr_indices, temp_expr_index)
            self._register_instruction(instruction, assignee, expression, within_inames)
        elif dim.has_single_child_dim_type:
            if not itree:
                subindex = Range(dim.child.size)
                subitree = None
            else:
                subindex, = itree.indices.keys()
                subitree, = itree.indices.values()
            self._traverse(instruction, dim.child, subindex, subitree, 0, tensor_expr_indices, temp_expr_index, within_inames)
        else:
            if isinstance(index, Slice):
                subdims = dim.children[index.value]
            else:
                subdims = (dim.children[index.value],)

            suboffset = 0
            for subdim, (idx, itree_) in zip(subdims, itree.indices.items()):
                self._traverse(instruction, subdim, idx, itree_, suboffset, tensor_expr_indices, temp_expr_index, within_inames)
                if itree_:
                    suboffset += itree_.size
                else:
                    suboffset += idx.size

    @functools.singledispatchmethod
    def handle_index(self, index):
        """Return inames and tensor pym expr for indices"""
        raise AssertionError

    @handle_index.register
    def _(self, index: Range):
        try:
            iname = self.existing_inames[index]
        except KeyError:
            iname = self.existing_inames.setdefault(index, self.name_generator.next("i"))
            self.domains.append(f"{{ [{iname}]: {index.start} <= {iname} < {index.stop} }}")
        return (iname,), pym.var(iname), (pym.var(iname),)

    @handle_index.register
    def _(self, index: IntIndex):
        return None, 0, (index.value,)

    @handle_index.register
    def _(self, index: Map):
        inames, temp_idxs, tensor_idxs = self.handle_index(index.from_dim)
        rinames, rtemp_idxs, rtensor_idxs = self.handle_index(index.to_dim)
        return inames + rinames, temp_idxs*index.to_dim.size + rtemp_idxs, (pym.subscript(pym.var("map"), tensor_idxs+rtensor_idxs),)

    @handle_index.register
    def _(self, index: LoopIndex):
        try:
            iname = self.existing_inames[index]
        except KeyError:
            iname = self.existing_inames.setdefault(index, self.name_generator.next("i"))
            self.domains.append(f"{{ [{iname}]: {index.domain.start} <= {iname} < {index.domain.stop} }}")
        return (iname,), 0, (pym.var(iname),)

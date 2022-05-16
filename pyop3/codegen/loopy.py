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
import pymbolic.primitives as pym

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


@dataclasses.dataclass(frozen=True)
class DomainContext:
    iname: str
    start: pym.Variable
    stop: pym.Variable

    def __str__(self):
        return f"{{ [{self.iname}]: {self.start} <= {self.iname} < {self.stop} }}"

    @property
    def shape(self):
        return self.stop - self.start



class _LoopyKernelBuilder:
    def __init__(self, expr):
        self.expr = expr

        self.mydomains = set()
        self.existing_inames = {}

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
        for insn in self.expr.instructions:
            self.register_instruction(insn)

            if not isinstance(insn, tlang.FunctionCall):
                self.register_temporary(insn.temporary)
                self.register_tensor(insn.tensor)

        breakpoint()
        translation_unit = lp.make_kernel(
            self.mydomains,
            self.instructions,
            self.kernel_data,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )

        return lp.merge((translation_unit, *self.subkernels))

    def register_temporary(self, name):
        if name in self.temporaries_dict:
            return

        self.temporaries_dict[name] = lp.TemporaryVariable(name)

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


    def get_function_rw(self, call, within_domains):
        reads = []
        writes = []
        for argument in call.arguments:
            temporary = self.function_temporaries[call][argument]
            access = argument.access

            ref = self.as_subarrayref(argument, temporary, within_domains)
            if access == pyop3.exprs.READ:
                reads.append(ref)
            elif access == pyop3.exprs.WRITE:
                writes.append(ref)
            elif access in {pyop3.exprs.INC, pyop3.exprs.RW}:
                reads.append(ref)
                writes.append(ref)
            else:
                raise NotImplementedError
        return tuple(reads), tuple(writes)

    def as_subarrayref(self, argument, temporary, within_domains):
        """Register an argument to a function."""
        inames = self.register_broadcast_domains(argument, within_domains)
        indices = tuple(pym.Variable(iname) for iname in inames.values())
        return lp.symbolic.SubArrayRef(
            indices, pym.Subscript(pym.Variable(temporary.name), indices)
        )

    @functools.singledispatchmethod
    def register_instruction(self, instruction: tlang.Instruction):
        raise TypeError

    @register_instruction.register
    def _(self, instruction: tlang.Read):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = dat[i, j, k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        converter = ReadConverter(instruction, self.existing_inames)
        self.existing_inames = converter.traverse()
        self.mydomains |= converter.domains
        self.instructions.extend(converter.instructions)

    @register_instruction.register
    def _(self, instruction: tlang.Zero):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = 0
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        converter = ZeroConverter(instruction, self.existing_inames)
        self.existing_inames = converter.traverse()
        self.mydomains |= converter.domains
        self.instructions.extend(converter.instructions)

    @register_instruction.register
    def _(self, instruction: tlang.Write):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        converter = WriteConverter(instruction, self.existing_inames)
        self.existing_inames = converter.traverse()
        self.mydomains |= converter.domains
        self.instructions.extend(converter.instructions)

    @register_instruction.register
    def _(self, instruction: tlang.Increment):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = dat[i, j, k, l] + t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        converter = IncrementConverter(instruction, self.existing_inames)
        self.existing_inames = converter.traverse()
        self.mydomains |= converter.domains
        self.instructions.extend(converter.instructions)

    @register_instruction.register
    def _(self, instruction: tlang.FunctionCall):
        # reads, writes = self.get_function_rw(call, within_inames)
        reads = tuple(pym.Variable(var) for var in instruction.reads)
        writes = tuple(pym.Variable(var) for var in instruction.writes)
        assignees = tuple(writes)
        expression = pym.Call(
            pym.Variable(instruction.function.loopy_kernel.default_entrypoint.name),
            tuple(reads),
        )

        within_inames = frozenset.union(*(self.get_within_inames(index) for index in instruction.within_indices))

        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=self.name_generator.generate("func"),
            within_inames=within_inames,
            depends_on=instruction.depends_on
        )
        self.instructions.append(call_insn)
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
                assignee = pym.Variable(temporary.name)
                # TODO Put into utility function
                expression = pym.Subscript(
                    pym.Variable(parameter.name),
                    tuple(pym.Variable(within_inames[index.space]) for index in parameter.indices)
                )
                self.register_assignment(
                    assignee, expression, within_inames=within_inames
                )
                return self.parameters_dict.setdefault(parameter, pym.Variable(temporary.name))
            else:
                return self.parameters_dict.setdefault(parameter, pym.Variable(parameter.name))
        elif isinstance(parameter, str):
            self.register_parameter(parameter, name=parameter, dtype=np.int32, shape=())
            return self.parameters_dict.setdefault(parameter, pym.Variable(parameter))
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

    def register_broadcast_domains(self, argument, within_inames):
        inames = {}
        for domain in argument.tensor.broadcast_domains:
            inames[domain] = self.register_domain(domain, within_inames)
        return inames


# TODO rename
class TensorToLoopyConverter:

    def __init__(self, instruction: tlang.Instruction, existing_inames):
        self.instruction = instruction

        self.domains = set()
        self.instructions = []
        self.dims = {}

        self.iname_dict = {}
        name_generator = pytools.UniqueNameGenerator(existing_names=set(existing_inames.values()))
        self.iname_generator = functools.partial(name_generator, "i")
        self.instruction_generator = functools.partial(name_generator, self.instruction_prefix)
        self.inames = existing_inames.copy()

    @abc.abstractmethod
    def resolve(self, *args, **kwargs):
        pass

    @property
    def temporary(self):
        return pym.Variable(self.instruction.temporary)

    @property
    def tensor(self):
        return pym.Variable(self.instruction.tensor.name)

    def register_instruction(self, assignee, expression, within_inames):
        self.instructions.append(
            lp.Assignment(assignee, expression, id=self.instruction.id,
                within_inames=within_inames, depends_on=self.instruction.depends_on)
        )

    def traverse(self):
        """Return an indexed expression of whatever the instruction is"""
        for index, itree in self.instruction.tensor.indices.indices.items():
            self._traverse(self.instruction.tensor.dim, index, itree)
        return self.inames

    def _traverse(self, dim: Dim, index, itree, tensor_expr_indices=CustomTuple(), temp_expr_indices=CustomTuple(), within_inames=frozenset()):


        # all indices not explicitly described are broadcast over
        # if not indices:
        #     indices = (Slice(),)

        # iname = f"i{next(self.iname_counter)}"
        # self.iname_dict[dim] = iname
        # temp_expr_indices += (pym.Variable(iname),)

        # handle the different indexing that needs to be done
        iname, temp_idx, tensor_index = self.handle_index(index)
        tensor_expr_indices += tensor_index
        temp_expr_indices += temp_idx

        if iname:
            within_inames |= {iname}

        # if leaf
        if not dim.children:
            assert not itree
            assignee, expression = self.resolve(tensor_expr_indices, temp_expr_indices)
            self.register_instruction(assignee, expression, within_inames)
        elif dim.has_single_child_dim_type:
            if not itree:
                subindex = Range(dim.child.size)
                subitree = None
            else:
                subindex, = itree.indices.keys()
                subitree, = itree.indices.values()
            self._traverse(dim.child, subindex, subitree, tensor_expr_indices, temp_expr_indices, within_inames)
        else:
            if isinstance(index, Slice):
                subdims = dim.children[index.value]
            else:
                subdims = (dim.children[index.value],)

            for subdim, (idx, itree_) in zip(subdims, itree.indices.items()):
                self._traverse(subdim, idx, itree_, tensor_expr_indices, temp_expr_indices, within_inames)


    @functools.singledispatchmethod
    def handle_index(self, index):
        """Return inames and tensor pym expr for indices"""
        raise AssertionError

    @handle_index.register
    def _(self, index: Range):
        try:
            iname = self.inames[index]
        except KeyError:
            iname = self.inames.setdefault(index, self.iname_generator())
            self.domains.add(f"{{ [{iname}]: {index.start} <= {iname} < {index.stop} }}")
        return iname, (pym.Variable(iname),), (pym.Variable(iname),)

    @handle_index.register
    def _(self, index: IntIndex):
        return None, (), (index.value,)

    @handle_index.register
    def _(self, index: Map):
        raise NotImplementedError

    @handle_index.register
    def _(self, index: LoopIndex):
        try:
            iname = self.inames[index]
        except KeyError:
            iname = self.inames.setdefault(index, self.iname_generator())
            self.domains.add(f"{{ [{iname}]: {index.domain.start} <= {iname} < {index.domain.stop} }}")
        return iname, (), (pym.Variable(iname),)


class ReadConverter(TensorToLoopyConverter):
    instruction_prefix = "read"
    counter = 0

    def resolve(self, tensor_indices, temp_indices):
        assignee = pym.Subscript(self.temporary, temp_indices)
        expression = pym.Subscript(self.tensor, tensor_indices)
        return assignee, expression


class ZeroConverter(TensorToLoopyConverter):
    instruction_prefix = "zero"
    counter = 0

    def resolve(self, _, temp_indices):
        assignee = pym.Subscript(self.temporary, temp_indices)
        expression = 0
        return assignee, expression


class WriteConverter(TensorToLoopyConverter):
    instruction_prefix = "write"
    counter = 0

    def resolve(self, tensor_indices, temp_indices):
        assignee = pym.Subscript(self.tensor, tensor_indices)
        expression = pym.Subscript(self.temporary, temp_indices)
        return assignee, expression


class IncrementConverter(TensorToLoopyConverter):
    instruction_prefix = "inc"
    counter = 0

    def resolve(self, tensor_indices, temp_indices):
        assignee = pym.Subscript(self.temporary, temp_indices)
        expression = pym.Sum((assignee, pym.Subscript(self.tensor, tensor_indices)))
        return assignee, expression

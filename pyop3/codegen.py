import collections
import dataclasses
import numbers
import enum
import functools
import itertools

import loopy as lp
import loopy.symbolic
import numpy as np
import pymbolic.primitives as pym

import pyop3.exprs
import pyop3.utils
from pyop3.tensors import Tensor, Index, Map
from pyop3.tensors import Slice, Space

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
    return _LoopyKernelBuilder(expr).build()


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

        self.domains = []
        self.instructions = []
        self.arguments_dict = {}
        self.parameter_dict = {}
        self.parameters_dict = {}
        self.subkernels = []
        self.temporaries = []
        self.domain_parameters = {}
        self.maps_dict = {}
        self.function_temporaries = collections.defaultdict(dict)
        self.name_generator = pyop3.utils.UniqueNameGenerator()

    @property
    def arguments(self):
        return tuple(self.arguments_dict.values())

    @property
    def parameters(self):
        return tuple(self.parameter_dict.values())

    @property
    def maps(self):
        return tuple(self.maps_dict.values())

    @property
    def kernel_data(self):
        return self.arguments + self.maps + tuple(self.temporaries) + self.parameters

    def build(self):
        self._fill_loopy_context(self.expr, within_inames={})

        translation_unit = lp.make_kernel(
            frozenset(str(dom) for dom in self.domains),
            self.instructions,
            self.kernel_data,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )

        return lp.merge((translation_unit, *self.subkernels))

    def register_temporary(self, **kwargs):
        name = self.name_generator.generate("t")
        self.temporaries.append(temporary := lp.TemporaryVariable(name, **kwargs))
        return temporary

    def generate_iname(self):
        return self.name_generator.generate("i")

    @functools.singledispatchmethod
    def _fill_loopy_context(self, expr, **kwargs):
        raise NotImplementedError

    @_fill_loopy_context.register
    def _(self, expr: pyop3.exprs.Loop, *, within_inames, **kwargs):
        new_within_inames = {}
        for index in expr.indices:
            new_within_inames[index.space] = self.register_domain(index.space, within_inames|new_within_inames)

        for statement in expr.statements:
            self._fill_loopy_context(
                statement,
                within_inames=within_inames|new_within_inames,
                **kwargs,
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

    @_fill_loopy_context.register
    def _(self, expr: pyop3.exprs.FunctionCall, within_inames, **kwargs):
        for argument in expr.arguments:
            if argument.tensor.order:
                self.register_broadcast_domains(argument, within_inames)
                shape = self.as_shape(argument.tensor.spaces)
            else:
                shape = ()

            temporary = self.register_temporary(dtype=argument.dtype, shape=shape)
            self.function_temporaries[expr][argument] = temporary

            if argument in self.arguments_dict:
                # TODO I currently assume that we only ever register arguments once.
                # this is not true if we repeat arguments to a kernel or if we have multiple kernels
                raise NotImplementedError

            self.arguments_dict[argument] = lp.GlobalArg(
                argument.name,
                dtype=argument.dtype,
                shape=self.as_orig_shape(argument.tensor.orig_shape)
            )

            for dim in argument.tensor.dims:
                self.register_maps(dim)

        gathers = self.register_gathers(expr, within_inames)
        call_insn = self.register_function_call(
            expr, within_inames, depends_on=frozenset(insn.id for insn in gathers)
        )
        _ = self.register_scatters(
            expr, within_inames, depends_on=frozenset({call_insn.id})
        )

    def register_gathers(self, call, within_domains, *, depends_on=frozenset()):
        gathers = []
        for argument in call.arguments:
            temporary = self.function_temporaries[call][argument]
            if argument.access == pyop3.exprs.READ:
                gathers.append(self.read_tensor(argument, temporary, within_domains))
            elif argument.access == pyop3.exprs.WRITE:
                gathers.append(self.zero_tensor(argument, temporary, within_domains))
            elif argument.access == pyop3.exprs.INC:
                gathers.append(self.zero_tensor(argument, temporary, within_domains))
            elif argument.access == pyop3.exprs.RW:
                gathers.append(self.read_tensor(argument, temporary, within_domains))
            else:
                raise NotImplementedError
        return tuple(gathers)

    def register_function_call(self, call, within_inames, *, depends_on=frozenset()):
        reads, writes = self.get_function_rw(call, within_inames)
        assignees = tuple(writes)
        expression = pym.Call(
            pym.Variable(call.function.loopy_kernel.default_entrypoint.name),
            tuple(reads),
        )
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=self.name_generator.generate("func"),
            within_inames=frozenset(within_inames.values()),
            depends_on=depends_on,
        )
        self.instructions.append(call_insn)
        self.subkernels.append(call.function.loopy_kernel)
        return call_insn

    def register_scatters(self, call, within_domains, *, depends_on=frozenset()):
        scatters = []
        for argument in call.arguments:
            temporary = self.function_temporaries[call][argument]
            if argument.access == pyop3.exprs.READ:
                continue
            elif argument.access == pyop3.exprs.WRITE:
                scatters.append(
                    self.write_tensor(argument, temporary, within_domains, depends_on)
                )
            elif argument.access == pyop3.exprs.INC:
                scatters.append(
                    self.inc_tensor(argument, temporary, within_domains, depends_on)
                )
            elif argument.access == pyop3.exprs.RW:
                scatters.append(
                    self.write_tensor(argument, temporary, within_domains, depends_on)
                )
            else:
                raise NotImplementedError
        return tuple(scatters)

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

    def register_assignment(
        self,
        assignee,
        expression,
        within_inames,
        depends_on=frozenset(),
        prefix="assign",
    ):
        instruction = lp.Assignment(
            assignee,
            expression,
            id=self.name_generator.generate(prefix),
            within_inames=frozenset(within_inames.values()),
            depends_on=depends_on,
        )
        self.instructions.append(instruction)
        return instruction

    def read_tensor(self, argument, temporary, within_inames, depends_on=frozenset()):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = dat[i, j, k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_inames = self.register_broadcast_domains(argument, within_inames)

        assignee = ExpressionGenerator.new_temporary(argument.tensor, temporary.name, within_inames | broadcast_inames)
        expression = ExpressionGenerator.indexed_tensor(argument.tensor, within_inames|broadcast_inames)

        return self.register_assignment(
            assignee, expression, within_inames|broadcast_inames, depends_on, "read"
        )

    def zero_tensor(self, argument, temporary, within_inames, depends_on=frozenset()):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = 0
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_inames = self.register_broadcast_domains(argument, within_inames)

        assignee = ExpressionGenerator.new_temporary(argument.tensor, temporary.name, within_inames | broadcast_inames)

        expression = 0
        return self.register_assignment(
            assignee, expression, within_inames|broadcast_inames, depends_on=depends_on, prefix="zero"
        )

    def write_tensor(self, argument, temporary, within_inames, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_inames = self.register_broadcast_domains(argument, within_inames)

        expression = ExpressionGenerator.new_temporary(argument.tensor, temporary.name, within_inames | broadcast_inames)
        assignee = ExpressionGenerator.indexed_tensor(argument.tensor, within_inames|broadcast_inames)

        return self.register_assignment(
            assignee, expression, within_inames|broadcast_inames, depends_on, "write"
        )

    def inc_tensor(self, argument, temporary, within_inames, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = dat[i, j, k, l] + t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_inames = self.register_broadcast_domains(argument, within_inames)
        assignee = ExpressionGenerator.indexed_tensor(argument.tensor, within_inames|broadcast_inames)

        temp_expression = ExpressionGenerator.new_temporary(argument.tensor, temporary.name, within_inames | broadcast_inames)
        expression = pym.Sum((assignee, temp_expression))

        return self.register_assignment(
            assignee, expression, within_inames|broadcast_inames, depends_on, "inc"
        )

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


class ExpressionGenerator:

    @staticmethod
    def new_temporary(tensor, name, inames):
        if tensor.broadcast_domains:
            return pym.Subscript(
                pym.Variable(name),
                tuple(pym.Variable(inames[dim]) for dim in tensor.broadcast_domains),
            )
        else:
            return pym.Variable(name)

    @classmethod
    def indexed_tensor(cls, tensor, inames):
        if tensor.dims:
            return pym.Subscript(
                pym.Variable(tensor.name),
                tuple(
                    cls.stack_subscripts(dim, inames)
                    for dim in tensor.dims
                ),
            )
        else:
            return pym.Variable(tensor.name)

    @classmethod
    def stack_subscripts(cls, dim, inames):
        """Convert an index tensor expression into a pymbolic expression"""
        if isinstance(dim, Space):
            if isinstance(dim, Map):
                iname = pym.Variable(inames[dim.to_space])
                subscripts = cls.stack_subscripts(dim.from_space, inames)
                return pym.Subscript(pym.Variable(dim.name), (subscripts, iname))
            else:
                iname = pym.Variable(inames[dim])
                return iname
        else:
            assert isinstance(dim, Index)
            return pym.Variable(inames[dim.space])

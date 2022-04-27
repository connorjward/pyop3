import collections
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
from pyop3.tensors import Tensor
from pyop3.tensors import Range

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


class _LoopyKernelBuilder:
    def __init__(self, expr):
        self.expr = expr

        self.domains_dict = {}
        self.instructions = []
        self.arguments_dict = {}
        self.parameter_dict = {}
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
    def domains(self):
        return frozenset(self.domains_dict.values())

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
        self._fill_loopy_context(self.expr, within_domains={})

        translation_unit = lp.make_kernel(
            self.domains,
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
    def _(self, expr: pyop3.exprs.Loop, *, within_domains, **kwargs):
        for index in expr.indices:
            iname = self.generate_iname()
            self.register_domain(iname, index.domain, within_domains)
            within_domains[index.domain] = iname

        for statement in expr.statements:
            self._fill_loopy_context(
                statement,
                within_domains,
                **kwargs,
            )

    def as_shape(self, domains, within_domains, *, first_is_none=False):
        shape = []
        for i, domain in enumerate(domains):
            if first_is_none and i == 0:
                shape.append(None)
            else:
                start, stop = self.domain_parameters[domain]
                # TODO This will fail for certain types of inputs
                shape.append(f"{stop}-{start}")
        return tuple(shape)

    @_fill_loopy_context.register
    def _(self, expr: pyop3.exprs.FunctionCall, within_domains, **kwargs):
        for argument in expr.arguments:
            if broadcast_domains := self.register_broadcast_domains(
                argument, within_domains
            ):
                shape = self.as_shape(broadcast_domains, within_domains)
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
                shape=self.as_shape(argument.tensor.broadcast_domains, within_domains, first_is_none=True)
            )

            for index in argument.indices:
                self.register_maps(index)

        gathers = self.register_gathers(expr, within_domains)
        call_insn = self.register_function_call(
            expr, within_domains, depends_on=frozenset(insn.id for insn in gathers)
        )
        _ = self.register_scatters(
            expr, within_domains, depends_on=frozenset({call_insn.id})
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

    def register_function_call(self, call, within_domains, *, depends_on=frozenset()):
        reads, writes = self.get_function_rw(call, within_domains)
        assignees = tuple(writes)
        expression = pym.Call(
            pym.Variable(call.function.loopy_kernel.default_entrypoint.name),
            tuple(reads),
        )
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=self.name_generator.generate("func"),
            within_inames=frozenset(within_domains.values()),
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
        broadcast_domains = self.register_broadcast_domains(argument, within_domains)
        indices = tuple(pym.Variable(iname) for iname in broadcast_domains.values())
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
            within_inames=within_inames,
            depends_on=depends_on,
        )
        self.instructions.append(instruction)
        return instruction

    def read_tensor(self, argument, temporary, within_domains, depends_on=frozenset()):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = dat[i, j, k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_domains = self.register_broadcast_domains(argument, within_domains)

        if broadcast_domains:
            assignee = pym.Subscript(
                pym.Variable(temporary.name),
                tuple(pym.Variable(iname) for iname in broadcast_domains.values()),
            )
        else:
            assignee = pym.Variable(temporary.name)

        if argument.indices or argument.tensor.broadcast_domains:
            expression = pym.Subscript(
                pym.Variable(argument.name),
                tuple(
                    self.stack_subscripts(index, within_domains | broadcast_domains)
                    for index in argument.tensor.indices + tuple(dom.to_range() for dom in argument.tensor.shape)
                ),
            )
        else:
            expression = pym.Variable(argument.name)

        within_inames = frozenset(
            {*within_domains.values(), *broadcast_domains.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on, "read"
        )

    def zero_tensor(self, argument, temporary, within_domains, depends_on=frozenset()):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = 0
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_domains = self.register_broadcast_domains(argument, within_domains)

        if broadcast_domains:
            assignee = pym.Subscript(
                pym.Variable(temporary.name),
                tuple(pym.Variable(iname) for iname in broadcast_domains.values()),
            )
        else:
            assignee = pym.Variable(temporary.name)

        expression = 0
        within_inames = frozenset(
            {*within_domains.values(), *broadcast_domains.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on=depends_on, prefix="zero"
        )

    def write_tensor(self, argument, temporary, within_domains, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_domains = self.register_broadcast_domains(argument, within_domains)

        if broadcast_domains:
            expression = pym.Subscript(
                pym.Variable(temporary.name),
                tuple(pym.Variable(iname) for iname in broadcast_domains.values()),
            )
        else:
            expression = pym.Variable(temporary.name)

        assignee = pym.Subscript(
            pym.Variable(argument.name),
            tuple(
                self.stack_subscripts(index, within_domains | broadcast_domains)
                for index in argument.tensor.indices + tuple(dom.to_range() for dom in argument.tensor.shape)
            ),
        )

        within_inames = frozenset(
            {*within_domains.values(), *broadcast_domains.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on, "write"
        )

    def inc_tensor(self, argument, temporary, within_domains, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = dat[i, j, k, l] + t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_domains = self.register_broadcast_domains(argument, within_domains)

        assignee = pym.Subscript(
            pym.Variable(argument.name),
            tuple(
                self.stack_subscripts(index, within_domains | broadcast_domains)
                for index in argument.tensor.indices + tuple(dom.to_range() for dom in argument.tensor.shape)
            ),
        )

        if broadcast_domains:
            expression = pym.Sum(
                (
                    assignee,
                    pym.Subscript(
                        pym.Variable(temporary.name),
                        tuple(
                            pym.Variable(iname) for iname in broadcast_domains.values()
                        ),
                    ),
                )
            )
        else:
            expression = pym.Sum((assignee, pym.Variable(temporary.name)))

        within_inames = frozenset(
            {*within_domains.values(), *broadcast_domains.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on, "inc"
        )

    def stack_subscripts(self, index, domain_map):
        """Convert an index tensor expression into a pymbolic expression"""
        iname_var = pym.Variable(domain_map[index.domain])
        is_map = index.is_vector and len(index.indices) == 1
        if is_map:
            subscripts = tuple(self.stack_subscripts(idx, domain_map) for idx in index.indices)
            return pym.Subscript(pym.Variable(index.name), (*subscripts, iname_var))
        else:
            return iname_var

    def register_maps(self, index):
        # TODO This seems like quite a hacky way to tell if index is a map
        # FIXME This whole bit needs a rethink
        is_map = index.is_vector and len(index.indices) == 1
        if is_map and index not in self.maps_dict:
            self.maps_dict[index] = lp.GlobalArg(
                index.name, dtype=np.int32, shape=(None, len(index.domain))
            )
            self.register_maps(index.indices[0])

    def register_parameter(self, parameter, **kwargs):
        return self.parameter_dict.setdefault(parameter, lp.GlobalArg(**kwargs))

    def register_domain_parameter(self, parameter, within_domains):
        """Return a pymbolic expression"""
        if isinstance(parameter, Tensor):
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
                    tuple(pym.Variable(within_domains[index.domain]) for index in parameter.indices)
                )
                self.register_assignment(
                    assignee, expression, frozenset(within_domains.values())
                )
                return temporary.name
            else:
                return parameter.name
        elif isinstance(parameter, str):
            self.register_parameter(parameter, name=parameter, dtype=np.int32, shape=())
            return parameter
        elif isinstance(parameter, numbers.Integral):
            return parameter
        else:
            raise ValueError

    def register_domain(self, iname: str, domain, within_domains):
        if domain in self.domains_dict:
            return

        start, stop = (self.register_domain_parameter(param, within_domains)
                       for param in [domain.start, domain.stop])
        self.domain_parameters[domain] = start, stop
        isl_domain = f"{{ [{iname}]: {start} <= {iname} < {stop} }}"
        assert isl_domain not in self.domains_dict.values()
        self.domains_dict[domain] = isl_domain

    def register_broadcast_domains(self, argument, within_domains):
        broadcast_domains = {}
        for domain in filter(lambda d: d not in within_domains, argument.tensor.all_domains):
            iname = self.generate_iname()
            self.register_domain(iname, domain, within_domains)
            broadcast_domains[domain] = iname
        return broadcast_domains

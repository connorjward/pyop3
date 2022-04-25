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
from pyop3.domains import Domain, Index, SparseDomain

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

        self.domains = set()
        self.instructions = []
        self.arguments_dict = {}
        self.parameters = []
        self.subkernels = []
        self.maps_dict = {}
        self.temporaries_dict = {}
        self.name_generator = pyop3.utils.UniqueNameGenerator()

    @property
    def arguments(self):
        return tuple(self.arguments_dict.values())

    @property
    def maps(self):
        return tuple(self.maps_dict.values())

    @property
    def temporaries(self):
        return tuple(self.temporaries_dict.values())

    @property
    def kernel_data(self):
        return self.arguments + self.maps + self.temporaries + tuple(self.parameters)

    def build(self):
        self._fill_loopy_context(self.expr, within_indices={})

        translation_unit = lp.make_kernel(
            self.domains,
            self.instructions,
            self.kernel_data,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )

        return lp.merge((translation_unit, *self.subkernels))

    def register_temporary(self, argument, **kwargs):
        if argument in self.temporaries_dict:
            raise AssertionError

        name = self.name_generator.generate("t")
        return self.temporaries_dict.setdefault(
            argument, lp.TemporaryVariable(name, **kwargs)
        )

    def generate_iname(self):
        return self.name_generator.generate("i")

    @functools.singledispatchmethod
    def _fill_loopy_context(self, expr, **kwargs):
        raise NotImplementedError

    @_fill_loopy_context.register
    def _(self, expr: pyop3.exprs.Loop, *, within_indices, **kwargs):
        for index in expr.indices:
            iname = self.generate_iname()
            self.register_domain(iname, index.domain, within_indices)
            within_indices[index] = iname

        for statement in expr.statements:
            self._fill_loopy_context(
                statement,
                within_indices,
                **kwargs,
            )

    @_fill_loopy_context.register
    def _(self, expr: pyop3.exprs.FunctionCall, within_indices, **kwargs):
        for argument in expr.arguments:
            # FIXME This currently fails because we don't register parameter shapes very nicely
            # try again in the morning...
            if broadcast_indices := self.register_broadcast_indices(
                argument, within_indices
            ):
                shape = tuple(index.domain.extent for index in broadcast_indices)
            else:
                shape = ()
            self.register_temporary(argument, dtype=argument.dtype, shape=shape)

            if argument in self.arguments_dict:
                # TODO I currently assume that we only ever register arguments once.
                # this is not true if we repeat arguments to a kernel or if we have multiple kernels
                raise NotImplementedError

            self.arguments_dict[argument] = lp.GlobalArg(
                argument.name,
                dtype=argument.dtype,
                shape=argument.tensor.orig_shape,
            )

            for index in itertools.chain(argument.indices, argument.tensor.broadcast_indices):
                self.register_maps(index)

        gathers = self.register_gathers(expr, within_indices)
        call_insn = self.register_function_call(
            expr, within_indices, depends_on=frozenset(insn.id for insn in gathers)
        )
        _ = self.register_scatters(
            expr, within_indices, depends_on=frozenset({call_insn.id})
        )

    def register_gathers(self, call, within_indices, *, depends_on=frozenset()):
        gathers = []
        for argument in call.arguments:
            temporary = self.temporaries_dict[argument]
            if argument.access == pyop3.exprs.READ:
                gathers.append(self.read_tensor(argument, temporary, within_indices))
            elif argument.access == pyop3.exprs.WRITE:
                gathers.append(self.zero_tensor(argument, temporary, within_indices))
            elif argument.access == pyop3.exprs.INC:
                gathers.append(self.zero_tensor(argument, temporary, within_indices))
            elif argument.access == pyop3.exprs.RW:
                gathers.append(self.read_tensor(argument, temporary, within_indices))
            else:
                raise NotImplementedError
        return tuple(gathers)

    def register_function_call(self, call, within_indices, *, depends_on=frozenset()):
        reads, writes = self.get_function_rw(call, within_indices)
        assignees = tuple(writes)
        expression = pym.Call(
            pym.Variable(call.function.loopy_kernel.default_entrypoint.name),
            tuple(reads),
        )
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=self.name_generator.generate("func"),
            within_inames=frozenset(within_indices.values()),
            depends_on=depends_on,
        )
        self.instructions.append(call_insn)
        self.subkernels.append(call.function.loopy_kernel)
        return call_insn

    def register_scatters(self, call, within_indices, *, depends_on=frozenset()):
        scatters = []
        for argument in call.arguments:
            temporary = self.temporaries_dict[argument]
            if argument.access == pyop3.exprs.READ:
                continue
            elif argument.access == pyop3.exprs.WRITE:
                scatters.append(
                    self.write_tensor(argument, temporary, within_indices, depends_on)
                )
            elif argument.access == pyop3.exprs.INC:
                scatters.append(
                    self.inc_tensor(argument, temporary, within_indices, depends_on)
                )
            elif argument.access == pyop3.exprs.RW:
                scatters.append(
                    self.write_tensor(argument, temporary, within_indices, depends_on)
                )
            else:
                raise NotImplementedError
        return tuple(scatters)

    def get_function_rw(self, call, within_indices):
        reads = []
        writes = []
        for argument in call.arguments:
            temporary = self.temporaries_dict[argument]
            access = argument.access

            ref = self.as_subarrayref(argument, temporary, within_indices)
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

    def as_subarrayref(self, argument, temporary, within_indices):
        """Register an argument to a function."""
        broadcast_indices = self.register_broadcast_indices(argument, within_indices)
        indices = tuple(pym.Variable(iname) for iname in broadcast_indices.values())
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

    def read_tensor(self, argument, temporary, within_indices, depends_on=frozenset()):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = dat[i, j, k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_indices = self.register_broadcast_indices(argument, within_indices)

        if broadcast_indices:
            assignee = pym.Subscript(
                pym.Variable(temporary.name),
                tuple(pym.Variable(iname) for iname in broadcast_indices.values()),
            )
        else:
            assignee = pym.Variable(temporary.name)

        if argument.indices or argument.tensor.broadcast_indices:
            expression = pym.Subscript(
                pym.Variable(argument.name),
                tuple(
                    self.stack_subscripts(index, within_indices | broadcast_indices)
                    for index in itertools.chain(
                        argument.tensor.indices, argument.tensor.broadcast_indices
                    )
                ),
            )
        else:
            expression = pym.Variable(argument.name)

        within_inames = frozenset(
            {*within_indices.values(), *broadcast_indices.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on, "read"
        )

    def zero_tensor(self, argument, temporary, within_indices, depends_on=frozenset()):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = 0
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_indices = self.register_broadcast_indices(argument, within_indices)

        if broadcast_indices:
            assignee = pym.Subscript(
                pym.Variable(temporary.name),
                tuple(pym.Variable(iname) for iname in broadcast_indices.values()),
            )
        else:
            assignee = pym.Variable(temporary.name)

        expression = 0
        within_inames = frozenset(
            {*within_indices.values(), *broadcast_indices.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on=depends_on, prefix="zero"
        )

    def write_tensor(self, argument, temporary, within_indices, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_indices = self.register_broadcast_indices(argument, within_indices)

        if broadcast_indices:
            expression = pym.Subscript(
                pym.Variable(temporary.name),
                tuple(pym.Variable(iname) for iname in broadcast_indices.values()),
            )
        else:
            expression = pym.Variable(temporary.name)

        assignee = pym.Subscript(
            pym.Variable(argument.name),
            tuple(
                self.stack_subscripts(index, within_indices | broadcast_indices)
                for index in itertools.chain(
                    argument.tensor.indices, argument.tensor.broadcast_indices
                )
            ),
        )

        within_inames = frozenset(
            {*within_indices.values(), *broadcast_indices.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on, "write"
        )

    def inc_tensor(self, argument, temporary, within_indices, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = dat[i, j, k, l] + t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        broadcast_indices = self.register_broadcast_indices(argument, within_indices)

        assignee = pym.Subscript(
            pym.Variable(argument.name),
            tuple(
                self.stack_subscripts(index, within_indices | broadcast_indices)
                for index in itertools.chain(
                    argument.tensor.indices, argument.tensor.broadcast_indices
                )
            ),
        )

        if broadcast_indices:
            expression = pym.Sum(
                (
                    assignee,
                    pym.Subscript(
                        pym.Variable(temporary.name),
                        tuple(
                            pym.Variable(iname) for iname in broadcast_indices.values()
                        ),
                    ),
                )
            )
        else:
            expression = pym.Sum((assignee, pym.Variable(temporary.name)))

        within_inames = frozenset(
            {*within_indices.values(), *broadcast_indices.values()}
        )
        return self.register_assignment(
            assignee, expression, within_inames, depends_on, "inc"
        )

    def stack_subscripts(self, index, index_map):
        iname = index_map[index]

        if isinstance(index.domain, SparseDomain):
            from_index = self.stack_subscripts(index.domain.parent, index_map)
            to_index = pym.Variable(iname)
            # equivalent to map0[map1[i, j], k]
            return pym.Subscript(pym.Variable(index.domain.name), (from_index, to_index))
        else:
            return pym.Variable(iname)

    def register_maps(self, index: Index):
        domain = index.domain
        if isinstance(domain, SparseDomain):
            if domain not in self.maps_dict:
                self.maps_dict[domain] = lp.GlobalArg(
                    domain.name, dtype=np.int32, shape=(None, domain.extent)
                )

            if domain.parent:
                self.register_maps(domain.parent)

    def register_domain(self, iname: str, domain: Domain, within_indices):
        domain_stop = domain.stop
        domain_step = domain.step
        for i, slice_param in enumerate([domain.start, domain.stop, domain.step]):
            # register tensor-like parameters
            if isinstance(slice_param, Tensor):
                # must be a scalar
                assert not slice_param.domains
                if not hasattr(self, "dodgy_counter"):
                    self.dodgy_counter = 0
                new_temp_name = f"mynewtemp{self.dodgy_counter}"
                self.dodgy_counter += 1

                self.parameters.append(
                        lp.GlobalArg(
                    slice_param.name,
                    dtype=np.int32,
                    shape=slice_param.orig_shape,
                ))
                self.temporaries_dict[slice_param] = lp.TemporaryVariable(new_temp_name)

                tensor = slice_param
                # write to temporary
                assignee = pym.Variable(new_temp_name)
                expression = pym.Subscript(pym.Variable(tensor.name), tuple(pym.Variable(within_indices[index]) for index in tensor.indices))
                self.register_assignment(
                    assignee, expression, frozenset(within_indices.values()), prefix="myhack"
                )

                # string output
                if tensor.indices:
                    if i == 1:
                        domain_stop = new_temp_name
                    elif i == 2:
                        domain_step = new_temp_name
                else:
                    if i == 1:
                        domain_stop = f"{tensor.name}"
                    elif i == 2:
                        domain_step = f"{tensor.name}"
            elif isinstance(slice_param, str):
                self.parameters.append(
                        lp.GlobalArg(
                    slice_param,
                    dtype=np.int32,
                    shape=(),
                ))

            else:
                import numbers
                if not isinstance(slice_param, numbers.Integral):
                    raise NotImplementedError

        isl_domain = f"[{iname}]: {domain.start} <= {iname} < {domain_stop}"
        if domain.step != 1:
            tmp_iname = iname + "_tmp"
            isl_domain += f" and (exists {tmp_iname}: {iname} = {domain_step}*{tmp_iname})"
        isl_domain = "{" + isl_domain + "}"

        if isl_domain in self.domains:
            raise ValueError
        self.domains.add(isl_domain)

    def register_broadcast_indices(self, argument, within_indices):
        broadcast_indices = {}
        for index in argument.tensor.broadcast_indices:
            iname = self.generate_iname()
            self.register_domain(iname, index.domain, within_indices)
            broadcast_indices[index] = iname
        return broadcast_indices

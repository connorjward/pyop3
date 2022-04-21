import dataclasses
import enum
import functools
from typing import Dict, List

import dtlutils
import loopy as lp
import loopy.symbolic
import pymbolic.primitives as pym

import pyop3.exprs


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


class UniqueNamer:
    def __init__(self):
        self.name_generators = {}

    def next(self, prefix="", suffix=""):
        try:
            namer = self.name_generators[(prefix, suffix)]
        except KeyError:
            namer = self.name_generators.setdefault(
                (prefix, suffix), dtlutils.NameGenerator(prefix, suffix)
            )
        return next(namer)


class _LoopyKernelBuilder:
    def __init__(self, expr):
        self.expr = expr

        self.domains = []
        self.instructions = []
        self.arguments_dict = {}
        self.subkernels = []
        self.maps = {}
        self.temporaries_dict = {}
        self.namer = UniqueNamer()

    @property
    def arguments(self):
        return tuple(self.arguments_dict.values())

    @property
    def temporaries(self):
        return tuple(self.temporaries_dict.values())

    @property
    def kernel_data(self):
        return self.arguments + tuple(self.maps.values()) + self.temporaries

    def build(self):
        self._fill_loopy_context(self.expr, within_inames={})

        translation_unit = lp.make_kernel(
            self.domains,
            self.instructions,
            self.kernel_data,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )

        return lp.merge((translation_unit,) + tuple(self.subkernels))

    def register_temporary(self, argument, **kwargs):
        if argument in self.temporaries_dict:
            raise AssertionError

        name = self.namer.next("t")
        return self.temporaries_dict.setdefault(
            argument, lp.TemporaryVariable(name, **kwargs)
        )

    @functools.singledispatchmethod
    def _fill_loopy_context(self, expr, **kwargs):
        raise NotImplementedError

    @_fill_loopy_context.register
    def _(self, expr: pyop3.exprs.Loop, *, within_inames, **kwargs):
        self.domains.append(expr.index.domain.loopy_domain)

        for statement in expr.statements:
            self._fill_loopy_context(
                statement,
                within_inames=within_inames | {expr.index: self.namer.next("i")},
                **kwargs,
            )

    @_fill_loopy_context.register
    def _(self, expr: pyop3.exprs.FunctionCall, *, within_inames, **kwargs):
        # register a temporary for each argument
        for arg, spec in zip(expr.arguments, expr.argspec):
            # TODO broadcast_indices should be a tuple so the shape is ordered/consistent
            if arg.broadcast:
                tshape = (arg.index.domain.extent,)
            else:
                tshape = ()
            self.register_temporary(arg, dtype=spec.dtype, shape=tshape)

            if arg in self.arguments_dict:
                # TODO I currently assume that we only ever register arguments once.
                # this is not true if we repeat arguments to a kernel or if we have multiple kernels
                raise NotImplementedError
            self.arguments_dict[arg] = lp.GlobalArg(arg.name, dtype=spec.dtype)

        gathers = self.register_gathers(expr, within_inames)
        call_insn = self.register_function_call(
            expr, within_inames, depends_on=frozenset(insn.id for insn in gathers)
        )
        _ = self.register_scatters(
            expr, within_inames, depends_on=frozenset({call_insn.id})
        )

    def register_gathers(self, call, within_inames, *, depends_on=frozenset()):
        gathers = []
        for arg, spec in zip(call.arguments, call.argspec):
            temporary = self.temporaries_dict[arg]
            if spec.access == pyop3.exprs.READ:
                gathers.append(self.read_tensor(arg, temporary, within_inames))
            elif spec.access == pyop3.exprs.WRITE:
                continue
            elif spec.access == pyop3.exprs.INC:
                gathers.append(self.zero_tensor(arg, temporary, within_inames))
            elif spec.access == pyop3.exprs.RW:
                gathers.append(self.read_tensor(arg, temporary, within_inames))
            else:
                raise NotImplementedError
        return tuple(gathers)

    def register_function_call(self, call, within_inames, *, depends_on=frozenset()):
        reads, writes = self.get_function_rw(call)
        assignees = tuple(writes)
        expression = pym.Call(
            pym.Variable(call.function.loopy_kernel.default_entrypoint.name),
            tuple(reads),
        )
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=self.namer.next("func"),
            within_inames=frozenset(within_inames.values()),
            depends_on=depends_on,
        )
        self.instructions.append(call_insn)
        self.subkernels.append(call.function.loopy_kernel)
        return call_insn

    def register_scatters(self, call, within_inames, *, depends_on=frozenset()):
        scatters = []
        for arg, spec in zip(call.arguments, call.argspec):
            temporary = self.temporaries_dict[arg]
            if spec.access == pyop3.exprs.READ:
                continue
            elif spec.access == pyop3.exprs.WRITE:
                scatters.append(
                    self.write_tensor(arg, temporary, within_inames, depends_on)
                )
            elif spec.access == pyop3.exprs.INC:
                scatters.append(
                    self.inc_tensor(arg, temporary, within_inames, depends_on)
                )
            elif spec.access == pyop3.exprs.RW:
                scatters.append(
                    self.write_tensor(arg, temporary, within_inames, depends_on)
                )
            else:
                raise NotImplementedError
        return tuple(scatters)

    def get_function_rw(self, call):
        reads = []
        writes = []
        for tensor, spec in zip(call.arguments, call.argspec):
            temporary = self.temporaries_dict[tensor]
            access = spec.access

            ref = self.as_subarrayref(tensor, temporary)
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

    def as_subarrayref(self, tensor, temporary):
        """Register an argument to a function."""
        # the following is necessary to get shapes happy between the caller and callee
        # it is forbidden to reuse inames so we declare new ones as well as new domains
        if tensor.broadcast:
            (myindex,) = tensor.indices
            iname = self.namer.next("i")
            indexvar = pym.Variable(iname)
            temp_var = pym.Subscript(pym.Variable(temporary.name), indexvar)
            myarg = lp.symbolic.SubArrayRef((indexvar,), temp_var)
            self.domains.append(
                f"{{ [{iname}]: 0 <= {iname} < {myindex.domain.extent} }}"
            )
        else:
            temp_var = pym.Variable(temporary.name)
            myarg = temp_var
        return myarg

    def read_tensor(self, tensor, temporary, within_inames):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = dat[i, j, k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        (index,) = tensor.indices
        within_inames_set = frozenset(within_inames.values())

        if tensor.broadcast:
            iname = self.namer.next("i")
            within_inames_set |= {iname}
            within_inames = within_inames | {index: iname}
            self.domains.append(f"{{ [{iname}]: 0<= {iname} < {index.domain.extent} }}")

        if tensor.broadcast:
            assignee = pym.Subscript(pym.Variable(temporary.name), pym.Variable(iname))
        else:
            assignee = pym.Variable(temporary.name)

        expression = pym.Subscript(
            pym.Variable(tensor.name), self.stack_subscripts(index, within_inames)
        )

        instruction = lp.Assignment(
            assignee,
            expression,
            id=self.namer.next("read"),
            within_inames=within_inames_set,
        )
        self.instructions.append(instruction)

        self.register_maps(index)
        return instruction

    def zero_tensor(self, tensor, temporary, within_inames):
        # so we want something like:
        # for i:
        #   for j:
        #     t[k, l] = 0
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        (index,) = tensor.indices
        within_inames_set = frozenset(within_inames.values())

        if tensor.broadcast:
            iname = self.namer.next("i")
            within_inames_set |= {iname}
            within_inames = within_inames | {index: iname}
            self.domains.append(f"{{ [{iname}]: 0<= {iname} < {index.domain.extent} }}")

        if tensor.broadcast:
            assignee = pym.Subscript(pym.Variable(temporary.name), pym.Variable(iname))
        else:
            assignee = pym.Variable(temporary.name)

        expression = 0

        instruction = lp.Assignment(
            assignee,
            expression,
            id=self.namer.next("zero"),
            within_inames=within_inames_set,
        )
        self.instructions.append(instruction)

        self.register_maps(index)
        return instruction

    def write_tensor(self, tensor, temporary, within_inames, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        (index,) = tensor.indices
        within_inames_set = frozenset(within_inames.values())

        if tensor.broadcast:
            iname = self.namer.next("i")
            within_inames_set |= {iname}
            within_inames = within_inames | {index: iname}
            self.domains.append(f"{{ [{iname}]: 0<= {iname} < {index.domain.extent} }}")

        if tensor.broadcast:
            expression = pym.Subscript(
                pym.Variable(temporary.name), pym.Variable(iname)
            )
        else:
            expression = pym.Variable(temporary.name)

        assignee = pym.Subscript(
            pym.Variable(tensor.name), self.stack_subscripts(index, within_inames)
        )

        instruction = lp.Assignment(
            assignee,
            expression,
            id=self.namer.next("write"),
            within_inames=within_inames_set,
            depends_on=depends_on,
        )
        self.instructions.append(instruction)

        self.register_maps(index)
        return instruction

    def inc_tensor(self, tensor, temporary, within_inames, depends_on):
        # so we want something like:
        # for i:
        #   for j:
        #     dat[i, j, k, l] = dat[i, j, k, l] + t[k, l]
        # in this case within_indices is i and j
        # we also need to register two new domains to loop over: k and l
        (index,) = tensor.indices
        within_inames_set = frozenset(within_inames.values())

        if tensor.broadcast:
            iname = self.namer.next("i")
            within_inames_set |= {iname}
            within_inames = within_inames | {index: iname}
            self.domains.append(f"{{ [{iname}]: 0<= {iname} < {index.domain.extent} }}")

        assignee = pym.Subscript(
            pym.Variable(tensor.name), self.stack_subscripts(index, within_inames)
        )

        if tensor.broadcast:
            expression = pym.Sum(
                (
                    assignee,
                    pym.Subscript(pym.Variable(temporary.name), pym.Variable(iname)),
                )
            )
        else:
            expression = pym.Sum((assignee, pym.Variable(temporary.name)))

        instruction = lp.Assignment(
            assignee,
            expression,
            id=self.namer.next("inc"),
            within_inames=within_inames_set,
            within_inames_is_final=True,
            depends_on=depends_on,
        )
        self.instructions.append(instruction)

        self.register_maps(index)
        return instruction

    def stack_subscripts(self, index, within_inames):
        try:
            iname = within_inames[index]
        except KeyError:
            iname = self.namer.next("i")

        if index.domain.parent_index:
            indices = self.stack_subscripts(
                index.domain.parent_index, within_inames
            ), pym.Variable(iname)
            return pym.Subscript(pym.Variable(index.domain.map.name), indices)
        else:
            return pym.Variable(iname)

    def register_maps(self, index):
        if (map_ := index.domain.map) and map_.name not in self.maps:
            self.maps[map_.name] = lp.GlobalArg(
                map_.name, dtype=map_.dtype, shape=(None, map_.arity)
            )

        if index.domain.parent_index:
            self.register_maps(index.domain.parent_index)

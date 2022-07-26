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
from pyop3.utils import CustomTuple, checked_zip, NameGenerator, rzip
from pyop3.tensors import Tensor, Index, Map, Dim, NonAffineMap, _compute_indexed_shape, _compute_indexed_shape2
from pyop3.tensors import Slice, AffineMap, index
from pyop3.codegen.tlang import to_tlang


class VariableCollector(pym.mapper.Collector):
    def map_variable(self, expr, *args, **kwargs):
        return {expr}


class VariableReplacer(pym.mapper.IdentityMapper):
    def __init__(self, handler, *args, **kwargs):
        self.handler = handler
        super().__init__(*args, **kwargs)

    def map_variable(self, expr):
        return self.handler(expr)


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
        self.domains = {}
        self.instructions = []
        self.kernel_data = []
        self.subkernels = []
        # self._within_inames = {}
        self.extents = {}
        self.assumptions = []

    def build(self, tlang_expr):
        self._namer.reset()
        self._build(tlang_expr, {})

        # breakpoint()
        domains = [self._make_domain(iname, start, stop, step) for iname, (start, stop, step) in self.domains.items()]
        translation_unit = lp.make_kernel(
            utils.unique(domains),
            utils.unique(self.instructions),
            utils.unique(self.kernel_data),
            assumptions=",".join(self.assumptions),
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
            name="mykernel",
        )
        tu = lp.merge((translation_unit, *self.subkernels))
        # import pdb; pdb.set_trace()
        return tu.with_entrypoints("mykernel")

    @functools.singledispatchmethod
    def _build(self, expr, within_loops):
        raise TypeError

    @_build.register
    def _(self, expr: exprs.Loop, within_loops):
        def collect_within_loops(idx):
            if isinstance(idx, NonAffineMap):
                within = {}
                for i in idx.tensor.indices:
                    within |= collect_within_loops(i)
                return within
            else:
                iname = self._namer.next("i")
                return {idx: iname}

        within_loops = {}
        for idxs in expr.index:
            for idx in idxs:
                within_loops |= collect_within_loops(idx)

        # import pdb; pdb.set_trace()

        for stmt in expr.statements:
            self._build(stmt, within_loops.copy())

    @_build.register
    def _(self, insn: tlang.Instruction, within_loops):
        self._make_instruction_context(insn, within_loops)


    @functools.singledispatchmethod
    def _make_instruction_context(self, instruction: tlang.Instruction, within_inames, **kwargs):
        raise TypeError


    @_make_instruction_context.register
    def _(self, call: tlang.FunctionCall, within_loops, **kwargs):
        subarrayrefs = {}
        extents = []
        # import pdb; pdb.set_trace()
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            # determine the right size of temporary - since loopy thinks everything
            # is flat just find the total size.
            temp_size = 1
            for shape in temp.shapes:
                for extent in shape:
                    if isinstance(extent, Tensor):
                        if (var := self.extents[extent.name]) not in extents:
                            extents.append(var)
                            self.assumptions.append(f"{var} <= {extent.max_value}")
                        extent = extent.max_value
                    temp_size *= extent

            # if a dimension is ragged then take the maximum size of it (and only
            # allocate once)
            temp_isize = 1
            for shape in temp.indexed_shapes:
                for extent in shape:
                    if isinstance(extent, Tensor):
                        extent = extent.max_value
                    temp_isize *= extent

            # assert temp.size == temp.indexed_size
            assert temp_size == temp_isize

            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            # import pdb; pdb.set_trace()
            self.domains[iname] = (0, temp_size, 1)
            self.kernel_data.append(lp.TemporaryVariable(temp.name, shape=(temp_size,)))

        assignees = tuple(subarrayrefs[var] for var in call.writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(subarrayrefs[var] for var in call.reads) + tuple(extents),
        )

        within_inames = frozenset(within_loops.values())
        depends_on = frozenset({f"{insn}*" for insn in call.depends_on})
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=call.id,
            within_inames=frozenset(within_inames),
            depends_on=depends_on
        )

        self.instructions.append(call_insn)
        self.subkernels.append(call.function.code)


    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, within_loops, scalar=False, domain_stack=None):
        within_loops = within_loops.copy()
        # import pdb; pdb.set_trace()
        for lidxs, ridxs in zip(assignment.lhs.indicess, assignment.rhs.indicess):
            # 1. Create a domain stack if not provided - this means that we can have consistent inames for the LHS and RHS
            if not domain_stack:
                domain_stack = self.create_domain_stack(assignment.lhs, assignment.rhs, lidxs, ridxs)

            # 2. generate LHS and RHS expressions (registering domains etc as you go)
            # The idea is that we are given a consistent stack of domains to share that are consumed
            # as the indices are traversed. We can then build a map between indices and inames which
            # we finally process to generate an expression.
            ldstack = domain_stack.copy()
            lwithin_loops = self.register_domains(assignment.lhs, lidxs, ldstack, within_loops)
            lexpr = self.handle_assignment(assignment.lhs, lidxs, lwithin_loops)

            rdstack = domain_stack.copy()
            rwithin_loops = self.register_domains(assignment.rhs, ridxs, rdstack, within_loops)
            rexpr = self.handle_assignment(assignment.rhs, ridxs, rwithin_loops)

            assert not ldstack and not rdstack
            domain_stack = ldstack

            lname = pym.var(assignment.lhs.name)
            if assignment.lhs.dim or not scalar:
                lhs = pym.subscript(lname, lexpr)
            else:
                lhs = lname

            if isinstance(assignment, tlang.Zero):
                rhs = 0
            else:
                rname = pym.var(assignment.rhs.name)
                if assignment.rhs.dim or not scalar:
                    rhs = pym.subscript(rname, rexpr)
                else:
                    rhs = rname

            # there are no ordering restrictions between assignments to the
            # same temporary - but this is only valid to declare if multiple insns are used
            if len(assignment.lhs.indicess) > 1:
                assert len(assignment.rhs.indicess) > 1
                no_sync_with = frozenset({(f"{assignment.id}*", "any")})
            else:
                no_sync_with = frozenset()

            within_inames = frozenset(list(within_loops.values()) + [iname for iname in domain_stack])
            # import pdb; pdb.set_trace()
            assign_insn = lp.Assignment(
                lhs, rhs,
                id=self._namer.next(f"{assignment.id}_"),
                within_inames=within_inames,
                depends_on=frozenset({f"{dep}*" for dep in assignment.depends_on}),
                no_sync_with=no_sync_with,
            )
            self.instructions.append(assign_insn)

        # register kernel arguments
        # TODO should really use assignment.{lhs,rhs} here...
        # TODO this is sorta repeated in FunctionCall handler.
        if assignment.temporary.dim:
            assert not scalar
            size = 1
            for shape in assignment.temporary.shapes:
                for extent in shape:
                    if isinstance(extent, Tensor):
                        extent = extent.max_value
                    size *= extent
            # temp_shape = (assignment.temporary.size,)
            temp_shape = (size,)  # must be 1D for loopy to be OK with ragged things
        elif not scalar:
            temp_shape = (1,)
        else:
            temp_shape = ()

        self.kernel_data += [
            lp.GlobalArg(assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None),

            lp.TemporaryVariable(assignment.temporary.name, shape=temp_shape)
        ]

        return domain_stack

    def register_new_domain(self, iname, index, within_loops):
        if isinstance(index.size, pym.primitives.Expression):
            if not isinstance(index.size, Tensor):
                raise NotImplementedError("need to think hard about more complicated expressions"
                                          "esp. sharing inames")
            size = self.register_extent(index.size, within_loops)
        else:
            size = index.size
        if iname in self.domains:
            assert self.domains[iname] == (0, size, 1)
        else:
            self.domains[iname] = (0, size, 1)

        return iname

    def create_domain_stack(self, lhs, rhs, lidxs, ridxs):
        """Create a consistent set of inames for lhs and rhs to use.

        We ignore any 'within' indices here as these will already exist and do
        not contribute to the shape.
        """
        # import pdb; pdb.set_trace()
        shapes = {_compute_indexed_shape2(lidxs), _compute_indexed_shape2(ridxs)}
        try:
            shape, = shapes
        except ValueError:
            raise ValueError("Shapes do not match")
        inames = [self._namer.next("i") for _ in shape]
        return inames

    def register_extent(self, extent, within_loops):
        if isinstance(extent, Tensor):
            # If we have a ragged thing then we need to create a scalar temporary
            # to hold its value.
            try:
                # TODO here we assume that we only index an extent tensor once. This
                # is a simplification.
                return self.extents[extent.name]
            except KeyError:
                # import pdb; pdb.set_trace()
                temp_name = self._namer.next("n")
                temp = Tensor.new(name=temp_name, dtype=np.int32)

                # make sure that the RHS reduces down to a scalar
                # TODO this should really be handled when extent.indices are set up
                # (in Tensor._parse_indices).
                # new_stencils = index(extent.indices)
                # extent = extent.copy(indices=new_stencils)
                insn = tlang.Read(extent, temp)
                self._make_instruction_context(insn, within_loops, scalar=True)

                return self.extents.setdefault(extent.name, pym.var(temp_name))
        else:
            return extent

    def handle_assignment(self, tensor, indices, within_loops):
        index_expr = 0
        if not tensor.dim:
            return index_expr
        dim = tensor.dim
        for i, index in enumerate(indices):
            assert dim is not None

            dim_expr = self._as_expr(index, within_loops)

            subdim_id = dim.labels.index(index.label)

            # import pdb; pdb.set_trace()
            # Every dim uses a section to map the dim index (from the slice/map + iname)
            # onto a location in the data structure. For nice regular data this can just be
            # the index multiplied by the size of the inner dims (e.g. dat[4*i + j]), but for
            # ragged things we need to always have a map for the outer dims.
            section_name = self._namer.next("sec")
            index_expr += pym.subscript(pym.var(section_name), dim_expr + dim.offsets[subdim_id])
            self.kernel_data.append(lp.GlobalArg(section_name, shape=None, dtype=np.int32))

            if subdims := tensor.dim.subdims:
                dim = subdims[subdim_id]

        return index_expr

    def register_domains(self, tensor, indices, dstack, within_loops):
        within_loops = within_loops.copy()

        # dim = tensor.dim.root

        for i, index in enumerate(indices):
            if isinstance(index, Slice):
                if index not in within_loops:
                    iname = dstack.pop(0)
                else:
                    iname = within_loops.pop(index)
                self.register_new_domain(iname, index, within_loops)
                within_loops[index] = iname

            elif isinstance(index, NonAffineMap):
                within_loops |= self.register_domains(index.tensor, index.tensor.indices, dstack, within_loops)
                self.kernel_data.append(lp.GlobalArg(index.tensor.name, shape=None, dtype=index.tensor.dtype))
            else:
                raise TypeError

        return within_loops

    @staticmethod
    def _make_domain(iname, start, stop, step):
        if start is None:
            start = 0
        if step is None:
            step = 1

        if step != 1:
            raise NotImplementedError

        assert all(param is not None for param in [start, stop, step])

        return f"{{ [{iname}]: {start} <= {iname} < {stop} }}"

    @functools.singledispatchmethod
    def _as_expr(self, index, *args):
        raise TypeError

    @_as_expr.register
    def _(self, index: Slice, within_loops):
        start = index.start or 0
        step = index.step or 1

        iname = within_loops[index]
        return pym.var(iname)*step + start

    @_as_expr.register
    def _(self, index: NonAffineMap, within_loops):
        myexpr = self.handle_assignment(index.tensor, index.tensor.indices, within_loops)
        return pym.subscript(pym.var(index.tensor.name), myexpr)


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
        name = pym.var(self.instruction.temporary.name)
        if self.instruction.temporary.dim:
            return pym.subscript(name, self.local_idxs + self.local_offset)
        else:
            return name



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

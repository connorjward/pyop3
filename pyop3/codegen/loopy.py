import abc
import collections
import copy
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
from pyop3.tensors import MultiArray, Index, ScalarAxis, Map, MultiAxis, NonAffineMap, _compute_indexed_shape, _compute_indexed_shape2
from pyop3.tensors import Slice, IndexFunction, index
from pyop3.codegen.tlang import to_tlang


class VariableCollector(pym.mapper.Collector):
    def map_variable(self, expr, *args, **kwargs):
        return {expr}


def merge_bins(bin1, bin2):
    new_bin = bin1.copy()
    for k, v in bin2.items():
        if k in bin1:
            new_bin[k].extend(v)
        else:
            new_bin[k] = v
    return new_bin


def index_tensor_with_within_loops(tensor, within_loops):
    within_loops = copy.deepcopy(within_loops)

    new_indicess = [[]]


    # within_loops should now be empty
    assert all(len(v) == 0 for v in within_loops.values())
    return tensor.copy(indicess=new_indicess)


def collect_labels(index):
    # import pdb; pdb.set_trace()
    if isinstance(index, NonAffineMap):
        return [label for idx in index.input_indices for label in collect_labels(idx)] + [index.label]
    else:
        return [index.label]


def compute_needed_size(index):
    if isinstance(index, NonAffineMap):
        return sum(compute_needed_size(idx) for idx in index.input_indices) + 1
    elif isinstance(index, IndexFunction):
        return len(index.vars)
    else:
        return 1


def truncate_within_loops(within_loops, indices):
    """Truncate within loops s.t. only the last n entries are included
    where n is the number of times a certain dim label appears in the indices.
    """
    # TODO I think it might be better to go through in reverse order somehow
    # that would require using reversed(indices)

    ninames_needed = sum(compute_needed_size(idx) for idx in indices)
    return within_loops[-ninames_needed:].copy()


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
        self._tensor_data = []
        self._section_data = []
        self._temp_kernel_data = []
        self.subkernels = []
        # self._within_inames = {}
        self.extents = {}
        self.assumptions = []

    @property
    def kernel_data(self):
        return self._tensor_data + self._section_data + self._temp_kernel_data

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

    def collect_within_loops(self, idx, within_loops):
        within_loops = copy.deepcopy(within_loops)

        if isinstance(idx, NonAffineMap):
            for i in idx.input_indices:
                within_loops = self.collect_within_loops(i, within_loops)

        iname = self._namer.next("i")
        within_loops.append(iname)
        self.register_new_domain(iname, idx, within_loops)
        return within_loops

    @_build.register
    def _(self, expr: exprs.Loop, within_loops):

        within_loops = []
        for idx in expr.index:
            within_loops = self.collect_within_loops(idx, within_loops)

        for stmt in expr.statements:
            self._build(stmt, copy.deepcopy(within_loops))

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
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            # determine the right size of temporary - since loopy thinks everything
            # is flat just find the total size.
            temp_size = 0
            for shape in temp.shapes:
                temp_size_ = 1
                for extent in shape:
                    if isinstance(extent, MultiArray):
                        if (var := self.extents[extent.name]) not in extents:
                            extents.append(var)
                            self.assumptions.append(f"{var} <= {extent.max_value}")
                        extent = extent.max_value
                    temp_size_ *= extent
                temp_size += temp_size_

            # if a dimension is ragged then take the maximum size of it (and only
            # allocate once)
            temp_isize = 0
            for shape in temp.indexed_shapes:
                temp_isize_ = 1
                for extent in shape:
                    if isinstance(extent, MultiArray):
                        extent = extent.max_value
                    temp_isize_ *= extent
                temp_isize += temp_isize_

            # assert temp.size == temp.indexed_size
            assert temp_size == temp_isize

            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            self.domains[iname] = (0, temp_size, 1)
            self._temp_kernel_data.append(lp.TemporaryVariable(temp.name, shape=(temp_size,)))

        assignees = tuple(subarrayrefs[var] for var in call.writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(subarrayrefs[var] for var in call.reads) + tuple(extents),
        )

        within_inames = frozenset(within_loops)
        depends_on = frozenset({f"{insn}*" for insn in call.depends_on})
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=call.id,
            within_inames=frozenset(within_inames),
            within_inames_is_final=True,
            depends_on=depends_on
        )

        self.instructions.append(call_insn)
        self.subkernels.append(call.function.code)


    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, within_loops, scalar=False, domain_stack=None):
        within_loops = copy.deepcopy(within_loops)
        for lidxs, ridxs in zip(assignment.lhs.indicess, assignment.rhs.indicess):
            # 1. Create a domain stack if not provided - this means that we can have consistent inames for the LHS and RHS
            if not domain_stack:
                domain_stack = self.create_domain_stack(assignment.lhs, assignment.rhs, lidxs, ridxs)

            # 2. generate LHS and RHS expressions (registering domains etc as you go)
            # The idea is that we are given a consistent stack of domains to share that are consumed
            # as the indices are traversed. We can then build a map between indices and inames which
            # we finally process to generate an expression.
            # import pdb; pdb.set_trace()
            ldstack = domain_stack.copy()
            lwithin_loops = self.register_domains(lidxs, ldstack, within_loops)
            lwithin_loops = truncate_within_loops(lwithin_loops, lidxs)
            lexpr = self.handle_assignment(assignment.lhs, lidxs, lwithin_loops)

            rdstack = domain_stack.copy()
            rwithin_loops = self.register_domains(ridxs, rdstack, within_loops)
            rwithin_loops = truncate_within_loops(rwithin_loops, ridxs)
            rexpr = self.handle_assignment(assignment.rhs, ridxs, rwithin_loops)

            assert not ldstack and not rdstack
            domain_stack = ldstack

            lname = pym.var(assignment.lhs.name)
            if assignment.lhs.shapes != ((),) or not scalar:
                lhs = pym.subscript(lname, lexpr)
            else:
                lhs = lname

            if isinstance(assignment, tlang.Zero):
                rhs = 0
            else:
                rname = pym.var(assignment.rhs.name)
                if assignment.rhs.shapes != ((),) or not scalar:
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

            within_inames = frozenset(within_loops) | set(domain_stack)
            assign_insn = lp.Assignment(
                lhs, rhs,
                id=self._namer.next(f"{assignment.id}_"),
                within_inames=within_inames,
                within_inames_is_final=True,
                depends_on=frozenset({f"{dep}*" for dep in assignment.depends_on}),
                no_sync_with=no_sync_with,
            )
            self.instructions.append(assign_insn)

        # register kernel arguments
        # TODO should really use assignment.{lhs,rhs} here...
        # TODO this is sorta repeated in FunctionCall handler.
        if assignment.temporary.shapes != ((),):
            assert not scalar
            size = 0
            for shape in assignment.temporary.shapes:
                size_ = 1
                for extent in shape:
                    if isinstance(extent, MultiArray):
                        extent = extent.max_value
                    size_ *= extent
                size += size_
            # temp_shape = (assignment.temporary.size,)
            temp_shape = (size,)  # must be 1D for loopy to be OK with ragged things
        elif not scalar:
            temp_shape = (1,)
        else:
            temp_shape = ()

        self._tensor_data.append(
            lp.GlobalArg(assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None),
        )
        self._temp_kernel_data.append(
            lp.TemporaryVariable(assignment.temporary.name, shape=temp_shape)
        )

        return domain_stack

    def register_new_domain(self, iname, index, within_loops):
        if isinstance(index.size, pym.primitives.Expression):
            if not isinstance(index.size, MultiArray):
                raise NotImplementedError("need to think hard about more complicated expressions"
                                          "esp. sharing inames")
            # remove the final iname matching index from within_loops as the index.size will
            # not see this/be outside of it
            exwithin_loops = copy.deepcopy(within_loops)
            exwithin_loops.pop()
            size = self.register_extent(index.size, exwithin_loops)
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
        shapes = {_compute_indexed_shape2(lidxs), _compute_indexed_shape2(ridxs)}
        try:
            shape, = shapes
        except ValueError:
            raise ValueError("Shapes do not match")
        inames = [self._namer.next("i") for _ in shape]
        return inames

    def register_extent(self, extent, within_loops):
        if isinstance(extent, MultiArray):
            # If we have a ragged thing then we need to create a scalar temporary
            # to hold its value.
            try:
                # TODO here we assume that we only index an extent tensor once. This
                # is a simplification.
                return self.extents[extent.name]
            except KeyError:
                temp_name = self._namer.next("n")
                temp = MultiArray.new(MultiAxis(ScalarAxis()), name=temp_name, dtype=np.int32)

                # make sure that the RHS reduces down to a scalar
                new_extent = extent.copy(indicess=(index(extent.indices),))

                insn = tlang.Read(new_extent, temp)
                self._make_instruction_context(insn, within_loops, scalar=True)

                return self.extents.setdefault(extent.name, pym.var(temp_name))
        else:
            return extent

    # I don't like needing the tensor here..  maybe I could attach the offset to the index?
    # using from_dim
    def handle_assignment(self, tensor, indices, within_loops):
        # do not copy - modify!
        saved_within_loops = copy.deepcopy(within_loops)

        index_expr = 0
        dim = tensor.dim

        for idx in indices:
            assert dim is not None

            dim_expr = self._as_expr(idx, within_loops)

            npart = idx.npart

            # Every dim uses a section to map the dim index (from the slice/map + iname)
            # onto a location in the data structure. For nice regular data this can just be
            # the index multiplied by the size of the inner dims (e.g. dat[4*i + j]), but for
            # ragged things we need to always have a map for the outer dims.
            part = dim.get_part(npart)
            layout = part.layout

            if isinstance(layout, IndexFunction):
                from_var, = layout.vars
                index_expr += pym.substitute(layout.expr, {from_var: dim_expr})
            elif isinstance(layout, MultiArray):
                myexpr = self.handle_assignment(layout, layout.indices, copy.deepcopy(saved_within_loops))
                index_expr += pym.subscript(pym.var(layout.name), myexpr)
                self._section_data.append(lp.GlobalArg(layout.name, shape=None, dtype=np.int32))
            else:
                raise TypeError

            if part.subdim:
                dim = part.subdim

        return index_expr

    def register_domains(self, indices, dstack, within_loops):
        within_loops = copy.deepcopy(within_loops)

        # I think ultimately all I need to do here is stick the things from dstack
        # onto the right dim labels in within_loops, and register all of the domains

        # first we stick any new inames into the right places
        for idx in indices:
            if isinstance(idx, NonAffineMap):
                within_loops = self.register_domains(idx.input_indices, dstack, within_loops)
                self._tensor_data.append(lp.GlobalArg(idx.tensor.name, shape=None, dtype=idx.tensor.dtype))

            if idx.is_loop_index:
                # should do this earlier (hard to do now due to ordering confusion)
                continue
            else:
                iname = dstack.pop(0)
                within_loops.append(iname)
                self.register_new_domain(iname, idx, within_loops)

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

        # TODO I would prefer to go from the end here. Maybe use a deque?
        iname = within_loops.pop(0)
        return pym.var(iname)*step + start

    @_as_expr.register
    def _(self, index: IndexFunction, within_loops):
        # use the innermost matching dims as the right inames
        varmap = {}
        for var in reversed(index.vars):
            iname = within_loops.pop(0)
            varmap[var] = pym.var(iname)

        res = pym.substitute(index.expr, varmap)
        return res

    @_as_expr.register
    def _(self, index: NonAffineMap, within_loops):
        # import pdb; pdb.set_trace()
        myexpr = self.handle_assignment(index.tensor, index.tensor.indices, within_loops)
        # iname = within_loops[index.label].pop(0)
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

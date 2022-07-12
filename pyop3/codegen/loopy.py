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
from pyop3.tensors import Tensor, Index, Map, Dim, UniformDim, MixedDim, NonAffineMap
from pyop3.tensors import Slice, indexed_shape, indexed_size_per_index_group, AffineMap, Stencil, StencilGroup, index_size
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


class LoopyKernelBuilder:
    def __init__(self):
        self._namer = MultiNameGenerator()
        self.domains = {}
        self.instructions = []
        self.kernel_data = []
        self.subkernels = []
        # self._within_inames = {}

    def build(self, tlang_expr):
        self._namer.reset()
        self._build(tlang_expr, {})

        # breakpoint()
        domains = [self._make_domain(iname, start, stop, step) for iname, (start, stop, step) in self.domains.items()]
        translation_unit = lp.make_kernel(
            utils.unique(domains),
            utils.unique(self.instructions),
            utils.unique(self.kernel_data),
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
        # be assertive
        stencil, = expr.index
        indices, = stencil

        def collect_within_loops(idx):
            if isinstance(idx, NonAffineMap):
                within_loops = {}
                for _, i in idx.tensor.stencils[0][0]:
                    within_loops |= collect_within_loops(i)
                return within_loops
            else:
                iname = self._namer.next("i")
                return {idx: iname}

        within_loops = {}
        for _, idx in indices:
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
        def mysize(t):
            stencil, = t.stencils
            indices, = stencil
            s = 1
            dim = t.dim.root
            for subdim_id, index in indices:
                start = index.start or 0
                stop = index.stop or dim.sizes[subdim_id]
                step = index.step or 1
                size = (stop - start) // step
                s *= size
                dim = t.dim.get_child(dim)
            return s

        # breakpoint()
        subarrayrefs = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            self.domains[iname] = (0, mysize(temp), 1)

        reads = tuple(subarrayrefs[var] for var in call.reads)
        writes = tuple(subarrayrefs[var] for var in call.writes)
        assignees = tuple(writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(reads),
        )

        # within_inames = frozenset([self._within_inames[iname] for iname in within_loops.values()])
        within_inames = frozenset(within_loops.values())

        kernel_data = [lp.TemporaryVariable(temp.name, shape=(mysize(temp),)) for temp in subarrayrefs]

        depends_on = frozenset({f"{insn}*" for insn in call.depends_on})
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=call.id,
            within_inames=frozenset(within_inames),
            depends_on=depends_on
        )

        self.instructions.append(call_insn)
        self.kernel_data.extend(kernel_data)
        self.subkernels.append(call.function.code)


    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, within_loops, scalar=False, domain_stack=None):
        within_loops = within_loops.copy()
        # within_inames = within_inames.copy()
        # breakpoint()
        assert len(assignment.lhs.stencils) == len(assignment.rhs.stencils)
        for lstencil, rstencil in zip(assignment.lhs.stencils, assignment.rhs.stencils):
            assert len(lstencil) == len(rstencil)
            for lidxs, ridxs in zip(lstencil, rstencil):
                # 1. Create a domain stack if not provided - this means that we can have consistent inames for the LHS and RHS
                if not domain_stack:
                    domain_stack = self.create_domain_stack(assignment.lhs, assignment.rhs, lidxs, ridxs, within_loops)

                # import pdb; pdb.set_trace()

                lexpr, ldstack = self.handle_assignment(assignment.lhs, lidxs, domain_stack, within_loops)
                rexpr, rdstack = self.handle_assignment(assignment.rhs, ridxs, domain_stack, within_loops)

                assert ldstack == rdstack
                domain_stack = ldstack

                lname = pym.var(assignment.lhs.name)
                if assignment.lhs.dim.root or not scalar:
                    lhs = pym.subscript(lname, lexpr)
                else:
                    lhs = lname

                if isinstance(assignment, tlang.Zero):
                    rhs = 0
                else:
                    rname = pym.var(assignment.rhs.name)
                    if assignment.rhs.dim.root or not scalar:
                        rhs = pym.subscript(rname, rexpr)
                    else:
                        rhs = rname

                within_inames = frozenset(list(within_loops.values()) + [iname for iname in domain_stack])
                # import pdb; pdb.set_trace()
                assign_insn = lp.Assignment(
                        lhs, rhs,
                        id=self._namer.next(f"{assignment.id}_"),
                        within_inames=within_inames,
                        depends_on=frozenset({f"{dep}*" for dep in assignment.depends_on}))
                self.instructions.append(assign_insn)

        # register kernel arguments
        # import pdb; pdb.set_trace()
        if assignment.temporary.dim.root or not scalar:
            stencil, = assignment.temporary.stencils
            indices, = stencil
            temp_size = 1
            dim = assignment.temporary.dim.root
            for subdim_id, index in indices:
                start = index.start or 0
                stop = index.stop or dim.sizes[subdim_id]
                step = index.step or 1
                size = (stop - start) // step
                temp_size *= size
                dim = assignment.temporary.dim.get_child(dim)
            temp_shape = (temp_size,)
        else:
            temp_shape =()

        self.kernel_data += [
            lp.GlobalArg(assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None),

            lp.TemporaryVariable(assignment.temporary.name, shape=temp_shape)
        ]

        return domain_stack

    def create_domain_stack(self, lhs, rhs, lidxs, ridxs, within_loops):
        # import pdb; pdb.set_trace()
        inames = self.create_domain_stack_inner(lhs, rhs, lidxs, ridxs)
        return inames

    def register_new_within_domain(self, dim, subdim_id, index, within_loops):
        iname = within_loops[index]

        # and try to register/check
        start = self.register_extent(index.start or 0, within_loops)
        dimsize = dim.sizes[subdim_id] if dim else None
        stop = self.register_extent(index.stop or dimsize, within_loops)
        step = self.register_extent(index.step or 1, within_loops)

        # import pdb; pdb.set_trace()
        if iname in self.domains:
            assert self.domains[iname] == (0, (stop - start) // step, 1)
        else:
            self.domains[iname] = (0, (stop - start) // step, 1)

        return iname

    def create_domain_stack_inner(self, lhs, rhs, lidxs, ridxs):
        shapes = {lhs.indexed_shape_per_indices(lidxs), rhs.indexed_shape_per_indices(ridxs)}
        try:
            shape, = shapes
        except ValueError:
            raise ValueError("Shapes do not match")
        inames = [self._namer.next("i") for _ in shape]
        for iname, extent in zip(inames, shape):
            assert iname not in self.domains
            self.domains[iname] = (0, extent, 1)
        return inames

    def register_extent(self, extent, within_loops):
        within_loops = within_loops.copy()
        within_loops.pop(list(within_loops.keys())[-1])
        if not hasattr(self, "_extents"):
            self._extents = {}
        if isinstance(extent, Tensor):
            try:
                return self._extents[extent]
            except KeyError:
                temp_name = self._namer.next("p")
                temp = Tensor(pyop3.utils.Tree(None), name=temp_name, dtype=np.int32)["fill"]

                indices = tuple((0, idx) for idx in list(within_loops.keys())[-extent.order:])
                stencils = StencilGroup([Stencil([indices])])
                insn = tlang.Read(extent[stencils], temp)
                self._make_instruction_context(insn, within_loops, scalar=True)

                return self._extents.setdefault(extent, pym.var(temp_name))
        else:
            return extent

    def handle_assignment(self, tensor, indices, domain_stack, within_loops):
        # import pdb; pdb.set_trace()
        dstack = domain_stack.copy()

        # if not tensor.dim:
            # assert not indices and not inames
            # return 0
        within_loops = within_loops.copy()
        index_expr = 0
        current_dim = tensor.dim.root
        for i, (subdim_id, index) in enumerate(indices):
            # import pdb; pdb.set_trace()
            assert current_dim is not None

            if isinstance(index, Slice):
                start = index.start or 0
                stop = index.stop or current_dim.sizes[subdim_id]
                step = index.step or 1

                if index in within_loops:
                    iname = self.register_new_within_domain(current_dim, subdim_id, index, within_loops)
                else:
                    iname = dstack.pop(0)
                    within_loops[index] = iname

                dim_expr = pym.var(iname)*step + start
            elif isinstance(index, NonAffineMap):
                myexpr, dstack = self.handle_assignment(index.tensor, index.tensor.stencils[0][0], domain_stack=dstack, within_loops=within_loops)
                self.kernel_data.append(lp.GlobalArg(index.tensor.name, shape=None, dtype=index.tensor.dtype))
                # import pdb; pdb.set_trace()
                # TODO I expect I need to put index -> dim_expr in within_loops so this is recorded
                dim_expr = pym.subscript(pym.var(index.tensor.name), myexpr)

                """
                Explanation for future self:

                We only want to be dealing with one side of the assignment here. Adding
                a scalar to write to suddenly makes the tensor stuff a lot more
                complicated. Instead we consume the loops we *already know about* to
                construct an appropriate expression.
                """
            else:
                raise NotImplementedError

            new_map_name = self._namer.next("sec")
            index_expr += pym.subscript(pym.var(new_map_name), dim_expr + current_dim.offsets[subdim_id])
            self.kernel_data.append(lp.GlobalArg(new_map_name, shape=None, dtype=np.int32))

            if subdims := tensor.dim.get_children(current_dim):
                current_dim = subdims[subdim_id]

        return index_expr, dstack

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

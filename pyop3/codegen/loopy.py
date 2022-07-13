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
from pyop3.tensors import Slice, AffineMap, Stencil, StencilGroup
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
        self.extents = {}

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
                for _, i in idx.tensor.indices:
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
        subarrayrefs = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            assert temp.size == temp.indexed_size
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            import pdb; pdb.set_trace()
            # self.domains[iname] = (0, temp.size, 1)

        assignees = tuple(subarrayrefs[var] for var in call.writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(subarrayrefs[var] for var in call.reads),
        )

        within_inames = frozenset(within_loops.values())

        kernel_data = [lp.TemporaryVariable(temp.name, shape=(temp.size,)) for temp in subarrayrefs]

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
        assert len(assignment.lhs.stencils) == len(assignment.rhs.stencils)
        for lstencil, rstencil in zip(assignment.lhs.stencils, assignment.rhs.stencils):
            assert len(lstencil) == len(rstencil)
            for lidxs, ridxs in zip(lstencil, rstencil):
                # 1. Create a domain stack if not provided - this means that we can have consistent inames for the LHS and RHS
                if not domain_stack:
                    domain_stack = self.create_domain_stack(assignment.lhs, assignment.rhs, lidxs, ridxs)

                # import pdb; pdb.set_trace()
                ldstack = domain_stack.copy()
                lwithin_loops = self.register_domains(assignment.lhs, lidxs, ldstack, within_loops)
                lexpr = self.handle_assignment(assignment.lhs, lidxs, lwithin_loops)

                rdstack = domain_stack.copy()
                rwithin_loops = self.register_domains(assignment.rhs, ridxs, rdstack, within_loops)
                rexpr = self.handle_assignment(assignment.rhs, ridxs, rwithin_loops)

                assert not ldstack and not rdstack
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
        if assignment.temporary.shape or not scalar:
            temp_shape = (assignment.temporary.size,)
        else:
            temp_shape = ()

        self.kernel_data += [
            lp.GlobalArg(assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None),

            lp.TemporaryVariable(assignment.temporary.name, shape=temp_shape)
        ]

        return domain_stack

    def register_new_domain(self, iname, dim, subdim_id, index, within_loops, parent_indices):
        # iname = within_loops[index]

        # and try to register/check
        start = self.register_extent(index.start or 0, within_loops, parent_indices)
        dimsize = dim.sizes[subdim_id] if dim else None
        stop = self.register_extent(index.stop or dimsize, within_loops, parent_indices)
        step = self.register_extent(index.step or 1, within_loops, parent_indices)

        # import pdb; pdb.set_trace()
        size = (stop - start) // step
        if iname in self.domains:
            assert self.domains[iname] == (0, size, 1)
        else:
            self.domains[iname] = (0, size, 1)

        return iname

    def create_domain_stack(self, lhs, rhs, lidxs, ridxs):
        shapes = {lhs.indexed_shape_per_indices(lidxs), rhs.indexed_shape_per_indices(ridxs)}
        try:
            shape, = shapes
        except ValueError:
            raise ValueError("Shapes do not match")
        inames = [self._namer.next("i") for _ in shape]
        return inames
        # for iname, extent in zip(inames, shape):
        #     assert iname not in self.domains
        #     self.domains[iname] = (0, extent, 1)
        # return inames

    def register_extent(self, extent, within_loops, parent_indices):
        # this is needed for within_inames - can it be removed?
        # doesn't seem generic wrt. inner ragged
        # within_loops = within_loops.copy()
        # within_loops.pop(list(within_loops.keys())[-1])
        # if not hasattr(self, "_extents"):
        #     self._extents = {}
        if isinstance(extent, Tensor):
            try:
                return self.extents[extent]
            except KeyError:
                temp_name = self._namer.next("n")
                temp = Tensor(pyop3.utils.Tree(None), name=temp_name, dtype=np.int32)["fill"]

                indices = parent_indices[-extent.order:]
                stencils = StencilGroup([Stencil([indices])])
                insn = tlang.Read(extent[stencils], temp)
                self._make_instruction_context(insn, within_loops, scalar=True)

                return self.extents.setdefault(extent, pym.var(temp_name))
        else:
            return extent

    def handle_assignment(self, tensor, indices, within_loops):
        # import pdb; pdb.set_trace()
        # dstack = domain_stack.copy()

        index_expr = 0
        current_dim = tensor.dim.root

        for i, (subdim_id, index) in enumerate(indices):
            # import pdb; pdb.set_trace()
            assert current_dim is not None

            dim_expr = self._as_expr(index, within_loops)

            new_map_name = self._namer.next("sec")
            index_expr += pym.subscript(pym.var(new_map_name), dim_expr + current_dim.offsets[subdim_id])
            self.kernel_data.append(lp.GlobalArg(new_map_name, shape=None, dtype=np.int32))

            if subdims := tensor.dim.get_children(current_dim):
                current_dim = subdims[subdim_id]

        return index_expr

    def register_domains(self, tensor, indices, dstack, within_loops):
        within_loops = within_loops.copy()

        dim = tensor.dim.root

        for i, (subdim_id, index) in enumerate(indices):
            if isinstance(index, Slice):
                if index not in within_loops:
                    iname = dstack.pop(0)
                else:
                    iname = within_loops.pop(index)
                self.register_new_domain(iname, dim, subdim_id, index, within_loops, indices[:i])
                within_loops[index] = iname

            elif isinstance(index, NonAffineMap):
                within_loops |= self.register_domains(index.tensor, index.tensor.indices, domain_stack=dstack, within_loops=within_loops)
                self.kernel_data.append(lp.GlobalArg(index.tensor.name, shape=None, dtype=index.tensor.dtype))
            else:
                raise TypeError

            if subdims := tensor.dim.get_children(dim):
                dim = subdims[subdim_id]

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

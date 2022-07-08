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
        import pdb; pdb.set_trace()
        return tu.with_entrypoints("mykernel")

    @functools.singledispatchmethod
    def _build(self, expr, within_loops):
        raise TypeError

    @_build.register
    def _(self, expr: exprs.Loop, within_loops):
        # be assertive
        stencil, = expr.index
        indices, = stencil

        within_loops = {}

        # TODO
        # this nasty within_inames stuff is required because within_loops only tells
        # us what iname should be used inside of the index expression. Since we have
        # maps that produce extra inames it does not know about those. Probably the
        # best way to handle this is to have a registry/map somewhere mapping
        # iname -> actual inames (so in most cases this is just identity).
        within_inames = set()

        for subdim_id, index in indices:
            iname = self._namer.next("i")
            within_loops[index] = iname

            if isinstance(index, NonAffineMap):
                indices = tuple((0, Slice()) for _ in range(index.tensor.order))
                for _, idx in indices:
                    within_loops[idx] = self._namer.next("i")
                temporary = Tensor(pyop3.utils.Tree(None), name=iname, dtype=np.int32)["fill"]
                indexed = index.tensor[StencilGroup([Stencil([indices])])]
                # import pdb; pdb.set_trace()
                mapassignment = tlang.Read(indexed, temporary)
                within_inames1 = self._make_instruction_context(mapassignment, within_loops, within_inames, scalar=True)
                within_inames |= within_inames1
            else:
                within_inames.add(iname)
            # if isinstance(index, Slice):
            #     if isinstance(index.stop, Tensor):
            #         temporary_name = self._namer.next("p")
            #         temporary = Tensor((), name=temporary_name, dtype=np.int32)
            #         indexed = self._index_the_map2(index.stop, within_loops)
            #         mapassignment = tlang.Read(indexed, temporary)
            #         self._make_instruction_context(mapassignment, within_loops, scalar=True)
            #         size = temporary_name
            #     else:
            #         size = (index.stop-index.start)//index.step
            #     iname = self._namer.next("i")
            #     within_loops[index] = iname
            #
            # elif isinstance(index, NonAffineMap):
            #     iname = self._namer.next("i")
            #     temporary = Tensor((), name=iname, dtype=np.int32)
            #     new_index = Slice(index.dim, 0, stop=index.arity)
            #     indexed = self._index_the_map(index.tensor, new_index)
            #     mapassignment = tlang.Read(indexed, temporary)
            #     self._make_instruction_context(mapassignment, within_loops)
            #     within_loops[index] = iname
            #     size = index.arity
            # else:
            #     raise AssertionError
            #
            # self.domains.append(self._make_domain(iname, 0, size, 1))

        for stmt in expr.statements:
            self._build(stmt, within_loops.copy(), within_inames)

    @_build.register
    def _(self, insn: tlang.Instruction, within_loops, within_inames):
        self._make_instruction_context(insn, within_loops, within_inames)


    @functools.singledispatchmethod
    def _make_instruction_context(self, instruction: tlang.Instruction, within_inames, **kwargs):
        raise TypeError


    @_make_instruction_context.register
    def _(self, call: tlang.FunctionCall, within_loops, within_inames, **kwargs):
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

        # FIXME This is not sufficient. We need to know that a map iname in fact contributes other 'within'
        # in other words, if we have a map contributing an iname 'i1', then we really
        # want to add the within_inames of the map assignment here...
        # within_inames = frozenset({iname for iname in within_loops.values()})
        # within_inames = frozenset([iname for iname in {*linames, *rinames} if iname in self.domains.keys()])

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
    def _(self, assignment: tlang.Assignment, within_loops, within_inames, scalar=False):
        within_loops = within_loops.copy()
        within_inames = within_inames.copy()
        # breakpoint()
        assert len(assignment.lhs.stencils) == len(assignment.rhs.stencils)
        for lstencil, rstencil in zip(assignment.lhs.stencils, assignment.rhs.stencils):
            assert len(lstencil) == len(rstencil)
            for lidxs, ridxs in zip(lstencil, rstencil):
                linames, rinames, within_inames1 = self.sync_inames(assignment.lhs.dim.root, assignment.rhs.dim.root, assignment.lhs.dim, assignment.rhs.dim, lidxs, ridxs, within_loops, within_inames)
                lexpr = self.handle_assignment(assignment.lhs, lidxs, linames, within_loops)
                rexpr = self.handle_assignment(assignment.rhs, ridxs, rinames, within_loops)

                # import pdb; pdb.set_trace()

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

                within_inames_priv = frozenset({*within_inames, *within_inames1})
                # import pdb; pdb.set_trace()
                assign_insn = lp.Assignment(
                        lhs, rhs,
                        id=self._namer.next(f"{assignment.id}_"),
                        within_inames=within_inames_priv,
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
                dim = assignment.tensor.dim.get_child(dim)
            temp_shape = (temp_size,)
        else:
            temp_shape =()

        self.kernel_data += [
            lp.GlobalArg(assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None),

            lp.TemporaryVariable(assignment.temporary.name, shape=temp_shape)
        ]

        return within_loops, {*within_inames, *within_inames1}

    def handle_assignment(self, tensor, indices, inames, within_loops):
        if not tensor.dim:
            assert not indices and not inames
            return 0
        within_loops = within_loops.copy()
        index_expr = 0
        current_dim = tensor.dim.root
        for (subdim_id, index), iname in zip(indices, inames):
            assert current_dim is not None

            if isinstance(index, Slice):
                start = index.start or 0
                step = index.step or 1
                dim_expr = pym.var(iname)*step + start
            elif isinstance(index, NonAffineMap):
                # temporary_name = self._namer.next("p")
                # temporary = Tensor(pyop3.utils.Tree(None), name=temporary_name, dtype=np.int32)["fill"]
                # # indexed = self._index_the_map2(stop, within_loops)
                # # FIXME this is wrong! - should reuse indices from parent stack
                # # import pdb; pdb.set_trace()
                # i = ridxs.index((rsubdim_id, ridx))
                # myindices = tuple((0, idx) for _, idx in ridxs[-stop.order-i:-i])
                # mystencils = StencilGroup([Stencil([myindices])])
                # indexed = stop[mystencils]
                # mapassignment = tlang.Read(indexed, temporary)
                # self._make_instruction_context(mapassignment, within_loops, scalar=True)
                # stop = pym.var(temporary_name)

                # temporary = Tensor(pyop3.utils.Tree(None), name=iname, dtype=np.int32)["fill"]
                # # indexed = self._index_the_map(index.tensor, within_loops)
                # mapassignment = tlang.Read(index.tensor["fill"], temporary)
                # self._make_instruction_context(mapassignment, within_loops, scalar=True)
                dim_expr = pym.var(iname)
            else:
                raise NotImplementedError

            # import pdb; pdb.set_trace()
            new_map_name = self._namer.next("sec")
            index_expr += pym.subscript(pym.var(new_map_name), dim_expr + current_dim.offsets[subdim_id])
            self.kernel_data.append(lp.GlobalArg(new_map_name, shape=None, dtype=np.int32))

            if subdims := tensor.dim.get_children(current_dim):
                current_dim = subdims[subdim_id]

        return index_expr

    def sync_inames(self, ldim, rdim, ldtree, rdtree, lidxs, ridxs, within_loops, within_inames):
        within_inames = within_inames.copy()
        linames = []
        rinames = []
        # import pdb; pdb.set_trace()
        for lidx, ridx in rzip(lidxs, ridxs):
            if lidx:
                lsubdim_id, lidx = lidx
                if ridx:
                    rsubdim_id, ridx = ridx
                    # assert lidx == ridx
                    assert lidx not in within_loops
                    assert ridx not in within_loops

                    iname = self._namer.next("i")

                    if isinstance(lidx, Slice):
                        start = lidx.start or 0
                        stop = lidx.stop or ldim.sizes[lsubdim_id]
                        step = lidx.step or 1
                        lsize = str((stop - start) // step)
                    elif isinstance(lidx, NonAffineMap):
                        lsize = str(ridx.arity)

                    if isinstance(ridx, Slice):
                        start = ridx.start or 0
                        stop = ridx.stop or rdim.sizes[rsubdim_id]
                        step = ridx.step or 1
                        rsize = str((stop - start) // step)
                        within_inames.add(iname)
                    elif isinstance(ridx, NonAffineMap):
                        rsize = str(ridx.arity)

                        stencil, = ridx.tensor.stencils
                        indices, = stencil
                        # new_indices = tuple((0, Slice()) for _ in range(ridx.tensor.order - len(indices)))
                        # for _, idx in indices:
                        #     if idx not in within_loops:
                        #         within_loops[idx] = self._namer.next("i")
                        temporary = Tensor(pyop3.utils.Tree(None), name=iname, dtype=np.int32)["fill"]
                        indexed = ridx.tensor[StencilGroup([Stencil([indices])])]
                        # import pdb; pdb.set_trace()
                        mapassignment = tlang.Read(indexed, temporary)
                        within_loops1, within_inames1 = self._make_instruction_context(mapassignment, within_loops, within_inames, scalar=True)
                        within_inames |= within_inames1

                    assert lsize == rsize

                    # import pdb; pdb.set_trace()
                    if all(isinstance(idx, Slice) for idx in [lidx, ridx]):
                        self.domains[iname] = (0, lsize, 1)
                    linames.append(iname)
                    rinames.append(iname)

                    if rsubdims := rdtree.get_children(rdim):
                        rdim = rsubdims[rsubdim_id]
                    else:
                        rdim = None
                else:
                    iname = within_loops[lidx]
                    if iname not in self.domains:
                        if isinstance(lidx, Slice):
                            start = lidx.start or 0
                            stop = lidx.stop or ldim.sizes[lsubdim_id]
                            step = lidx.step or 1
                            size = str((stop - start) // step)
                            self.domains[iname] = (0, size, 1)
                    linames.append(iname)

                if lsubdims := ldtree.get_children(ldim):
                    ldim = lsubdims[lsubdim_id]
                else:
                    ldim = None
            else:
                if ridx:
                    rsubdim_id, ridx = ridx
                    iname = within_loops[ridx]
                    if iname not in self.domains:
                        if isinstance(ridx, Slice):
                            start = ridx.start or 0
                            stop = ridx.stop or rdim.sizes[rsubdim_id]

                            # ragged checks
                            if isinstance(stop, Tensor):
                                temporary_name = self._namer.next("p")
                                temporary = Tensor(pyop3.utils.Tree(None), name=temporary_name, dtype=np.int32)["fill"]
                                i = ridxs.index((rsubdim_id, ridx))
                                myindices = tuple((0, idx) for _, idx in ridxs[-stop.order-i:-i])
                                mystencils = StencilGroup([Stencil([myindices])])
                                indexed = stop[mystencils]
                                mapassignment = tlang.Read(indexed, temporary)
                                self._make_instruction_context(mapassignment, within_loops, scalar=True)
                                stop = pym.var(temporary_name)

                            step = ridx.step or 1

                            size = str((stop - start) // step)
                            self.domains[iname] = (0, size, 1)
                            within_inames.add(iname)
                    rinames.append(iname)

                    if rsubdims := rdtree.get_children(rdim):
                        rdim = rsubdims[rsubdim_id]
                    else:
                        rdim = None
                else:
                    raise AssertionError
        return linames, rinames, within_inames

    def _index_the_map(self, tensor, new_index):
        """
            Traverse the tensor dim tree and create new indices/loops as required to get
            it down to a scalar.
        """
        stencil, = tensor.stencils
        indices, = stencil
        stencils = StencilGroup([Stencil([indices + (new_index,)])])
        return tensor.copy(stencils=stencils)

    def _index_the_map2(self, tensor, within_loops):
        # do a nasty backtracking algorithm to search from the bottom of the tree
        ptr = 1
        while ptr <= len(within_loops):
            try:
                indices = list(within_loops.keys())[-ptr:]
            except:
                raise ValueError

            # import pdb; pdb.set_trace()
            # now check
            is_valid = True
            dim = tensor.dim.root
            for idx in indices:
                if not dim or idx.dim != dim:
                    is_valid = False
                    break
            # also fail if there are still unindexed dimensions
            if tensor.dim.get_child(dim):
                is_valid = False

            if is_valid:
                stencils = StencilGroup([Stencil([tuple(indices)])])
                return tensor.copy(stencils=stencils)
            ptr += 1

        raise ValueError

    def get_stratum(self, dim):
        ptr = 0
        for i, size in enumerate(dim.sizes):
            if dim.offset == ptr:
                return i
            ptr += size

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

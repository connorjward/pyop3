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
from pyop3.utils import MultiNameGenerator, NameGenerator, strictly_all
from pyop3.utils import PrettyTuple, checked_zip, NameGenerator, rzip
from pyop3.tensors import MultiArray, Map, MultiAxis, NonAffineMap, _compute_indexed_shape, _compute_indexed_shape2, IndirectLayoutFunction, AxisPart
from pyop3.tensors import Slice, IndexFunction, index, MultiIndexCollection, MultiIndex, AffineLayoutFunction, TypedIndex, IndexSet
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
    # import pdb; pdb.set_trace()

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
        self.domains = []
        self.instructions = []
        self._tensor_data = {}
        self._section_data = []
        self._temp_kernel_data = []
        self.subkernels = []
        # self._within_inames = {}
        self.extents = {}
        self.assumptions = []

        self._within_typed_indices = []  # a stack
        self._within_loop_index_names = []  # a stack

        self._part_id_namer = NameGenerator("mypartid")
        self._loop_index_names = {}

        self._latest_insn = {}
        self._active_insn = None

    @property
    def kernel_data(self):
        return list(self._tensor_data.values()) + self._section_data + self._temp_kernel_data

    def build(self, tlang_expr):
        self._namer.reset()
        self._build(tlang_expr, {})

        # breakpoint()
        # domains = [self._make_domain(iname, start, stop, step) for iname, (start, stop, step) in self.domains.items()]
        translation_unit = lp.make_kernel(
            self.domains,
            self.instructions,
            self.kernel_data,
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

        # within_loops = []
        # for idx in expr.index:
        #     within_loops = self.collect_within_loops(idx, within_loops)
        # this might break with composition - need to track which indices (not multiindices) are in use.
        multi_idx_collection = expr.index
        assert isinstance(multi_idx_collection, MultiIndexCollection)

        if len(multi_idx_collection.multi_indices) > 1:
            raise NotImplementedError(
            """In this case we need to follow different codegen pathways internally. I
               expect this to cause big issues with how we identify the indices of the
               arrays themselves as dat[p] where p can be one of many things is hard to
               identify.
            """)

        # register inames (also needs to be done for packing loops)
        for multi_idx in multi_idx_collection:
            for i, typed_idx in enumerate(multi_idx):
                # do this before creating the new iname
                extent = self.register_extent(
                    typed_idx.iset.size,
                    list(self._loop_index_names.values()),
                    list(self._loop_index_names.values()),
                )

                loop_index_name = self._namer.next("i")
                self._loop_index_names[typed_idx] = loop_index_name

                domain_str = f"{{ [{loop_index_name}]: 0 <= {loop_index_name} < {extent} }}"
                self.domains.append(domain_str)

                self._within_typed_indices.append(typed_idx)
                self._within_loop_index_names.append(loop_index_name)

            # we need to build a separate set of instructions for each multi-index
            # in the collection.
            # e.g. if we are looping over interior facets of an extruded mesh - the
            # iterset is two multi-indices: edges+hfacets and verts+vfacets
            for stmt in expr.statements:
                # self._build(stmt, copy.deepcopy(within_loops))
                self._build(stmt, self._loop_index_names)

            for typed_idx in multi_idx:
                self._within_typed_indices.pop()
                self._within_loop_index_names.pop()

    def _get_within_inames(self):
        # since we want to pop we need a list of lists
        return frozenset({
            iname
            for iname in self._within_loop_index_names
        })

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
            temp_size = temp.axes.alloc_size
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            self.domains.append(f"{{ [{iname}]: 0 <= {iname} < {temp_size} }}")
            assert temp.name in [d.name for d in self._temp_kernel_data]

        assignees = tuple(subarrayrefs[var] for var in call.writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(subarrayrefs[var] for var in call.reads) + tuple(extents),
        )

        insn_id = f"{call.id}_0"
        depends_on = frozenset({self._latest_insn[id] for id in call.depends_on})

        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=insn_id,
            within_inames=self._get_within_inames(),
            within_inames_is_final=True,
            depends_on=depends_on
        )

        self.instructions.append(call_insn)
        self.subkernels.append(call.function.code)
        self._latest_insn[call.id] = insn_id

    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, outer_loop_indices):
        jnames = self.register_loops(assignment.indices, outer_loop_indices)
        within_inames = list(outer_loop_indices.values()) + [j for j in jnames if j not in outer_loop_indices.values()]
        # import pdb; pdb.set_trace()
        self._generate_assignment_insn(assignment, jnames, within_inames)

    def _generate_assignment_insn(self, assignment, jnames, within_inames, scalar=False):
        # import pdb; pdb.set_trace()
        self._active_insn = assignment

        # validate that the number of inames specified is the right number for the
        # indexed structure provided
        # e.g. self.check_inames()
        if any(isinstance(idx, Map) for idx in assignment.indices):
            raise NotImplementedError("""
not expected.
we can only hit this situation with something like map composition where the inames are already
declared. we just need to traverse properly to check that we have the right number of inames.
            """)

        # we don't always need all the indices (e.g. if we permute the inner dimension)
        # just use the first n outer ones
        depth = len(assignment.indices.typed_indices)
        jnames = jnames[-depth:]

        # import pdb; pdb.set_trace()
        # depends_on = depends_on | {f"{dep}*" for dep in assignment.depends_on}

        # 2. Generate map code
        lpnames, ljnames = self.generate_index_insns(assignment.indices, jnames, assignment.lhs.name, assignment, within_inames)

        # reset here to avoid loop clashes - these don't need to sync
        self._latest_insn.pop(self._active_insn.id)

        rpnames, rjnames = self.generate_index_insns(assignment.indices, jnames, assignment.rhs.name, assignment, within_inames)

        # 3. Generate layout code
        loffset = self.make_offset_expr(
            assignment.lhs, lpnames, ljnames, f"{assignment.id}_loffset", within_inames,
        )
        roffset = self.make_offset_expr(
            assignment.rhs, rpnames, rjnames, f"{assignment.id}_roffset", within_inames,
        )

        # import pdb; pdb.set_trace()
        # 4. Emit assignment instruction
        self.generate_assignment_insn(assignment, assignment.lhs, loffset,
                                      assignment.rhs, roffset,
                                      scalar=scalar, within_inames=within_inames)

        # Register data
        # TODO should only do once at a higher point (merge tlang and loopy stuff)
        # i.e. make tlang a preprocessing step
        indexed = assignment.tensor
        unindexed = assignment.temporary
        if indexed.name not in self._tensor_data:
            self._tensor_data[indexed.name] = lp.GlobalArg(
                indexed.name, dtype=indexed.dtype, shape=None
            )

        # import pdb; pdb.set_trace()
        if unindexed.name not in [d.name for d in self._temp_kernel_data]:
            if scalar:
                shape = ()
            else:
                shape = (unindexed.axes.alloc_size,)
            self._temp_kernel_data.append(
                lp.TemporaryVariable(unindexed.name, shape=shape)
            )

        return

        # everything below shouldn't work now...
        raise Exception("shouldn't touch below code")
        # there are no ordering restrictions between assignments to the
        # same temporary - but this is only valid to declare if multiple insns are used
        # if len(assignment.lhs.indicess) > 1:
        #     assert len(assignment.rhs.indicess) > 1
        #     no_sync_with = frozenset({(f"{assignment.id}*", "any")})
        # else:
        #     no_sync_with = frozenset()

    def make_offset_expr(self, array, part_names, jnames, insn_prefix, within_inames):
        """create an instruction of the form

        off = 0
        if p1 == 0:
            off += f(i1)
            if p2 == 0:
                off += g(i1, i2)
            else:
                off += h(i1, i2)
        else:
            off += k(i1)

        returning the name of 'off'
        """
        # if array.name == "dat1":
        #     import pdb; pdb.set_trace()
        assert len(part_names) == len(jnames)


        # start at the root
        axis = array.axes
        offset_var_name = f"{array.name}_off"
        self._temp_kernel_data.append(
            lp.TemporaryVariable(offset_var_name, shape=(), dtype=np.uintp)
        )


        # need to create and pass within_inames
        inames_attr = f"inames={':'.join(within_inames)}"

        init_insn_id = self._namer.next(insn_prefix)

        depends_on = frozenset({self._latest_insn[id] for id in self._active_insn.depends_on})
        if self._active_insn.id in self._latest_insn:
            depends_on |= {self._latest_insn[self._active_insn.id]}

        stmts = [f"{offset_var_name} = 0 {{{inames_attr},dep={':'.join(depends_on)},id={init_insn_id}}}"]
        self._latest_insn[self._active_insn.id] = init_insn_id

        new_stmts = self.make_offset_expr_inner(offset_var_name, axis, part_names, jnames, within_inames, insn_prefix)
        stmts.extend(new_stmts)
        self.instructions.append("\n".join(stmts))
        return offset_var_name

    def make_offset_expr_inner(self, offset_var_name, axis, part_names,
            jnames, within_inames, insn_prefix, depth=0):
        assert axis.nparts > 0

        stmts = []
        if axis.nparts == 1: # if statement not needed

            # do not emit a statement if the loop isn't needed
            if isinstance(axis.part.count, MultiArray) or (isinstance(axis.part.count, numbers.Integral) and axis.part.count > 1):
                new_stmts = self.emit_layout_insns(
                    axis.part.layout_fn,
                    offset_var_name,
                    jnames[:depth+1],
                    within_inames,
                    insn_prefix,
                    depth
                )
                stmts += new_stmts

            # TODO indent statements for readability
            subaxis = axis.part.subaxis
            if subaxis:
                substmts = self.make_offset_expr_inner(offset_var_name,
                        subaxis, part_names, jnames, within_inames, insn_prefix, depth+1)
                stmts.extend(substmts)
        else:
            for i, axis_part in enumerate(axis.parts):
                # decide whether to use if, else if, or else

                if axis_part.count > 1:
                    if i == 0:
                        stmts.append(f"if {part_names[depth]} == {i}")
                    elif i == axis.nparts - 1:
                        stmts.append("else")
                    else:
                        stmts.append(f"else if {part_names[depth]} == {i}")

                    newstmts = self.emit_layout_insns(
                        axis_part.layout_fn,
                        offset_var_name, jnames[:depth+1], within_inames,
                        insn_prefix, depth
                    )
                    stmts += newstmts

                # recurse (and indent?)
                subaxis = axis_part.subaxis
                if subaxis:
                    newstmts = self.make_offset_expr_inner(
                        offset_var_name,
                        subaxis, part_names, jnames, within_inames,
                        insn_prefix, depth+1
                    )
                    stmts.extend(newstmts)
            stmts.append("end")
        return stmts

    def emit_layout_insns(self, layout_fn, offset_var, jnames, within_inames, insn_prefix, depth):
        """
        TODO
        """
        insn_id = self._namer.next(insn_prefix)
        within_inames_attr = f"inames={':'.join(within_inames)}"

        depends_on = frozenset({self._latest_insn[id] for id in self._active_insn.depends_on})
        if self._active_insn.id in self._latest_insn:
            depends_on |= {self._latest_insn[self._active_insn.id]}


        # TODO singledispatch!
        if isinstance(layout_fn, IndirectLayoutFunction):
            layout_var = self.register_scalar_assignment(layout_fn.data, jnames, within_inames)

            # generate the instructions
            stmts = [
                f"{offset_var} = {offset_var} + {layout_var} "
                f"{{{within_inames_attr},dep={':'.join(depends_on)},id={insn_id}}}"
            ]


            # register the data
            # layout_arg = lp.GlobalArg(layout_fn.data.name, np.uintp, (None,), is_input=True, is_output=False)
            # self._tensor_data[layout_fn.data.name] = layout_arg
        else:
            if not isinstance(layout_fn, AffineLayoutFunction):
                try:
                    start = layout_fn.data[0]
                    step, = set(layout_fn.data[1:] - layout_fn.data[:-1])
                except ValueError:
                    if len(layout_fn.data) == 1:
                        start = layout_fn.data[0]
                        step = 1
                    else:
                        raise AssertionError("should hit this")
            else:
                start = layout_fn.start
                step = layout_fn.step

            # iname = useable_inames[-1]
            iname = jnames[-1]

            stmts = [
                f"{offset_var} = {offset_var} + {iname}*{step} + {start} "
                f"{{{within_inames_attr},dep={':'.join(depends_on)},id={insn_id}}}"
            ]

        self._latest_insn[self._active_insn.id] = insn_id
        return stmts

    def generate_assignment_insn(
            self,
            assignment,
            lhs,
            loffset,
            rhs,
            roffset,
            scalar,
            within_inames,
        ):
        # the final instruction needs to come after all offset-determining instructions

        # handle scalar assignment (for loop bounds)
        if scalar:
            if assignment.lhs == assignment.temporary:
                loffset = None
            else:
                assert assignment.rhs == assignment.temporary
                roffset = None

        if loffset is None:
            lexpr = pym.var(lhs.name)
        else:
            lexpr = pym.subscript(pym.var(lhs.name), pym.var(loffset))

        if roffset is None:
            rexpr = pym.var(rhs.name)
        else:
            rexpr = pym.subscript(pym.var(rhs.name), pym.var(roffset))

        if isinstance(assignment, tlang.Zero):
            # import pdb; pdb.set_trace()
            rexpr = 0
        elif isinstance(assignment, tlang.Increment):
            rexpr = lexpr + rexpr

        # there are no ordering restrictions between assignments to the
        # same temporary - but this is only valid to declare if multiple insns are used
        # if len(assignment.lhs.indicess) > 1:
        #     assert len(assignment.rhs.indicess) > 1
        #     no_sync_with = frozenset({(f"{assignment.id}*", "any")})
        # else:
        #     no_sync_with = frozenset()

        insn_id = self._namer.next(f"{assignment.id}_")
        depends_on = frozenset({self._latest_insn[id] for id in self._active_insn.depends_on})
        if self._active_insn.id in self._latest_insn:
            depends_on |= {self._latest_insn[self._active_insn.id]}

        assign_insn = lp.Assignment(
            lexpr, rexpr,
            id=insn_id,
            within_inames=frozenset(within_inames),
            within_inames_is_final=True,
            depends_on=depends_on,
            # no_sync_with=no_sync_with,
        )
        self.instructions.append(assign_insn)
        self._latest_insn[self._active_insn.id] = insn_id

    def generate_index_insns(self, multi_index, inames, prefix, context, within_inames):
        """This instruction needs to somehow traverse the indices and axes and probably
        return variable representing the parts and index name (NOT inames) for the LHS
        and RHS. Could possibly be done separately for each as the inames are known.

        Something like 'if the index is not a map yield something equal to the iname, if
        it is a map then yield something like j0 = map0[i0, i1] (and generate the right
        instructions).
        """
        # import pdb; pdb.set_trace()
        # since popping (probably not so) - remove this line?
        # inames = inames.copy()

        idx, *subidxs = multi_index

        pname = self._namer.next(f"{prefix}_p")
        jname = self._namer.next(f"{prefix}_j")

        # not needed as can be done at top of loop
        # depends_on = frozenset({self._latest_insn[id] for id in self._active_insn.depends_on})
        if self._active_insn.id in self._latest_insn:
            depends_on = frozenset({self._latest_insn[self._active_insn.id]})
        else:
            depends_on = frozenset()


        if isinstance(idx, Map):
            map_inames, inames = inames[:idx.depth], inames[idx.depth:]
            raise NotImplementedError("need to emit some clever code")
            assign_insn = ...
        else:
            if subidxs:
                myinames = frozenset(within_inames[:-len(subidxs)])
            else:
                myinames = frozenset(within_inames)
            iname, *subinames = inames
            # TODO handle step sizes etc here
            part_insn = lp.Assignment(
                pym.var(pname), idx.part_label,
                id=self._namer.next(f"{context.id}_"),
                within_inames=myinames,
                depends_on=depends_on,
                # no_sync_with=no_sync_with,
            )
            index_insn = lp.Assignment(
                pym.var(jname), pym.var(iname),
                id=self._namer.next(f"{context.id}_"),
                within_inames=myinames,
                depends_on=depends_on,
                # no_sync_with=no_sync_with,
            )
        self.instructions.append(part_insn)
        self.instructions.append(index_insn)

        self._latest_insn[self._active_insn.id] = index_insn.id

        self._temp_kernel_data.extend([
            lp.TemporaryVariable(name, shape=(), dtype=np.uintp)
            for name in [jname, pname]
        ])

        if subidxs:
            subp, subj = self.generate_index_insns(subidxs, subinames, prefix, context, within_inames)
            return (pname,) + subp, (jname,) + subj
        else:
            return (pname,), (jname,)

    def register_loops(self, indices, outer_loop_indices):
        """
        Returns
        -------
        A stack of inames.
        """
        inames = collections.deque()

        if not indices:
            return list(inames)

        idx, *subidxs = indices

        if isinstance(idx, Map):
            inames.extendleft(self.register_loops([idx.from_], outer_loop_indices))

        if idx in outer_loop_indices:
            iname = outer_loop_indices[idx]
        else:
            iname = self._namer.next("i")

            # register the domain
            domain_str = f"{{ [{iname}]: 0 <= {iname} < {idx.iset.size} }}"
            self.domains.append(domain_str)

        inames.append(iname)

        return list(inames) + self.register_loops(subidxs, outer_loop_indices)

    def register_extent(self, extent, jnames, within_inames):
        if isinstance(extent, MultiArray):
            # If we have a ragged thing then we need to create a scalar temporary
            # to hold its value.
            temp_var = self.register_scalar_assignment(extent, jnames, within_inames)
            return str(temp_var)
        else:
            return extent

    def register_scalar_assignment(self, array, jnames, within_inames):
        # import pdb; pdb.set_trace()
        temp_name = self._namer.next("n")
        # need to create a scalar multi-axis with the same depth
        tempid = "mytempid" + str(0)
        tempaxis = MultiAxis([AxisPart(1, id=tempid)])
        oldtempid = tempid

        for d in range(1, array.depth):
            tempid = "mytempid" + str(d)
            tempaxis = tempaxis.add_subaxis(oldtempid, MultiAxis([AxisPart(1, id=tempid)], parent=tempaxis.set_up()))
            oldtempid = tempid

        temp = MultiArray.new(
            tempaxis.set_up(),
            name=temp_name,
            dtype=np.int32,
        )

        # track old assignment
        old_assignment = self._active_insn

        if self._active_insn:
            depends_on = self._active_insn.depends_on
        else:
            depends_on = frozenset()
        insn = tlang.Read(array, temp, array.indices.multi_indices[0], depends_on=depends_on)
        self._generate_assignment_insn(insn, jnames, within_inames, scalar=True)

        # restore
        self._active_insn = old_assignment

        return pym.var(temp_name)

    # I don't like needing the tensor here..  maybe I could attach the offset to the index?
    # using from_dim
    def handle_assignment(self, tensor, indices, within_loops):
        index_expr = 0
        mainoffset = 0

        for axis, idx in zip(tensor.select_axes(indices), indices):
            part = axis.get_part(idx.npart)
            iname = within_loops.pop(0)

            myexpr = self._as_expr(idx, iname, within_loops)

            if axis.permutation:
                raise AssertionError("should be old code")
                offsets, _ = part.layout
                mainoffset += pym.subscript(pym.var(offsets.name), pym.var(iname))
                self._section_data.append(lp.GlobalArg(offsets.name, shape=None, dtype=np.int32))
                index_expr *= part.size
            else:
                # Every dim uses a section to map the dim index (from the slice/map + iname)
                # onto a location in the data structure. For nice regular data this can just be
                # the index multiplied by the size of the inner dims (e.g. dat[4*i + j]), but for
                # ragged things we need to always have a map for the outer dims.
                # import pdb; pdb.set_trace()
                layout, offset = part.layout

                mainoffset += offset

                if isinstance(layout, numbers.Integral):
                    index_expr = index_expr * layout + myexpr
                elif isinstance(layout, MultiArray):
                    # TODO hack to avoid inserting nnzc[0] in places
                    # I think to resolve this I need to think about how to do
                    # permuted_inner_and_ragged which currently fails for another reason
                    if index_expr != 0:
                        index_expr = pym.subscript(pym.var(layout.name), index_expr) + myexpr
                        self._section_data.append(lp.GlobalArg(layout.name, shape=None, dtype=np.int32))
                    else:
                        index_expr = myexpr
                else:
                    raise TypeError

        return index_expr + mainoffset

    def register_domains(self, indices, dstack, within_loops):
        raise Exception("shouldnt touch")
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
    def _(self, index: Slice, iname, within_loops):
        start = index.start or 0
        step = index.step or 1
        return pym.var(iname)*step + start

    @_as_expr.register
    def _(self, index: IndexFunction, iname, within_loops):
        # use the innermost matching dims as the right inames
        varmap = {}

        # hack to reinsert iname
        within_loops.insert(0, iname)
        for var in reversed(index.vars):
            iname = within_loops.pop(0)
            varmap[var] = pym.var(iname)

        res = pym.substitute(index.expr, varmap)
        return res

    @_as_expr.register
    def _(self, index: NonAffineMap, iname, within_loops):
        # hack to reinsert iname
        within_loops.insert(0, iname)
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

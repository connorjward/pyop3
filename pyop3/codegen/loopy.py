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
from pyop3.utils import PrettyTuple, checked_zip, NameGenerator, rzip
from pyop3.tensors import MultiArray, Map, MultiAxis, NonAffineMap, _compute_indexed_shape, _compute_indexed_shape2
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

        self._within_multi_indices = []  # a stack
        self._within_loop_index_names = []  # a stack

        self._part_id_namer = NameGenerator("mypartid")
        self._loop_index_names = {}

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


        # register inames (also needs to be done for packing loops)
        for multi_idx in multi_idx_collection:
            self._within_multi_indices.append(multi_idx)

            new_loop_index_names = []
            for typed_idx in multi_idx:
                loop_index_name = self._namer.next("i")
                self._loop_index_names[typed_idx] = loop_index_name
                domain_str = f"{{ [{loop_index_name}]: 0 <= {loop_index_name} < {typed_idx.iset.size} }}"
                self.domains.append(domain_str)
                new_loop_index_names.append(loop_index_name)

            self._within_loop_index_names.append(new_loop_index_names)

            # we need to build a separate set of instructions for each multi-index
            # in the collection.
            # e.g. if we are looping over interior facets of an extruded mesh - the
            # iterset is two multi-indices: edges+hfacets and verts+vfacets
            for stmt in expr.statements:
                # self._build(stmt, copy.deepcopy(within_loops))
                self._build(stmt, None)

            self._within_multi_indices.pop()

    def _get_within_inames(self):
        # since we want to pop we need a list of lists
        return frozenset({
            iname
            for loop_index_names in self._within_loop_index_names
            for iname in loop_index_names
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

        depends_on = frozenset({f"{insn}*" for insn in call.depends_on})
        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=call.id,
            within_inames=self._get_within_inames(),
            within_inames_is_final=True,
            depends_on=depends_on
        )

        self.instructions.append(call_insn)
        self.subkernels.append(call.function.code)


    def generate_insn(self, assignment, lhs, loffset, rhs, roffset, depends_on):
        lexpr = pym.subscript(pym.var(lhs.name), loffset)
        rexpr = pym.subscript(pym.var(rhs.name), roffset)

        if isinstance(assignment, tlang.Zero):
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

        assign_insn = lp.Assignment(
            lexpr, rexpr,
            id=self._namer.next(f"{assignment.id}_"),
            within_inames=self._get_within_inames(),
            within_inames_is_final=True,
            depends_on=depends_on|frozenset({f"{dep}*" for dep in assignment.depends_on}),
            # no_sync_with=no_sync_with,
        )
        self.instructions.append(assign_insn)


    def make_offset_expr(self, array, part_names, loop_index_names, insn_prefix, depends_on):
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
        # import pdb; pdb.set_trace()
        assert len(part_names) == len(loop_index_names)


        # start at the root
        axis = array.axes
        offset_var_name = f"{array.name}_off"

        # need to create and pass within_inames
        inames_attr = f"inames={':'.join(self._get_within_inames())}"

        init_insn_id = self._namer.next(insn_prefix)
        depends_on = {f"{dep}*" for dep in depends_on} | {init_insn_id}

        stmts = [f"{offset_var_name} = 0 {{{inames_attr},id={init_insn_id}}}"]

        new_stmts, subdeps = self.make_offset_expr_inner(offset_var_name, axis, part_names, loop_index_names, inames_attr, depends_on, insn_prefix)
        stmts.extend(new_stmts)
        depends_on |= subdeps
        self.instructions.append("\n".join(stmts))
        return offset_var_name, frozenset(depends_on)

    def make_offset_expr_inner(self, offset_var_name, axis, part_names,
            loop_index_names, inames_attr, depends_on, insn_prefix, depth=0):
        assert axis.nparts > 0

        if not depends_on:
            depends_on = set()

        stmts = []
        if axis.nparts == 1:
            # if statement not needed

            # handle layout function here
            new_stmts, subdeps = self.emit_layout_insns(axis.part.layout_fn,
                offset_var_name, loop_index_names, inames_attr, depends_on.copy(), insn_prefix, depth)
            stmts += new_stmts
            depends_on |= subdeps
            # recurse (and indent?)
            subaxis = axis.parts[0].subaxis
            if subaxis:
                substmts, moredeps = self.make_offset_expr_inner(offset_var_name,
                        subaxis, part_names, loop_index_names, inames_attr, depends_on, insn_prefix, depth+1)
                depends_on |= moredeps
                stmts.extend(substmts)
        else:
            for i, axis_part in enumerate(axis.parts):
                # decide whether to use if, else if, or else
                # import pdb; pdb.set_trace()
                if i == 0:
                    stmts.append(f"if {part_names[depth]} == {i}")
                elif i == axis.nparts - 1:
                    stmts.append("else")
                else:
                    stmts.append(f"else if {part_names[depth]} == {i}")

                newstmts, subdeps = self.emit_layout_insns(
                    axis_part.layout_fn,
                    offset_var_name, loop_index_names, inames_attr,
                    depends_on, insn_prefix, depth
                )
                stmts += newstmts
                depends_on |= subdeps

                # recurse (and indent?)
                subaxis = axis_part.subaxis
                if subaxis:
                    newstmts, moredeps = self.make_offset_expr_inner(
                        offset_var_name,
                        subaxis, part_names, loop_index_names, inames_attr,
                        depends_on, insn_prefix, depth+1
                    )
                    depends_on |= moredeps
                    stmts.extend(newstmts)
            stmts.append("end")

        # import pdb; pdb.set_trace()
        return stmts, frozenset(depends_on)

    def emit_layout_insns(self, layout_fn, offset_var, inames, inames_attr, depends_on, insn_prefix, depth):
        """
        TODO
        """
        # if layout_fn is None then we skip this part (used for temporaries eg)
        if not layout_fn:
            return [], set()

        # the layout can depend on the inames *outside* of the current axis - not inside
        useable_inames = inames[:depth+1]

        insn_id = self._namer.next(insn_prefix)

        # assume layout function is an affine layout for now
        # this means we don't need to, for example, use multiple inames
        if not isinstance(layout_fn, AffineLayoutFunction):
            try:
                start = layout_fn.data[0]
                step, = set(layout_fn.data[1:] - layout_fn.data[:-1])
            except ValueError:
                if len(layout_fn.data) == 1:
                    start = layout_fn.data[0]
                    step = 1
                else:
                    raise NotImplementedError("non-const stride in layout not handled - need"
                        "to register data structure etc")
        else:
            start = layout_fn.start
            step = layout_fn.step

        iname = useable_inames[-1]

        stmts = [f"{offset_var} = {offset_var} + {iname}*{step} + {start} {{{inames_attr},dep={':'.join(dep for dep in depends_on)},id={insn_id}}}"]
        return stmts, {insn_id}

    # NEXT: temporaries are using the wrong indices - should be scalar...
    def cleverly_recurse(
            self,
            assignment,
            indexed,
            indexed_axis,
            indexed_part_id_names,
            indexed_loop_index_names,
            unindexed,
            unindexed_axis,
            unindexed_part_id_names,
            unindexed_loop_index_names,
            multi_index_collections,
        ):
        """
        For an assignment collect the inames associated with the indexed and unindexed (temp)
        bits. These will then be used to derive the correct offset expression.

        I think this will break if we need to index any maps or ragged things - would need to
        maintain some iname registry

        indexed_within: inames associated with the indexed bit
        """
        # because I'm lazy elsewhere... (make tuples!)
        indexed_part_id_names = indexed_part_id_names.copy()
        indexed_loop_index_names = indexed_loop_index_names.copy()
        unindexed_part_id_names = unindexed_part_id_names.copy()
        unindexed_loop_index_names = unindexed_loop_index_names.copy()

        def is_bottom_of_tree(ax):
            if not ax:
                return True
            elif ax.nparts == 1 and ax.part.size == 1:
                return True
            else:
                return False

        # if at the bottom of the tree generate instructions and terminate
        if is_bottom_of_tree(indexed_axis):
            assert is_bottom_of_tree(indexed_axis) and is_bottom_of_tree(unindexed_axis), "must both be false"
            assert not multi_index_collections

            indexed_offset, indexed_depends_on = self.make_offset_expr(
                indexed, indexed_part_id_names, indexed_loop_index_names, assignment.id, assignment.depends_on,
            )
            unindexed_offset, unindexed_depends_on = self.make_offset_expr(
                unindexed, unindexed_part_id_names, unindexed_loop_index_names, assignment.id, assignment.depends_on,
            )

            # the final instruction needs to come after all offset-determining instructions
            depends_on = indexed_depends_on | unindexed_depends_on

            self._temp_kernel_data.append(
                lp.TemporaryVariable(indexed_offset, shape=(), dtype=np.uintp)
            )
            self._temp_kernel_data.append(
                lp.TemporaryVariable(unindexed_offset, shape=(), dtype=np.uintp)
            )

            # FIXME
            if isinstance(assignment, (tlang.Read, tlang.Zero)):
                lhs = unindexed
                loffset = unindexed_offset
                rhs = indexed
                roffset = indexed_offset
            elif isinstance(assignment, (tlang.Write, tlang.Increment)):
                lhs = indexed
                loffset = indexed_offset
                rhs = unindexed
                roffset = unindexed_offset
            else:
                raise TypeError

            self.generate_insn(assignment, lhs, pym.var(loffset), rhs, pym.var(roffset), depends_on)

            if indexed.name not in self._tensor_data:
                self._tensor_data[indexed.name] = lp.GlobalArg(
                    indexed.name, dtype=indexed.dtype, shape=None
                )

            # import pdb; pdb.set_trace()
            if unindexed.name not in [d.name for d in self._temp_kernel_data]:
                self._temp_kernel_data.append(
                    lp.TemporaryVariable(unindexed.name, shape=(unindexed.axes.alloc_size,))
                )
            return

        ###

        if multi_index_collections:
            multi_idx_collection, *subidx_collections = multi_index_collections
        else:
            # then take all of the rest of the shape
            multi_idx_collection = MultiIndexCollection([
                MultiIndex([
                    TypedIndex(p, IndexSet(indexed_axis.parts[p].size))
                    for p in range(indexed_axis.nparts)
                ])
            ])
            subidx_collections = []

        assert isinstance(multi_idx_collection, MultiIndexCollection)

        if isinstance(multi_idx_collection, Map):
            raise NotImplementedError


        # need to collect loop index names (inames) and part names
        # each multi-index is a different branch for generating code (as the inner
        # dimensions can be different)
        for i, multi_idx in enumerate(multi_idx_collection):
            is_loop_index = multi_idx in self._within_multi_indices
            new_inames = []
            for typed_idx in multi_idx:
                if is_loop_index:
                    indexed_loop_index_name = self._loop_index_names[typed_idx]
                    unindexed_loop_index_name = self._loop_index_names[typed_idx]
                else:
                    # register a new domain
                    loop_index_name = self._namer.next("i")
                    self._loop_index_names[typed_idx] = loop_index_name
                    domain_str = f"{{ [{loop_index_name}]: 0 <= {loop_index_name} < {typed_idx.iset.size} }}"
                    self.domains.append(domain_str)
                    indexed_loop_index_name = loop_index_name
                    unindexed_loop_index_name = loop_index_name
                    new_inames.append(loop_index_name)

                # note that the names must be different for indexed and unindexed as
                # these variables get transformed by the map etc.
                indexed_axis = indexed_axis.parts[typed_idx.part].subaxis
                indexed_part_id_name = self._part_id_namer.next()
                indexed_part_id_names.append(indexed_part_id_name)
                indexed_loop_index_names.append(indexed_loop_index_name)

                unindexed_axis = unindexed_axis.parts[i].subaxis
                unindexed_part_id_name = self._part_id_namer.next()
                unindexed_part_id_names.append(unindexed_part_id_name)
                unindexed_loop_index_names.append(unindexed_loop_index_name)

                # TODO indices should probably not be the indices directly as maps
                # need to transform them

                # Register parts
                # TODO also loop_index_names
                self._temp_kernel_data.extend([
                    lp.TemporaryVariable(name, shape=(), dtype=np.uintp)
                    for name in [indexed_part_id_name, unindexed_part_id_name]
                ])

            # add new inames to the stack
            self._within_loop_index_names.append(new_inames)

            # doesn't return anything
            self.cleverly_recurse(
                assignment,
                indexed, indexed_axis,
                indexed_part_id_names,
                indexed_loop_index_names,
                unindexed, unindexed_axis,
                unindexed_part_id_names,
                unindexed_loop_index_names,
                subidx_collections,
            )

            self._within_loop_index_names.pop()

    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, within_loops, scalar=False, domain_stack=None):
        # so basically one of the LHS or RHS should not have any indices. We can therefore
        # use the one that does to generate the domains

        # now for some clever recursion
        self.cleverly_recurse(
            assignment,
            assignment.tensor, assignment.tensor.axes, [], [],
            assignment.temporary, assignment.temporary.axes, [], [],
            assignment.tensor.indices
        )

        return

        # everything below shouldn't work now...
        raise Exception("shouldn't touch below code")
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

        if assignment.tensor.name not in self._tensor_data:
            self._tensor_data[assignment.tensor.name] = lp.GlobalArg(
                assignment.tensor.name, dtype=assignment.tensor.dtype, shape=None
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
                temp = MultiArray.new(MultiAxis(ScalarAxisPart()), name=temp_name, dtype=np.int32)

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

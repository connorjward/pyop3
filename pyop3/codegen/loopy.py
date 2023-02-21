import abc
import collections
import copy
import dataclasses
import numbers
import enum
import functools
import itertools

from typing import Dict, Any, Tuple, FrozenSet, Sequence, Optional

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
from pyop3.tensors import MultiArray, Map, MultiAxis, _compute_indexed_shape, _compute_indexed_shape2, AxisPart
from pyop3.tensors import index, MultiIndexCollection, MultiIndex, TypedIndex, AffineLayoutFunction, IndirectLayoutFunction


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
    return _make_loopy_kernel(expr)


def to_c(expr):
    program = to_loopy(expr)
    return lp.generate_code_v2(program).device_code()


@dataclasses.dataclass
class IndexRegistryItem:
    # index: TypedIndex
    label: str
    iname: str
    jnames: Sequence[str]
    is_loop_index: bool = False
    registry: Optional[Sequence["IndexRegistryItem"]] = None
    """Need this if we have maps."""
    within_inames: FrozenSet = frozenset()

    def __postinit__(self):
        assert self.label is not None


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

        self._part_id_namer = NameGenerator("mypartid")

        self._latest_insn = {}
        self._active_insn = None
        self._temp_name_generator = NameGenerator("t")

    @property
    def kernel_data(self):
        return list(self._tensor_data.values()) + self._section_data + self._temp_kernel_data

    def build(self, tlang_expr):
        self._namer.reset()
        self._build(tlang_expr)

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
    def _build(self, expr, *args, **kwargs):
        raise TypeError

    @_build.register
    def _(self, expr: exprs.Loop, index_registry=None, loop_indices=frozenset()):
        if not index_registry:
            index_registry = []

        multi_indices = expr.index
        assert isinstance(multi_indices, MultiIndexCollection)

        if len(multi_indices.multi_indices) > 1:
            raise NotImplementedError(
            """In this case we need to follow different codegen pathways internally. I
               expect this to cause big issues with how we identify the indices of the
               arrays themselves as dat[p] where p can be one of many things is hard to
               identify.
            """)

        # save these as they are needed fresh per-loop
        outer_loop_indices = loop_indices

        for multi_idx in multi_indices:
            # 1. register loops
            # loop_indices = outer_loop_indices | self.collect_loop_indices(multi_idx)

            new_index_registry = self.register_loops2(multi_idx, index_registry, True)

            # import pdb; pdb.set_trace()

            # 2. emit instructions
            # we need to build a separate set of instructions for each multi-index
            # in the collection.
            # e.g. if we are looping over interior facets of an extruded mesh - the
            # iterset is two multi-indices: edges+hfacets and verts+vfacets
            for stmt in expr.statements:
                self._build(stmt, new_index_registry)


    def collect_loop_indices(self, multi_index):
        loop_indices = set()
        for typed_idx in multi_index:
            if isinstance(typed_idx, Map):
                loop_indices |= self.collect_loop_indices(typed_idx.from_multi_index)
            loop_indices |= {typed_idx}
        return frozenset(loop_indices)


    # index registry is just a list of IndexRegistryEntry items
    def register_loops2(self, multi_index, index_registry, called_by_loop):
        # import pdb; pdb.set_trace()
        eaten_registry = index_registry.copy()
        # 1. partition the existing index registry
        registry_per_index = {}
        for typed_idx in multi_index:
            registry = []
            for part_label in typed_idx.consumed_labels:
                try:
                    item = utils.popwhen(
                        lambda entry: entry.label == part_label, eaten_registry)
                    registry.append(item)
                except KeyError:
                    pass
            registry_per_index[typed_idx] = registry

        # 2. construct a new index registry
        new_registry = []
        for typed_idx in multi_index:
            # don't re-register loops (never failed to pop an index)
            if len(registry_per_index[typed_idx]) == len(typed_idx.consumed_labels):
                new_registry.extend(registry_per_index[typed_idx])
                continue

            # for now - not sure how to do this for partially indexed things
            assert len(registry_per_index[typed_idx]) == 0

            if isinstance(typed_idx, Map):
                """
                if we hit a map we want to handle its from_multi_index first, then
                deal with it itself.

                it will always yield depth-many active_inames and emit a single loop
                of size arity.

                No. Actually I think it will *consume* depth-many active_inames and
                yield a single new active iname

                No. It will yield that many active_inames but also need to somehow
                account for the number that are consumed. This is different to depth
                I think.
                """
                sub_registry = self.register_loops2(
                    typed_idx.from_multi_index, registry_per_index[typed_idx], called_by_loop)
            else:
                sub_registry = None

            # do this before creating the new iname
            # FIXME shouldn't this use the stuff from the map if required?
            extent = self.register_extent(
                typed_idx.size,
                new_registry,
                # within_inames,
            )

            # each index can only emit a single new loop
            iname = self._namer.next("i")
            # but can yield depth-many new jnames (which are pretend inames)
            jnames = tuple(self._namer.next("j") for _ in range(typed_idx.depth))

            domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
            self.domains.append(domain_str)

            # import pdb; pdb.set_trace()
            if sub_registry:
                within_inames = frozenset({iname}) | {item for entry in sub_registry for item in entry.within_inames}
            else:
                within_inames = frozenset({iname})

            registry_item = IndexRegistryItem(typed_idx.part_label, iname, jnames, is_loop_index=called_by_loop, registry=sub_registry, within_inames=within_inames)
            new_registry.append(registry_item)
        return new_registry

    @_build.register
    def _(self, call: exprs.FunctionCall, index_registry):
        insns = self.expand_function_call(call, index_registry)

        for insn in insns:
            self._make_instruction_context(insn, index_registry)

    def expand_function_call(self, call, index_registry):
        """
        Turn an exprs.FunctionCall into a series of assignment instructions etc.
        Handles packing/accessor logic.
        """

        temporaries = {}
        for arg in call.arguments:
            # create an appropriate temporary
            dims = self._construct_temp_dims(arg.tensor.axes, arg.tensor.indices, index_registry)
            temporary = MultiArray.new(
                dims, name=self._temp_name_generator.next(), dtype=arg.tensor.dtype,
            )
            temporaries[arg] = temporary

        gathers = self.make_gathers(temporaries)
        newcall = self.make_function_call(
            call, temporaries,
            depends_on=frozenset(gather.id for gather in gathers)
        )
        scatters = self.make_scatters(temporaries, depends_on=frozenset({newcall.id}))

        return (*gathers, newcall, *scatters)

    def _construct_temp_dims(self, axis, multi_indices, index_registry):
        """Return a multi-axis describing the temporary shape.

        Parameters
        ----------
        multi_indices : ???
            Iterable of :class:`MultiIndex`.
        """
        """
        Can have something like [:5, map2(map1(c))] which should return a temporary
        with shape (5, |map1|, |map2|, ...) where ... is whatever the bottom part of
        the tensor is.

        To do this we start with the first index (the slice) to generate the root axis.
        Then we encounter map2 so we first handle map1. Since c is a within_index we
        disregard it.
        """
        # deprecated since we compress this in __getitem__
        # multi-index collections example: [closure(c0), closure(c1)]
        # if multi_index_collections:
        #     multi_idx_collection, *subidx_collections = multi_index_collections
        # else:
        #     # then take all of the rest of the shape
        #     multi_idx_collection = MultiIndexCollection([
        #         MultiIndex([
        #             TypedIndex(p, IndexSet(axis.parts[p].count))
        #             for p in range(axis.nparts)
        #         ])
        #     ])
        #     subidx_collections = []
        #

        assert isinstance(multi_indices, MultiIndexCollection)

        ###


        # each multi-index yields an adjacent axis part
        # e.g. for mixed (with 2 spaces) you would have a temporary with 2 parts
        # similarly this would be the expected behaviour for interior facets of extruded
        # meshes - the outer axis would be split in two parts because the DoFs come from
        # both the vertical and horizontal facets and these require separate multi-indices
        axis_parts = []
        for multi_idx in multi_indices:
            axis_part = self.make_temp_axis_part_per_multi_index(multi_idx, index_registry)
            axis_parts.append(axis_part)

        return MultiAxis(axis_parts).set_up()

    def make_temp_axis_part_per_multi_index(
            self, multi_index, index_registry, return_final_part_id=False):
        """TODO

        Returns
        -------
        An iterable of :class:`AxisPart`.

        Remember that whatever we get here is "linear" - maybe we could make a list of
        axis parts and glue together at the end?
        No it's not linear - if we don't index into a multi-part subaxis.

        Wait! We should already have fully indexed the array so multi-index should be
        the right length.
        """
        index_registry = index_registry.copy()
        idx, *subidxs = multi_index
        try:
            entry = utils.popwhen(lambda e: e.label == idx.part_label, index_registry)
            has_scalar_shape = entry.is_loop_index
        except KeyError:
            has_scalar_shape = False
        new_part = AxisPart(idx.size, indexed=has_scalar_shape, label=idx.part_label)

        if subidxs:
            result = self.make_temp_axis_part_per_multi_index(subidxs, index_registry, return_final_part_id=return_final_part_id)
            if return_final_part_id:
                subpart, bottom_part_id = result
            else:
                subpart = result
            new_part = new_part.copy(subaxis=MultiAxis([subpart]))
        else:
            bottom_part_id = new_part.id

        if isinstance(idx, Map):
            super_part, bottom_part_id = self.make_temp_axis_part_per_multi_index(idx.from_multi_index, loop_indices, return_final_part_id=True)
            # wont work as need part IDs
            new_part = super_part.add_subaxis(bottom_part_id, MultiAxis([new_part]))

        if return_final_part_id:
            return new_part, bottom_part_id
        else:
            return new_part


        ############################
        is_loop_index = depth < len(active_inames)
        temp_axis_part_id = self._namer.next("mypart")
        size = multi_idx.typed_indices[0].size
        temp_axis_part  = AxisPart(
            size,
            indexed=is_loop_index,
            id=temp_axis_part_id,
        )
        old_temp_axis_part_id = temp_axis_part_id

        # track the position in the array as this tells us whether or not we
        # need to recurse.
        # we need to track this throughout because the types of the typed_idx
        # tells us which bits of the hierarchy are 'below'.
        current_axis = axis.find_part(multi_idx.typed_indices[0].part_label).subaxis
        depth += 1

        # each typed index is a subaxis of the original
        for typed_idx in multi_idx.typed_indices[1:]:
            is_loop_index = depth < len(active_inames)
            temp_axis_part_id = self._namer.next("mypart")
            size = typed_idx.size
            temp_subaxis  = MultiAxis(
                [AxisPart(
                    size,
                    indexed=is_loop_index,
                    id=temp_axis_part_id
                )],
            )
            temp_axis_part = temp_axis_part.add_subaxis(old_temp_axis_part_id, temp_subaxis)
            old_temp_axis_part_id = temp_axis_part_id

            current_axis = current_axis.find_part(typed_idx.part_label).subaxis
            depth += 1

        temp_axis_parts.append(temp_axis_part)
        temp_axis_base_ids.append(temp_axis_part_id)

        # if we still have a current axis then we haven't hit the bottom of the
        # tree and more shape is needed
        if current_axis:
            subaxis = self._construct_temp_dims(current_axis, subidx_collections, depth+1)
            # add this subaxis to each part we have so far
            # recall that the number of parts we have is equal to the number of multi-index
            # collections that we have
            temp_axis_parts = [
                pt.add_subaxis(ptid, subaxis)
                for pt, ptid in checked_zip(temp_axis_parts, temp_axis_base_ids)
            ]


    def make_gathers(self, temporaries, **kwargs):
        return tuple(
            self.make_gather(arg, temp, idxs, **kwargs)
            for arg, temp in temporaries.items()
            for idxs in arg.tensor.indices.multi_indices
        )

    def make_gather(self, argument, temporary, indices, **kwargs):
        # TODO cleanup the ids
        if argument.access in {exprs.READ, exprs.RW}:
            return tlang.Read(argument.tensor, temporary, indices, **kwargs)
        elif argument.access in {exprs.WRITE, exprs.INC}:
            return tlang.Zero(argument.tensor, temporary, indices, **kwargs)
        else:
            raise NotImplementedError

    def make_function_call(self, call, temporaries, **kwargs):
        assert all(arg.access in {exprs.READ, exprs.WRITE, exprs.INC, exprs.RW} for arg in call.arguments)

        reads = tuple(
            # temporaries[arg] for arg in call.arguments
            temp for arg, temp in temporaries.items()
            if arg.access in {exprs.READ, exprs.INC, exprs.RW}
        )
        writes = tuple(
            temp for arg, temp in temporaries.items()
            # temporaries[arg] for arg in call.arguments
            if arg.access in {exprs.WRITE, exprs.INC, exprs.RW}
        )
        return tlang.FunctionCall(call.function, reads, writes, **kwargs)

    def make_scatters(self, temporaries, **kwargs):
        return tuple(filter(
            None,
            (
                self.make_scatter(arg, temp, idxs, **kwargs)
                for arg, temp in temporaries.items()
                for idxs in arg.tensor.indices.multi_indices
            )
        ))

    def make_scatter(self, argument, temporary, indices, **kwargs):
        if argument.access == exprs.READ:
            return None
        elif argument.access in {exprs.WRITE, exprs.RW}:
            return tlang.Write(argument.tensor, temporary, indices, **kwargs)
        elif argument.access == exprs.INC:
            return tlang.Increment(argument.tensor, temporary, indices, **kwargs)
        else:
            raise AssertionError

    @functools.singledispatchmethod
    def _make_instruction_context(self, instruction: tlang.Instruction, *args, **kwargs):
        raise TypeError


    @_make_instruction_context.register
    def _(self, call: tlang.FunctionCall, index_registry):
        subarrayrefs = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            temp_size = temp.axes.alloc_size
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            self.domains.append(f"{{ [{iname}]: 0 <= {iname} < {temp_size} }}")
            assert temp.name in [d.name for d in self._temp_kernel_data]

        # we need to pass sizes through if they are only known at runtime (ragged)
        extents = {}
        # traverse the temporaries
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            extents |= self.collect_extents(
                temp.root,
                index_registry,
                # active_inames,
                # within_inames,
            )

        # NOTE: If we register an extent to pass through loopy will complain
        # unless we register it as an assumption of the local kernel (e.g. "n <= 3")

        assignees = tuple(subarrayrefs[var] for var in call.writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(subarrayrefs[var] for var in call.reads) + tuple(extents.values()),
        )

        insn_id = f"{call.id}_0"
        depends_on = frozenset({self._latest_insn[id] for id in call.depends_on})
        # within_inames = frozenset(within_loops.values())

        within_inames = functools.reduce(
            frozenset.__or__,
            [entry.within_inames for entry in index_registry
             if entry.is_loop_index]
        )

        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=insn_id,
            within_inames=within_inames,
            within_inames_is_final=True,
            depends_on=depends_on
        )

        self.instructions.append(call_insn)
        self.subkernels.append(call.function.code)
        self._latest_insn[call.id] = insn_id

    def collect_extents(self, axis, index_registry, depth=0):
        extents = {}

        for part in axis.parts:
            if isinstance(part.count, MultiArray) and not part.indexed:
                assert depth >= 1
                extent = self.register_scalar_assignment(part.count, index_registry)
                extents[part.count] = extent

            if part.subaxis:
                extents_ = self.collect_extents(part.subaxis, index_registry, depth+1)
                if any(array in extents for array in extents_):
                    raise ValueError("subaxes should not be using the same nnz structures (I think)")
                extents |= extents_

        return extents

    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, index_registry):
        new_registry = self.register_loops2(assignment.indices, index_registry, False)

        self._generate_assignment_insn(assignment, new_registry)

    def _generate_assignment_insn(self, assignment, index_registry, scalar=False):
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
        # depth = len(assignment.indices.typed_indices)
        # jnames = jnames[-depth:]

        # import pdb; pdb.set_trace()
        # depends_on = depends_on | {f"{dep}*" for dep in assignment.depends_on}

        index_registry_copy = index_registry.copy()
        selected = [utils.popwhen(lambda e: e.label == idx.part_label, index_registry_copy)
                    for idx in assignment.indices]

        within_inames = functools.reduce(frozenset.__or__, [e.within_inames for e in selected])

        # 2. Generate map code
        lpnames, ljnames = self.generate_index_insns(assignment.lhs.axes, assignment.indices, index_registry, assignment.lhs.name, assignment, within_inames)

        # reset here to avoid loop clashes - these don't need to sync
        self._latest_insn.pop(self._active_insn.id)

        rpnames, rjnames = self.generate_index_insns(assignment.rhs.axes, assignment.indices, index_registry, assignment.rhs.name, assignment, within_inames)

        # 3. Generate layout code
        # import pdb; pdb.set_trace()
        loffset = self.make_offset_expr(
            assignment.lhs, lpnames, index_registry, f"{assignment.id}_loffset", within_inames,
        )
        roffset = self.make_offset_expr(
            assignment.rhs, rpnames, index_registry, f"{assignment.id}_roffset", within_inames,
        )

        # 4. Emit assignment instruction
        self.generate_assignment_insn(assignment, assignment.lhs, loffset,
                                      assignment.rhs, roffset,
                                      scalar=scalar, within_inames=within_inames)

        # Register data
        # TODO should only do once at a higher point
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

    def make_offset_expr(self, array, part_names, index_registry, insn_prefix, within_inames):
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
        # assert len(part_names) == len(jnames)

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

        new_stmts = self.make_offset_expr_inner(offset_var_name, axis, part_names, index_registry, within_inames, insn_prefix)
        stmts.extend(new_stmts)
        self.instructions.append("\n".join(stmts))
        return offset_var_name

    def make_offset_expr_inner(self, offset_var_name, axis, part_names,
            index_registry, within_inames, insn_prefix, depth=0):
        assert axis.nparts > 0

        stmts = []
        if axis.nparts == 1: # if statement not needed

            # do not emit a statement if the loop isn't needed
            if isinstance(axis.part.count, MultiArray) or (isinstance(axis.part.count, numbers.Integral) and axis.part.count > 1):
                new_stmts = self.emit_layout_insns(
                    axis.part.layout_fn,
                    0,  # not really a label
                    offset_var_name,
                    index_registry,
                    within_inames,
                    insn_prefix,
                    depth
                )
                stmts += new_stmts

            # TODO indent statements for readability
            subaxis = axis.part.subaxis
            if subaxis:
                substmts = self.make_offset_expr_inner(offset_var_name,
                        subaxis, part_names, index_registry, within_inames, insn_prefix, depth+1)
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
                        stmts.append("else")
                        stmts.append(f"if {part_names[depth]} == {i}")

                    newstmts = self.emit_layout_insns(
                        axis_part.layout_fn,
                        i,  # not really a label
                        offset_var_name, index_registry, within_inames,
                        insn_prefix, depth
                    )
                    stmts += newstmts

                # recurse (and indent?)
                subaxis = axis_part.subaxis
                if subaxis:
                    newstmts = self.make_offset_expr_inner(
                        offset_var_name,
                        subaxis, part_names, index_registry, within_inames,
                        insn_prefix, depth+1
                    )
                    stmts.extend(newstmts)

            # number of "end" statements that need emitting
            for _ in range(axis.nparts-1):
                stmts.append("end")
        # import pdb; pdb.set_trace()
        return stmts

    def emit_layout_insns(self, layout_fn, label, offset_var, index_registry, within_inames, insn_prefix, depth):
        """
        TODO
        """
        if layout_fn == "null layout":
            return []

        insn_id = self._namer.next(insn_prefix)
        within_inames_attr = f"inames={':'.join(within_inames)}"

        depends_on = frozenset({self._latest_insn[id] for id in self._active_insn.depends_on})
        if self._active_insn.id in self._latest_insn:
            depends_on |= {self._latest_insn[self._active_insn.id]}


        # TODO singledispatch!
        if isinstance(layout_fn, IndirectLayoutFunction):
            layout_var = self.register_scalar_assignment(layout_fn.data, index_registry)

            # generate the instructions
            stmts = [
                f"{offset_var} = {offset_var} + {layout_var} "
                f"{{{within_inames_attr},dep={':'.join(depends_on)},id={insn_id}}}"
            ]


            # register the data
            # layout_arg = lp.GlobalArg(layout_fn.data.name, np.uintp, (None,), is_input=True, is_output=False)
            # self._tensor_data[layout_fn.data.name] = layout_arg
        elif isinstance(layout_fn, AffineLayoutFunction):
            start = layout_fn.start
            step = layout_fn.step

            if isinstance(start, MultiArray):
                start = self.register_scalar_assignment(layout_fn.start, index_registry)

            # import pdb; pdb.set_trace()
            # this isn't quite good enough - I would previously truncate this and always
            # take the last entry. Here we hit duplicates. How do I distinguish the
            # particular index entry that I want?
            # entry = utils.single_valued(e for e in index_registry if e.index.part_label == label)
            iname = index_registry[depth].iname

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

    def generate_index_insns(self, axis, multi_index, index_registry, prefix, context, within_inames):
        """This instruction needs to somehow traverse the indices and axes and probably
        return variable representing the parts and index name (NOT inames) for the LHS
        and RHS. Could possibly be done separately for each as the inames are known.

        Something like 'if the index is not a map yield something equal to the iname, if
        it is a map then yield something like j0 = map0[i0, i1] (and generate the right
        instructions).
        """
        registry = index_registry.copy()

        idx, *subidxs = multi_index

        entry = utils.popwhen(lambda it: it.label == idx.part_label, registry)

        pname = self._namer.next(f"{prefix}_p")
        # jname = self._namer.next(f"{prefix}_j")
        jnames = entry.jnames

        # not needed as can be done at top of loop
        depends_on = frozenset({self._latest_insn[id] for id in self._active_insn.depends_on})
        if self._active_insn.id in self._latest_insn:
            depends_on |= frozenset({self._latest_insn[self._active_insn.id]})

        npart = [part.label for part in axis.parts].index(idx.part_label)

        if isinstance(idx, Map):
            map_inames, inames = inames[:idx.depth], inames[idx.depth:]
            raise NotImplementedError("need to emit some clever code")
            assign_insn = ...
        else:
            iname = entry.iname
            part_insn = lp.Assignment(
                pym.var(pname), npart,
                id=self._namer.next(f"{context.id}_"),
                within_inames=within_inames,
                depends_on=depends_on,
                # no_sync_with=no_sync_with,
            )
            jname, = jnames
            index_insn = lp.Assignment(
                pym.var(jname), pym.var(iname),
                id=self._namer.next(f"{context.id}_"),
                within_inames=within_inames,
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
            assert axis.parts[npart].subaxis
            subp, subj = self.generate_index_insns(axis.parts[npart].subaxis, subidxs, registry, prefix, context, within_inames)
            return (pname,) + subp, (jname,) + subj
        else:
            return (pname,), (jname,)

    def register_loops(self, indices, active_inames, within_inames, depth=0):
        """
        Take a multi-index and register any extra loops needed to fill "shape" for the
        array.

        Maybe this could be done when temporaries are registered?

        Returns
        -------
        active_inames, within_inames
        """
        # not true if there are maps
        # assert  len(indices) >= len(active_inames)

        # at bottom
        if depth == len(indices):
            return active_inames, within_inames

        if not active_inames:
            active_inames = ()

        inames = collections.deque()

        idx = indices.typed_indices[depth]

        try:
            iname = active_inames[depth]
        except IndexError:
            iname = None

        if isinstance(idx, Map):
            raise NotImplementedError
            inames.extendleft(self.register_loops([idx.from_], active_inames, within_inames))

        if not iname:
            # register the domain
            iname = self._namer.next("i")

            # import pdb; pdb.set_trace()

            # TODO: index with depth??
            if isinstance(idx.size, MultiArray):
                assert depth > 0

                extent = self.register_extent(
                    idx.size,
                    active_inames[:depth],
                    within_inames,
                )
            else:
                extent = self.register_extent(
                    idx.size,
                    None,
                    within_inames,
                )

            domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
            self.domains.append(domain_str)


            active_inames += (iname,)
            within_inames |= {iname}

        return self.register_loops(indices, active_inames, within_inames, depth+1)

    def register_extent(self, extent, index_registry):
        if isinstance(extent, MultiArray):
            # If we have a ragged thing then we need to create a scalar temporary
            # to hold its value.
            temp_var = self.register_scalar_assignment(extent, index_registry)
            return str(temp_var)
        else:
            return extent

    def register_scalar_assignment(self, array, index_registry):
        temp_name = self._namer.next("n")
        # need to create a scalar multi-axis with the same depth
        # TODO is it not better to just "fully index" this thing?
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
        self._generate_assignment_insn(insn, index_registry, scalar=True)

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

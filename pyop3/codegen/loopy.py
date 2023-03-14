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
from pyop3.tensors import MultiArray, Map, MultiAxis, _compute_indexed_shape, _compute_indexed_shape2, AxisPart, TerminalMultiIndex
from pyop3.tensors import index, MultiIndexCollection, MultiIndex, TypedIndex, AffineLayoutFunction, IndirectLayoutFunction, Range


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
    def _(self, expr: exprs.Loop, within_multi_index_groups=None, depends_on=frozenset()):
        # within_multi_index_groups is a map from multi-index groups to inames
        # every multi-index group (which have unique IDs) may only appear once
        """
        We need to key by group rather than multi-index because that is what we
        use in our loop definitions. For example:

            loop(f := extruded_mesh.interior_facets, dat[f])

        f is a multi-index group because it can be (base cell, extr edge) or
        (base vert,extr edge). When we generate code for the dat we need to look up the
        inames using f, rather than the selected multi-index.

        This ties in well with the "path" logic I have already done for maps.
        """
        if within_multi_index_groups is None:
            within_multi_index_groups = {}  # TODO should make persistent to avoid accidental side-effects

        if expr.index in within_multi_index_groups:
            raise ValueError("Loop indices cannot be reused in nested loops")


        # this isn't sufficient
        """
        want something like:
            
            for all the possible variations from expr.index:
                do the inner bit

        but what does this mean for assignment? when we have conflicting indices?
        but actually the indices can be identical
        either identical or declared outside...
        not true! closure(c) for example is different
        just the inner bits that would be the same - can use same axis labels though

        do I need to loop over them together?
        """
        for updated_within_migs, updated_deps in self.expand_multi_index_group(expr.index, within_multi_index_groups, depends_on, prefix=expr.id):
            # I kind of want a paradigm like:
            # new_state = handle_single_multi_index(old_state)
            # means that there is less external bookkeeping though care needs to be taken
            # with for loops like here

            # bitsandpieces = self.register_loops2(multi_idx, index_registry, loop_indices)
            # for midx, inames in bitsandpieces.items():
            #     index_registry.append(midx, inames)
            #
            # loop_indices = outer_loop_indices + self.collect_loop_indices(multi_idx)

            # new_index_registry = self.register_loops2(multi_idx, index_registry, True)

            # import pdb; pdb.set_trace()

            # 2. emit instructions
            for stmt in expr.statements:
                self._build(stmt, updated_within_migs, updated_deps)

    def collect_loop_indices(self, multi_index):
        if isinstance(multi_index, Map):
            return [multi_index] + self.collect_loop_indices(multi_index.from_index)
        else:
            assert isinstance(multi_index, TerminalMultiIndex)
            return [multi_index]


    def register_loops2(self, multi_index, index_registry, loop_indices):
        """
        Register a bunch of loops related to a multi-index and return a mapping from index
        to inames

        TODO
        This should be done as part of the same pass as the rest of the codegen! We can diverge
        if we are looping over fancy maps
        """
        assert isinstance(multi_index, MultiIndex)
        if multi_index in loop_indices:
            return {}

        within_inames = set()
        for idx in loop_indices:
            within_iname, = index_registry[idx]
            within_inames |= {within_iname}
        within_inames = frozenset(within_inames)

        # hack for now - needs more thought
        depends_on = frozenset()

        if isinstance(multi_index, Map):
            if len(multi_index.paths) > 1:
                raise NotImplementedError("wrong type of recursion for this - need to wrap "
                                          "it all together")

            extra_dict = self.register_loops2(multi_index.from_index, index_registry, loop_indices)

            # a map will yield a single extra loop
            # TODO encode the arity
            inames = []
            if len(multi_index.paths) > 1:
                # if we have multiple paths then need to diverge here, complicated!
                raise NotImplementedError("hard")
            path, = multi_index.paths
            extent = self.register_extent(
                path.arity,
                inames,
                within_inames,
                depends_on,
            )

            iname = self._namer.next("i")
            inames.append(iname)
            within_inames |= {iname}

            domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
            self.domains.append(domain_str)
            import pdb; pdb.set_trace()
            return {multi_index: inames} | extra_dict
        else:
            assert isinstance(multi_index, TerminalMultiIndex)
            inames = []
            for idx in multi_index:

                # do this before creating the new iname
                extent = self.register_extent(
                    idx.size,
                    inames,
                    within_inames,
                    depends_on,
                )

                # each index can only emit a single new loop
                iname = self._namer.next("i")
                inames.append(iname)
                within_inames |= {iname}

                domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
                self.domains.append(domain_str)

            return {multi_index: inames}

    @_build.register
    def _(self, call: exprs.FunctionCall, within_multi_index_groups, depends_on):
        # I think I'd prefer to do this in a separate earlier pass?
        insns = self.expand_function_call(call, within_multi_index_groups, depends_on)

        for insn in insns:
            self._make_instruction_context(insn, within_multi_index_groups, depends_on)

    def expand_function_call(self, call, index_registry, loop_indices):
        """
        Turn an exprs.FunctionCall into a series of assignment instructions etc.
        Handles packing/accessor logic.
        """

        temporaries = {}
        for arg in call.arguments:
            # create an appropriate temporary
            # temporary = self._temp_name_generator.next()
            dims = self._construct_temp_dims(arg.tensor.axes, arg.tensor.indices, index_registry, loop_indices)
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

    def _construct_temp_dims(self, axis, multi_indices, index_registry, loop_indices):
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

        assert not isinstance(multi_indices, MultiIndexCollection)
        assert all(isinstance(item, MultiIndexCollection) for item in multi_indices)

        ###


        # each multi-index collection yields an adjacent axis part
        # e.g. for mixed (with 2 spaces) you would have a temporary with 2 parts
        # similarly this would be the expected behaviour for interior facets of extruded
        # meshes - the outer axis would be split in two parts because the DoFs come from
        # both the vertical and horizontal facets and these require separate multi-indices
        axis_parts = []
        for multi_idxs in multi_indices:
            axis_part = self.make_temp_axis_part_per_multi_index_collection(axis, multi_idxs, index_registry, loop_indices)
            axis_parts.append(axis_part)

        # import pdb; pdb.set_trace()

        return MultiAxis(axis_parts).set_up()

    def make_temp_axis_part_per_multi_index_collection(
            self, axis, multi_indices, index_registry, loop_indices, return_final_part_id=False):
        current_array_axis = axis
        top_temp_part = None
        current_bottom_part_id = None
        for multi_idx in multi_indices:
            temp_axis_part, bottom_part_id = self.make_temp_axes_per_multi_index(multi_idx, index_registry, loop_indices)

            if not top_temp_part:
                top_temp_part = temp_axis_part
                current_bottom_part_id = bottom_part_id
            else:
                top_temp_part = top_temp_part.add_subaxis(current_bottom_part_id, MultiAxis([temp_axis_part]))
                current_bottom_part_id = bottom_part_id

            if isinstance(multi_idx, Map):
                if len(multi_idx.paths) > 1:
                    raise NotImplementedError("need to branch recurse here")

                # need this loop since a map can yield multiple axis parts
                for part_label in multi_idx.paths[0].to_axes:
                    current_array_axis = current_array_axis._parts_by_label[part_label].subaxis
            else:
                assert isinstance(multi_idx, TerminalMultiIndex)
                for typed_idx in multi_idx:
                    current_array_axis = current_array_axis._parts_by_label[typed_idx.part_label].subaxis

        # axis might not be at the bottom so we'd need to shove on some extra shape
        # TODO: untested if multi-part
        if current_array_axis:
            # oh this is elegant
            top_temp_part = top_temp_part.add_subaxis(current_bottom_part_id, current_array_axis)

        return top_temp_part

    def make_temp_axes_per_multi_index(self, multi_index, index_registry, loop_indices):

        top_part = None
        current_bottom_part_id = None

        if multi_index in loop_indices:
            if isinstance(multi_index, Map):
                return self.make_temp_axes_per_multi_index(multi_index.from_multi_index, index_registry, loop_indices)
            else:
                assert isinstance(multi_index, TerminalMultiIndex)
                for typed_idx in multi_index:
                    new_part = AxisPart(typed_idx.size, indexed=True, label=typed_idx.part_label)

                    if not top_part:
                        top_part = new_part
                        current_bottom_part_id = new_part.id
                    else:
                        top_part = top_part.add_subaxis(current_bottom_part_id, MultiAxis([new_part]))
                        current_bottom_part_id = new_part.id
        else:
            raise NotImplementedError

        return top_part, current_bottom_part_id

    def make_gathers(self, temporaries, **kwargs):
        return tuple(
            self.make_gather(arg, temp, idxs, **kwargs)
            for arg, temp in temporaries.items()
            for idxs in arg.tensor.indices
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
                for idxs in arg.tensor.indices
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
    def _(self, call: tlang.FunctionCall, index_registry, loop_indices):
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
        inames = []
        within_inames = set()
        for idx in loop_indices:
            within_inames_, = index_registry[idx]
            inames.extend(within_inames_)
            within_inames |= {*within_inames_}
        within_inames = frozenset(within_inames)
        depends_on = frozenset()  # might be wrong
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            extents |= self.collect_extents(
                temp.root,
                inames,
                within_inames,
                depends_on,
            )

        # NOTE: If we register an extent to pass through loopy will complain
        # unless we register it as an assumption of the local kernel (e.g. "n <= 3")

        assignees = tuple(subarrayrefs[var] for var in call.writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(subarrayrefs[var] for var in call.reads) + tuple(extents.values()),
        )

        insn_id = f"{call.id}_0"
        depends_on = frozenset({f"{id}_*" for id in call.depends_on})
        within_inames = self.collect_within_inames(MultiIndexCollection([]), index_registry, loop_indices)

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

    def collect_extents(self, axis, inames, within_inames, depends_on, depth=0):
        extents = {}

        for part in axis.parts:
            if isinstance(part.count, MultiArray) and not part.indexed:
                assert depth >= 1
                extent = self.register_scalar_assignment(part.count, inames[:depth], within_inames, depends_on)
                extents[part.count] = extent

            if part.subaxis:
                extents_ = self.collect_extents(part.subaxis, inames, within_inames | {inames[depth]}, depends_on, depth+1)
                if any(array in extents for array in extents_):
                    raise ValueError("subaxes should not be using the same nnz structures (I think)")
                extents |= extents_

        return extents

    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, within_multi_index_groups, depends_on):
        for updated_within_migs, updated_deps in \
                self.generate_index_insns([assignment.lhs.indices, assignment.rhs.indices],
                                          within_multi_index_groups, depends_on):

            """
            at this point should have registered all of the different ways we can have loops

            the bit below just needs to connect the indices to the axis parts to emit the
            right layout instructions and then the right assignment.
            """

            raise NotImplementedError

            self.emit_assignment_insn(
                assignment,
                array_part_labels+array_pt_labels,
                temp_part_labels+temp_pt_labels,
                array_inames+array_jnames,
                temp_inames+temp_jnames,
                within_inames|within, depends_on|deps)

    def emit_assignment_insn(
            self, assignment,
            array_pt_labels, temp_pt_labels,
            array_jnames, temp_jnames,
            within_inames, depends_on,
            scalar=False
    ):

        # layout instructions - must be emitted innermost to make sense (reset appropriately)
        array_offset = f"{assignment.array.name}_ptr"
        temp_offset = f"{assignment.temporary.name}_ptr"
        self._temp_kernel_data.extend([
            lp.TemporaryVariable(name, shape=(), dtype=np.uintp)
            for name in [array_offset, temp_offset]
        ])
        array_offset_insn = lp.Assignment(
            pym.var(array_offset), 0,
            id=self._namer.next("insn"),
            within_inames=within_inames,
            depends_on=depends_on,
        )
        temp_offset_insn = lp.Assignment(
            pym.var(temp_offset), 0,
            id=self._namer.next("insn"),
            within_inames=within_inames,
            depends_on=depends_on,
        )
        self.instructions.append(array_offset_insn)
        self.instructions.append(temp_offset_insn)
        depends_on |= {array_offset_insn.id, temp_offset_insn.id}

        array_axis = assignment.array.axes
        for i, pt_label in enumerate(array_pt_labels):
            array_npart = [pt.label for pt in array_axis.parts].index(pt_label)
            array_part = array_axis.parts[array_npart]

            deps = self.emit_layout_insns(
                array_part.layout_fn,
                array_offset, array_jnames[:i+1],
                within_inames, depends_on,
            )
            depends_on |= deps
            array_axis = array_part.subaxis

        if not scalar:
            temp_axis = assignment.temporary.axes
            for i, pt_label in enumerate(temp_pt_labels):
                temp_npart = [pt.label for pt in temp_axis.parts].index(pt_label)
                temp_part = temp_axis.parts[temp_npart]

                deps = self.emit_layout_insns(
                    temp_part.layout_fn,
                    temp_offset, temp_jnames[:i+1],
                    within_inames, depends_on,
                )
                depends_on |= deps
                temp_axis = temp_part.subaxis

        self.generate_assignment_insn_inner(
            assignment, assignment.temporary, temp_offset,
            assignment.tensor, array_offset,
            scalar=scalar, depends_on=depends_on, within_inames=within_inames)

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

    def emit_layout_insns(self, layout_fn, offset_var, jnames, within_inames, depends_on):
        """
        TODO
        """
        if layout_fn == "null layout":
            return frozenset()

        # TODO singledispatch!
        if isinstance(layout_fn, IndirectLayoutFunction):
            # add 1 to depth here since we want to use the actual index for these, but not for start
            layout_var = self.register_scalar_assignment(layout_fn.data, jnames, within_inames, depends_on)
            expr = pym.var(offset_var)+layout_var

            # register the data
            # done in register_scalar_assignment?
            # layout_arg = lp.GlobalArg(layout_fn.data.name, np.uintp, (None,), is_input=True, is_output=False)
            # self._tensor_data[layout_fn.data.name] = layout_arg
        elif isinstance(layout_fn, AffineLayoutFunction):
            start = layout_fn.start
            step = layout_fn.step

            if isinstance(start, MultiArray):
                # drop the last jname
                start = self.register_scalar_assignment(layout_fn.start, jnames[:-1], within_inames, depends_on)

            jname = pym.var(jnames[-1])
            expr = pym.var(offset_var) + jname*step + start
        else:
            raise NotImplementedError

        insn = lp.Assignment(
            offset_var, expr,
            id=self._namer.next("insn"),
            within_inames=within_inames,
            depends_on=depends_on,
        )

        self.instructions.append(insn)

        return frozenset({insn.id})

    def generate_assignment_insn_inner(
            self,
            assignment,
            temp,
            toffset,
            array,
            aoffset,
            scalar,
            depends_on,
            within_inames,
        ):
        # the final instruction needs to come after all offset-determining instructions

        # # handle scalar assignment (for loop bounds)
        # if scalar:
        #     if assignment.lhs == assignment.temporary:
        #         loffset = None
        #     else:
        #         assert assignment.rhs == assignment.temporary
        #         roffset = None

        if scalar:
            texpr = pym.var(temp.name)
        else:
            texpr = pym.subscript(pym.var(temp.name), pym.var(toffset))

        aexpr = pym.subscript(pym.var(array.name), pym.var(aoffset))

        if assignment.array is assignment.lhs:
            lexpr = aexpr
            rexpr = texpr
        else:
            lexpr = texpr
            rexpr = aexpr

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

        insn_id = self._namer.next(f"{assignment.id}_")
        # depends_on = frozenset({f"{id}_*" for id in assignment.depends_on})
        # if self._active_insn.id in self._latest_insn:
        #     depends_on |= {self._latest_insn[self._active_insn.id]}

        assign_insn = lp.Assignment(
            lexpr, rexpr,
            id=insn_id,
            within_inames=frozenset(within_inames),
            within_inames_is_final=True,
            depends_on=depends_on,
            # no_sync_with=no_sync_with,
        )
        self.instructions.append(assign_insn)

    def generate_index_insns(
            self,
            indicess,  # iterable of an iterable of multi-index groups
            within_multi_index_groups,
            depends_on):

        if not utils.is_single_valued(len(idxs) for idxs in indicess):
            raise NotImplementedError("Need to be clever about having different lengths"
                                      "of indices for LHS and RHS")

        # this is a zip
        current_index_groups = []
        later_index_groupss = []
        for indices in indicess:
            current_group, *later_groups = indices
            current_index_groups.append(current_group)
            later_index_groupss.append(later_groups)

        state = []
        expansion = self.expand_multi_index_group(
            current_index_groups, within_multi_index_groups, depends_on)
        for updated_within_migs, updated_deps in expansion:
            subresult = self.generate_index_insns(
                later_index_groupss,
                updated_within_migs, updated_deps,
            )
            state.extend(subresult)
        return tuple(state)

    def expand_multi_index_group(
            self, multi_index_group, within_multi_index_groups, depends_on, prefix="mig"):
        """
        N.B. multi_index_group can actually be an iterable of LHS, RHS etc

        Deciding what to do when we have unit-sized axes is *hard*.

        E.g. mdat[0, :] = dat[:] should be valid
         or dat[p, :] = t[:] (i.e. ignore outer axis for rhs traversal)
        """
        if multi_index_group in within_multi_index_groups:
            return within_multi_index_groups, depends_on

        if isinstance(multi_idx, Map):
            # map expression
            if len(multi_idx.data) > 1:
                raise NotImplementedError("should really attach data to paths")

            result = []
            subresult = self.expand_multi_index_group(
                multi_idx.from_index,
                within_multi_index_groups,
                depends_on
            )
            for updated_within_migs, updated_deps in subresult:
                # maybe one could yield multiple paths - similar to the non-map case
                selected_path, = [path for path in multi_idx.paths if path.from_axes == aplabels]

                if selected_path.selector:
                    raise NotImplementedError("This is the right place to split codegen")

                # register a new loop
                iname = self._namer.next("i")
                extent = self.register_extent(
                    selected_path.arity,
                    ???,
                    updated_deps,
                )
                domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
                self.domains.append(domain_str)

                index_info = tuple(
                    (label, self._namer.next(f"{prefix}_j"))
                    for label in selected_path.to_axes)


                # FIXME This is specific to a TabulatedMap
                expr, deps = self.register_scalar_assignment(multi_idx.data[0], ajnames, within_inames, depends_on)

                index_insn = lp.Assignment(
                    pym.var(array_jname), expr,
                    id=self._namer.next(f"{assignment.id}_"),
                    within_inames=within_inames,
                    depends_on=updated_deps|deps,
                    # no_sync_with=no_sync_with,
                )
                self.instructions.append(index_insn)
                self._temp_kernel_data.extend([
                    lp.TemporaryVariable(name, shape=(), dtype=np.uintp)
                    for name in [array_jname]
                ])
                new_deps = frozenset({index_insn.id})

                within_migs = updated_within_migs | {multi_index_group: (index_info, {iname})}

                newthing = (within_migs, updated_deps|new_deps)
                result.append(newthing)
            return tuple(result)
        else:
            # if we aren't a map then iterate over all paths
            result = []
            for multi_idx in multi_index_group.multi_indices:
                res = self.register_terminal_multi_index(
                    multi_idx,
                    within_multi_index_groups,
                    depends_on,
                )
                result.append(res)
            return tuple(result)

    def register_terminal_multi_index(self, assignment, multi_idx, inames, within_inames, depends_on):
        # Note: At the moment we are assuming that all of the loops have already
        # been registered

        assert len(inames) == len(multi_idx)

        temp_pt_labels = []
        temp_jnames = []
        array_pt_labels = []
        array_jnames = []
        for typed_idx, iname in utils.checked_zip(multi_idx, inames):
            array_jname = self._namer.next(f"{assignment.array.name}_j")

            index_insn = lp.Assignment(
                # pym.var(array_iname), pym.var(iname)*typed_idx.step+typed_idx.start,
                pym.var(array_jname), pym.var(iname),
                id=self._namer.next(f"{assignment.id}_"),
                within_inames=within_inames,
                depends_on=depends_on,
                # no_sync_with=no_sync_with,
            )

            self.instructions.append(index_insn)
            new_deps = frozenset({index_insn.id})

            self._temp_kernel_data.extend([
                lp.TemporaryVariable(name, shape=(), dtype=np.uintp)
                for name in [array_jname]
            ])

            array_pt_labels.append(typed_idx.part_label)
            array_jnames.append(array_jname)

            temp_pt_labels.append(typed_idx.part_label)
            temp_jnames.append(iname)
        return tuple(temp_pt_labels), tuple(array_pt_labels), tuple(temp_jnames), tuple(array_jnames), within_inames, new_deps

    def register_extent(self, extent, inames, within_inames, depends_on):
        if isinstance(extent, MultiArray):
            temp_var = self.register_scalar_assignment(extent, inames, within_inames, depends_on)
            return str(temp_var)
        else:
            return extent

    def register_scalar_assignment(self, array, inames, within_inames, depends_on):
        temp_name = self._namer.next("n")
        # need to create a scalar multi-axis with the same depth
        # TODO is it not better to just "fully index" this thing?
        array_axis = array.axes
        tempid = "mytempid" + str(0)
        tempaxis = MultiAxis([AxisPart(1, id=tempid, label=array_axis.part.label)])
        oldtempid = tempid

        array_part_labels = [array_axis.part.label]
        temp_part_labels = [array_axis.part.label]

        for d in range(1, array.depth):
            tempid = "mytempid" + str(d)
            subaxis = MultiAxis([AxisPart(1, id=tempid, label=array_axis.part.label)])
            tempaxis = tempaxis.add_subaxis(oldtempid, subaxis)
            oldtempid = tempid

            array_part_labels.append(array_axis.part.label)
            temp_part_labels.append(array_axis.part.label)

            array_axis = array_axis.part.subaxis

        temp = MultiArray.new(
            tempaxis.set_up(),
            name=temp_name,
            dtype=np.int32,
        )

        # make sure to use the right inames
        if array.depth == 1:
            inames = [inames[-1]]
        else:
            # we can only index things with all or one of the inames
            assert len(inames) == array.depth

        indices = "not a thing here"
        insn = tlang.Read(array, temp, indices, depends_on=depends_on)

        self.emit_assignment_insn(
            insn, array_part_labels, temp_part_labels, inames, None,
            within_inames=within_inames,
            depends_on=depends_on,
            scalar=True
        )

        return pym.var(temp_name)


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

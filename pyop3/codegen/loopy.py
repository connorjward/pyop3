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
from pyop3.tensors import MultiIndexCollection, MultiIndex, TypedIndex, AffineLayoutFunction, IndirectLayoutFunction, Range, Path, TabulatedMap, MapNode, TabulatedMapNode, RangeNode


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


def drop_within(myindex, within_indices):
    """So the idea here is that this index has multiple children so we can't
    just add a new path entry and move to the child. We need to know which
    child to move to. Fortunately we have already decided which child we
    need in the loop bit above. We have registered the "selected path" of
    this index inside of within_indices.
    """
    if myindex is None:
        return None, PrettyTuple(), PrettyTuple()

    # import pdb; pdb.set_trace()
    mylabels, myjnames = [], []
    while myindex.id in within_indices:
        labels, jnames = within_indices[myindex.id]
        if isinstance(myindex, RangeNode):
            mylabels.extend(labels)
            myjnames.extend(jnames)
        else:
            mylabels = mylabels[:-len(myindex.from_labels)] + list(labels)
            myjnames = myjnames[:-len(myindex.from_labels)] + list(jnames)

        if len(myindex.children) == 0:
            return None, PrettyTuple(mylabels), PrettyTuple(myjnames)

        if isinstance(myindex, MapNode):
            to_labels = myindex.to_labels
        else:
            to_labels = ((myindex.label,),)

        newindex = None
        for ch, lbls in checked_zip(myindex.children, to_labels):
            if lbls == labels:
                assert newindex is None
                newindex = ch
        assert newindex is not None
        myindex = newindex

    return myindex, PrettyTuple(mylabels), PrettyTuple(myjnames)


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
    def _(self, loop: exprs.Loop, within_multi_index_groups=None, within_inames=frozenset(), depends_on=frozenset()):
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

        for index in loop.index:
            self.build_loop(
                    loop, index, within_multi_index_groups, within_inames, depends_on)

    def build_loop(self, loop, index, within_indices, within_inames, depends_on, existing_labels=PrettyTuple(), existing_jnames=PrettyTuple()):
        """
        note: there is no need to track a current axis here. We just need to register
        loops and associated inames. We also need part labels because it informs
        the maps what to do.

        The lack of axes distinguishes this function from the one needed for assignments?

        The only difference I can see is that here we are registering within_migs as we go
        (which is likely fine to do for the other case), and that the final action is different.

        Tree visitor approach is nice. Pass some callable to execute right at the bottom.
        """
        if isinstance(index, MapNode) and index.selector:
            raise NotImplementedError

        iname = self._namer.next("i")
        extent = self.register_extent(index.size, within_indices, within_inames, depends_on)
        domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
        self.domains.append(domain_str)

        # set these below (singledispatch me)
        index_insns = None
        new_labels = None
        new_jnames = None
        jnames = None
        if isinstance(index, RangeNode):
            jname = self._namer.next("j")
            self._temp_kernel_data.append(
                lp.TemporaryVariable(jname, shape=(), dtype=np.uintp))

            index_insn = lp.Assignment(
                pym.var(jname), pym.var(iname)*index.step+index.start,
                id=self._namer.next("myid_"),
                within_inames=within_inames,
                depends_on=depends_on,
                # no_sync_with=no_sync_with,
            )
            index_insns = [index_insn]
            new_labels = existing_labels | index.label
            new_jnames = existing_jnames | jname
            jnames = (jname,)
            new_within = {index.id: ((index.label,), (jname,))}
        elif isinstance(index, TabulatedMapNode):
            # NOTE: some maps can produce multiple jnames (but not this one)
            jname = self._namer.next("j")
            self._temp_kernel_data.append(
                lp.TemporaryVariable(jname, shape=(), dtype=np.uintp))

            temp_labels = list(existing_labels)
            temp_jnames = list(existing_jnames)
            assert len(index.from_labels) == 1
            assert len(index.to_labels) == 1
            for label in index.from_labels:
                assert temp_labels.pop() == label
                temp_jnames.pop()

            to_label, = index.to_labels
            new_labels = existing_labels | to_label
            new_jnames = existing_jnames | jname
            jnames = (jname,)
            new_within = {index.id: ((to_label,), (jname,))}

            expr = self.register_scalar_assignment(
                index.data, new_labels, new_jnames, within_inames, depends_on)

            index_insn = lp.Assignment(
                pym.var(jname),
                expr,
                id=self._namer.next("myid_"),
                within_inames=within_inames,
                depends_on=depends_on)

            index_insns = [index_insn]
        else:
            raise AssertionError

        assert index_insns is not None
        assert new_labels is not None
        assert new_jnames is not None
        assert jnames is not None
        self.instructions.extend(index_insns)
        new_deps = frozenset({insn.id for insn in index_insns})

        if index.children:
            for subindex in index.children:
                self.build_loop(
                    loop, subindex,
                    within_indices|new_within,
                    within_inames|{iname},
                    depends_on|new_deps,
                    new_labels, new_jnames)
        else:
            for stmt in loop.statements:
                self._build(
                    stmt, within_indices|new_within, within_inames|{iname}, depends_on|new_deps)


    def build_assignment(self, assignment, lindex, rindex, within_indices, within_inames,
                         depends_on,
                         llabels=PrettyTuple(),
                         ljnames=PrettyTuple(),
                         rlabels=PrettyTuple(),
                         rjnames=PrettyTuple()):
        """
        Note: lindex and rindex can be different lengths if we include within_indices
        """

        # catch within_indices
        # ah - need to keep tracking the right part labels though!
        import pdb; pdb.set_trace()
        lindex, innerllabels, innerljnames = drop_within(lindex, within_indices)
        rindex, innerrlabels, innerrjnames = drop_within(rindex, within_indices)


        llabels = PrettyTuple(llabels+innerllabels)
        ljnames = PrettyTuple(ljnames+innerljnames)
        rlabels = PrettyTuple(rlabels+innerrlabels)
        rjnames = PrettyTuple(rjnames+innerrjnames)

        iname = None
        if not strictly_all(idx is None for idx in [lindex, rindex]):
            size = utils.single_valued([lindex.size, rindex.size])

            iname = self._namer.next("i")
            extent = self.register_extent(size, within_indices, within_inames, depends_on)
            domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
            self.domains.append(domain_str)

            def myinnerfunc(index, existing_labels, existing_jnames):
                # set these below (singledispatch me)
                index_insns = None
                new_labels = None
                new_jnames = None
                jnames = None
                if isinstance(index, RangeNode):
                    jname = self._namer.next("j")
                    self._temp_kernel_data.append(
                        lp.TemporaryVariable(jname, shape=(), dtype=np.uintp))

                    index_insn = lp.Assignment(
                        pym.var(jname), pym.var(iname)*index.step+index.start,
                        id=self._namer.next("myid_"),
                        within_inames=within_inames|{iname},
                        depends_on=depends_on,
                        # no_sync_with=no_sync_with,
                    )
                    index_insns = [index_insn]
                    new_labels = existing_labels | index.label
                    new_jnames = existing_jnames | jname
                    jnames = (jname,)
                    new_within = {index.id: ((index.label,), (jname,))}
                elif isinstance(index, TabulatedMapNode):
                    # NOTE: some maps can produce multiple jnames (but not this one)
                    jname = self._namer.next("j")
                    self._temp_kernel_data.append(
                        lp.TemporaryVariable(jname, shape=(), dtype=np.uintp))

                    # FIXME use labels and jnames?
                    raise NotImplementedError
                    expr, deps = self.register_scalar_assignment(
                        index.data, path, within_inames, depends_on)

                    index_insn = lp.Assignment(
                        pym.var(jname),
                        expr,
                        id=self._namer.next("myid_"),
                        within_inames=within_inames,
                        depends_on=depends_on)

                    index_insns = [index_insn]

                    temp_labels = list(existing_labels)
                    temp_jnames = list(existing_jnames)
                    assert len(index.from_labels) == 1
                    assert len(index.to_labels) == 1
                    for label in index.from_labels:
                        assert temp_labels.pop() == label
                        temp_jnames.pop()

                    to_label, = index.to_labels
                    new_path = PrettyTuple(temp_path) | (to_label, jname)
                    jnames = (jname,)
                    new_within = {index.id: ((to_label,), (jname,))}
                else:
                    raise AssertionError

                assert index_insns is not None
                assert new_labels is not None
                assert new_jnames is not None
                assert jnames is not None
                self.instructions.extend(index_insns)
                new_deps = frozenset({insn.id for insn in index_insns})

                return new_labels, new_jnames, new_within, new_deps

            lthings = myinnerfunc(lindex, llabels, ljnames)
            rthings = myinnerfunc(rindex, rlabels, rjnames)

            if strictly_all([lindex.children, rindex.children]):
                for lsubindex, rsubindex in checked_zip(lindex.children, rindex.children):
                    self.build_assignment(
                        assignment, lsubindex, rsubindex,
                        within_indices|lthings[2]|rthings[2],
                        within_inames|{iname},
                        depends_on|lthings[3]|rthings[3],
                        lthings[0], lthings[1], rthings[0], rthings[1])

                terminate = False
            else:
                terminate = True
        else:
            lthings = [llabels, ljnames]  # other indices arent needed
            rthings = [rlabels, rjnames]  # other indices arent needed
            terminate = True

        if terminate:
            lhs_part_labels, lhs_jnames = lthings[0], lthings[1]
            rhs_part_labels, rhs_jnames = rthings[0], rthings[1]

            if assignment.lhs is assignment.array:
                array_part_labels = lhs_part_labels
                array_jnames = lhs_jnames
                temp_part_labels = rhs_part_labels
                temp_jnames = rhs_jnames
            else:
                temp_part_labels = lhs_part_labels
                temp_jnames = lhs_jnames
                array_part_labels = rhs_part_labels
                array_jnames = rhs_jnames

            # import pdb; pdb.set_trace()
            extra_inames = {iname} if iname else set()
            extra_deps = frozenset({f"{id}_*" for id in assignment.depends_on})
            self.emit_assignment_insn(
                assignment,
                array_part_labels,
                temp_part_labels,
                array_jnames,
                temp_jnames,
                within_inames|extra_inames,
                depends_on|extra_deps)


    @_build.register
    def _(self, call: exprs.FunctionCall, within_multi_index_groups, within_inames, depends_on):
        # I think I'd prefer to do this in a separate earlier pass?
        insns = self.expand_function_call(call, within_multi_index_groups, depends_on)

        for insn in insns:
            self._make_instruction_context(insn, within_multi_index_groups, within_inames, depends_on)

    def expand_function_call(self, call, within_indices, depends_on):
        """
        Turn an exprs.FunctionCall into a series of assignment instructions etc.
        Handles packing/accessor logic.
        """

        temporaries = {}
        for arg in call.arguments:
            # create an appropriate temporary
            # temporary = self._temp_name_generator.next()
            parts = []
            indices = []
            for idx in arg.tensor.indices:
                new_part, newidx = self._construct_temp_dims(idx, within_indices)
                parts.append(new_part)
                indices.append(newidx)

            # I don't think we can ever get some None and some not
            if strictly_all(part is None for part in parts):
                assert all(idx is None for idx in indices)
                axes = None
                indices = ()
            else:
                axes = MultiAxis(parts).set_up()

            temporary = MultiArray.new(
                axes, indices=indices, name=self._temp_name_generator.next(), dtype=arg.tensor.dtype,
            )
            temporaries[arg] = temporary

        gathers = self.make_gathers(temporaries)
        newcall = self.make_function_call(
            call, temporaries,
            depends_on=frozenset(gather.id for gather in gathers)
        )
        scatters = self.make_scatters(temporaries, depends_on=frozenset({newcall.id}))

        return (*gathers, newcall, *scatters)

    def _construct_temp_dims(self, index, within_indices):
        """Return a multi-axis describing the temporary shape.

        Parameters
        ----------
        multi_indices : ???
            Iterable of :class:`MultiIndex`.

        Note: an index roughly corresponds to an axis part so we return single axis
        parts from this function
        """
        # import pdb; pdb.set_trace()
        index, newlabels, newjnames = drop_within(index, within_indices)

        # scalar case
        if index is None:
            return None, None

        if len(index.children) > 0:
            subparts = []
            new_subidxs = []
            for subidx in index.children:
                subpart, new_subidx = self._construct_temp_dims(subidx, within_indices)
                subparts.append(subpart)
                new_subidxs.append(new_subidx)
            subaxis = MultiAxis(subparts)
        else:
            subaxis = None
            new_subidxs = []

        label = self._namer.next("mylabel")
        new_part = AxisPart(index.size, label=label, subaxis=subaxis)
        new_index = RangeNode(label, index.size, children=new_subidxs)

        return new_part, new_index


    def make_temp_axis_part_per_multi_index_collection(
            self, axis, multi_indices, within_migs, return_final_part_id=False):
        current_array_axis = axis
        top_temp_part = None
        current_bottom_part_id = None
        for multi_idx in multi_indices:
            temp_axis_part, bottom_part_id = self.make_temp_axes_per_multi_index(multi_idx, within_migs)

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

    def make_gathers(self, temporaries, **kwargs):
        return tuple(
            self.make_gather(arg, temp, **kwargs)
            for arg, temp in temporaries.items()
        )

    def make_gather(self, argument, temporary, **kwargs):
        # TODO cleanup the ids
        if argument.access in {exprs.READ, exprs.RW}:
            return tlang.Read(argument.tensor, temporary, **kwargs)
        elif argument.access in {exprs.WRITE, exprs.INC}:
            return tlang.Zero(argument.tensor, temporary, **kwargs)
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
                self.make_scatter(arg, temp, **kwargs)
                for arg, temp in temporaries.items()
            )
        ))

    def make_scatter(self, argument, temporary, **kwargs):
        if argument.access == exprs.READ:
            return None
        elif argument.access in {exprs.WRITE, exprs.RW}:
            return tlang.Write(argument.tensor, temporary, **kwargs)
        elif argument.access == exprs.INC:
            return tlang.Increment(argument.tensor, temporary, **kwargs)
        else:
            raise AssertionError

    @functools.singledispatchmethod
    def _make_instruction_context(self, instruction: tlang.Instruction, *args, **kwargs):
        raise TypeError


    @_make_instruction_context.register
    def _(self, call: tlang.FunctionCall, within_indices, within_inames, depends_on):
        # import pdb; pdb.set_trace()

        subarrayrefs = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            temp_size = temp.alloc_size
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp, iname)
            self.domains.append(f"{{ [{iname}]: 0 <= {iname} < {temp_size} }}")
            assert temp.name in [d.name for d in self._temp_kernel_data]

        # we need to pass sizes through if they are only known at runtime (ragged)
        extents = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            for index in temp.indices:
                extents |= self.collect_extents(index, within_indices, within_inames, depends_on)

        # NOTE: If we register an extent to pass through loopy will complain
        # unless we register it as an assumption of the local kernel (e.g. "n <= 3")

        assignees = tuple(subarrayrefs[var] for var in call.writes)
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(subarrayrefs[var] for var in call.reads) + tuple(extents.values()),
        )

        insn_id = f"{call.id}_0"
        depends_on = frozenset({f"{id}_*" for id in call.depends_on})

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

    def collect_extents(self, index, within_indices, within_inames, depends_on):
        extents = {}

        if isinstance(index.size, MultiArray):
            # TODO This will overwrite if we have duplicates
            extent = self.register_extent(index.size, within_indices, within_inames, depends_on)
            extents[index.size] = pym.var(extent)

        if index.children:
            for child in index.children:
                extents |= self.collect_extents(child, within_indices, within_inames, depends_on)

        return extents

    @_make_instruction_context.register
    def _(self, assignment: tlang.Assignment, within_indices, within_inames, depends_on):
        # TODO are these always the same length?
        lindices = assignment.lhs.indices or (None,)
        rindices = assignment.rhs.indices or (None,)

        for lindex, rindex in checked_zip(lindices, rindices):
            self.build_assignment(assignment, lindex, rindex, within_indices, within_inames, depends_on)

    def emit_assignment_insn(
            self, assignment,
            array_pt_labels, temp_pt_labels,
            array_jnames, temp_jnames,
            within_inames, depends_on,
            scalar=False
    ):

        # layout instructions - must be emitted innermost to make sense (reset appropriately)
        array_offset = self._namer.next(f"{assignment.array.name}_ptr")
        temp_offset = self._namer.next(f"{assignment.temporary.name}_ptr")
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
                array_part.layout_fn, array_offset,
                array_pt_labels[:i+1], array_jnames[:i+1],
                within_inames, depends_on,
            )
            depends_on |= deps
            array_axis = array_part.subaxis

        assert array_axis is None

        if not scalar:
            temp_axis = assignment.temporary.axes
            for i, pt_label in enumerate(temp_pt_labels):
                temp_npart = [pt.label for pt in temp_axis.parts].index(pt_label)
                temp_part = temp_axis.parts[temp_npart]

                deps = self.emit_layout_insns(
                    temp_part.layout_fn,
                    temp_offset, temp_pt_labels[:i+1], temp_jnames[:i+1],
                    within_inames, depends_on,
                )
                depends_on |= deps
                temp_axis = temp_part.subaxis

            # TODO make scalar stuff work a lot better
            # assert temp_axis is None

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
                shape = (unindexed.alloc_size,)
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

    def emit_layout_insns(self, layout_fn, offset_var, part_labels, jnames, within_inames, depends_on):
        """
        TODO
        """
        if layout_fn == "null layout":
            return frozenset()

        # TODO singledispatch!
        if isinstance(layout_fn, IndirectLayoutFunction):
            # we can either index with just the lowest index or all of them
            if layout_fn.data.depth == 1:
                mylabels = part_labels[-1:]
                myjnames = jnames[-1:]
            else:
                assert layout_fn.data.depth == utils.single_valued(map(len, [part_labels, jnames]))
                mylabels = part_labels
                myjnames = jnames

            layout_var = self.register_scalar_assignment(
                layout_fn.data, mylabels, myjnames, within_inames, depends_on)
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
                start = self.register_scalar_assignment(
                    layout_fn.start, part_labels[:-1], jnames[:-1], within_inames, depends_on)

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

    def register_extent(self, extent, within_indices, within_inames, depends_on):
        if isinstance(extent, MultiArray):
            labels, jnames = [], []
            index, = extent.indices
            while True:
                new_labels, new_jnames = within_indices[index.id]
                labels.extend(new_labels)
                jnames.extend(new_jnames)
                if index.children:
                    index = index.child  # must be linear
                else:
                    break

            temp_var = self.register_scalar_assignment(
                extent, labels, jnames, within_inames, depends_on)
            return str(temp_var)
        else:
            return extent

    def register_scalar_assignment(self, array, part_labels, jnames, within_inames, depends_on):
        temp_name = self._namer.next("n")
        temp = MultiArray(None, name=temp_name, dtype=np.int32)
        insn = tlang.Read(array, temp)

        temp_pt_labels = ()
        temp_jnames = ()

        # TODO Think about dependencies
        self.emit_assignment_insn(
            insn, part_labels, temp_pt_labels, jnames, temp_jnames,
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

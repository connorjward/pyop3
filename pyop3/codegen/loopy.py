import abc
import collections
import copy
import dataclasses
import enum
import functools
import itertools
import numbers
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import loopy as lp
import loopy.symbolic
import numpy as np
import pymbolic as pym
import pytools

from pyop3 import exprs, tlang, utils
from pyop3.distarray.multiarray import MultiArray
from pyop3.multiaxis import (
    AffineLayoutFunction,
    AffineMapNode,
    AxisPart,
    IdentityMapNode,
    IndexTree,
    IndirectLayoutFunction,
    MapNode,
    MultiAxis,
    MultiAxisComponent,
    MultiAxisTree,
    RangeNode,
    TabulatedMapNode,
)
from pyop3.tree import NullRootNode, NullRootTree
from pyop3.utils import (
    MultiNameGenerator,
    NameGenerator,
    PrettyTuple,
    StrictlyUniqueSet,
    checked_zip,
    just_one,
    strictly_all,
)


class VariableCollector(pym.mapper.Collector):
    def map_variable(self, expr, *args, **kwargs):
        return {expr}


# @dataclasses.dataclass(frozen=True)
# class CodegenContext:
#     indices:


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
        return (
            list(self._tensor_data.values())
            + self._section_data
            + self._temp_kernel_data
        )

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
            options=lp.Options(check_dep_resolution=False),
        )
        tu = lp.merge((translation_unit, *self.subkernels))
        # import pdb; pdb.set_trace()
        return tu.with_entrypoints("mykernel")

    @functools.singledispatchmethod
    def _build(self, expr, *args, **kwargs):
        raise TypeError

    @_build.register
    def _(
        self,
        loop: exprs.Loop,
        within_indices=None,
        within_inames=frozenset(),
        depends_on=frozenset(),
    ):
        if not within_indices:
            within_indices = {}

        for index in loop.index.children(loop.index.root):
            self.build_loop(
                loop,
                loop.index,
                index,
                within_indices,
                within_inames,
                depends_on,
            )

    def build_loop(
        self,
        loop,
        itree,
        index,
        within_indices,
        within_inames,
        depends_on,
        existing_labels=PrettyTuple(),
        existing_jnames=PrettyTuple(),
    ):
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
        extent = self.register_extent(
            index.size, within_indices, within_inames, depends_on
        )
        domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
        self.domains.append(domain_str)

        # pass around mutable or immutable things?
        # new_labels and new_jnames are *new* objects that *replace* existing
        # new_within and new_deps are *added* to the existing ones
        # we are doing a tree traversal, what is the generic solution here?
        # probably always do a full replacement - that will work for both
        # wrap into an object?
        # don't tie it to the kernel builder as that could cause issues between forks
        new_labels, new_jnames, new_within, new_deps = self.myinnerfunc(
            iname,
            index,
            existing_labels,
            existing_jnames,
            within_indices,
            within_inames,
            depends_on,
        )

        if children := itree.children(index):
            for subindex in children:
                self.build_loop(
                    loop,
                    itree,
                    subindex,
                    within_indices | new_within,
                    within_inames | {iname},
                    depends_on | new_deps,
                    new_labels,
                    new_jnames,
                )
        else:
            for stmt in loop.statements:
                self._build(
                    stmt,
                    within_indices | new_within,
                    within_inames | {iname},
                    depends_on | new_deps,
                )

    def myinnerfunc(
        self,
        iname,
        index,
        existing_labels,
        existing_jnames,
        within_indices,
        within_inames,
        depends_on,
    ):
        # set these below (singledispatch me)
        index_insns = None
        new_labels = None
        new_jnames = None
        jnames = None
        new_within = None

        if index.id in within_indices:
            # import pdb; pdb.set_trace()

            labels, jnames = within_indices[index.id]
            if isinstance(index, RangeNode):
                index_insns = []
                new_labels = existing_labels + labels
                new_jnames = existing_jnames + jnames
                jnames = "not used"
                new_within = {}
            else:
                index_insns = []
                temp_labels = list(existing_labels)
                temp_jnames = list(existing_jnames)
                assert len(index.from_labels) == 1
                assert len(index.to_labels) == 1
                for label in index.from_labels:
                    assert temp_labels.pop() == label
                    temp_jnames.pop()

                new_labels = PrettyTuple(temp_labels) + labels
                new_jnames = PrettyTuple(temp_jnames) + jnames
                jnames = "not used"
                new_within = {}

        elif isinstance(index, RangeNode):
            jname = self._namer.next("j")
            self._temp_kernel_data.append(
                lp.TemporaryVariable(jname, shape=(), dtype=np.uintp)
            )

            index_insn = lp.Assignment(
                pym.var(jname),
                pym.var(iname) * index.step + index.start,
                id=self._namer.next("myid_"),
                within_inames=within_inames | {iname},
                depends_on=depends_on,
                # no_sync_with=no_sync_with,
            )
            index_insns = [index_insn]
            new_labels = existing_labels | index.label
            new_jnames = existing_jnames | jname
            jnames = (jname,)
            new_within = {index.id: ((index.label,), (jname,))}

        elif isinstance(index, IdentityMapNode):
            index_insns = []
            new_labels = existing_labels
            new_jnames = existing_jnames
            jnames = ()
            new_within = {}

        elif isinstance(index, AffineMapNode):
            jname = self._namer.next("j")
            self._temp_kernel_data.append(
                lp.TemporaryVariable(jname, shape=(), dtype=np.uintp)
            )

            subst_rules = {
                var: pym.var(j)
                for var, j in checked_zip(
                    index.expr[0][:-1],
                    existing_jnames[-len(index.from_labels) :],
                )
            }
            subst_rules |= {index.expr[0][-1]: pym.var(iname)}

            expr = pym.substitute(index.expr[1], subst_rules)

            index_insn = lp.Assignment(
                pym.var(jname),
                expr,
                id=self._namer.next("myid_"),
                within_inames=within_inames | {iname},
                depends_on=depends_on,
                # no_sync_with=no_sync_with,
            )
            index_insns = [index_insn]

            temp_labels = list(existing_labels)
            temp_jnames = list(existing_jnames)
            assert len(index.from_labels) == 1
            assert len(index.to_labels) == 1
            for label in index.from_labels:
                assert temp_labels.pop() == label
                temp_jnames.pop()

            (to_label,) = index.to_labels
            new_labels = PrettyTuple(temp_labels) | to_label
            new_jnames = PrettyTuple(temp_jnames) | jname
            jnames = (jname,)
            new_within = {index.id: ((to_label,), (jname,))}

        elif isinstance(index, TabulatedMapNode):
            # NOTE: some maps can produce multiple jnames (but not this one)
            jname = self._namer.next("j")
            self._temp_kernel_data.append(
                lp.TemporaryVariable(jname, shape=(), dtype=np.uintp)
            )

            # import pdb; pdb.set_trace()

            # find the right target label for the map (assume can't be multi-part)
            mapaxis = index.data.axes
            mappartid = NullRootNode.ID
            for l in existing_labels:
                (_part_,) = [pt for pt in mapaxis.children(mappartid) if l == pt.label]
                mappartid = _part_.id
            # not sure about this check
            # assert mapaxis.nparts == 1
            # assert not mapaxis.children(mappartid)
            (mypart,) = mapaxis.children(mappartid)
            map_labels = existing_labels | mypart.label
            map_jnames = existing_jnames | iname
            expr = self.register_scalar_assignment(
                index.data,
                dict(checked_zip(map_labels, map_jnames)),
                # map_labels,
                # map_jnames,
                within_inames | {iname},
                depends_on,
            )

            index_insn = lp.Assignment(
                pym.var(jname),
                expr,
                id=self._namer.next("myid_"),
                within_inames=within_inames | {iname},
                depends_on=depends_on,
            )

            index_insns = [index_insn]

            temp_labels = list(existing_labels)
            temp_jnames = list(existing_jnames)
            assert len(index.from_labels) == 1
            assert len(index.to_labels) == 1
            for label in index.from_labels:
                assert temp_labels.pop() == label
                temp_jnames.pop()

            (to_label,) = index.to_labels
            new_labels = PrettyTuple(temp_labels) | to_label
            new_jnames = PrettyTuple(temp_jnames) | jname
            jnames = (jname,)
            new_within = {index.id: ((to_label,), (jname,))}
        else:
            raise AssertionError

        assert index_insns is not None
        assert new_labels is not None
        assert new_jnames is not None
        assert jnames is not None
        assert new_within is not None
        self.instructions.extend(index_insns)
        new_deps = frozenset({insn.id for insn in index_insns})

        return new_labels, new_jnames, new_within, new_deps

    def build_assignment(
        self,
        assignment,
        lindex,
        rindex,
        within_indices,
        within_inames,
        depends_on,
        llabels=PrettyTuple(),
        ljnames=PrettyTuple(),
        rlabels=PrettyTuple(),
        rjnames=PrettyTuple(),
    ):
        lres = [(lindex, (), ())]
        rres = [(rindex, (), ())]

        for (lindex, innerllabels, innerljnames), (
            rindex,
            innerrlabels,
            innerrjnames,
        ) in utils.checked_zip(lres, rres):
            # import pdb; pdb.set_trace()

            llabels_ = PrettyTuple(llabels + innerllabels)
            ljnames_ = PrettyTuple(ljnames + innerljnames)
            rlabels_ = PrettyTuple(rlabels + innerrlabels)
            rjnames_ = PrettyTuple(rjnames + innerrjnames)

            iname = None
            # size = utils.single_valued([lindex.size, rindex.size])
            # now need to catch one-sized things here
            lsize = lindex.size if lindex.id not in within_indices else 1
            rsize = rindex.size if rindex.id not in within_indices else 1
            size = utils.single_valued([lsize, rsize])

            iname = self._namer.next("i")
            extent = self.register_extent(
                size, within_indices, within_inames, depends_on
            )
            domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
            self.domains.append(domain_str)

            lthings = self.myinnerfunc(
                iname,
                lindex,
                llabels_,
                ljnames_,
                within_indices,
                within_inames,
                depends_on,
            )
            rthings = self.myinnerfunc(
                iname,
                rindex,
                rlabels_,
                rjnames_,
                within_indices,
                within_inames,
                depends_on,
            )

            lchildren = assignment.lhs.index.children(lindex)
            rchildren = assignment.rhs.index.children(rindex)
            if strictly_all([lchildren, rchildren]):
                for lsubindex, rsubindex in checked_zip(lchildren, rchildren):
                    self.build_assignment(
                        assignment,
                        lsubindex,
                        rsubindex,
                        within_indices | lthings[2] | rthings[2],
                        within_inames | {iname},
                        depends_on | lthings[3] | rthings[3],
                        lthings[0],
                        lthings[1],
                        rthings[0],
                        rthings[1],
                    )

            else:
                lhs_part_labels, lhs_jnames = lthings[0], lthings[1]
                rhs_part_labels, rhs_jnames = rthings[0], rthings[1]

                # refactoring - part labels always match jnames - tie together?
                # map part label to jname?
                assert len(lhs_part_labels) == len(lhs_jnames)
                assert len(rhs_part_labels) == len(rhs_jnames)

                lhs_labels_to_jnames = {
                    l: j for l, j in checked_zip(lhs_part_labels, lhs_jnames)
                }
                rhs_labels_to_jnames = {
                    l: j for l, j in checked_zip(rhs_part_labels, rhs_jnames)
                }

                if assignment.lhs is assignment.array:
                    array_labels_to_jnames = lhs_labels_to_jnames
                    temp_labels_to_jnames = rhs_labels_to_jnames
                else:
                    temp_labels_to_jnames = lhs_labels_to_jnames
                    array_labels_to_jnames = rhs_labels_to_jnames

                extra_inames = {iname} if iname else set()
                extra_deps = frozenset({f"{id}_*" for id in assignment.depends_on})

                ###

                array_offset, array_deps = self.emit_assignment_insn(
                    assignment.array.data.name,
                    assignment.array.data.axes,
                    array_labels_to_jnames,
                    within_inames | extra_inames,
                    depends_on | extra_deps,
                )
                temp_offset, temp_deps = self.emit_assignment_insn(
                    assignment.temporary.data.name,
                    assignment.temporary.data.axes,
                    temp_labels_to_jnames,
                    within_inames | extra_inames,
                    depends_on | extra_deps,
                )

                array = assignment.array.data
                temporary = assignment.temporary.data
                temp_expr = pym.subscript(pym.var(temporary.name), pym.var(temp_offset))
                array_expr = pym.subscript(pym.var(array.name), pym.var(array_offset))

                if isinstance(assignment, tlang.Read):
                    lexpr = temp_expr
                    rexpr = array_expr
                elif isinstance(assignment, tlang.Write):
                    lexpr = array_expr
                    rexpr = temp_expr
                elif isinstance(assignment, tlang.Increment):
                    lexpr = array_expr
                    rexpr = array_expr + temp_expr
                elif isinstance(assignment, tlang.Zero):
                    lexpr = temp_expr
                    rexpr = 0
                else:
                    raise NotImplementedError

                self.generate_assignment_insn_inner(
                    lexpr,
                    rexpr,
                    assignment.id,
                    depends_on=depends_on | extra_deps | array_deps | temp_deps,
                    within_inames=within_inames | extra_inames,
                )

    @_build.register
    def _(
        self,
        call: exprs.FunctionCall,
        within_indices,
        within_inames,
        depends_on,
    ):
        # I think I'd prefer to do this in a separate earlier pass?
        insns = self.expand_function_call(call, within_indices, depends_on)

        for insn in insns:
            self._make_instruction_context(
                insn, within_indices, within_inames, depends_on
            )

    def expand_function_call(self, call, within_indices, depends_on):
        """
        Turn an exprs.FunctionCall into a series of assignment instructions etc.
        Handles packing/accessor logic.
        """

        temporaries = {}
        for arg in call.arguments:
            # create an appropriate temporary
            # we need the indices here because the temporary shape needs to be indexed
            # by the same indices as the original array
            # is this definitely true???
            if not isinstance(arg.tensor.data, MultiArray):
                raise NotImplementedError(
                    "Need to handle indices to create temp shape differently"
                )
            itree = IndexTree()
            axtree = MultiAxisTree()
            for idx in arg.tensor.index.children(arg.tensor.index.root):
                self._construct_temp_dims(
                    axtree, itree, arg.tensor.index, idx, within_indices
                )

            axtree.set_up()
            temporary = MultiArray(
                axtree,
                name=self._temp_name_generator.next(),
                dtype=arg.tensor.data.dtype,
            )
            temporaries[arg] = temporary[itree]
            # import pdb; pdb.set_trace()

        gathers = self.make_gathers(temporaries)
        newcall = self.make_function_call(
            call, temporaries, depends_on=frozenset(gather.id for gather in gathers)
        )
        scatters = self.make_scatters(temporaries, depends_on=frozenset({newcall.id}))

        return (*gathers, newcall, *scatters)

    def _construct_temp_dims(
        self,
        axtree,
        itree,
        currentindices,
        index,
        within_indices,
        path=None,
        iid=NullRootNode.ID,
    ):
        """Return an iterable of axis parts per index

        Parameters
        ----------
        multi_indices : ???
            Iterable of :class:`MultiIndex`.

        Note: an index roughly corresponds to an axis part so we return single axis
        parts from this function
        """
        result = [(index, (), ())]

        path = path or {}

        # TODO do I need to track the labels?
        for index, _, _ in result:
            if index.id in within_indices:
                size = 1
            else:
                size = index.size
            new_part = MultiAxisComponent(size)
            new_axis = MultiAxis([new_part])
            if path:
                axtree.add_node(new_axis, path)
            else:
                axtree.add_node(new_axis)

            new_index = RangeNode((new_axis.name, new_part.label), size)
            itree.add_node(new_index, parent=iid)

            if children := currentindices.children(index):
                for subidx in children:
                    self._construct_temp_dims(
                        axtree,
                        itree,
                        currentindices,
                        subidx,
                        within_indices,
                        path | {new_axis.label: new_part.label},
                        new_index.id,
                    )

    def make_gathers(self, temporaries, **kwargs):
        return tuple(
            self.make_gather(arg, temp, **kwargs) for arg, temp in temporaries.items()
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
        assert all(
            arg.access in {exprs.READ, exprs.WRITE, exprs.INC, exprs.RW}
            for arg in call.arguments
        )

        reads = tuple(
            # temporaries[arg] for arg in call.arguments
            temp
            for arg, temp in temporaries.items()
            if arg.access in {exprs.READ, exprs.INC, exprs.RW}
        )
        writes = tuple(
            temp
            for arg, temp in temporaries.items()
            # temporaries[arg] for arg in call.arguments
            if arg.access in {exprs.WRITE, exprs.INC, exprs.RW}
        )
        return tlang.FunctionCall(call.function, reads, writes, **kwargs)

    def make_scatters(self, temporaries, **kwargs):
        return tuple(
            filter(
                None,
                (
                    self.make_scatter(arg, temp, **kwargs)
                    for arg, temp in temporaries.items()
                ),
            )
        )

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
    def _make_instruction_context(
        self, instruction: tlang.Instruction, *args, **kwargs
    ):
        raise TypeError

    @_make_instruction_context.register
    def _(self, call: tlang.FunctionCall, within_indices, within_inames, depends_on):
        # import pdb; pdb.set_trace()

        subarrayrefs = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            temp_size = temp.data.alloc_size
            iname = self._namer.next("i")
            subarrayrefs[temp] = as_subarrayref(temp.data, iname)
            self.domains.append(f"{{ [{iname}]: 0 <= {iname} < {temp_size} }}")
            assert temp.data.name in [d.name for d in self._temp_kernel_data]

        # we need to pass sizes through if they are only known at runtime (ragged)
        extents = {}
        for temp in utils.unique(itertools.chain(call.reads, call.writes)):
            for index in temp.index.children(temp.index.root):
                extents |= self.collect_extents(
                    temp.index, index, within_indices, within_inames, depends_on
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

        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=insn_id,
            within_inames=within_inames,
            within_inames_is_final=True,
            depends_on=depends_on,
        )

        self.instructions.append(call_insn)
        self.subkernels.append(call.function.code)

    def collect_extents(self, itree, index, within_indices, within_inames, depends_on):
        extents = {}

        if isinstance(index.size, MultiArray):
            # TODO This will overwrite if we have duplicates
            extent = self.register_extent(
                index.size, within_indices, within_inames, depends_on
            )
            extents[index.size] = pym.var(extent)

        if children := itree.children(index):
            for child in children:
                extents |= self.collect_extents(
                    itree, child, within_indices, within_inames, depends_on
                )

        return extents

    @_make_instruction_context.register
    def _(
        self, assignment: tlang.Assignment, within_indices, within_inames, depends_on
    ):
        if not isinstance(assignment.tensor.data, MultiArray):
            raise NotImplementedError(
                "probably want to dispatch here if we hit a PetscMat etc"
            )

        # Register data
        # TODO should only do once at a higher point
        indexed = assignment.tensor.data
        unindexed = assignment.temporary.data
        if indexed.name not in self._tensor_data:
            self._tensor_data[indexed.name] = lp.GlobalArg(
                indexed.name, dtype=indexed.dtype, shape=None
            )

        # import pdb; pdb.set_trace()
        if unindexed.name not in [d.name for d in self._temp_kernel_data]:
            shape = (unindexed.alloc_size,)
            self._temp_kernel_data.append(
                lp.TemporaryVariable(unindexed.name, shape=shape)
            )

        for lindex, rindex in checked_zip(
            assignment.lhs.index.children(assignment.lhs.index.root),
            assignment.rhs.index.children(assignment.rhs.index.root),
        ):
            self.build_assignment(
                assignment,
                lindex,
                rindex,
                within_indices,
                within_inames,
                depends_on,
                (),
                (),
                (),
                (),
            )

    def emit_assignment_insn(
        self,
        array_name,
        array_axes,  # can be None
        labels_to_jnames,
        within_inames,
        depends_on,
        scalar=False,
    ):
        # layout instructions - must be emitted innermost to make sense (reset appropriately)
        offset = self._namer.next(f"{array_name}_ptr")
        self._temp_kernel_data.append(
            lp.TemporaryVariable(offset, shape=(), dtype=np.uintp)
        )
        array_offset_insn = lp.Assignment(
            pym.var(offset),
            0,
            id=self._namer.next("insn"),
            within_inames=within_inames,
            depends_on=depends_on,
        )
        self.instructions.append(array_offset_insn)
        depends_on |= {array_offset_insn.id}

        if not scalar:
            axes = array_axes
            axis = axes.root
            while axis:
                component = just_one(
                    cpt
                    for cpt in axis.components
                    if (axis.name, cpt.label) in labels_to_jnames
                )

                deps = self.emit_layout_insns(
                    axis,
                    component,
                    offset,
                    labels_to_jnames,
                    within_inames,
                    depends_on,
                )
                depends_on |= deps
                axis = axes.child_by_label(axis, component.label)

        return offset, depends_on

    def emit_layout_insns(
        self, axis, axis_part, offset_var, labels_to_jnames, within_inames, depends_on
    ):
        """
        TODO
        """
        layout_fn = axis_part.layout_fn

        if layout_fn == "null layout":
            return frozenset()

        # TODO singledispatch!
        if isinstance(layout_fn, IndirectLayoutFunction):
            # we can either index with just the lowest index or all of them
            # if layout_fn.data.axes.rootless_depth == 1:
            #     mylabels = part_labels[-1:]
            #     myjnames = jnames[-1:]
            # else:
            #     assert layout_fn.data.axes.rootless_depth == utils.single_valued(
            #         map(len, [part_labels, jnames])
            #     )
            #     mylabels = part_labels
            #     myjnames = jnames

            layout_var = self.register_scalar_assignment(
                layout_fn.data, labels_to_jnames, within_inames, depends_on
            )
            expr = pym.var(offset_var) + layout_var

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
                    layout_fn.start,
                    labels_to_jnames,
                    # part_labels[:-1],
                    # jnames[:-1],
                    within_inames,
                    depends_on,
                )

            jname = pym.var(labels_to_jnames[(axis.name, axis_part.label)])
            expr = pym.var(offset_var) + jname * step + start
        else:
            raise NotImplementedError

        insn = lp.Assignment(
            offset_var,
            expr,
            id=self._namer.next("insn"),
            within_inames=within_inames,
            depends_on=depends_on,
        )

        self.instructions.append(insn)

        return frozenset({insn.id})

    def generate_assignment_insn_inner(
        self,
        lexpr,
        rexpr,
        assignment_id,  # FIXME
        depends_on,
        within_inames,
    ):
        insn_id = self._namer.next(f"{assignment_id}_")

        # there are no ordering restrictions between assignments to the
        # same temporary
        no_sync_with = frozenset({(f"{assignment_id}*", "any")})

        assign_insn = lp.Assignment(
            lexpr,
            rexpr,
            id=insn_id,
            within_inames=frozenset(within_inames),
            within_inames_is_final=True,
            depends_on=depends_on,
            no_sync_with=no_sync_with,
        )
        self.instructions.append(assign_insn)

    def generate_index_insns(
        self,
        indicess,  # iterable of an iterable of multi-index groups
        within_multi_index_groups,
        depends_on,
    ):
        if not utils.is_single_valued(len(idxs) for idxs in indicess):
            raise NotImplementedError(
                "Need to be clever about having different lengths"
                "of indices for LHS and RHS"
            )

        # this is a zip
        current_index_groups = []
        later_index_groupss = []
        for indices in indicess:
            current_group, *later_groups = indices
            current_index_groups.append(current_group)
            later_index_groupss.append(later_groups)

        state = []
        expansion = self.expand_multi_index_group(
            current_index_groups, within_multi_index_groups, depends_on
        )
        for updated_within_migs, updated_deps in expansion:
            subresult = self.generate_index_insns(
                later_index_groupss,
                updated_within_migs,
                updated_deps,
            )
            state.extend(subresult)
        return tuple(state)

    def register_extent(self, extent, within_indices, within_inames, depends_on):
        if isinstance(extent, MultiArray):
            labels, jnames = [], []
            index = just_one(extent.indices.children(extent.indices.root))
            while True:
                new_labels, new_jnames = within_indices[index.id]
                labels.extend(new_labels)
                jnames.extend(new_jnames)
                if children := extent.indices.children(index):
                    index = just_one(children)  # must be linear
                else:
                    break

            labels_to_jnames = dict(checked_zip(labels, jnames))

            temp_var = self.register_scalar_assignment(
                extent, labels_to_jnames, within_inames, depends_on
            )
            return str(temp_var)
        else:
            return extent

    def register_scalar_assignment(
        self, array, array_labels_to_jnames, within_inames, depends_on
    ):
        # Register data
        # TODO should only do once at a higher point
        if array.name not in self._tensor_data:
            self._tensor_data[array.name] = lp.GlobalArg(
                array.name, dtype=array.dtype, shape=None
            )

        temp_name = self._namer.next("n")
        self._temp_kernel_data.append(lp.TemporaryVariable(temp_name, shape=()))

        array_offset, array_deps = self.emit_assignment_insn(
            array.name, array.axes, array_labels_to_jnames, within_inames, depends_on
        )
        # TODO: Does this function even do anything when I use a scalar?
        temp_offset, temp_deps = self.emit_assignment_insn(
            temp_name, None, {}, within_inames, depends_on, scalar=True
        )

        lexpr = pym.var(temp_name)
        rexpr = pym.subscript(pym.var(array.name), pym.var(array_offset))

        self.generate_assignment_insn_inner(
            lexpr,
            rexpr,
            self._namer.next("_scalar_assign"),
            depends_on=depends_on | array_deps | temp_deps,
            within_inames=within_inames,
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
    return lp.symbolic.SubArrayRef(index, pym.subscript(pym.var(temporary.name), index))


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

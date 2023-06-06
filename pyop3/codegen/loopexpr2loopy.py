import abc
import collections
import copy
import dataclasses
import enum
import functools
import itertools
import numbers
import operator
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import loopy as lp
import loopy.symbolic
import numpy as np
import pymbolic as pym
import pytools

from pyop3 import tlang, utils
from pyop3.axis import (
    AffineLayoutFunction,
    Axis,
    AxisTree,
    IndirectLayoutFunction,
    MultiAxis,
    MultiAxisComponent,
    MultiAxisTree,
)
from pyop3.distarray import IndexedMultiArray, MultiArray
from pyop3.index import (
    AffineMap,
    IdentityMap,
    Index,
    IndexTree,
    Map,
    Range,
    TabulatedMap,
)
from pyop3.loopexpr import INC, READ, RW, WRITE, FunctionCall, Loop
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


def compile(expr):
    return _make_loopy_kernel(expr)


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
        # breakpoint()
        return tu.with_entrypoints("mykernel")

    @functools.singledispatchmethod
    def _build(self, expr, *args, **kwargs):
        raise TypeError

    @_build.register
    def _(
        self,
        loop: Loop,
        within_indices=None,
        within_inames=frozenset(),
        depends_on=frozenset(),
    ):
        if not within_indices:
            within_indices = {}

        for index in loop.index.root.indices:
            self.build_loop(
                loop,
                loop.index,
                loop.index.root,
                index,
                within_indices,
                within_inames,
                depends_on,
            )

    def build_loop(
        self,
        loop,
        itree,
        multi_index,
        index,
        within_indices,
        within_inames,
        depends_on,
        existing_labels=PrettyTuple(),
        existing_jnames=PrettyTuple(),
        path=None,
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
        if isinstance(index, Map) and index.selector:
            raise NotImplementedError

        path = path or {}

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
            multi_index,
            index,
            existing_labels,
            existing_jnames,
            within_indices,
            within_inames,
            depends_on,
        )

        if child := itree.find_node(
            path | {multi_index.label: multi_index.index(index)}
        ):
            for subindex in child.indices:
                self.build_loop(
                    loop,
                    itree,
                    child,
                    subindex,
                    within_indices | new_within,
                    within_inames | {iname},
                    depends_on | new_deps,
                    new_labels,
                    new_jnames,
                    path | {multi_index.label: multi_index.index(index)},
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
        multi_index,
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

        # breakpoint()
        if multi_index.label in within_indices:
            labels, jnames = within_indices[multi_index.label]
            if isinstance(index, Range):
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

        elif isinstance(index, Range):
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
            new_labels = existing_labels | index.path
            new_jnames = existing_jnames | jname
            jnames = (jname,)
            new_within = {multi_index.label: ((index.path,), (jname,))}

        elif isinstance(index, IdentityMap):
            index_insns = []
            new_labels = existing_labels
            new_jnames = existing_jnames
            jnames = ()
            new_within = {}

        elif isinstance(index, AffineMap):
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
            new_within = {multi_index.label: ((to_label,), (jname,))}

        elif isinstance(index, TabulatedMap):
            # NOTE: some maps can produce multiple jnames (but not this one)
            jname = self._namer.next("j")
            self._temp_kernel_data.append(
                lp.TemporaryVariable(jname, shape=(), dtype=np.uintp)
            )

            map_labels = existing_labels | (index.data.data.axes.leaf.label, 0)
            map_jnames = existing_jnames | iname
            expr = self.register_scalar_assignment(
                index.data.data,
                dict(checked_zip(map_labels, map_jnames)),
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
            new_within = {multi_index.label: ((to_label,), (jname,))}
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
        lmulti_index,
        rmulti_index,
        lindex,
        rindex,
        mycounter,
        within_indices,
        within_inames,
        depends_on,
        llabels=PrettyTuple(),
        ljnames=PrettyTuple(),
        rlabels=PrettyTuple(),
        rjnames=PrettyTuple(),
        lindex_path=None,
        rindex_path=None,
    ):
        lindex_path = lindex_path or {}
        rindex_path = rindex_path or {}

        llabels_ = PrettyTuple(llabels)
        ljnames_ = PrettyTuple(ljnames)
        rlabels_ = PrettyTuple(rlabels)
        rjnames_ = PrettyTuple(rjnames)

        iname = None
        # size = utils.single_valued([lindex.size, rindex.size])
        # now need to catch one-sized things here
        if lmulti_index.label in within_indices:
            lsize = 1
        else:
            lsize = lindex.size
        if rmulti_index.label in within_indices:
            rsize = 1
        else:
            rsize = rindex.size
        size = utils.single_valued([lsize, rsize])

        iname = self._namer.next("i")
        extent = self.register_extent(size, within_indices, within_inames, depends_on)
        domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
        self.domains.append(domain_str)

        lthings = self.myinnerfunc(
            iname,
            lmulti_index,
            lindex,
            llabels_,
            ljnames_,
            within_indices,
            within_inames,
            depends_on,
        )
        rthings = self.myinnerfunc(
            iname,
            rmulti_index,
            rindex,
            rlabels_,
            rjnames_,
            within_indices,
            within_inames,
            depends_on,
        )

        lchild = assignment.lhs.index.find_node((lmulti_index.id, mycounter))
        rchild = assignment.rhs.index.find_node((rmulti_index.id, mycounter))
        if strictly_all([lchild, rchild]):
            for mynewcounter, (lsubindex, rsubindex) in enumerate(
                checked_zip(lchild.indices, rchild.indices)
            ):
                self.build_assignment(
                    assignment,
                    lchild,
                    rchild,
                    lsubindex,
                    rsubindex,
                    mynewcounter,  # FIXME do the loop at the top of this function, not "outside"
                    within_indices | lthings[2] | rthings[2],
                    within_inames | {iname},
                    depends_on | lthings[3] | rthings[3],
                    lthings[0],
                    lthings[1],
                    rthings[0],
                    rthings[1],
                    lindex_path | {lmulti_index.label: mycounter},
                    rindex_path | {rmulti_index.label: mycounter},
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
                depends_on=depends_on
                | assignment.depends_on
                | extra_deps
                | array_deps
                | temp_deps,
                within_inames=within_inames | extra_inames,
            )

    @_build.register
    def _(
        self,
        call: FunctionCall,
        within_indices,
        within_inames,
        depends_on,
    ):
        # I think I'd prefer to do this in a separate earlier pass?
        # when we construct the function call?
        insns = self.expand_function_call(
            call, within_indices, within_inames, depends_on
        )

        for insn in insns:
            self._make_instruction_context(
                insn, within_indices, within_inames, depends_on
            )

    def expand_function_call(self, call, within_indices, within_inames, depends_on):
        """
        Turn an exprs.FunctionCall into a series of assignment instructions etc.
        Handles packing/accessor logic.
        """

        temporaries = {}
        subarrayrefs = {}
        extents = {}

        # loopy args can contain ragged params too
        loopy_args = call.function.code.default_entrypoint.args[: len(call.arguments)]
        for loopy_arg, arg, spec in checked_zip(
            loopy_args, call.arguments, call.argspec
        ):
            # create an appropriate temporary
            # we need the indices here because the temporary shape needs to be indexed
            # by the same indices as the original array
            # is this definitely true??? think so. because it gives us the right loops
            # but we only really need it to determine "within" or not...
            if not isinstance(arg.data, MultiArray):
                # think PetscMat etc
                raise NotImplementedError(
                    "Need to handle indices to create temp shape differently"
                )

            axes = self._axes_from_index_tree(arg.index, within_indices)
            temporary = MultiArray(
                axes,
                name=self._temp_name_generator.next(),
                dtype=arg.data.dtype,
            )
            indexed_temp = temporary[...]
            temporaries[arg] = (indexed_temp, spec.access)

            # Register data
            if arg.data.name not in self._tensor_data:
                self._tensor_data[arg.data.name] = lp.GlobalArg(
                    arg.data.name, dtype=arg.data.dtype, shape=None
                )

            if loopy_arg.shape is None:
                shape = (temporary.alloc_size,)
            else:
                if np.prod(loopy_arg.shape, dtype=int) != temporary.alloc_size:
                    raise RuntimeError("Shape mismatch between inner and outer kernels")
                shape = loopy_arg.shape

            self._temp_kernel_data.append(
                lp.TemporaryVariable(temporary.name, shape=shape)
            )

            # subarrayref nonsense/magic
            indices = []
            for s in shape:
                iname = self._namer.next("i")
                indices.append(pym.var(iname))
                self.domains.append(f"{{ [{iname}]: 0 <= {iname} < {s} }}")
            indices = tuple(indices)

            subarrayrefs[arg] = lp.symbolic.SubArrayRef(
                indices, pym.subscript(pym.var(temporary.name), indices)
            )

            # we need to pass sizes through if they are only known at runtime (ragged)
            # NOTE: If we register an extent to pass through loopy will complain
            # unless we register it as an assumption of the local kernel (e.g. "n <= 3")
            for cidx in range(indexed_temp.index.root.degree):
                extents |= self.collect_extents(
                    indexed_temp.index,
                    indexed_temp.index.root,
                    cidx,
                    within_indices,
                    within_inames,
                    depends_on,
                )

        assignees = tuple(
            subarrayrefs[arg]
            for arg, spec in checked_zip(call.arguments, call.argspec)
            if spec.access in {WRITE, RW, INC}
        )
        expression = pym.primitives.Call(
            pym.var(call.function.code.default_entrypoint.name),
            tuple(
                subarrayrefs[arg]
                for arg, spec in checked_zip(call.arguments, call.argspec)
                if spec.access in {READ, RW, INC}
            )
            + tuple(extents.values()),
        )

        # TODO Refactor this
        gathers = self.make_gathers(temporaries)

        insn_id = self._namer.next(call.name)
        deps = frozenset({f"{gather.id}_*" for gather in gathers}) | depends_on

        call_insn = lp.CallInstruction(
            assignees,
            expression,
            id=insn_id,
            within_inames=within_inames,
            within_inames_is_final=True,
            depends_on=deps,
        )

        self.instructions.append(call_insn)
        self.subkernels.append(call.function.code)

        scatters = self.make_scatters(
            temporaries, depends_on=depends_on | frozenset({insn_id})
        )

        return (*gathers, *scatters)

    # TODO This algorithm is pretty much identical to fill_shape
    def _axes_from_index_tree(self, index_tree, within_indices, index_path=None):
        index_path = index_path or {}

        components = []
        subroots = []
        bits = {}
        multi_index = index_tree.find_node(index_path)
        indexed = multi_index.label in within_indices
        for i, index in enumerate(multi_index.components):
            components.append(MultiAxisComponent(index.size))

            if index_tree.find_node(index_path | {multi_index.label: i}):
                subaxes = self._axes_from_index_tree(
                    index_tree,
                    within_indices,
                    index_path | {multi_index.label: i},
                )
                subroots.append(subaxes.root)
                bits |= subaxes.parent_to_children
            else:
                subroots.append(None)

        root = Axis(components, label=multi_index.label, indexed=indexed)
        return AxisTree(root, {root.id: subroots} | bits)

    def make_gathers(self, temporaries, **kwargs):
        return tuple(
            self.make_gather(arg, temp, access, **kwargs)
            for arg, (temp, access) in temporaries.items()
        )

    def make_gather(self, argument, temporary, access, **kwargs):
        # TODO cleanup the ids
        if access in {READ, RW}:
            return tlang.Read(argument, temporary, **kwargs)
        elif access in {WRITE, INC}:
            return tlang.Zero(argument, temporary, **kwargs)
        else:
            raise NotImplementedError

    def make_scatters(self, temporaries, **kwargs):
        return tuple(
            filter(
                None,
                (
                    self.make_scatter(arg, temp, access, **kwargs)
                    for arg, (temp, access) in temporaries.items()
                ),
            )
        )

    def make_scatter(self, argument, temporary, access, **kwargs):
        if access == READ:
            return None
        elif access in {WRITE, RW}:
            return tlang.Write(argument, temporary, **kwargs)
        elif access == INC:
            return tlang.Increment(argument, temporary, **kwargs)
        else:
            raise AssertionError

    @functools.singledispatchmethod
    def _make_instruction_context(
        self, instruction: tlang.Instruction, *args, **kwargs
    ):
        raise TypeError

    def collect_extents(
        self, itree, index, component_index, within_indices, within_inames, depends_on
    ):
        component = index.components[component_index]
        extents = {}

        if isinstance(component.size, IndexedMultiArray):
            # TODO This will overwrite if we have duplicates
            extent = self.register_extent(
                component.size, within_indices, within_inames, depends_on
            )
            extents[component.size] = pym.var(extent)

        if subidx := itree.find_node((index.id, component_index)):
            for cidx in range(subidx.degree):
                extents |= self.collect_extents(
                    itree, subidx, cidx, within_indices, within_inames, depends_on
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

        for mycounter, (lindex, rindex) in enumerate(
            checked_zip(
                assignment.lhs.index.root.indices,
                assignment.rhs.index.root.indices,
            )
        ):
            self.build_assignment(
                assignment,
                assignment.lhs.index.root,
                assignment.rhs.index.root,
                lindex,
                rindex,
                mycounter,
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
            path = PrettyTuple()
            while axis:
                component, component_index = just_one(
                    (cpt, i)
                    for i, cpt in enumerate(axis.components)
                    if (axis.label, i) in labels_to_jnames
                )

                path |= component_index

                # set the component index to always 0 for the inner linear layout
                linear_labels_to_jnames = {
                    (label, 0): jname for (label, _), jname in labels_to_jnames.items()
                }
                axis = axes.find_node((axis.id, component_index))

            deps = self.emit_layout_insns(
                axes,
                axis,
                component,
                component_index,
                offset,
                linear_labels_to_jnames,
                within_inames,
                depends_on,
                path,
            )
            depends_on |= deps

        return offset, depends_on

    def emit_layout_insns(
        self,
        axes,
        axis,
        axis_part,
        component_index,
        offset_var,
        labels_to_jnames,
        within_inames,
        depends_on,
        path,
    ):
        """
        TODO
        """
        expr = pym.var(offset_var)
        remaining_labels = set(x for x, y in labels_to_jnames.keys())
        for layout_fn in axes.layouts[path]:
            # TODO singledispatch!
            if isinstance(layout_fn, IndirectLayoutFunction):
                layout_var = self.register_scalar_assignment(
                    layout_fn.data, labels_to_jnames, within_inames, depends_on
                )
                expr += layout_var
            elif isinstance(layout_fn, AffineLayoutFunction):
                start = layout_fn.start
                step = layout_fn.step

                if isinstance(start, MultiArray):
                    assert False, "dropping support for this"
                    # drop the last jname
                    start = self.register_scalar_assignment(
                        layout_fn.start,
                        labels_to_jnames,
                        within_inames,
                        depends_on,
                    )

                jname = pym.var(labels_to_jnames[(layout_fn.consumed_label, 0)])
                expr += jname * step + start
            else:
                raise NotImplementedError

            remaining_labels -= layout_fn.consumed_labels

        # sometimes we have more labels than we want - I should probably stop that from happening
        # assert len(remaining_labels) == 0

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
        if isinstance(extent, IndexedMultiArray):
            labels, jnames = [], []
            index = extent.index.root
            while index:
                component = just_one(index.components)
                new_labels, new_jnames = within_indices[index.label]
                labels.extend(new_labels)
                jnames.extend(new_jnames)
                index = extent.index.find_node((index.id, 0))

            labels_to_jnames = {
                (label, 0): jname for ((label, _), jname) in checked_zip(labels, jnames)
            }

            temp_var = self.register_scalar_assignment(
                extent.data, labels_to_jnames, within_inames, depends_on
            )
            return str(temp_var)
        else:
            assert isinstance(extent, numbers.Integral)
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

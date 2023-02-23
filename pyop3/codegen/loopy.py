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
        # index_registry is a map from indices to inames
        if not index_registry:
            index_registry = utils.MultiQueue()

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

        # we need to build a separate set of instructions for each multi-index
        # in the collection.
        # e.g. if we are looping over interior facets of an extruded mesh - the
        # iterset is two multi-indices: edges+hfacets and verts+vfacets
        for multi_idx in multi_indices:
            if multi_idx in loop_indices:
                raise ValueError("Loops cannot reuse indices")

            # 1. register loops

            # in theory we could reuse the same multi-index twice (e.g. dat[p, p]) so we
            # need a clever stack I think
            bitsandpieces = self.register_loops2(multi_idx, index_registry, loop_indices)
            for midx, inames in bitsandpieces.items():
                index_registry.append(midx, inames)

            loop_indices = outer_loop_indices | {multi_idx}

            # new_index_registry = self.register_loops2(multi_idx, index_registry, True)

            # import pdb; pdb.set_trace()

            # 2. emit instructions
            for stmt in expr.statements:
                self._build(stmt, index_registry, loop_indices)

            index_registry.pop(multi_idx) # context manager?


    def collect_loop_indices(self, multi_index):
        loop_indices = set()
        for typed_idx in multi_index:
            if isinstance(typed_idx, Map):
                loop_indices |= self.collect_loop_indices(typed_idx.from_multi_index)
            loop_indices |= {typed_idx}
        return frozenset(loop_indices)


    # index registry is just a list of IndexRegistryEntry items
    def register_loops2(self, multi_index, index_registry, loop_indices):
        """
        Register a bunch of loops related to a multi-index and return a mapping from index
        to inames
        """
        # debug
        assert not isinstance(loop_indices, bool)

        assert isinstance(multi_index, MultiIndex)

        # import pdb; pdb.set_trace()
        # eaten_registry = index_registry.copy()
        # 1. partition the existing index registry
        # registry_per_index = {}
        # for typed_idx in multi_index:
        #     registry = []
        #     for part_label in typed_idx.consumed_labels:
        #         try:
        #             item = utils.popwhen(
        #                 lambda entry: entry.label == part_label, eaten_registry)
        #             registry.append(item)
        #         except KeyError:
        #             pass
        #     registry_per_index[typed_idx] = registry

        # 2. construct a new index registry
        # new_registry = []

        if multi_index in loop_indices:
            return {}

        if isinstance(multi_index, Map):
            # recurse
            raise NotImplementedError

        inames = []
        for idx in multi_index:

            # do this before creating the new iname
            # FIXME shouldn't this use the stuff from the map if required?
            extent = self.register_extent(
                idx.size,
                "notathing",
                # new_registry,
                # within_inames,
            )

            # each index can only emit a single new loop
            iname = self._namer.next("i")
            inames.append(iname)

            domain_str = f"{{ [{iname}]: 0 <= {iname} < {extent} }}"
            self.domains.append(domain_str)

        # import pdb; pdb.set_trace()

        return {multi_index: inames}

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
        # return new_registry

    @_build.register
    def _(self, call: exprs.FunctionCall, index_registry, loop_indices):
        insns = self.expand_function_call(call, index_registry, loop_indices)

        for insn in insns:
            self._make_instruction_context(insn, index_registry, loop_indices)

    def expand_function_call(self, call, index_registry, loop_indices):
        """
        Turn an exprs.FunctionCall into a series of assignment instructions etc.
        Handles packing/accessor logic.
        """

        temporaries = {}
        for arg in call.arguments:
            # create an appropriate temporary
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

            for typed_idx in multi_idx:
                current_array_axis = current_array_axis._parts_by_label[typed_idx.part_label].subaxis

        # axis might not be at the bottom so we'd need to shove on some extra shape
        # TODO: untested if multi-part
        if current_array_axis:
            raise NotImplementedError("need to tackle trailing shape")

        return top_temp_part

    def make_temp_axes_per_multi_index(self, multi_index, index_registry, loop_indices):

        top_part = None
        current_bottom_part_id = None

        if multi_index in loop_indices:
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

        index_registry = index_registry.copy()
        idx, *subidxs = multi_index
        try:
            entry = utils.popwhen(lambda e: e.label == idx.part_label, index_registry)
            has_scalar_shape = entry.is_loop_index
        except KeyError:
            has_scalar_shape = False
        new_part = AxisPart(idx.size, indexed=has_scalar_shape, label=idx.part_label)

        if subidxs:
            result = self.make_temp_axis_part_per_multi_index_collection(
                subidxs, index_registry, loop_indices, return_final_part_id=return_final_part_id)
            if return_final_part_id:
                subpart, bottom_part_id = result
            else:
                subpart = result
            new_part = new_part.copy(subaxis=MultiAxis([subpart]))
        else:
            bottom_part_id = new_part.id

        if isinstance(idx, Map):
            raise NotImplementedError
            # super_part, bottom_part_id = self.make_temp_axis_part_per_multi_index(idx.from_multi_index, loop_indices, return_final_part_id=True)
            # # wont work as need part IDs
            # new_part = super_part.add_subaxis(bottom_part_id, MultiAxis([new_part]))

        return new_part, bottom_part_id

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
    def _(self, assignment: tlang.Assignment, index_registry, loop_indices):
        index_registry = index_registry.copy()

        for multi_idx in assignment.indices:
            bitsandpieces = self.register_loops2(multi_idx, index_registry, loop_indices)

            for midx, inames in bitsandpieces.items():
                index_registry.append(midx, inames)

        self._generate_assignment_insn(assignment, index_registry, loop_indices)

    def _generate_assignment_insn(self, assignment, index_registry, loop_indices, scalar=False):
        # import pdb; pdb.set_trace()

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
        depends_on = frozenset({f"{dep}_*" for dep in assignment.depends_on})

        # index_registry_copy = index_registry.copy()
        # selected = [utils.popwhen(lambda e: e.label == idx.part_label, index_registry_copy)
        #             for idx in assignment.indices]
        #
        # within_inames = functools.reduce(frozenset.__or__, [e.within_inames for e in selected])
        within_inames = NotImplemented

        # 2. Generate map code
        lpnames, ljnames = self.generate_index_insns(assignment.lhs.axes, assignment.indices, index_registry, assignment.lhs.name, assignment, within_inames)

        rpnames, rjnames = self.generate_index_insns(assignment.rhs.axes, assignment.indices, index_registry, assignment.rhs.name, assignment, within_inames)

        within_inames = self.collect_within_inames(assignment.indices, index_registry, loop_indices)

        self.emit_assignment_insn(assignment, lpnames, ljnames, rpnames, rjnames, depends_on, within_inames, scalar=scalar)

    def collect_within_inames(self, multi_indices, index_registry, loop_indices):


        assert isinstance(multi_indices, MultiIndexCollection)

        index_registry = index_registry.copy()
        within_inames = set()

        # register loop_indices first
        for multi_idx in loop_indices:
            inames = index_registry.popfirst(multi_idx)
            within_inames |= set(inames)

        for multi_idx in multi_indices:
            # if a multi-index is also a loop index then it can only be registered once
            if multi_idx in loop_indices:
                continue

            if isinstance(multi_idx, Map):
                raise NotImplementedError
                # remove from the registry for consistency
                index_registry.popfirst(multi_idx)

                # should use from_multi_indices instead (the inames here aren't registered)
                # hack
                frommidx = MultiIndexCollection([multi_idx.from_multi_index])
                within_inames |= self.collect_within_inames(frommidx, index_registry, loop_indices)

            else:
                inames = index_registry.popfirst(multi_idx)
                within_inames |= set(inames)

        # If we register multi-indices multiple times (e.g. dat1[map0, map0]) then they
        # should be fully consumed here - actually the whole thing should be empty
        assert all(len(v) == 0 for v in index_registry._data.values())

        return frozenset(within_inames)

    def emit_assignment_insn(
            self, assignment,
            lpnames, ljnames, rpnames, rjnames, depends_on,
            within_inames, scalar):

        """
        This needs to be a separate function because when we index things like layouts
        and extents we already know that the shapes will agree (and they are single part)
        so we just need to use the inames that we are given.
        """

        # 3. Generate layout code
        # import pdb; pdb.set_trace()
        loffset, ldeps = self.make_offset_expr(
            assignment.lhs, lpnames, ljnames, f"{assignment.id}_loffset", depends_on, within_inames,
        )
        roffset, rdeps = self.make_offset_expr(
            assignment.rhs, rpnames, rjnames, f"{assignment.id}_roffset", depends_on, within_inames,
        )

        depends_on |= {*ldeps, *rdeps}

        # 4. Emit assignment instruction
        self.generate_assignment_insn_inner(assignment, assignment.lhs, loffset,
                                      assignment.rhs, roffset,
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

    def make_offset_expr(self, array, part_names, inames, insn_prefix, depends_on, within_inames):
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

        stmts = [f"{offset_var_name} = 0 {{{inames_attr},dep={':'.join(depends_on)},id={init_insn_id}}}"]

        new_stmts, new_deps = self.make_offset_expr_inner(offset_var_name, axis, part_names, inames, depends_on, within_inames, insn_prefix)
        stmts.extend(new_stmts)
        self.instructions.append("\n".join(stmts))
        return offset_var_name, new_deps

    def make_offset_expr_inner(self, offset_var_name, axis, part_names,
            inames, depends_on, within_inames, insn_prefix, depth=0):
        assert axis.nparts > 0

        new_deps = set()

        stmts = []
        if axis.nparts == 1: # if statement not needed

            # do not emit a statement if the loop isn't needed
            if isinstance(axis.part.count, MultiArray) or (isinstance(axis.part.count, numbers.Integral) and axis.part.count > 1):
                new_stmts, deps = self.emit_layout_insns(
                    axis.part.layout_fn,
                    0,  # not really a label
                    offset_var_name,
                    inames,
                    depends_on,
                    within_inames,
                    insn_prefix,
                    depth
                )
                stmts += new_stmts
                new_deps |= deps

            # TODO indent statements for readability
            subaxis = axis.part.subaxis
            if subaxis:
                substmts, deps = self.make_offset_expr_inner(offset_var_name,
                        subaxis, part_names, inames, within_inames, insn_prefix, depth+1)
                stmts.extend(substmts)
                new_deps |= deps
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

                    newstmts, deps = self.emit_layout_insns(
                        axis_part.layout_fn,
                        i,  # not really a label
                        offset_var_name, inames, depends_on, within_inames,
                        insn_prefix, depth
                    )
                    stmts += newstmts
                    new_deps |= deps

                # recurse (and indent?)
                subaxis = axis_part.subaxis
                if subaxis:
                    newstmts, deps = self.make_offset_expr_inner(
                        offset_var_name,
                        subaxis, part_names, inames, within_inames,
                        insn_prefix, depth+1
                    )
                    stmts.extend(newstmts)
                    new_deps |= deps

            # number of "end" statements that need emitting
            for _ in range(axis.nparts-1):
                stmts.append("end")

        return stmts, frozenset(new_deps)

    def emit_layout_insns(self, layout_fn, label, offset_var, inames, depends_on, within_inames, insn_prefix, depth):
        """
        TODO
        """
        if layout_fn == "null layout":
            return [], frozenset()

        insn_id = self._namer.next(insn_prefix)
        within_inames_attr = f"inames={':'.join(within_inames)}"

        # TODO singledispatch!
        if isinstance(layout_fn, IndirectLayoutFunction):
            # add 1 to depth here since we want to use the actual index for these, but not for start
            layout_var = self.register_scalar_assignment(layout_fn.data, inames[:depth+1])

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
                start = self.register_scalar_assignment(layout_fn.start, inames)

            # import pdb; pdb.set_trace()
            # this isn't quite good enough - I would previously truncate this and always
            # take the last entry. Here we hit duplicates. How do I distinguish the
            # particular index entry that I want?
            # entry = utils.single_valued(e for e in index_registry if e.index.part_label == label)
            iname = inames[depth]

            stmts = [
                f"{offset_var} = {offset_var} + {iname}*{step} + {start} "
                f"{{{within_inames_attr},dep={':'.join(depends_on)},id={insn_id}}}"
            ]

        # self._latest_insn[self._active_insn.id] = insn_id
        return stmts, frozenset({insn_id})

    def generate_assignment_insn_inner(
            self,
            assignment,
            lhs,
            loffset,
            rhs,
            roffset,
            scalar,
            depends_on,
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
        # self._latest_insn[self._active_insn.id] = insn_id

    def generate_index_insns(self, axis, multi_indices, index_registry, prefix, context, within_inames):
        """This instruction needs to somehow traverse the indices and axes and probably
        return variable representing the parts and index name (NOT inames) for the LHS
        and RHS. Could possibly be done separately for each as the inames are known.

        Something like 'if the index is not a map yield something equal to the iname, if
        it is a map then yield something like j0 = map0[i0, i1] (and generate the right
        instructions).
        """


        """
        Really the inames should already exist and be registered - here we only really
        need to collect the ones that we want."""

        assert isinstance(multi_indices, MultiIndexCollection)
        index_registry = index_registry.copy()

        # import pdb; pdb.set_trace()

        multi_idx, *submulti_indices = multi_indices

        pnames = [self._namer.next(f"{prefix}_p") for _ in range(len(multi_idx))]
        inames = index_registry.pop(multi_idx)

        assert len(pnames) == len(inames)

        # not needed as can be done at top of loop
        # depends_on = frozenset({self._latest_insn[id] for id in self._active_insn.depends_on})
        # if self._active_insn.id in self._latest_insn:
        #     depends_on |= frozenset({self._latest_insn[self._active_insn.id]})

        # npart = [part.label for part in axis.parts].index(idx.part_label)

        # if isinstance(idx, Map):
        #     raise NotImplementedError("need to emit some clever code")
        #     map_inames, inames = inames[:idx.depth], inames[idx.depth:]
        #     assign_insn = ...
        # else:
        #     iname = entry.iname
        #     part_insn = lp.Assignment(
        #         pym.var(pname), npart,
        #         id=self._namer.next(f"{context.id}_"),
        #         within_inames=within_inames,
        #         depends_on=depends_on,
        #         # no_sync_with=no_sync_with,
        #     )
        #     jname, = jnames
        #     index_insn = lp.Assignment(
        #         pym.var(jname), pym.var(iname),
        #         id=self._namer.next(f"{context.id}_"),
        #         within_inames=within_inames,
        #         depends_on=depends_on,
        #         # no_sync_with=no_sync_with,
        #     )
        # self.instructions.append(part_insn)
        # self.instructions.append(index_insn)
        #
        # self._latest_insn[self._active_insn.id] = index_insn.id
        #
        # self._temp_kernel_data.extend([
        #     lp.TemporaryVariable(name, shape=(), dtype=np.uintp)
        #     for name in [jname, pname]
        # ])

        if submulti_indices:
            raise NotImplementedError
            assert axis.parts[npart].subaxis
            subp, subi = self.generate_index_insns(axis.parts[npart].subaxis, subidxs, registry, prefix, context, within_inames)
            return pnames + subp, inames + subi
        else:
            return pnames, inames

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
            raise NotImplementedError
            # If we have a ragged thing then we need to create a scalar temporary
            # to hold its value.
            temp_var = self.register_scalar_assignment(extent, index_registry, within_inames)
            return str(temp_var)
        else:
            return extent

    def register_scalar_assignment(self, array, inames, within_inames):
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
        self.emit_assignment_insn(insn, None, None, None, inames, within_inames, scalar=True)

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

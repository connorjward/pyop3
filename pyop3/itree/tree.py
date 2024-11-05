from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import itertools
import functools
import math
import numbers
import sys
from functools import cached_property
from typing import Any, Collection, Hashable, Mapping, Sequence, Type, cast, Optional

import numpy as np
import pymbolic as pym
from pyop3.exceptions import Pyop3Exception
import pytools
from immutabledict import ImmutableOrderedDict
from pyrsistent import PMap, freeze, pmap, thaw

from pyop3.array import Dat
from pyop3.array.base import Array
from pyop3.axtree import (
    Axis,
    AxisComponent,
    AxisTree,
    AxisForest,
    AxisVar,
    ContextSensitive,
    LoopIterable,
)
from pyop3.axtree.layout import _as_int
from pyop3.axtree.tree import (
    AxisTree,
    Expression,
    AxisVar, Operator,
    ContextSensitiveLoopIterable,
    IndexedAxisTree,
    LoopIndexVar,
)
from pyop3.dtypes import IntType, get_mpi_dtype
from pyop3.expr_visitors import replace
from pyop3.lang import KernelArgument
from pyop3.tree import (
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    MutableLabelledTreeMixin,
    postvisit,
)
from pyop3.utils import (
    Identified,
    Labelled,
    as_tuple,
    expand_collection_of_iterables,
    strict_zip,
    single_valued,
    just_one,
    checked_zip,
    merge_dicts,
    strictly_all,
)

bsearch = pym.var("mybsearch")

class InvalidIndexTargetException(Pyop3Exception):
    """Exception raised when we try to match index information to a mismatching axis tree."""





# NOTE: index trees are not really labelled trees. The component labels are always
# nonsense. Instead I think they should just advertise a degree and then attach
# to matching index (instead of label).
class IndexTree(MutableLabelledTreeMixin, LabelledTree):
    def __init__(self, node_map=pmap()):
        super().__init__(node_map)

    @classmethod
    def from_nest(cls, nest):
        # NOTE: This may fail when we have maps which produce multiple axis trees.
        # This is because the map has multiple equivalent interpretations so the
        # correct one is not known at this point. Instead one should use
        # IndexForest.from_nest(...).
        root, node_map = cls._from_nest(nest)
        node_map.update({None: [root]})
        return cls(node_map)

    @classmethod
    def from_iterable(cls, iterable):
        # All iterable entries must be indices for now as we do no parsing
        root, *rest = iterable
        node_map = {None: (root,)}
        parent = root
        for index in rest:
            node_map.update({parent.id: (index,)})
            parent = index
        return cls(node_map)


class SliceComponent(LabelledNodeComponent, abc.ABC):
    def __init__(self, component, *, label=None):
        super().__init__(label)
        self.component = component


class AffineSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"start", "stop", "step", "label_was_none"}

    # use None for the default args here since that agrees with Python slices
    def __init__(self, component, start=None, stop=None, step=None, *, label=None, **kwargs):
        label_was_none = label is None

        super().__init__(component, label=label, **kwargs)
        # could be None here
        self.start = start if start is not None else 0
        self.stop = stop
        # could be None here
        self.step = step if step is not None else 1

        # hack to force a relabelling
        self.label_was_none = label_was_none

    @property
    def datamap(self) -> PMap:
        return pmap()

    @property
    def is_full(self):
        return self.start == 0 and self.stop is None and self.step == 1


class SubsetSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"array"}

    def __init__(self, component, array, **kwargs):
        super().__init__(component, **kwargs)
        self.array = array

    @property
    def datamap(self) -> PMap:
        return self.array.datamap


# alternative name, better or worse?
Subset = SubsetSliceComponent


class MapComponent(pytools.ImmutableRecord, Labelled, abc.ABC):
    fields = {"target_axis", "target_component", "label"}

    def __init__(self, target_axis, target_component, *, label=None):
        pytools.ImmutableRecord.__init__(self)
        Labelled.__init__(self, label)
        self.target_axis = target_axis
        self.target_component = target_component

    @property
    @abc.abstractmethod
    def arity(self):
        pass

    @property
    def target_path(self) -> PMap:
        return pmap({self.target_axis: self.target_component})


# TODO: Implement AffineMapComponent
class TabulatedMapComponent(MapComponent):
    fields = MapComponent.fields | {"array", "arity"}

    def __init__(self, target_axis, target_component, array, *, arity=None, label=None):
        # determine the arity from the provided array
        if arity is None:
            leaf_axis, leaf_clabel = array.axes.leaf
            leaf_cidx = leaf_axis.component_index(leaf_clabel)
            arity = leaf_axis.components[leaf_cidx].count

        super().__init__(target_axis, target_component, label=label)
        self.array = array
        self._arity = arity

    @property
    def arity(self):
        return self._arity

    # old alias
    @property
    def data(self):
        return self.array

    @functools.cached_property
    def datamap(self):
        return self.array.datamap


class Index(MultiComponentLabelledNode):
    pass


class AxisIndependentIndex(Index):
    @property
    @abc.abstractmethod
    def axes(self) -> IndexedAxisTree:
        pass

    @property
    def component_labels(self) -> tuple:
        return tuple(i for i, _ in enumerate(self.axes.leaves))


# class ContextFreeIndex(Index, ContextFree, abc.ABC):
#     # The following is unimplemented but may prove useful
#     # @property
#     # def axes(self):
#     #     return self._tree.axes
#     #
#     # @property
#     # def target_paths(self):
#     #     return self._tree.target_paths
#     #
#     # @cached_property
#     # def _tree(self):
#     #     """
#     #
#     #     Notes
#     #     -----
#     #     This method will deliberately not work for slices since slices
#     #     require additional existing axis information in order to be valid.
#     #
#     #     """
#     #     return as_index_tree(self)
#     @abc.abstractmethod
#     def restrict(self, paths):
#         """Return a restricted index with only the paths provided.
#
#         The resulting index will have its components ordered as given by ``paths``.
#
#         """
#
#
#
# class ContextSensitiveIndex(Index, ContextSensitive, abc.ABC):
#     def __init__(self, context_map, *, id=None):
#         Index.__init__(self, id)
#         ContextSensitive.__init__(self, context_map)



class LoopIndex(Index, KernelArgument):
    """
    Parameters
    ----------
    iterset: AxisTree or ContextSensitiveAxisTree (!!!)
        Only add context later on

    """
    dtype = IntType
    fields = Index.fields - {"label"}

    def __init__(self, iterset, *, id=None):
        assert len(iterset.leaves) >= 1

        super().__init__(label=id, id=id)
        self.iterset = iterset

    @property
    def kernel_dtype(self):
        return self.dtype

    # NOTE: should really just be 'degree' or similar, labels do not really make sense for
    # index trees
    @property
    def component_labels(self) -> tuple:
        if not self.is_context_free:  # TODO: decorator?
            # custom exception type
            raise ValueError("only valid (context-free) in single component case")

        return (0,)

    @property
    def is_context_free(self):
        return len(self.iterset.leaves) == 1

    @cached_property
    def axes(self) -> IndexedAxisTree:
        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        # NOTE: same as _index_axes_index

        # Example:
        # If we assume that the loop index has target expressions
        #     AxisVar("a") * 2     and       AxisVar("b")
        # then this will return
        #     LoopIndexVar(p, "a") * 2      and LoopIndexVar(p, "b")
        replace_map = {
            None: (
                {axis.label: component_label for axis, component_label in self.iterset.path_with_nodes(self.iterset.leaf).items()},
                {axis.label: LoopIndexVar(self, axis.label) for axis, component_label in self.iterset.path_with_nodes(self.iterset.leaf).items()},
            )
        }

        targets = []
        for equivalent_targets in self.iterset.paths_and_exprs:
            new_path = {}
            new_exprs = {}
            for (orig_path, orig_exprs) in equivalent_targets.values():
                new_path.update(orig_path)
                for axis_label, orig_expr in orig_exprs.items():
                    new_exprs[axis_label] = replace(orig_expr, AxisTree(), replace_map)
            new_path = pmap(new_path)
            new_exprs = pmap(new_exprs)
            targets.append(pmap({None: (new_path, new_exprs)}))
        return IndexedAxisTree({}, unindexed=None, targets=targets)

    # TODO: don't think this is useful any more, certainly a confusing name
    @property
    def leaf_target_paths(self):
        """

        Unlike with maps and slices, loop indices are single-component (so return a 1-tuple)
        but that component can target differently labelled axes (so the tuple entry is an n-tuple).

        """
        assert self.is_context_free

        equivalent_paths = []
        leaf_axis, leaf_component_label = self.iterset.leaf
        leaf_key = (leaf_axis.id, leaf_component_label)
        for targets_acc in self.iterset.targets_acc:
            equivalent_path, _ = targets_acc[leaf_key]
            equivalent_paths.append(equivalent_path)
        equivalent_paths = tuple(equivalent_paths)

        return (equivalent_paths,)


    # @cached_property
    # def local_index(self):
    #     return LocalLoopIndex(self)

    # @property
    # def i(self):
    #     return self.local_index

    # @property
    # def paths(self):
    #     return tuple(self.iterset.path(*leaf) for leaf in self.iterset.leaves)
    #
    # NOTE: This is confusing terminology. A loop index can be context-sensitive
    # in two senses:
    # 1. axes.index() is context-sensitive if axes is multi-component
    # 2. axes[p].index() is context-sensitive if p is context-sensitive
    # I think this can be resolved by considering axes[p] and axes as "iterset"
    # and handling that separately.
    def with_context(self, context, *args):
        iterset = self.iterset.with_context(context)
        source_path, path = context[self.id]

        # think I want this sorted...
        slices = []
        axis = iterset.root
        while axis is not None:
            cpt = source_path[axis.label]
            slices.append(Slice(axis.label, AffineSliceComponent(cpt)))
            axis = iterset.child(axis, cpt)

        # the iterset is a single-component full slice of the overall iterset
        iterset_ = iterset[slices]
        # return eLoopIndex(iterset_, source_path, path, id=self.id)
        return LoopIndex(iterset_, id=self.id)

    # unsure if this is required
    @property
    def datamap(self):
        return self.iterset.datamap


class InvalidIterationSetException(Pyop3Exception):
    pass


class ScalarIndex(Index):
    fields = {"axis", "component", "value", "id"}

    def __init__(self, axis, component, value, *, id=None):
        super().__init__(axis, component_labels=["XXX"], id=id)
        self.axis = axis
        self.component = component
        self.value = value

    @property
    def leaf_target_paths(self):
        return ((freeze({self.axis: self.component}),),)


# TODO I want a Slice to have "bits" like a Map/CalledMap does
class Slice(Index):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    fields = Index.fields | {"axis", "slices", "label"}

    def __init__(self, axis, slices, *, id=None, label=None):
        slices = as_tuple(slices)

        super().__init__(label=label, id=id)
        self.axis = axis
        self.slices = slices

    @property
    def components(self):
        return self.slices

    @property
    def component_labels(self) -> tuple:
        return tuple(s.label for s in self.slices)

    @cached_property
    def leaf_target_paths(self):
        # We return a collection of 1-tuples because each slice component
        # targets only a single (axis, component) pair. There are no
        # 'equivalent' target paths.
        return tuple(
            (pmap({self.axis: subslice.component}),)
            for subslice in self.slices
        )

    @property
    def expanded(self) -> tuple:
        return (self,)

    def restrict(self, paths):
        new_slice_components = []
        for path in paths:
            found = False
            for slice_component in self.slices:
                if pmap({self.label: slice_component.label}) == path:
                    new_slice_components.append(slice_component)
                    found = True
            if not found:
                raise ValueError("Invalid path provided")

        return type(self)(self.axis, new_slice_components, label=self.label)

    @property
    def datamap(self):
        return merge_dicts([s.datamap for s in self.slices])


# class DuplicateIndexException(Pyop3Exception):
#     pass


class Map(pytools.ImmutableRecord):
    """

    Parameters
    ----------
    connectivity :
        The mappings from input to output for the map. This must be provided as
        an iterable of mappings because the map can both map from *entirely different*
        indices (e.g. multi-component loops that expand to different
        context-free indices) and *semantically equivalent* indices (e.g. a loop
        over ``axes[subset].index()`` has two possible sets of paths and index
        expressions and the map may map from one or both of these but the
        result should be the same). Accordingly, the ``connectivity`` argument
        should provide the different indices as different entries in the iterable,
        and the equivalent indices as different entries in each mapping.

        NOTE: I think this is dead now

        In fact I think to understand the situation we need to consider the following:

        closure(mesh.cells.index()) is hard because mesh.cells is an indexed view of mesh.points,
        and so the loop index carries information on both about. We can feasibly have
        closure(point) AND closure(cell) being separately valid mappings and we don't know
        which we want until we have a target set of axes to make a choice. We therefore want
        to propagate both for as long as possible.
        We could similarly imagine a scenario where closure(cell) yields POINTS, not cells,
        edges and vertices. What do we do then??? That is similar in that we get different
        axis trees that we want to propagate to the end!

        With this in mind, connectivity is therefore the map:

        {
            input_index_label: [
                [*possible component outputs],
                [*possible component outputs]
            ]
        }

        for example, closure gives
        {
            points: [
                [points],
            ]
            cells: [
                [cells, edges, vertices],
                [points],
            ]
            edges: [
                [edges, vertices],
                [points],
            ]
            ...
        }

        but this is really hard because indexing things now gives different AXIS TREES,
        not just different expressions! Indexing therefore must produce an axis forest...

    """

    fields = {"connectivity", "name"}

    counter = 0

    def __init__(self, connectivity, name=None) -> None:
        # if not has_unique_entries(k for m in maps for k in m.connectivity.keys()):
        #     raise DuplicateIndexException("The keys for each map given to the multi-map may not clash")

        super().__init__()
        self.connectivity = ImmutableOrderedDict(connectivity)

        # TODO delete entirely
        if name is None:
            # lazy unique name
            name = f"_Map_{self.counter}"
            self.counter += 1
        self.name = name

    def __call__(self, index):
        # If the input index is context-free then we should return something context-free
        # TODO: Should be encoded in some mixin type
        # if isinstance(index, ContextFreeIndex):
        # if isinstance(index, (ContextFreeIndex, ContextFreeCalledMap)):
        if False:
            return ContextFreeCalledMap(self, index)

            # equiv_domainss = tuple(frozenset(mappings.keys()) for mappings in self.connectivity)
            #
            # map_targets = []
            # empty = True
            # for equiv_call_index_targets in index.leaf_target_paths:
            #
            #     domain_index = None
            #     for call_index_target in equiv_call_index_targets:
            #         for i, equiv_domains in enumerate(equiv_domainss):
            #             if call_index_target in equiv_domains:
            #                 assert domain_index in {None, i}
            #                 domain_index = i
            #
            #     if domain_index is None:
            #         continue
            #
            #     empty = False
            #
            #     equiv_mappings = self.connectivity[domain_index]
            #     ntargets = single_valued(len(mcs) for mcs in equiv_mappings.values())
            #
            #     for itarget in range(ntargets):
            #         equiv_map_targets = []
            #         for call_index_target in equiv_call_index_targets:
            #             if call_index_target not in equiv_domainss[domain_index]:
            #                 continue
            #
            #             orig_component = equiv_mappings[call_index_target][itarget]
            #
            #             # We need to be careful with the slice here because the source
            #             # label needs to match the generated axis later on.
            #             orig_array = orig_component.array
            #             leaf_axis, leaf_component_label = orig_array.axes.leaf
            #             myslice = Slice(leaf_axis.label, [AffineSliceComponent(leaf_component_label, label=leaf_component_label)], label=self.name)
            #             newarray = orig_component.array[index, myslice]
            #
            #             indexed_component = orig_component.copy(array=newarray)
            #             equiv_map_targets.append(indexed_component)
            #         equiv_map_targets = tuple(equiv_map_targets)
            #         map_targets.append(equiv_map_targets)
            #
            # if empty:
            #     import warnings
            #     warnings.warn(
            #         "Provided index is not recognised by the map, so the "
            #         "resulting axes will be empty."
            #     )
            #
            # return ContextFreeCalledMap(self, index, map_targets)
        else:
            return CalledMap(self, index)

    @cached_property
    def datamap(self):
        data = {}
        for bit in self.connectivity.values():
            for map_cpt in bit:
                data.update(map_cpt.datamap)
        return pmap(data)


class ContextSensitiveException(Pyop3Exception):
    """Exception raised when an index is sensitive to the loop index."""


class UnspecialisedCalledMapException(Pyop3Exception):
    """Exception raised when an unspecialised map is used in place of a specialised one.

    This is important for cases like closure(cell) where the result can be either
    a set of points, or sets of cells, edges, and vertices. We say that it is 'unspecialised'
    because it cannot be put into an `IndexTree` and instead should yield two trees as
    an `IndexForest`.

    """


class CalledMap(AxisIndependentIndex, Identified, Labelled, LoopIterable):
    def __init__(self, map, from_index, *, id=None, label=None):
        Identified.__init__(self, id=id)
        Labelled.__init__(self, label=label)
        self.map = map
        self.index = from_index

        # this is an old alias
        self.from_index = from_index

    def __getitem__(self, indices):
        raise NotImplementedError("TODO")
        # figure out the current loop context, just a single loop index
        # from_index = self.from_index
        # while isinstance(from_index, CalledMap):
        #     from_index = from_index.from_index
        # existing_loop_contexts = tuple(
        #     freeze({from_index.id: path}) for path in from_index.paths
        # )
        #
        # index_forest = {}
        # for existing_context in existing_loop_contexts:
        #     axes = self.with_context(existing_context)
        #     index_forest.update(
        #         as_index_forest(indices, axes=axes, loop_context=existing_context)
        #     )
        #
        # array_per_context = {}
        # for loop_context, index_tree in index_forest.items():
        #     indexed_axes = index_axes(index_tree, loop_context, self.axes)
        #
        #     (
        #         target_paths,
        #         index_exprs,
        #         layout_exprs,
        #     ) = _compose_bits(
        #         self.axes,
        #         self.target_paths,
        #         self.index_exprs,
        #         None,
        #         indexed_axes,
        #         indexed_axes.target_paths,
        #         indexed_axes.index_exprs,
        #         indexed_axes.layout_exprs,
        #     )
        #
        #     array_per_context[loop_context] = Dat(
        #         indexed_axes,
        #         data=self.array,
        #         layouts=self.layouts,
        #         target_paths=target_paths,
        #         index_exprs=index_exprs,
        #         name=self.name,
        #         max_value=self.max_value,
        #     )
        # return ContextSensitiveMultiArray(array_per_context)

    # def index(self) -> LoopIndex:
    #     context_map = {
    #         ctx: index_axes(itree, ctx) for ctx, itree in as_index_forest(self).items()
    #     }
    #     context_sensitive_axes = ContextSensitiveAxisTree(context_map)
    #     return LoopIndex(context_sensitive_axes)

    # def iter(self, outer_loops=()):
    #     loop_context = merge_dicts(
    #         iter_entry.loop_context for iter_entry in outer_loops
    #     )
    #     cf_called_map = self.with_context(loop_context)
    #     return iter_axis_tree(
    #         self.index(),
    #         cf_called_map.axes,
    #         cf_called_map.target_paths,
    #         cf_called_map.index_exprs,
    #         outer_loops,
    #     )

    @property
    def name(self):
        return self.map.name

    @property
    def connectivity(self):
        return self.map.connectivity

    @cached_property
    def axes(self) -> IndexedAxisTree:
        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        input_axes = self.index.axes
        axes_ = AxisTree(input_axes.node_map)
        targets = {}
        for input_leaf in input_axes.leaves:

            # something, something make linear, needed for replace to work properly
            if input_axes.is_empty:
                linear_input_axes = AxisTree(input_axes.node_map)
            else:
                raise NotImplementedError

            if input_leaf is None:
                leaf_key = None
            else:
                input_leaf_axis, input_leaf_component = input_leaf
                leaf_key = (input_leaf_axis.id, input_leaf_component.label)

            # equivalent_inputs = tuple(
            #     targets_acc[leaf_key] for targets_acc in input_axes.targets_acc
            # )
            equivalent_inputs = input_axes.targets

            found = False
            for input_targets in equivalent_inputs:

                # traverse the input axes and collect the final target path
                if input_axes.is_empty:
                    input_path, input_exprs = input_targets[None]
                else:
                    raise NotImplementedError

                if input_path in self.connectivity:
                    if found:
                        raise UnspecialisedCalledMapException(
                            "Multiple (equivalent) input paths have been found that "
                            "are accepted by the map. This ambiguity makes it impossible "
                            "to form an IndexTree."
                        )

                    if len(self.connectivity[input_path]) > 1:
                        raise UnspecialisedCalledMapException(
                            "Multiple (equivalent) output paths are generated by the map. "
                            "This ambiguity makes it impossible to form an IndexTree."
                        )

                    found = True
                    output_spec = just_one(self.connectivity[input_path])

                    # make a method
                    subaxis, subtargets = _make_leaf_axis_from_called_map_new(
                        self.name, output_spec, linear_input_axes, input_targets,
                    )

                    axes_ = axes_.add_axis(subaxis, leaf_key)
                    targets.update(subtargets)

            if not found:
                assert False

        targets = pmap(targets)

        # Since maps are necessarily restricted to a single interpretation we only
        # have one possible interpretation of the targets.
        targets = (targets,)
        return IndexedAxisTree(axes_.node_map, None, targets=targets)

    @property
    def is_context_free(self) -> bool:
        return self.index.is_context_free

    # NOTE: nothing about this is specific to an index
    @property
    def leaf_target_paths(self) -> tuple:
        targets_acc = just_one(self.axes.targets_acc)

        paths = []
        for leaf in self.axes.leaves:
            if leaf is None:
                leaf_key = None
            else:
                leaf_axis, leaf_component_label = leaf
                leaf_key = (leaf_axis.id, leaf_component_label)

            equivalent_paths = (targets_acc[leaf_key][0],)
            paths.append(equivalent_paths)
        return tuple(paths)

    @cached_property
    def expanded(self):
        """Return a `tuple` of maps specialised to possible inputs and outputs.

        This is necessary because the input index may match with multiple possible
        map inputs, and the map may have multiple possible outputs for each input.

        For example, closure(cell) matches inputs of points and cells, and has output
        cells, edges, and vertices, and separately points.

        """
        restricted_maps = []
        for index in self.call_index.expanded:
            for input_path in index.leaf_target_paths:
                for output_spec in self.connectivity[input_path]:
                    restricted_connectivity = {input_path: (output_spec,)}
                    restricted_map = Map(restricted_connectivity, self.name)(index)
                    restricted_maps.append(restricted_map)
        return tuple(restricted_maps)


    @cached_property
    def _source_paths(self):
        return tuple(p for p, _ in self.connectivity)

    @property
    def _connectivity_dict(self):
        return pmap(self.connectivity)

    # TODO cleanup
    def with_context(self, context, axes=None):
        raise NotImplementedError
        # maybe this line isn't needed?
        # cf_index = self.from_index.with_context(context, axes)
        cf_index = self.from_index
        leaf_target_paths = tuple(
            freeze({mcpt.target_axis: mcpt.target_component})
            for path in cf_index.leaf_target_paths
            for mcpt in self.map.connectivity[path]
            # if axes is None we are *building* the axes from this map
            if axes is None
            or axes.is_valid_path(
                {mcpt.target_axis: mcpt.target_component}, complete=False
            )
        )
        if len(leaf_target_paths) == 0:
            raise RuntimeError
        return ContextFreeCalledMap(self.map, cf_index, leaf_target_paths, id=self.id)

    @property
    def name(self) -> str:
        return self.map.name

    # NOTE: This can now probably go and use .axes instead

    # def iter(self, eager=False):
    #     if eager:
    #         raise NotImplementedError
    #
    #     index_forest = as_index_forest(self)
    #     assert index_forest.keys() == {pmap()}
    #     index_tree = index_forest[pmap()]
    #     iterset = index_axes(index_tree, pmap())
    #
    #     # The loop index from a context-free map can be context-sensitive if it
    #     # has multiple components.
    #     if len(iterset.leaves) == 1:
    #         path = iterset.path(*iterset.leaf)
    #         target_path = {}
    #         for ax, cpt in iterset.path_with_nodes(*iterset.leaf).items():
    #             target_path.update(iterset.target_paths.get((ax.id, cpt), {}))
    #         return ContextFreeLoopIndex(iterset, path, target_path)
    #     else:
    #         return LoopIndex(iterset)

    def restrict(self, paths):
        breakpoint()
        ...

    # is this ever used?
    # @property
    # def components(self):
    #     return self.map.connectivity[self.index.target_paths]

    # @property
    # def leaf_target_paths(self):
    #     return tuple(tuple(mc.target_path for mc in equiv_mcs) for equiv_mcs in self.targets)

    # @cached_property
    # def axes(self):
    #     return self._axes_info[0]
    #
    # @cached_property
    # def target_paths(self):
    #     return self._axes_info[1]
    #
    # @cached_property
    # def index_exprs(self):
    #     return self._axes_info[2]
    #
    # @cached_property
    # def layout_exprs(self):
    #     return self._axes_info[3]
    #
    # # TODO This is bad design, unroll the traversal and store as properties
    # @cached_property
    # def _axes_info(self):
    #     return _index_axes_index(self, (), prev_axes=None)


# remove this old class, loop contexts are now implicit
# ContextFreeCalledMap = CalledMap


class ContextSensitiveCalledMap(ContextSensitiveLoopIterable):
    pass


class InvalidIndexException(Pyop3Exception):
    pass


@functools.singledispatch
def _index_axes_index(index, *args, **kwargs):
    """TODO.

    Case 1: loop indices

    Assume we have ``axis[p]`` with ``p`` a `ContextFreeLoopIndex`.
    If p came from other_axis[::2].iter(), then it has *2* possible
    target paths and expressions: over the indexed or unindexed trees.
    Therefore when we index axis with p we must account for this, hence all
    indexing operations return a tuple of possible, equivalent, targets.

    Then, when we combine it all together, if we imagine having 2 loop indices
    like this, then we need the *product* of them to enumerate all possible
    targets.

    """
    raise TypeError(f"No handler provided for {type(index)}")


@_index_axes_index.register
def _(
    cf_loop_index: LoopIndex,
    **_,
):
    # This function should return {None: [(path0, expr0), (path1, expr1)]}
    # where path0 and path1 are "equivalent"
    # This entails in inversion of loop_index.iterset.targets which has the form
    # [
    #   {key: (path0, expr0), ...},
    #   {key: (path1, expr1), ...}
    # ]

    axes = AxisTree()
    # target_paths = freeze({None: cf_loop_index.leaf_target_paths})

    # Example:
    # If we assume that the loop index has target expressions
    #     AxisVar("a") * 2     and       AxisVar("b")
    # then this will return
    #     LoopIndexVar(p, "a") * 2      and LoopIndexVar(p, "b")
    # replacer = LoopIndexReplacer(cf_loop_index)
    replace_map = {
        (axis.id, component_label): (
            {axis.label: component_label},
            {axis.label: LoopIndexVar(cf_loop_index, axis.label)},
        )
        for axis, component_label in cf_loop_index.iterset.path_with_nodes(cf_loop_index.iterset.leaf).items()
    }
    targets = {None: []}
    for equivalent_targets in cf_loop_index.iterset.paths_and_exprs:
        new_path = {}
        new_exprs = {}
        for (orig_path, orig_exprs) in equivalent_targets.values():
            new_path.update(orig_path)
            for axis_label, orig_expr in orig_exprs.items():
                new_exprs[axis_label] = replace(orig_expr, cf_loop_index.iterset, replace_map)
        new_path = pmap(new_path)
        new_exprs = pmap(new_exprs)
        targets[None].append((new_path, new_exprs))
    targets[None] = tuple(targets[None])

    return (
        axes,
        pmap(targets),
        {},
        (),
        {},
    )


@_index_axes_index.register
def _(index: ScalarIndex, **_):
    target_path_and_exprs = pmap({None: ((index.leaf_target_paths, pmap({index.axis: index.value})),)})
    # index_exprs = pmap({None: (,)})
    layout_exprs = pmap({None: 0})
    return (
        AxisTree(),
        target_path_and_exprs,
        # index_exprs,
        layout_exprs,
        (),
        {},
    )


@_index_axes_index.register
# def _(slice_: Slice, *, parent_indices, prev_axes, **_):
def _(slice_: Slice, *, prev_axes, **_):
    # If we are just taking a component from a multi-component array,
    # e.g. mesh.points["cells"], then relabelling the axes just leads to
    # needless confusion. For instance if we had
    #
    #     myslice0 = Slice("mesh", AffineSliceComponent("cells", step=2))
    #
    # then mesh.points[myslice0] would work but mesh.points["cells"][myslice0]
    # would fail.
    # As a counter example, if we have non-trivial subsets then this sort of
    # relabelling is essential for things to make sense. If we have two subsets:
    #
    #     subset0 = Slice("mesh", Subset("cells", [1, 2, 3]))
    #
    # and
    #
    #     subset1 = Slice("mesh", Subset("cells", [4, 5, 6]))
    #
    # then mesh.points[subset0][subset1] is confusing, should subset1 be
    # assumed to work on the already sliced axis? This can be a major source of
    # confusion for things like interior facets in Firedrake where the first slice
    # happens in one function and the other happens elsewhere. We hit situations like
    #
    #     mesh.interior_facets[interior_facets_I_want]
    #
    # conflicts with
    #
    #     mesh.interior_facets[facets_I_want]
    #
    # where one subset is given with facet numbering and the other with interior
    # facet numbering. The labels are the same so identifying this is really difficult.
    #
    # We fix this here by requiring that non-full slices perform a relabelling and
    # full slices do not. A full slice is defined to be a slice where all of the
    # components are affine with start 0, stop None and step 1. The components must
    # also not already have a label since that would take precedence.
    #
    is_full = all(
        isinstance(s, AffineSliceComponent) and s.is_full and s.label_was_none
        for s in slice_.slices
    )
    # NOTE: We should be able to eagerly return here?

    if is_full:
        axis_label = slice_.axis
    else:
        # breakpoint()
        axis_label = slice_.label

    components = []
    target_path_per_subslice = []
    index_exprs_per_subslice = []
    layout_exprs_per_subslice = []

    # If there are multiple axes that match the slice then they must be
    # identical (apart from their ID, which is ignored in equality checks).
    target_axis = single_valued(
        ax for ax in prev_axes.nodes if ax.label == slice_.axis
    )

    for i, slice_component in enumerate(slice_.slices):
        target_component = just_one(
            c for c in target_axis.components if c.label == slice_component.component
        )

        if isinstance(slice_component, AffineSliceComponent):
            if isinstance(target_component.count, Dat):
                if (
                    slice_component.start != 0
                    or slice_component.step != 1
                ):
                    raise NotImplementedError("TODO")

                # need to index things using replace() instead of parent_indices
                # NOTE: is it strictly necessary to index here?
                size = target_component.count

                if slice_component.stop is None:
                    if len(parent_indices) == 0:
                        size = target_component.count
                    else:
                        # It is not necessarily the case that all of parent_indices is
                        # required to index count.
                        # TODO: Unify with if len(parent_indices) == 0, should work for both
                        size = target_component.count.getitem(parent_indices, allow_unused=True)
                else:
                    size = slice_component.stop

            else:
                if slice_component.stop is None:
                    stop = target_component.count
                else:
                    stop = slice_component.stop

                # if target_cpt.distributed:
                if False:
                    pass
                    # if subslice.start != 0 or subslice.step != 1:
                    #     raise NotImplementedError
                    #
                    # owned_count = min(target_cpt.owned_count, stop)
                    # count = stop
                    #
                    # if owned_count == count:
                    #     size = count
                    # else:
                    #     size = (owned_count, count)
                else:
                    size = math.ceil((stop - slice_component.start) / slice_component.step)

        else:
            assert isinstance(slice_component, Subset)
            size = slice_component.array.axes.leaf_component.count

        # if slice_.label == "closure":
        #     breakpoint()

        if is_full:
            component_label = slice_component.component
        else:
            # TODO: Ideally the default labels here would be integers if not
            # somehow provided. Perhaps the issue stems from the fact that the label
            # attribute is used for two things: identifying paths in the index tree
            # and labelling the resultant axis component.
            component_label = slice_component.label

        cpt = AxisComponent(size, label=component_label, unit=target_component.unit)
        components.append(cpt)

    axis = Axis(components, label=axis_label)
    axes = AxisTree(axis)

    for i, slice_component in enumerate(slice_.slices):
        # if is_full:
        if False:
            target_path_per_subslice.append({})
            index_exprs_per_subslice.append({})
            layout_exprs_per_subslice.append({})
        else:
            target_path_per_subslice.append(pmap({slice_.axis: slice_component.component}))

            newvar = AxisVar(axis)
            # layout_var = AxisVar()
            if isinstance(slice_component, AffineSliceComponent):
                index_exprs_per_subslice.append(
                    freeze(
                        {
                            slice_.axis: newvar * slice_component.step + slice_component.start,
                        }
                    )
                )
                # layout_exprs_per_subslice.append(
                #     pmap({slice_.label: (layout_var - slice_component.start) // slice_component.step})
                # )
            else:
                assert isinstance(slice_component, Subset)

                # below is also used for maps - cleanup
                # subset_array = slice_component.array
                # subset_axes = subset_array.axes
                #
                # if isinstance(subset_axes, IndexedAxisTree):
                #     raise NotImplementedError("Need more paths, not just 2")
                #
                # # must be single component
                # assert subset_axes.leaf

                # my_target_path = merge_dicts(just_one(subset_axes.paths).values())
                # old_index_exprs = merge_dicts(just_one(subset_axes.index_exprs).values())
                #
                # my_index_exprs = {}
                # index_expr_replace_map = {subset_axes.leaf_axis.label: newvar}
                # replacer = IndexExpressionReplacer(index_expr_replace_map)
                # for axlabel, index_expr in old_index_exprs.items():
                #     my_index_exprs[axlabel] = replacer(index_expr)
                # subset_var = ArrayVar(slice_component.array, my_index_exprs, my_target_path)
                # NOTE: This replacement could probably be done more eagerly
                subset_axes = slice_component.array.axes
                assert subset_axes.is_linear and subset_axes.depth == 1
                subset_axis = subset_axes.root
                linear_axis = Axis([axis.components[i]], axis.label)
                linear_axes = AxisTree(linear_axis)
                replace_path = {subset_axis.label: subset_axis.component.label}
                replace_exprs = {subset_axis.label: AxisVar(axis)}
                replace_bits = {(linear_axis.id, axis.components[i].label): (replace_path, replace_exprs)}

                index_exprs_per_subslice.append(freeze({slice_.axis: replace(slice_component.array, linear_axes, replace_bits)}))

    target_path_per_component = {}
    index_exprs_per_component = {}
    # layout_exprs_per_component = {}
    for cpt, target_path, index_exprs in strict_zip(
        components,
        target_path_per_subslice,
        index_exprs_per_subslice,
    ):
        target_path_per_component[axis.id, cpt.label] = ((freeze(target_path), freeze(index_exprs)),)

    return (
        axes,
        target_path_per_component,
        {},
        (),  # no outer loops
        {},
    )


@_index_axes_index.register
def _(
    called_map: CalledMap,
    **kwargs,
):
    # compress the targets here
    compressed_targets = {}
    for leaf_axis, leaf_component_label in called_map.axes.leaves:
        leaf_key = (leaf_axis.id, leaf_component_label)
        compressed_targets[leaf_key] = tuple(t[leaf_key] for t in called_map.axes.targets)

    return called_map.axes, compressed_targets, {}, (), {}


def _make_leaf_axis_from_called_map_new(map_name, output_spec, linear_input_axes, input_paths_and_exprs):
    components = []
    for map_output in output_spec:
        # NOTE: This could probably be done more eagerly.
        arity = replace(map_output.arity, linear_input_axes, input_paths_and_exprs)
        component = AxisComponent(arity, label=map_output.label)
        components.append(component)
    axis = Axis(components, label=map_name)

    targets = {}
    for component, map_output in strict_zip(components, output_spec):
        if not isinstance(map_output, TabulatedMapComponent):
            raise NotImplementedError("Currently we assume only arrays here")

        linear_axis = Axis(component, axis.label)
        linear_axes = linear_input_axes.add_axis(linear_axis, linear_input_axes.leaf)

        map_output_leaf = map_output.array.axes.leaf
        leaf_axis, leaf_component_label = map_output_leaf

        paths_and_exprs = input_paths_and_exprs | {(linear_axis.id, component.label): (pmap({leaf_axis.label: leaf_component_label}), pmap({leaf_axis.label: AxisVar(leaf_axis)}))}

        target_path = pmap({map_output.target_axis: map_output.target_component})

        # target_exprs = pmap({map_output.target_axis: replace(map_output.array, linear_axes, paths_and_exprs)})
        target_exprs = pmap({map_output.target_axis: replace(map_output.array, linear_axes, paths_and_exprs)})
        targets[axis.id, component.label] = (target_path, target_exprs)
    targets = pmap(targets)

    return (axis, targets)


def index_axes(
    index_tree: Union[IndexTree, Ellipsis],
    loop_context,
    axes: Optional[Union[AxisTree, AxisForest]] = None,
# ) -> AxisForest:
    ):
    """Build an axis tree from an index tree.

    Parameters
    ----------
    axes :
        An axis tree that is being indexed. This argument is not always needed
        if, say, we are constructing the iteration set for the expression
        ``map(p).index()``. If not provided then some indices (e.g. unbounded
        slices) will no longer work.

    Returns
    -------
    AxisTree :
        The new axis tree.

    plus target paths and target exprs

    """
    assert isinstance(axes, (AxisTree, IndexedAxisTree))

    if index_tree is Ellipsis:
        return axes

    (
        indexed_axes,
        indexed_target_paths_and_exprs_compressed,
        _,
        _,
        partial_linear_index_trees,
    ) = _index_axes(
        index_tree,
        loop_indices=loop_context,
        prev_axes=axes,
    )

    indexed_target_paths_and_exprs = expand_compressed_target_paths(indexed_target_paths_and_exprs_compressed)

    # If the original axis tree is unindexed then no composition is required.
    if isinstance(axes, AxisTree):
        return IndexedAxisTree(
            indexed_axes.node_map,
            axes,
            targets=indexed_target_paths_and_exprs + (indexed_axes._source_path_and_exprs,),
            layout_exprs={},
            outer_loops=indexed_axes.outer_loops,
        )

    all_target_paths_and_exprs = []
    for orig_path in axes.paths_and_exprs:

        match_found = False
        for indexed_path_and_exprs in indexed_target_paths_and_exprs:
            # catch invalid indexed_target_paths_and_exprs
            if not _index_info_targets_axes(indexed_axes, indexed_path_and_exprs, axes):
                continue

            assert not match_found, "don't expect multiple hits"
            target_path_and_exprs = compose_targets(axes, orig_path, indexed_axes, indexed_path_and_exprs)
            match_found = True
            all_target_paths_and_exprs.append(target_path_and_exprs)
        assert match_found, "must hit once"

    # TODO: reorder so the if statement captures the composition and this line is only needed once
    return IndexedAxisTree(
        indexed_axes.node_map,
        axes.unindexed,
        # targets=all_target_paths_and_exprs | {indexed_axes._source_path_and_exprs},
        targets=all_target_paths_and_exprs,
        layout_exprs={},
        outer_loops=indexed_axes.outer_loops,
    )



def restrict_targets(targets, indexed_axes: AxisTree, unindexed_axes: AxisTree, *, axis=None) -> PMap:
    restricted = collections.defaultdict(dict)

    if axis is None:
        for target_set in targets.get(None, ()):
            restricted[None].update(_matching_target(target_set, unindexed_axes))

        if indexed_axes.is_empty:
            # nothing more to be done
            return freeze(restricted)
        else:
            axis = indexed_axes.root

    # Make the type checker happy
    axis = cast(Axis, axis)

    for component in axis.components:
        axis_key = (axis.id, component.label)
        if axis_key in targets:
            target_set = targets[axis_key]
            restricted[axis_key].update(_matching_target(target_set, unindexed_axes))

        if subaxis := indexed_axes.child(axis, component):
            subrestricted = restrict_targets(targets, indexed_axes, unindexed_axes, axis=subaxis)
            restricted.update(subrestricted)

    return freeze(restricted)


def _matching_target(targets: PMap, orig_axes: AxisTree) -> PMap:
    # NOTE: Ideally this should return just_one(...) instead of
    # single_valued(...) but because we don't have pass-through
    # indexing yet we sometimes get duplicate axes inside targets.
    # NOTE: But passthrough indexing is now wrong to try and do...
    return single_valued(
        t for t in targets
        if all(ax in orig_axes.node_labels for ax in t.keys())
    )


# NOTE: Arguably this does not need to be done eagerly. The targets can be per-axis
# until later.
def accumulate_targets(targets, indexed_axes, *, axis=None, target_acc=None):
    """Traverse indexed_axes and accumulate per-axis ``targets`` as we go down."""
    if indexed_axes.is_empty:
        return targets

    targets_merged = {}

    if strictly_all(x is None for x in {axis, target_acc}):
        axis = indexed_axes.root
        target_acc = targets.get(None, pmap())
        targets_merged[None] = target_acc

    # To make the type checker happy
    axis = cast(Axis, axis)

    for component in axis.components:
        key = (axis.id, component.label)
        target_acc_ = target_acc | targets.get(key, {})
        targets_merged[key] = target_acc_

        if subaxis := indexed_axes.child(axis, component):
            targets_merged.update(
                accumulate_targets(targets, indexed_axes, axis=subaxis, target_acc=target_acc_)
            )
    return freeze(targets_merged)


def _index_axes(
    index_tree,
    *,
    loop_indices,  # NOTE: I don't think that this is ever needed, remove?
    prev_axes,
    index=None,
    # parent_indices=None,
):
    if index is None:
        index = index_tree.root
        # parent_indices = ()

    # Make the type checker happy
    index = cast(Index, index)
    # parent_indices = cast(tuple, parent_indices)

    axes_per_index, target_path_per_cpt_per_index, _, outer_loops, _ = _index_axes_index(
        index,
        loop_indices=loop_indices,
        prev_axes=prev_axes,
        # parent_indices=parent_indices,
    )

    target_path_per_cpt_per_index = dict(target_path_per_cpt_per_index)

    if axes_per_index:
        leafkeys = axes_per_index.leaves
    else:
        leafkeys = [None]

    subaxes = {}
    # partial_index_trees = {}
    for leafkey, subindex in strict_zip(
        leafkeys, index_tree.node_map[index.id]
    ):
        if subindex is None:
            continue

        subtree, subpathsandexprs, _, subouterloops, subpartialindextree = _index_axes(
            index_tree,
            loop_indices=loop_indices,
            prev_axes=prev_axes,
            index=subindex,
        )
        subaxes[leafkey] = subtree

        if None in subpathsandexprs:
            # in this case we need to tweak subpathsandexprs to point at the parent instead
            breakpoint()

        else:
            target_path_per_cpt_per_index.update(subpathsandexprs)

        outer_loops += subouterloops

    target_path_per_component = ImmutableOrderedDict(target_path_per_cpt_per_index)

    # This this is no longer necessary
    axes = AxisTree(axes_per_index.node_map)
    for k, subax in subaxes.items():
        if subax is not None:
            if axes:
                axes = axes.add_subtree(subax, *k)
            else:
                axes = AxisTree(subax.node_map)

    return (
        axes,
        target_path_per_component,
        {},
        outer_loops,
        "anything"
    )


# NOTE: should be similar to index_exprs
# def compose_targets(orig_axes, orig_target_paths_and_exprs, indexed_axes, indexed_target_paths_and_exprs, partial_linear_index_trees, *, axis=None, indexed_target_paths_acc=None, visited_orig_axes=None):
# def compose_targets(orig_axes, orig_target_paths_and_exprs, indexed_axes, indexed_target_paths_and_exprs, *, axis=None, indexed_target_paths_acc=None, visited_orig_axes=None, indexed_target_exprs_acc=None):
def compose_targets(orig_axes, orig_target_paths_and_exprs, indexed_axes, indexed_target_paths_and_exprs, *, axis=None, indexed_axes_acc=None, indexed_target_paths_and_exprs_acc=None, visited_orig_axes=None):
    """

    Traverse ``indexed_axes``, picking up bits from indexed_target_paths and keep
    trying to address orig_axes.paths with it. If there is a hit then we take that
    bit of the original target path into the new location.

    We *do not* accumulate things as we go. The final result should be the map

    { (indexed_axis, component) -> ((target_path1 | target_path2, ...), (targetexpr1 | targetexpr2)), ... }

    Things are complicated by the fact that not all of the targets from indexed_target_paths
    will resolve. Imagine axisB[p] where p is from axisA[::2].iter(). p targets 2 things and
    only one will match with axisB. We need to check for this outside the function.

    ---

    indexed_axes_acc is a *linear* version of the visited indexed axis tree. It provides the necessary information for replace(...) to interpret the target_paths_and_exprs

    """
    assert not orig_axes.is_empty

    composed_target_paths_and_exprs = collections.defaultdict(dict)

    if axis is None:  # strictly_all
        visited_orig_axes = frozenset()

        indexed_axes_acc = AxisTree()
        indexed_target_paths_and_exprs_acc = {None: indexed_target_paths_and_exprs.get(None, (pmap(), pmap()))}

        # special handling for None entries
        none_mapped_target_path = {}
        none_mapped_target_exprs = {}

        orig_none_mapped_target_path, orig_none_mapped_target_exprs = orig_target_paths_and_exprs.get(None, ({}, {}))

        none_mapped_target_path |= orig_none_mapped_target_path
        for orig_axis_label, orig_index_expr in orig_none_mapped_target_exprs.items():
            none_mapped_target_exprs[orig_axis_label] = replace(orig_index_expr, indexed_axes_acc, indexed_target_paths_and_exprs_acc)

        # make sure to add existing None entries - these will not require composition I think?

        # debug
        # if not indexed_axes.is_empty and indexed_axes.root.id == "_id_Axis_413":
        #     breakpoint()

        # copied from below, refactor

        # does the accumulated path match a part of orig_axes?
        accumulated_path = merge_dicts(p for p, _ in indexed_target_paths_and_exprs_acc.values())
        # if orig_axes.is_valid_path(indexed_target_paths_acc):
        if orig_axes.is_valid_path(accumulated_path):
            # if so then add previously unvisited node values to the composed target path for the current axis
            # for orig_axis, orig_component in orig_axes.detailed_path(indexed_target_paths_acc).items():
            for orig_axis, orig_component in orig_axes.detailed_path(accumulated_path).items():
                if orig_axis in visited_orig_axes:
                    continue
                visited_orig_axes |= {orig_axis}

                orig_key = (orig_axis.id, orig_component.label)
                if orig_key in orig_target_paths_and_exprs:
                    orig_target_path, orig_target_exprs = orig_target_paths_and_exprs[orig_key]

                    none_mapped_target_path |= orig_target_path
                    for orig_axis_label, orig_index_expr in orig_target_exprs.items():
                        none_mapped_target_exprs[orig_axis_label] = replace(orig_index_expr, indexed_axes_acc, indexed_target_paths_and_exprs_acc)

        composed_target_paths_and_exprs[None] = (
            pmap(none_mapped_target_path), pmap(none_mapped_target_exprs)
        )

        if indexed_axes.is_empty:
            return freeze(composed_target_paths_and_exprs)
        else:
            axis = indexed_axes.root

    for component in axis.components:
        # partial_index_tree = partial_linear_index_trees[axis.id, component.label]

        linear_axis = Axis([component], axis.label)
        indexed_axes_acc_ = indexed_axes_acc.add_axis(linear_axis, indexed_axes_acc.leaf)

        indexed_target_paths_and_exprs_acc_ = indexed_target_paths_and_exprs_acc | {(linear_axis.id, component.label): indexed_target_paths_and_exprs[axis.id, component.label]}
        # indexed_target_paths_acc_ = indexed_target_paths_acc | indexed_target_path
        # indexed_target_exprs_acc_ = indexed_target_exprs_acc | indexed_target_exprs

        visited_orig_axes_ = visited_orig_axes

        # does the accumulated path match a part of orig_axes?
        accumulated_path = merge_dicts(p for p, _ in indexed_target_paths_and_exprs_acc_.values())
        # if orig_axes.is_valid_path(indexed_target_paths_acc_):
        if orig_axes.is_valid_path(accumulated_path):
            # if so then add previously unvisited node values to the composed target path for the current axis
            # for orig_axis, orig_component in orig_axes.detailed_path(indexed_target_paths_acc_).items():
            for orig_axis, orig_component in orig_axes.detailed_path(accumulated_path).items():
                if orig_axis in visited_orig_axes:
                    continue
                visited_orig_axes_ |= {orig_axis}

                orig_key = (orig_axis.id, orig_component.label)

                if orig_key in orig_target_paths_and_exprs:
                    orig_target_path, orig_target_exprs = orig_target_paths_and_exprs[orig_key]

                    # now index exprs
                    new_exprs = {}
                    for orig_axis_label, orig_index_expr in orig_target_exprs.items():
                        # orig_index_expr_new = apply_index_tree_to_expr(orig_index_expr, partial_index_tree)
                        # orig_index_expr_new = replace(orig_index_expr, indexed_axes_acc_, indexed_target_paths_and_exprs)
                        # new_exprs[orig_axis_label] = replacer(orig_index_expr)
                        new_exprs[orig_axis_label] = replace(orig_index_expr, indexed_axes_acc_, indexed_target_paths_and_exprs_acc_)

                    composed_target_paths_and_exprs[axis.id, component.label] = (orig_target_path, freeze(new_exprs))

        # now recurse
        if subaxis := indexed_axes.child(axis, component):
            composed_target_paths_ = compose_targets(
                orig_axes,
                orig_target_paths_and_exprs,
                indexed_axes,
                indexed_target_paths_and_exprs,
                # partial_linear_index_trees,
                axis=subaxis,
                indexed_axes_acc=indexed_axes_acc_,
                indexed_target_paths_and_exprs_acc=indexed_target_paths_and_exprs_acc_,
                visited_orig_axes=visited_orig_axes_,
            )
            composed_target_paths_and_exprs.update(composed_target_paths_)

    return freeze(composed_target_paths_and_exprs)


def _index_info_targets_axes(indexed_axes, index_info, orig_axes) -> bool:
    """Return whether the index information targets the original axis tree.

    This is useful for when multiple interpretations of axis information are
    provided (e.g. with loop indices) and we want to filter for the right one.

    """
    for indexed_leaf in indexed_axes.leaves:
        none_target_path, _ = index_info.get(None, (pmap(), pmap()))
        target_path_acc = dict(none_target_path)
        for axis, component_label in indexed_axes.path_with_nodes(indexed_leaf).items():
            target_path, _ = index_info.get((axis.id, component_label), (pmap(), pmap()))
            target_path_acc |= target_path

        if not orig_axes.is_valid_path(target_path_acc):
            return False

    return True


# TODO: just get rid of this, assuming the new system works
def expand_compressed_target_paths(compressed_target_paths):
    return expand_collection_of_iterables(compressed_target_paths)


# NOTE: Think this should just be a replace
@functools.singledispatch
def apply_index_tree_to_expr(obj, indices):
    if isinstance(obj, Dat):
        return obj[indices]
    else:
        raise TypeError


@apply_index_tree_to_expr.register
def _(var: AxisVar, indices):
    return var


def compose_axes(orig_axes, indexed_axes, indexed_target_paths, indexed_target_exprs):
    assert not orig_axes.is_empty

    composed_target_paths = []
    composed_target_exprs = []
    for orig_paths, orig_index_exprs in strict_zip(
        orig_axes.paths, orig_axes.index_exprs
    ):
        composed_target_paths_, composed_target_exprs_ = _compose_axes(
            orig_axes,
            orig_paths,
            orig_index_exprs,
            indexed_axes,
            indexed_target_paths,
            indexed_target_exprs,
        )
        composed_target_paths.append(composed_target_paths_)
        composed_target_exprs.append(composed_target_exprs_)
    return IndexedAxisTree(
        indexed_axes.node_map,
        orig_axes.unindexed,
        target_paths=composed_target_paths,
        target_exprs=composed_target_exprs,
        layout_exprs={},
        outer_loops=indexed_axes.outer_loops,
    )


def _compose_axes(
        orig_axes,
        orig_paths,
        orig_index_exprss,
        indexed_axes,
        indexed_target_paths,
        indexed_target_exprss,
        *,
        indexed_axis=None,
):
    # This code attaches a target_path/target_expr to every node in the tree. Is
    # this strictly necessary?

    composed_target_paths = collections.defaultdict(dict)
    composed_target_exprss = collections.defaultdict(dict)

    if indexed_axis is None:
        # Keep the bits that are already indexed out.
        composed_target_paths[None].update(orig_paths.get(None, {}))
        composed_target_exprss[None].update(orig_index_exprss.get(None, {}))

        indexed_target_path = indexed_target_paths.get(None, {})
        indexed_target_exprs = indexed_target_exprss.get(None, {})
        if orig_axes.is_valid_path(indexed_target_path):
            orig_axis, orig_component = orig_axes._node_from_path(indexed_target_path)
            orig_key = (orig_axis.id, orig_component.label)

            # 1. Determine target paths.
            composed_target_paths[None] = orig_paths[orig_key]

            # 2. Determine target expressions. This is done via an *inside* substitution.
            orig_index_exprs = orig_index_exprss.get(orig_key, {})
            # replacer = IndexExpressionReplacer(indexed_target_exprs)
            for orig_axis_label, orig_index_expr in orig_index_exprs.items():
                # composed_target_exprss[None][orig_axis_label] = replacer(orig_index_expr)
                composed_target_exprss[None][orig_axis_label] = replace(orig_index_expr, indexed_target_path, indexed_target_exprs)
                print("DDD", composed_target_exprss[None][orig_axis_label])

        if indexed_axes.is_empty:
            # Can do nothing more, stop here.
            return (freeze(composed_target_paths), freeze(composed_target_exprss))
        else:
            indexed_axis = indexed_axes.root

    for indexed_component in indexed_axis.components:
        indexed_key = (indexed_axis.id, indexed_component.label)
        indexed_target_path = indexed_target_paths.get(indexed_key, {})
        indexed_target_exprs = indexed_target_exprss.get(indexed_key, {})

        if orig_axes.is_valid_path(indexed_target_path):
            orig_axis, orig_component = orig_axes._node_from_path(indexed_target_path)
            orig_key = (orig_axis.id, orig_component.label)

            # 1. Determine target_paths.
            composed_target_paths[indexed_key] = orig_paths.get(orig_key, {})

            # 2. Determine index_exprs.
            orig_index_exprs = orig_index_exprss.get(orig_key, {})
            # replacer = IndexExpressionReplacer(indexed_target_exprs)
            for orig_axis_label, orig_index_expr in orig_index_exprs.items():
                composed_target_exprss[indexed_key][orig_axis_label] = replace(orig_index_expr, indexed_target_path, indexed_target_exprs)

            # 3. Determine layout_exprs...
            # ...
            # now do the layout expressions, this is simpler since target path magic isnt needed
            # compose layout expressions, this does an *outside* substitution
            # so the final replace map is src -> h(final)
            # we start with src -> f(intermediate)
            # and intermediate -> g(final)

            # only do this if we are indexing an axis tree, not an array
            # if prev_layout_exprs is not None:
            #     full_replace_map = merge_dicts(
            #         [
            #             prev_layout_exprs.get((tgt_ax.id, tgt_cpt.label), pmap())
            #             for tgt_ax, tgt_cpt in detailed_path.items()
            #         ]
            #     )
            #     for ikey, layout_expr in new_partial_layout_exprs.items():
            #         # always 1:1 for layouts
            #         mykey, myvalue = just_one(layout_expr.items())
            #         mytargetpath = just_one(itarget_paths[ikey].keys())
            #         # layout_expr_replace_map = {
            #         #     mytargetpath: full_replace_map[mytargetpath]
            #         # }
            #         layout_expr_replace_map = full_replace_map
            #         new_layout_expr = IndexExpressionReplacer(layout_expr_replace_map)(
            #             myvalue
            #         )
            #
            #         # this is a trick to get things working in Firedrake, needs more
            #         # thought to understand what is going on
            #         if ikey in layout_exprs and mykey in layout_exprs[ikey]:
            #             assert layout_exprs[ikey][mykey] == new_layout_expr
            #         else:
            #             layout_exprs[ikey][mykey] = new_layout_expr

        if indexed_subaxis := indexed_axes.child(indexed_axis, indexed_component):
            (
                subtarget_paths,
                subtarget_exprs,
            ) = _compose_axes(
                orig_axes,
                orig_paths,
                orig_index_exprss,
                indexed_axes,
                indexed_target_paths,
                indexed_target_exprss,
                indexed_axis=indexed_subaxis,
            )
            composed_target_paths.update(subtarget_paths)
            composed_target_exprss.update(subtarget_exprs)

    return (
        freeze(composed_target_paths),
        freeze(composed_target_exprss),
    )


@dataclasses.dataclass(frozen=True)
class IndexIteratorEntry:
    index: LoopIndex
    source_path: PMap
    target_path: PMap
    source_exprs: PMap
    target_exprs: PMap

    @property
    def loop_context(self):
        return freeze({self.index.id: (self.source_path, self.target_path)})

    @property
    def replace_map(self):
        return freeze(
            {self.index.id: merge_dicts([self.source_exprs, self.target_exprs])}
        )

    @property
    def target_replace_map(self):
        return freeze(
            {
                self.index.id: {ax: expr for ax, expr in self.target_exprs.items()},
            }
        )

    @property
    def source_replace_map(self):
        return freeze(
            {
                self.index.id: {ax: expr for ax, expr in self.source_exprs.items()},
            }
        )


def iter_axis_tree(
    loop_index: LoopIndex,
    axes: AxisTree,
    target_paths,
    index_exprs,
    outer_loops=(),
    axis=None,
    path=pmap(),
    indices=pmap(),
    target_path=None,
    index_exprs_acc=None,
    no_index=False,
):
    from pyop3.expr_visitors import evaluate as eval_expr

    # this is a hack, sometimes things are indexed
    if no_index:
        indices = indices | merge_dicts(
            iter_entry.source_exprs for iter_entry in outer_loops
        )
        outer_replace_map = {}
    else:
        outer_replace_map = merge_dicts(
            iter_entry.replace_map for iter_entry in outer_loops
        )
    if target_path is None:
        assert index_exprs_acc is None
        target_path = target_paths.get(None, pmap())

        # Substitute the index exprs, which map target to source, into
        # indices, giving target index exprs
        myindex_exprs = index_exprs.get(None, pmap())
        # evaluator = ExpressionEvaluator(indices, outer_replace_map)
        new_exprs = {}
        for axlabel, index_expr in myindex_exprs.items():
            new_exprs[axlabel] = eval_expr(index_expr, indices)
        index_exprs_acc = freeze(new_exprs)

    if axes.is_empty:
        source_path = pmap()
        source_exprs = pmap()
        yield IndexIteratorEntry(
            loop_index, source_path, target_path, source_exprs, index_exprs_acc
        )
        return

    axis = axis or axes.root

    for component in axis.components:
        # for efficiency do these outside the loop
        path_ = path | {axis.label: component.label}
        target_path_ = target_path | target_paths.get((axis.id, component.label), {})
        myindex_exprs = index_exprs.get((axis.id, component.label), pmap())
        subaxis = axes.child(axis, component)

        # bit of a hack, I reckon this can go as we can just get it from component.count
        # inside as_int
        if isinstance(component.count, Dat):
            mypath = component.count.axes.target_path.get(None, {})
            myindices = component.count.axes.target_exprs.get(None, {})
            if not component.count.axes.is_empty:
                for cax, ccpt in component.count.axes.path_with_nodes(
                    *component.count.axes.leaf
                ).items():
                    mypath.update(component.count.axes.target_path.get((cax.id, ccpt), {}))
                    myindices.update(
                        component.count.axes.target_exprs.get((cax.id, ccpt), {})
                    )

            mypath = freeze(mypath)
            myindices = freeze(myindices)
            replace_map = indices
        else:
            mypath = pmap()
            myindices = pmap()
            replace_map = None

        for pt in range(
            _as_int(
                component.count,
                replace_map,
                loop_indices=outer_replace_map,
            )
        ):
            new_exprs = {}
            # evaluator = ExpressionEvaluator(
            #     indices | {axis.label: pt}, outer_replace_map
            # )
            for axlabel, index_expr in myindex_exprs.items():
                # new_index = evaluator(index_expr)
                # new_index = eval_expr(index_expr, path_, indices | {axis.label: pt})
                new_index = eval_expr(index_expr, indices | {axis.label: pt})
                new_exprs[axlabel] = new_index
            index_exprs_ = index_exprs_acc | new_exprs
            indices_ = indices | {axis.label: pt}
            if subaxis:
                yield from iter_axis_tree(
                    loop_index,
                    axes,
                    target_paths,
                    index_exprs,
                    outer_loops,
                    subaxis,
                    path_,
                    indices_,
                    target_path_,
                    index_exprs_,
                )
            else:
                yield IndexIteratorEntry(
                    loop_index, path_, target_path_, indices_, index_exprs_
                )


class ArrayPointLabel(enum.IntEnum):
    CORE = 0
    ROOT = 1
    LEAF = 2


class IterationPointType(enum.IntEnum):
    CORE = 0
    ROOT = 1
    LEAF = 2


# TODO This should work for multiple loop indices. One should really pass a loop expression.
def partition_iterset(index: LoopIndex, arrays):
    """Split an iteration set into core, root and leaf index sets.

    The distinction between these is as follows:

    * CORE: May be iterated over without any communication at all.
    * ROOT: Requires a leaf-to-root reduction (i.e. up-to-date SF roots).
    * LEAF: Requires a root-to-leaf broadcast (i.e. up-to-date SF leaves) and also up-to-date roots.

    The partitioning algorithm basically loops over the iteration set and marks entities
    in turn. Any entries whose stencils touch an SF leaf are marked LEAF and any that do
    not touch leaves but do roots are marked ROOT. Any remaining entities do not require
    the SF and are marked CORE.

    """
    from pyop3.array import Mat

    # take first
    # if index.iterset.depth > 1:
    #     raise NotImplementedError("Need a good way to sniff the parallel axis")
    paraxis = index.iterset.root

    # FIXME, need indices per component
    if len(paraxis.components) > 1:
        raise NotImplementedError

    # at a minimum this should be done per multi-axis instead of per array
    is_root_or_leaf_per_array = {}
    for array in arrays:
        # skip matrices
        # really nasty hack for now to handle indexed mats
        if isinstance(array, Mat) or not hasattr(array, "buffer"):
            continue

        # skip purely local arrays
        if not array.buffer.is_distributed:
            continue

        sf = array.buffer.sf  # the dof sf

        # mark leaves and roots
        is_root_or_leaf = np.full(sf.size, ArrayPointLabel.CORE, dtype=np.uint8)
        is_root_or_leaf[sf.iroot] = ArrayPointLabel.ROOT
        is_root_or_leaf[sf.ileaf] = ArrayPointLabel.LEAF

        is_root_or_leaf_per_array[array.name] = is_root_or_leaf

    labels = np.full(paraxis.size, IterationPointType.CORE, dtype=np.uint8)
    # for p in index.iterset.iter():
    #     # hack because I wrote bad code and mix up loop indices and itersets
    #     p = dataclasses.replace(p, index=index)
    # for p in index.iter():
    #     parindex = p.source_exprs[paraxis.label]
    #     assert isinstance(parindex, numbers.Integral)
    #
    #     for array in arrays:
    #         # same nasty hack
    #         if isinstance(array, (Mat, Sparsity)) or not hasattr(array, "buffer"):
    #             continue
    #         # skip purely local arrays
    #         if not array.buffer.is_distributed:
    #             continue
    #         if labels[parindex] == IterationPointType.LEAF:
    #             continue
    #
    #         # loop over stencil
    #         array = array.with_context({index.id: (p.source_path, p.target_path)})
    #
    #         for q in array.axes.iter({p}):
    #             # offset = array.axes.offset(q.target_exprs, q.target_path)
    #             offset = array.axes.offset(q.source_exprs, q.source_path, loop_exprs=p.replace_map)
    #
    #             point_label = is_root_or_leaf_per_array[array.name][offset]
    #             if point_label == ArrayPointLabel.LEAF:
    #                 labels[parindex] = IterationPointType.LEAF
    #                 break  # no point doing more analysis
    #             elif point_label == ArrayPointLabel.ROOT:
    #                 assert labels[parindex] != IterationPointType.LEAF
    #                 labels[parindex] = IterationPointType.ROOT
    #             else:
    #                 assert point_label == ArrayPointLabel.CORE
    #                 pass

    parcpt = just_one(paraxis.components)  # for now

    # I don't think this is working - instead everything touches a leaf
    # core = just_one(np.nonzero(labels == IterationPointType.CORE))
    # root = just_one(np.nonzero(labels == IterationPointType.ROOT))
    # leaf = just_one(np.nonzero(labels == IterationPointType.LEAF))
    # core = np.asarray([], dtype=IntType)
    # root = np.asarray([], dtype=IntType)
    # leaf = np.arange(paraxis.size, dtype=IntType)

    # hack to check things
    core = np.asarray([0], dtype=IntType)
    root = np.asarray([1], dtype=IntType)
    leaf = np.arange(2, paraxis.size, dtype=IntType)

    subsets = []
    for data in [core, root, leaf]:
        # Constant? no, rank_equal=False
        size = Dat(
            AxisTree(), data=np.asarray([len(data)]), dtype=IntType
        )
        subset = Dat(
            Axis([AxisComponent(size, parcpt.label)], paraxis.label), data=data
        )
        subsets.append(subset)
    subsets = tuple(subsets)
    return "not used", subsets

    # make a new iteration set over just these indices
    # index with just core (arbitrary)

    # need to use the existing labels here
    mysubset = Slice(
        paraxis.label,
        [Subset(parcpt.label, subsets[0], label=parcpt.label)],
        label=paraxis.label,
    )
    new_iterset = index.iterset[mysubset]

    return index.copy(iterset=new_iterset), subsets

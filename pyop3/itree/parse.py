import collections
import functools
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

from immutabledict import ImmutableOrderedDict
from pyrsistent import PMap, pmap

from pyop3.array.harray import Dat
from pyop3.axtree import AxisTree
from pyop3.axtree.tree import BaseAxisTree
from pyop3.exceptions import Pyop3Exception
from pyop3.itree.tree import CalledMap, IndexTree, LoopIndex, Slice, AffineSliceComponent, ScalarIndex, Index, Map
from pyop3.utils import OrderedSet, expand_collection_of_iterables, strictly_all, strict_zip, single_valued, just_one


class IncompletelyIndexedException(Pyop3Exception):
    """Exception raised when an axis tree is incompletely indexed by an index tree/forest."""


# NOTE: Now really should be plural: 'forests'
# NOTE: Is this definitely the case? I think at the moment I always return just a single
# tree per context.
def as_index_forests(forest: Any, /, axes: BaseAxisTree, *, strict: bool = False) -> PMap:
    """Return a collection of index trees, split by loop context.

    Parameters
    ----------
    forest :
        The object representing an indexing operation.
    axes :
        The axis tree to which the indexing is being applied.
    strict :
        Flag indicating whether or not additional slices should be added
        implicitly. If `False` then extra slices are added to fill up any
        unindexed shape. If `True` then providing an insufficient set of
        indices will raise an exception.

    Returns
    -------
    index_forest
        A mapping from loop contexts to a tuple of equivalent index trees. Loop
        contexts are represented by the mapping ``{loop index id: iterset path}``.

        Multiple index trees are needed because maps are able to yield multiple
        equivalent index trees.
    """
    if forest is Ellipsis:
        return ImmutableOrderedDict({pmap(): (forest,)})

    forests = {}
    compressed_loop_contexts = collect_loop_contexts(forest)
    # Pass `pmap` as the mapping type because we do not care about the ordering
    # of `loop_context` (though we *do* care about the order of iteration).
    for loop_context in expand_collection_of_iterables(compressed_loop_contexts, mapping_type=pmap):
        forest_ = _as_index_forest(forest, axes, loop_context)
        matched_forest = []

        found_match = False
        for index_tree in forest_:
            if strict:
                # Make sure that `axes` are completely indexed by each of the index
                # forests. Note that, since the index trees in a forest represent
                # 'equivalent' indexing operations, only one of them is expected to work.
                if not _index_tree_completely_indexes_axes(index_tree, axes):
                    continue
            else:
                # Add extra slices to make sure that index tree targets
                # all the axes in `axes`
                # FIXME: needs try-except
                index_tree = _complete_index_tree(index_tree, axes)

            if found_match:
                # Each of the index trees in a forest are considered
                # 'equivalent' in that they represent semantically
                # equivalent operations, differing only in the axes that
                # they target. For example, the loop index
                #
                #     p = axis[::2].iter()
                #
                # will target *both* the unindexed `axis`, as well as the
                # intermediate indexed axis `axis[::2]`. There are therefore
                # two index trees in play.
                #
                # For maps I think that it is possible for us to have clashes
                # in the target axes (e.g. points -> points and cells -> points).
                # If we ever hit this we will need to think a bit.
                raise NotImplementedError(
                    "Found multiple matching index trees, I thought this "
                    "day might come eventually"
                )

            matched_forest.append(index_tree)
            found_match = True

        if not found_match:
            raise IncompletelyIndexedException(
                "Index forest does not correctly index the axis tree"
            )

        forests[loop_context] = tuple(matched_forest)
    return ImmutableOrderedDict(forests)


# old alias, remove
as_index_forest = as_index_forests


@functools.singledispatch
def collect_loop_contexts(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@collect_loop_contexts.register(IndexTree)
def _(index_tree: IndexTree, /) -> OrderedSet:
    loop_contexts = OrderedSet()
    for index in index_tree.nodes:
        loop_contexts |= collect_loop_contexts(index)

    assert len(loop_contexts) < 2, "By definition an index tree cannot be context-sensitive"
    return loop_contexts


@collect_loop_contexts.register(LoopIndex)
def _(loop_index: LoopIndex, /) -> OrderedSet:
    if not isinstance(loop_index.iterset, BaseAxisTree):
        raise NotImplementedError("Need to think about context-sensitive itersets and add them here")

    return OrderedSet({
        (
            loop_index.id,
            tuple(
                loop_index.iterset.source_path[axis.id, component_label]
                for axis, component_label in loop_index.iterset.leaves
            )
        )
    })


@collect_loop_contexts.register(CalledMap)
def _(called_map: CalledMap, /) -> OrderedSet:
    return collect_loop_contexts(called_map.index)


@collect_loop_contexts.register(numbers.Number)
@collect_loop_contexts.register(str)
@collect_loop_contexts.register(slice)
@collect_loop_contexts.register(Slice)
@collect_loop_contexts.register(ScalarIndex)
def _(index: Any, /) -> OrderedSet:
    return OrderedSet()


@collect_loop_contexts.register(Sequence)
def _(seq: Sequence, /) -> OrderedSet:
    loop_contexts = OrderedSet()
    for item in seq:
        loop_contexts |= collect_loop_contexts(item)
    return loop_contexts


@functools.singledispatch
def _as_index_forest(obj: Any, /, *args, **kwargs) -> tuple[IndexTree]:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_as_index_forest.register(IndexTree)
def _(index_tree: IndexTree, /, *args, **kwargs) -> tuple[IndexTree]:
    return (index_tree,)


@_as_index_forest.register(Index)
def _(index: Index, /, axes, loop_context) -> tuple[IndexTree]:
    cf_indices = _as_context_free_indices(index, loop_context)
    return tuple(IndexTree(cf_index) for cf_index in cf_indices)


@_as_index_forest.register(Sequence)
def _(seq: Sequence, /, axes, loop_context) -> tuple[IndexTree]:
    raise NotImplementedError
    # The indices can contain a mixture of 'true' indices (i.e. subclasses of
    # `Index`) and 'sugar' indices (e.g. integers, strings and slices). The former
    # may be used in any order since they declare the axes they target whereas
    # the latter are order dependent.
    first, *rest = seq
    # for cf_index in _as_context_free_indices(first, axes

    index_trees = {}
    indices_and_contexts = _collect_indices_and_contexts(indices, loop_context=loop_context)
    for context, cf_indices in indices_and_contexts.items():
        index_forest = _index_forest_from_iterable(cf_indices, axes=axes)
        index_trees[context] = index_forest
    return index_trees


@_as_index_forest.register(Dat)
def _(dat: Dat, /, *args, **kwargs) -> tuple[IndexTree]:
    raise NotImplementedError
    # NOTE: This is the same behaviour as for slices
    parent = axes._node_from_path(path)
    if parent is not None:
        parent_axis, parent_cpt = parent
        target_axis = axes.child(parent_axis, parent_cpt)
    else:
        target_axis = axes.root

    if target_axis.degree > 1:
        raise ValueError(
            "Passing arrays as indices is only allowed when there is no ambiguity"
        )

    slice_cpt = Subset(target_axis.component.label, arg)
    slice_ = Slice(target_axis.label, [slice_cpt])
    return {pmap(): IndexTree(slice_)}


@_as_index_forest.register(slice)
@_as_index_forest.register(str)
@_as_index_forest.register(numbers.Integral)
def _(index: Any, /, axes, loop_context) -> tuple[IndexTree]:
    desugared = _desugar_index(index, axes)
    return _as_index_forest(desugared, axes, loop_context)


@functools.singledispatch
def _desugar_index(obj: Any, *args, **kwargs) -> Index:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_desugar_index.register(numbers.Integral)
def _(int_: numbers.Integral, /, axes, *, parent=None) -> Index:
    axis = axes.child(*parent) if parent else axes.root
    if len(axis.components) > 1:
        # Multi-component axis: take a slice from a matching component.
        component = just_one(c for c in axis.components if c.label == int_)
        if component.unit:
            index = ScalarIndex(axis.label, component.label, 0)
        else:
            index = Slice(axis.label, [AffineSliceComponent(component.label, label=component.label)], label=axis.label)
    else:
        # Single-component axis: return a scalar index.
        component = just_one(axis.components)
        index = ScalarIndex(axis.label, component.label, int_)
    return index


@_desugar_index.register(slice)
def _(slice_: slice, /, axes, *, parent=None) -> Index:
    axis = axes.child(*parent) if parent else axes.root
    if axis.degree > 1:
        # badindexexception?
        raise ValueError(
            "Cannot slice multi-component things using generic slices, ambiguous"
        )

    return Slice(
        axis.label,
        [AffineSliceComponent(axis.component.label, slice_.start, slice_.stop, slice_.step)]
    )


@_desugar_index.register(str)
def _(label, /, axes, *, parent=None):
    # Take a full slice of a component with a matching label
    axis = axes.child(*parent) if parent else axes.root
    component = just_one(c for c in axis.components if c.label == label)

    # If the component is marked as "unit" then indexing in this way will
    # fully consume the axis.
    # NOTE: Perhaps it would just be better to always do this if the axis
    # is one-sized?
    if component.unit:
        index = ScalarIndex(axis.label, component.label, 0)
    else:
        index = Slice(axis.label, [AffineSliceComponent(component.label, label=component.label)], label=axis.label)
    return index


@functools.singledispatch
def _expand_index(obj: Any):
    raise TypeError


@_expand_index.register(Index)
def _(index):
    return index.expanded


@_expand_index.register(numbers.Integral)
@_expand_index.register(str)
@_expand_index.register(slice)
def _(value):
    return (value,)


def _index_forest_from_iterable(indices, *, axes):
    restricted_indicess = itertools.product(*map(_expand_index, indices))
    return tuple(
        _index_tree_from_iterable(restricted_indices, axes=axes)
        for restricted_indices in restricted_indicess
    )


def _index_tree_from_iterable(indices, *, axes, parent=None, unhandled_target_paths=None):
    if strictly_all(x is None for x in {parent, unhandled_target_paths}):
        parent = (None, None)
        unhandled_target_paths = pmap()

    unhandled_target_paths_mut = dict(unhandled_target_paths)
    parent_axis, parent_component = parent
    while True:
        axis = axes.child(parent_axis, parent_component)

        if axis is None or axis.label not in unhandled_target_paths_mut:
            break
        else:
            parent_axis = axis
            parent_component = unhandled_target_paths_mut.pop(parent_axis.label)
    parent = (parent_axis, parent_component)

    index, *subindices = indices

    skip_index = False
    if isinstance(index, (ContextFreeIndex, ContextFreeCalledMap)):
        if strictly_all(
            not any(
                strictly_all(ax in axes.node_labels for ax in target_path.keys())
                for target_path in equiv_paths
            )
            for equiv_paths in index.leaf_target_paths 
        ):
            skip_index = True
    else:
        try:
            index = _desugar_index(index, axes=axes, parent=parent)
        except InvalidIndexException:
            skip_index = True

    if skip_index:
        if subindices:
            index_tree = _index_tree_from_iterable(
                subindices, 
                axes=axes,
                parent=parent,
                unhandled_target_paths=unhandled_target_paths,
            )
        else:
            index_tree = IndexTree()

    else:
        index_tree = IndexTree(index)
        for component_label, equiv_target_paths in strict_zip(
            index.component_labels, index.leaf_target_paths
        ):
            # Here we only care about targeting the most recent axis tree.
            unhandled_target_paths_ = unhandled_target_paths | equiv_target_paths[-1]

            if subindices:
                subindex_tree = _index_tree_from_iterable(
                    subindices, 
                    axes=axes,
                    parent=parent,
                    unhandled_target_paths=unhandled_target_paths_,
                )
                index_tree = index_tree.add_subtree(subindex_tree, index, component_label, uniquify_ids=True)

    return index_tree


# TODO: This function needs overhauling to work in more cases.
def _complete_index_tree(
    index_tree: IndexTree, axes: AxisTree, *, index=None, possible_target_paths_acc=None,
) -> IndexTree:
    """Add extra slices to the index tree to match the axes.

    Notes
    -----
    This function is currently only capable of adding additional slices if
    they are "innermost".

    """
    if strictly_all(x is None for x in {index, possible_target_paths_acc}):
        index = index_tree.root
        possible_target_paths_acc = (pmap(),)

    index_tree_ = IndexTree(index)

    for component_label, equivalent_target_paths in strict_zip(
        index.component_labels, index.leaf_target_paths
    ):
        possible_target_paths_acc_ = tuple(
            possible_target_path | target_path
            for possible_target_path in possible_target_paths_acc
            for target_path in equivalent_target_paths
        )

        if subindex := index_tree.child(index, component_label):
            subtree = _complete_index_tree(
                index_tree,
                axes,
                index=subindex,
                possible_target_paths_acc=possible_target_paths_acc_,
            )
        else:
            # At the bottom of the index tree, add any extra slices if needed.
            subtree = _complete_index_tree_slices(axes, possible_target_paths_acc_)

        index_tree_ = index_tree_.add_subtree(subtree, index, component_label)

    return index_tree_


def _complete_index_tree_slices(axes, target_paths, *, axis=None) -> IndexTree:
    if axis is None:
        axis = axes.root

    # If the label of the current axis exists in any of the target paths then
    # that means that an index already exists that targets that axis, and
    # hence no slice need be produced.
    # At the same time, we can also trim the target paths since we know that
    # we can exclude any that do not use that axis label.
    target_paths_ = tuple(tp for tp in target_paths if axis.label in tp)

    if len(target_paths_) == 0:
        # Axis not found, need to emit a slice
        slice_ = Slice(
            axis.label, [AffineSliceComponent(c.label) for c in axis.components]
        )
        index_tree = IndexTree(slice_)

        for axis_component, slice_component_label in strict_zip(
            axis.components, slice_.component_labels
        ):
            if subaxis := axes.child(axis, axis_component):
                subindex_tree = _complete_index_tree_slices(axes, target_paths, axis=subaxis)
                index_tree = index_tree.add_subtree(subindex_tree, slice_, slice_component_label)

        return index_tree
    else:
        # Axis found, pass things through
        target_component = single_valued(tp[axis.label] for tp in target_paths_)
        if subaxis := axes.child(axis, target_component):
            return _complete_index_tree_slices(axes, target_paths_, axis=subaxis)
        else:
            # At the bottom, no more slices needed
            return IndexTree()


def  _index_tree_completely_indexes_axes(index_tree: IndexTree, axes, *, index=None, possible_target_paths_acc=None) -> bool:
    """Return whether the index tree completely indexes the axis tree.

    This is done by traversing the index tree and collecting the possible target
    paths. At the leaf of the tree we then check whether or not any of the
    possible target paths correspond to a valid path to a leaf of the axis tree.

    """
    if strictly_all(x is None for x in {index, possible_target_paths_acc}):
        index = index_tree.root
        possible_target_paths_acc = (pmap(),)

    for component_label, equivalent_target_paths in strict_zip(
        index.component_labels, index.leaf_target_paths
    ):
        possible_target_paths_acc_ = tuple(
            possible_target_path_acc | possible_target_path
            for possible_target_path_acc in possible_target_paths_acc
            for possible_target_path in equivalent_target_paths
        )

        if subindex := index_tree.child(index, component_label):
            if not _index_tree_completely_indexes_axes(
                index_tree,
                axes,
                index=subindex,
                possible_target_paths_acc=possible_target_paths_acc_,
            ):
                return False
        else:
            if all(tp not in axes.leaf_paths for tp in possible_target_paths_acc_):
                return False
    return True


@functools.singledispatch
def _as_context_free_indices(obj: Any, /, loop_context: Mapping) -> Index:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_as_context_free_indices.register(Slice)
@_as_context_free_indices.register(ScalarIndex)
def _(index, /, loop_context: Mapping) -> tuple[Index]:
    return (index,)


@_as_context_free_indices.register(LoopIndex)
def _(loop_index: LoopIndex, /, loop_context) -> tuple[LoopIndex]:
    if loop_index.is_context_free:
        return (loop_index,)
    else:
        path = loop_context[loop_index.id]

        leaf = loop_index.iterset._node_from_path(path)
        slices = [
            Slice(axis_label, [AffineSliceComponent(component_label, label=component_label)], label=axis_label)
            for axis_label, component_label in loop_index.iterset.path(leaf, ordered=True)
        ]

        # TODO: should accept the iterable directly
        slices_tree = IndexTree.from_iterable(slices)

        linear_iterset = loop_index.iterset[slices_tree]
        return (loop_index.copy(iterset=linear_iterset),)


@_as_context_free_indices.register(CalledMap)
def _(called_map, /, loop_context):
    cf_maps = []
    cf_indices = _as_context_free_indices(called_map.from_index, loop_context)

    # loop over semantically equivalent indices
    for cf_index in cf_indices:

        # imagine that we have
        #
        #   {
        #      x -> [[a], [b, c]],
        #      y -> [[a], [d]],
        #   }
        #
        # ie xmaps to *either* [a] or [b, c] and y maps to either [a] or [d]
        # then we want to end up with
        #
        #   {
        #     x -> [[a]],
        #     y -> [[a]],
        #   }
        #   and
        #   {
        #     x -> [[b, c]],
        #     y -> [[a]],
        #   }
        #    etc
        #
        # In effect for a concrete set of inputs having a concrete set of outputs
        #
        # Note that this gets more complicated in cases like
        #
        #   { x -> [[a]], y -> [[a]] }
        #
        # where we assume x and y to be "equivalent".
        # because if two equivalent input paths map to the same output then they can
        # be considered equivalent in the final axis tree.
        #
        # This is later work.

        possibilities = []
        for equivalent_input_paths in cf_index.leaf_target_paths:
            found = False
            for input_path in equivalent_input_paths:
                if input_path in called_map.connectivity:
                    found = True
                    for output_spec in called_map.connectivity[input_path]:
                        possibilities.append((input_path, output_spec))
            assert found, "must be at least one matching path"

        if len(possibilities) > 1:
            # list(itertools.product(possibilities))
            raise NotImplementedError("Need to think about taking the product of these")
        else:
            input_path, output_spec = just_one(possibilities)
            restricted_connectivity = {input_path: (output_spec,)}
            restricted_map = Map(restricted_connectivity, called_map.name)(cf_index)
            cf_maps.append(restricted_map)
    return tuple(cf_maps)

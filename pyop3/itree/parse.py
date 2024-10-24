import collections
import functools
import numbers
from typing import Any

from pyrsistent import PMap

from pyop3.axtree import AxisTree
from pyop3.axtree.tree import BaseAxisTree
from pyop3.itree.tree import CalledMap, IndexTree, LoopIndex, Slice, AffineSliceComponent, ScalarIndex, Index, AbstractLoopIndex, LocalLoopIndex
from pyop3.utils import OrderedSet, expand_collection_of_iterables


# NOTE: Now really should be plural: 'forests'
def as_index_forest(forest: Any, /, axes: BaseAxisTree, *, strict: bool = False) -> PMap:
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
    assert forest is not Ellipsis, "Ellipses should be handled before calling"

    forests = {}
    loop_contexts = _collect_loop_contexts(forest)
    for loop_context in expand_collection_of_iterables(loop_contexts):
        forest_ = _as_index_forest(forest, axes, loop_context=loop_context)
        forests[loop_context] = forest_

    breakpoint()

    # If axes are provided then check that the index tree is compatible
    # and add extra slices if required.
    if axes is not None:
        forest_ = {}
        for ctx, trees in forest.items():
            checked_trees = []
            for tree in trees:
                if not strict:
                    # NOTE: This function doesn't always work. In particular if
                    # the loop index is from an already indexed axis. This
                    # requires more thought but for now just use the strict version
                    # and provide additional information elsewhere.
                    tree = _complete_index_tree(tree, axes)

                if not _index_tree_is_complete(tree, axes):
                    raise ValueError("Index tree does not completely index axes")

                checked_trees.append(tree)
            forest_[ctx] = checked_trees
        forest = forest_

        # # TODO: Clean this up, and explain why it's here.
        # forest_ = {}
        # for ctx, index_tree in forest.items():
        #     # forest_[ctx] = index_tree.copy(outer_loops=axes.outer_loops)
        #     forest_[ctx] = index_tree
        # forest = forest_
    return forest


@functools.singledispatch
def _collect_loop_contexts(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_collect_loop_contexts.register(IndexTree)
def _(index_tree: IndexTree, /) -> OrderedSet:
    loop_contexts = OrderedSet()
    for index in index_tree.nodes:
        loop_contexts |= _collect_loop_contexts(index)

    assert len(loop_contexts) < 2, "By definition an index tree cannot be context-sensitive"
    return loop_contexts


@_collect_loop_contexts.register(LoopIndex)
def _(loop_index: LoopIndex, /) -> OrderedSet:
    if not isinstance(loop_index.iterset, BaseAxisTree):
        raise NotImplementedError("Need to think about context-sensitive itersets and add them here")

    return OrderedSet({(loop_index.id, loop_index.iterset.paths)})


@_collect_loop_contexts.register(CalledMap)
def _(called_map: CalledMap, /) -> OrderedSet:
    return _collect_loop_contexts(called_map.index)


@_collect_loop_contexts.register(str)
@_collect_loop_contexts.register(Slice)
@_collect_loop_contexts.register(ScalarIndex)
def _(index: Any, /) -> OrderedSet:
    return OrderedSet()


@functools.singledispatch
def _as_index_forest(arg: Any, *, axes, **_):

    # if isinstance(arg, HierarchicalArray):
    if False:
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
    else:
        raise TypeError(f"No handler provided for {type(arg).__name__}")


@_as_index_forest.register
def _(forest: collections.abc.Mapping, **kwargs):
    return forest


@_as_index_forest.register
def _(index_tree: IndexTree, **_):
    return {pmap(): (index_tree,)}


# @_as_index_forest.register
# def _(index: ContextFreeIndex, **_):
#     return {pmap(): (IndexTree(index),)}


@_as_index_forest.register(LoopIndex)
@_as_index_forest.register(CalledMap)
def _(index, *, loop_context, **_):
    unpacked = _as_context_free_index(index, loop_context=loop_context)
    forest = {
        context: tuple(IndexTree(idx) for idx in idxs)
        for context, idxs in unpacked.items()
    }
    return forest


@_as_index_forest.register
def _(indices: collections.abc.Sequence, *, axes, loop_context):
    # The indices can contain a mixture of "true" indices (i.e. subclasses of
    # Index) and "sugar" indices (e.g. integers, strings and slices). The former
    # may be used in any order since they declare the axes they target whereas
    # the latter are order dependent.
    # To add another complication, the "true" indices may also be context-sensitive:
    # what they produce is dependent on the state of the outer loops. We therefore
    # need to unpack this to produce a different index tree for each possible
    # context.

    index_trees = {}
    indices_and_contexts = _collect_indices_and_contexts(indices, loop_context=loop_context)
    for context, cf_indices in indices_and_contexts.items():
        index_forest = _index_forest_from_iterable(cf_indices, axes=axes)
        index_trees[context] = index_forest
    return index_trees


@_as_index_forest.register(slice)
@_as_index_forest.register(str)
@_as_index_forest.register(numbers.Integral)
def _(index, **kwargs):
    return _as_index_forest([index], **kwargs)



@functools.singledispatch
def _desugar_index(index: Any, **_):
    raise TypeError(f"No handler defined for {type(index).__name__}")


@_desugar_index.register
def _(int_: numbers.Integral, *, axes, parent, **_):
    axis = axes.child(*parent)
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


@_desugar_index.register
def _(slice_: slice, *, axes, parent, **_):
    axis = axes.child(*parent)
    if axis.degree > 1:
        # badindexexception?
        raise ValueError(
            "Cannot slice multi-component things using generic slices, ambiguous"
        )

    return Slice(
        axis.label,
        [AffineSliceComponent(axis.component.label, slice_.start, slice_.stop, slice_.step)]
    )


@_desugar_index.register
def _(label: str, *, axes, parent, **_):
    # Take a full slice of a component with a matching label
    axis = axes.child(*parent)
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


def _index_tree_is_complete(index_tree: IndexTree, axes: AxisTree, *, index=None, possible_target_paths_acc=None) -> bool:
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
            if not _index_tree_is_complete(
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
def _as_context_free_index(arg, **_):
    raise TypeError


# @_as_context_free_index.register(ContextFreeIndex)
# @_as_context_free_index.register(ContextFreeCalledMap)
# def _(cf_index, **kwargs):
#     return {pmap(): (cf_index,)}


# TODO This function can definitely be refactored
@_as_context_free_index.register(AbstractLoopIndex)
@_as_context_free_index.register(LocalLoopIndex)
def _(index, *, loop_context, **kwargs):
    local = isinstance(index, LocalLoopIndex)

    cf_indices = {}
    if isinstance(index.iterset, ContextSensitive):
        for context, axes in index.iterset.context_map.items():
            if axes.is_empty:
                source_path = pmap()
                target_path = axes.target_paths.get(None, pmap())

                context_ = (
                    loop_context | context | {index.id: (source_path, target_path)}
                )

                cf_indices[context_] = index.with_context(context_)
            else:
                for leaf in axes.leaves:
                    source_path = axes.path(*leaf)
                    target_path = axes.target_paths.get(None, pmap())
                    for axis, cpt in axes.path_with_nodes(
                        *leaf, and_components=True
                    ).items():
                        target_path |= axes.target_paths.get((axis.id, cpt.label), {})

                    context_ = (
                        loop_context | context | {index.id: (source_path, target_path)}
                    )

                    cf_index = index.with_context(context_)
                    forest[context_] = IndexTree(cf_index)
    else:
        assert isinstance(index.iterset, ContextFree)
        for leaf in index.iterset.leaves:
            slices = [
                Slice(axis_label, [AffineSliceComponent(component_label, label=component_label)], label=axis_label)
                for axis_label, component_label in index.iterset.path(leaf, ordered=True)
            ]
            linear_iterset = index.iterset[slices]

            # source_path = index.iterset.path(leaf_axis, leaf_cpt)
            # target_path = index.iterset.target_path.get(None, pmap())
            # for axis, cpt in index.iterset.path_with_nodes(
            #     leaf_axis, leaf_cpt, and_components=True
            # ).items():
            #     target_path |= index.iterset.target_paths[axis.id, cpt.label]
            # # TODO cleanup
            # my_id = index.id if not local else index.loop_index.id
            # context = loop_context | {index.id: (source_path, target_path)}
            context = loop_context | {index.id: "anything"}

            cf_index = ContextFreeLoopIndex(linear_iterset, id=index.id)

            cf_indices[context] = cf_index
    return cf_indices


@_as_context_free_index.register(CalledMap)
def _(called_map, **kwargs):
    cf_maps = {}
    cf_indicess = _as_context_free_index(called_map.from_index, **kwargs)
    # loop over different "outer loop contexts"
    for context, cf_indices in cf_indicess.items():
        cf_maps[context] = []
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
                cf_maps[context].append(restricted_map)
    return freeze(cf_maps)


def _collect_indices_and_contexts(indices, *, loop_context):
    """
    Syntactic sugar indices (i.e. integers, strings, slices) are
    treated differently here
    because they must be handled (in order) later on.
    """
    index, *subindices = indices
    collected = {}

    if isinstance(index, Index):
        raise NotImplementedError("This is harder now we track equivalent paths")
        for context, cf_index in _as_context_free_index(
            index, loop_context=loop_context
        ).items():
            if subindices:
                subcollected = _collect_indices_and_contexts(
                    subindices,
                    loop_context=loop_context | context,
                )
                for subcontext, cf_subindices in subcollected.items():
                    collected[subcontext] = (cf_index,) + cf_subindices
            else:
                collected[context] = (cf_index,)

    else:
        if subindices:
            subcollected = _collect_indices_and_contexts(subindices, loop_context=loop_context)
            for subcontext, cf_subindices in subcollected.items():
                collected[subcontext] = (index,) + cf_subindices
        else:
            collected[pmap()] = (index,)

    return pmap(collected)

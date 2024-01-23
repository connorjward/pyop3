from __future__ import annotations

import abc
import collections
import functools
import itertools

from pyrsistent import freeze, pmap

from pyop3.array import ContextSensitiveMultiArray, HierarchicalArray
from pyop3.axtree import Axis, AxisTree, ContextFree
from pyop3.buffer import NullBuffer
from pyop3.itree import Map, TabulatedMapComponent
from pyop3.lang import (
    INC,
    READ,
    RW,
    WRITE,
    AddAssignment,
    CalledFunction,
    ContextAwareLoop,
    Instruction,
    Loop,
    ReplaceAssignment,
    Terminal,
)
from pyop3.utils import UniqueNameGenerator, checked_zip, just_one


# TODO Is this generic for other parsers/transformers? Esp. lower.py
class Transformer(abc.ABC):
    @abc.abstractmethod
    def apply(self, expr):
        pass


class LoopContextExpander(Transformer):
    def apply(self, expr: Instruction):
        return self._apply(expr, context=pmap())

    @functools.singledispatchmethod
    def _apply(self, expr: Instruction, **kwargs):
        raise TypeError(f"No handler provided for {type(expr).__name__}")

    @_apply.register
    def _(self, loop: Loop, *, context):
        cf_iterset = loop.index.iterset.with_context(context)
        source_paths = cf_iterset.leaf_paths
        target_paths = cf_iterset.leaf_target_paths
        assert len(source_paths) == len(target_paths)

        if len(source_paths) == 1:
            # single component iterset, no branching required
            source_path = just_one(source_paths)
            target_path = just_one(target_paths)

            context_ = context | {loop.index.id: (source_path, target_path)}
            statements = {
                source_path: tuple(
                    self._apply(stmt, context=context_) for stmt in loop.statements
                )
            }
        else:
            assert len(source_paths) > 1
            statements = {}
            for source_path, target_path in checked_zip(source_paths, target_paths):
                context_ = context | {loop.index.id: (source_path, target_path)}
                statements[source_path] = tuple(
                    self._apply(stmt, context=context_) for stmt in loop.statements
                )

        return ContextAwareLoop(
            loop.index.copy(iterset=cf_iterset),
            statements,
        )

    @_apply.register
    def _(self, terminal: Terminal, *, context):
        cf_args = [a.with_context(context) for a in terminal.arguments]
        return terminal.with_arguments(cf_args)


def expand_loop_contexts(expr: Instruction):
    return LoopContextExpander().apply(expr)


class ImplicitPackUnpackExpander(Transformer):
    def __init__(self):
        self._name_generator = UniqueNameGenerator()

    def apply(self, expr):
        return self._apply(expr)

    @functools.singledispatchmethod
    def _apply(self, expr: Any):
        raise NotImplementedError(f"No handler provided for {type(expr).__name__}")

    # TODO Can I provide a generic "operands" thing? Put in the parent class?
    @_apply.register
    def _(self, loop: ContextAwareLoop):
        return (
            loop.copy(
                statements={
                    ctx: [stmt_ for stmt in stmts for stmt_ in self._apply(stmt)]
                    for ctx, stmts in loop.statements.items()
                }
            ),
        )

    @_apply.register
    def _(self, terminal: Terminal):
        gathers = []
        scatters = []
        arguments = []
        for arg, intent in terminal.kernel_arguments:
            assert isinstance(
                arg, ContextFree
            ), "Loop contexts should already be expanded"
            if _requires_pack_unpack(arg):
                # this is a nasty hack - shouldn't reuse layouts from arg.axes
                axes = AxisTree(arg.axes.parent_to_children)
                temporary = HierarchicalArray(
                    axes,
                    data=NullBuffer(arg.dtype),  # does this need a size?
                    name=self._name_generator("t"),
                )

                if intent == READ:
                    gathers.append(ReplaceAssignment(temporary, arg))
                elif intent == WRITE:
                    gathers.append(ReplaceAssignment(temporary, 0))
                    scatters.append(ReplaceAssignment(arg, temporary))
                elif intent == RW:
                    gathers.append(ReplaceAssignment(temporary, arg))
                    scatters.append(ReplaceAssignment(arg, temporary))
                else:
                    assert intent == INC
                    gathers.append(ReplaceAssignment(temporary, 0))
                    scatters.append(AddAssignment(arg, temporary))

                arguments.append(temporary)

            else:
                arguments.append(arg)

        return (*gathers, terminal.with_arguments(arguments), *scatters)


# TODO check this docstring renders correctly
def expand_implicit_pack_unpack(expr: Instruction):
    """Expand implicit pack and unpack operations.

    An implicit pack/unpack is something of the form

    .. code::
        kernel(dat[f(p)])

    In order for this to work the ``dat[f(p)]`` needs to be packed
    into a temporary. Assuming that its intent in ``kernel`` is
    `pyop3.WRITE`, we would expand this function into

    .. code::
        tmp <- [0, 0, ...]
        kernel(tmp)
        dat[f(p)] <- tmp

    Notes
    -----
    For this routine to work, any context-sensitive loops must have
    been expanded already (with `expand_loop_contexts`). This is
    because context-sensitive arrays may be packed into temporaries
    in some contexts but not others.

    """
    return ImplicitPackUnpackExpander().apply(expr)


def _requires_pack_unpack(arg):
    # TODO in theory packing isn't required for arrays that are contiguous,
    # but this is hard to determine
    return isinstance(arg, HierarchicalArray)


# *below is old untested code*
#
# def compress(iterset, map_func, *, uniquify=False):
#     # TODO Ultimately we should be able to generate code for this set of
#     # loops. We would need to have a construct to describe "unique packing"
#     # with hash sets like we do in the Python version. PETSc have PetscHSetI
#     # which I think would be suitable.
#
#     if not uniquify:
#         raise NotImplementedError("TODO")
#
#     iterset = iterset.as_tree()
#
#     # prepare size arrays, we want an array per target path per iterset path
#     sizess = {}
#     for leaf_axis, leaf_clabel in iterset.leaves:
#         iterset_path = iterset.path(leaf_axis, leaf_clabel)
#
#         # bit unpleasant to have to create a loop index for this
#         sizes = {}
#         index = iterset.index()
#         cf_map = map_func(index).with_context({index.id: iterset_path})
#         for target_path in cf_map.leaf_target_paths:
#             if iterset.depth != 1:
#                 # TODO For now we assume iterset to have depth 1
#                 raise NotImplementedError
#             # The axes of the size array correspond only to the specific
#             # components selected from iterset by iterset_path.
#             clabels = (just_one(iterset_path.values()),)
#             subiterset = iterset[clabels]
#
#             # subiterset is an axis tree with depth 1, we only want the axis
#             assert subiterset.depth == 1
#             subiterset = subiterset.root
#
#             sizes[target_path] = HierarchicalArray(
#                 subiterset, dtype=IntType, prefix="nnz"
#             )
#         sizess[iterset_path] = sizes
#     sizess = freeze(sizess)
#
#     # count sizes
#     for p in iterset.iter():
#         entries = collections.defaultdict(set)
#         for q in map_func(p.index).iter({p}):
#             # we expect maps to only output a single target index
#             q_value = just_one(q.target_exprs.values())
#             entries[q.target_path].add(q_value)
#
#         for target_path, points in entries.items():
#             npoints = len(points)
#             nnz = sizess[p.source_path][target_path]
#             nnz.set_value(p.source_path, p.source_exprs, npoints)
#
#     # prepare map arrays
#     flat_mapss = {}
#     for iterset_path, sizes in sizess.items():
#         flat_maps = {}
#         for target_path, nnz in sizes.items():
#             subiterset = nnz.axes.root
#             map_axes = AxisTree.from_nest({subiterset: Axis(nnz)})
#             flat_maps[target_path] = HierarchicalArray(
#                 map_axes, dtype=IntType, prefix="map"
#             )
#         flat_mapss[iterset_path] = flat_maps
#     flat_mapss = freeze(flat_mapss)
#
#     # populate compressed maps
#     for p in iterset.iter():
#         entries = collections.defaultdict(set)
#         for q in map_func(p.index).iter({p}):
#             # we expect maps to only output a single target index
#             q_value = just_one(q.target_exprs.values())
#             entries[q.target_path].add(q_value)
#
#         for target_path, points in entries.items():
#             flat_map = flat_mapss[p.source_path][target_path]
#             leaf_axis, leaf_clabel = flat_map.axes.leaf
#             for i, pt in enumerate(sorted(points)):
#                 path = p.source_path | {leaf_axis.label: leaf_clabel}
#                 indices = p.source_exprs | {leaf_axis.label: i}
#                 flat_map.set_value(path, indices, pt)
#
#     # build the actual map
#     connectivity = {}
#     for iterset_path, flat_maps in flat_mapss.items():
#         map_components = []
#         for target_path, flat_map in flat_maps.items():
#             # since maps only target a single axis, component pair
#             target_axlabel, target_clabel = just_one(target_path.items())
#             map_component = TabulatedMapComponent(
#                 target_axlabel, target_clabel, flat_map
#             )
#             map_components.append(map_component)
#         connectivity[iterset_path] = map_components
#     return Map(connectivity)
#
#
# def split_loop(loop: Loop, path, tile_size: int) -> Loop:
#     orig_loop_index = loop.index
#
#     # I think I need to transform the index expressions of the iterset?
#     # or get a new iterset? let's try that
#     # It will not work because then the target path would change and the
#     # data structures would not know what to do.
#
#     orig_index_exprs = orig_loop_index.index_exprs
#     breakpoint()
#     # new_index_exprs
#
#     new_loop_index = orig_loop_index.copy(index_exprs=new_index_exprs)
#     return loop.copy(index=new_loop_index)

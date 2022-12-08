import dataclasses
import numpy as np
import itertools
import functools
import pytools
import pyop3.utils
from pyop3 import exprs, tensors, tlang
from pyop3.tensors import *
from typing import Any
import pymbolic as pym


def to_tlang(expr):
    return MultiArrayLangKernelBuilder(expr).build()


@dataclasses.dataclass
class MultiArrayLangKernel:
    instructions: Any


class MultiArrayLangKernelBuilder:
    def __init__(self, expr):
        self._expr = expr
        self._instructions = []

        name_generator = pyop3.utils.MultiNameGenerator()
        self.name_generator = name_generator
        self._iname_generator = functools.partial(name_generator.next, "i")
        self._temp_name_generator = functools.partial(name_generator.next, "t")

        # new thing - track stack of indices so arguments know their shape
        # TODO this can probably be merged with the stuff in pyop3.codegen.loopy
        self._within_multi_index_collections = []

    def build(self) -> MultiArrayLangKernel:
        new_expr = self._inspect(self._expr)
        return new_expr

    @functools.singledispatchmethod
    def _inspect(self, expr, **kwargs):
        raise AssertionError

    @_inspect.register
    def _(self, loop: exprs.Loop):
        # assert isinstance(expr.index, tensors.Indexed)

        # push index to stack
        self._within_multi_index_collections.append(loop.index)

        if not len(loop.statements) == 1:
            raise NotImplementedError

        # only 1 statement currently supported
        stmt, = loop.statements
        if not isinstance(stmt, exprs.FunctionCall):
            # no support yet for nested loops as determining the
            # interleaving of the stencils is quite complicated.
            # It can maybe be resolved by inspecting Indexed nodes
            # but I think a better solution is to defer stencils until later.
            raise NotImplementedError

        stmts = self._inspect(stmt)
        new_expr = loop.copy(statements=stmts)

        # pop from stack (nasty! - use set?)
        self._within_multi_index_collections.pop()
        return new_expr


    @_inspect.register
    def _(self, expr: exprs.FunctionCall):
        # Important: we only need to construct temporaries if the tensor is indexed
        # (i.e. tensor.indices is not None)
        # therefore temporaries should not have any indices
        temporaries = {}
        for arg in expr.arguments:
            dims = self._construct_temp_dims(arg.tensor.axes, arg.tensor.indices)
            dims = dims.set_up()
            temporaries[arg] = tensors.MultiArray.new(
                dims, name=self._temp_name_generator(), dtype=arg.tensor.dtype,
            )

        gathers = self.make_gathers(temporaries)
        call = self.make_function_call(
            expr, temporaries,
            depends_on=frozenset(gather.id for gather in gathers)
        )
        scatters = self.make_scatters(temporaries, depends_on=frozenset({call.id}))

        return (*gathers, call, *scatters)

    def prepend_map(self, map_, after_axis):
        parts = []
        # 1. stick after_axis onto each map output part
        for i, pt in enumerate(map_.parts):
            new_part = tensors.AxisPart(map_.sizes[i])
            new_part.add_subaxis(after_axis)
            parts.append(new_part)

        new_axis = tensors.MultiAxis(parts)

        if map_.from_ in self._within_multi_index_collections:
            return self.prepend_map(map_.from_, new_axis)
        else:
            return new_axis

    def _construct_temp_dims(self, axis, multi_index_collections):
        """Return a multi-axis describing the temporary shape."""
        """
        Can have something like [:5, map2(map1(c))] which should return a temporary
        with shape (5, |map1|, |map2|, ...) where ... is whatever the bottom part of
        the tensor is.

        To do this we start with the first index (the slice) to generate the root axis.
        Then we encounter map2 so we first handle map1. Since c is a within_index we
        disregard it.
        """
        # multi-index collections example: [closure(c0), closure(c1)]
        if multi_index_collections:
            multi_idx_collection, *subidx_collections = multi_index_collections
        else:
            # then take all of the rest of the shape
            multi_idx_collection = MultiIndexCollection([
                MultiIndex([
                    TypedIndex(p, IndexSet(axis.parts[p].size))
                    for p in range(axis.nparts)
                ])
            ])
            subidx_collections = []

        assert isinstance(multi_idx_collection, tensors.MultiIndexCollection)

        ###

        is_loop_index = multi_idx_collection in self._within_multi_index_collections

        # each multi-index yields an adjacent axis part
        temp_axis_parts = []
        for multi_idx in multi_idx_collection:
            # if the index exists then the temporary has a single entry per part
            temp_axis_part_size = 1 if is_loop_index else multi_idx.typed_indices[0].iset.size
            # TODO we should have a more graceful way to include an axis if loop index or not.
            temp_axis_part_id = self.name_generator.next("mypart")
            temp_axis_part  = tensors.AxisPart(
                temp_axis_part_size,
                id=temp_axis_part_id,
            )
            old_temp_axis_part_id = temp_axis_part_id

            # track the position in the array as this tells us whether or not we
            # need to recurse.
            # we need to track this throughout because the types of the typed_idx
            # tells us which bits of the hierarchy are 'below'.
            current_axis = axis.parts[multi_idx.typed_indices[0].part].subaxis

            # each typed index is a subaxis of the original
            for typed_idx in multi_idx.typed_indices[1:]:
                temp_axis_part_id = self.name_generator.next("mypart")
                temp_subaxis  = tensors.MultiAxis(
                    tensors.AxisPart(
                        typed_idx.iset.size,
                        id=temp_axis_part_id
                    )
                )
                temp_axis_part = temp_axis_part.add_subaxis0(old_temp_axis_part_id, temp_subaxis)
                old_temp_axis_part_id = temp_axis_part_id

                current_axis = current_axis.parts[typed_idx.part].subaxis

            # if we still have a current axis then we haven't hit the bottom of the
            # tree and more shape is needed
            if current_axis:
                subaxis = self._construct_temp_dims(current_axis, subidx_collections)
                temp_axis_part = temp_axis_part.add_subaxis(temp_axis_part_id, subaxis)

            temp_axis_parts.append(temp_axis_part)

        temp_axis = tensors.MultiAxis(temp_axis_parts)

        # if we are using a map then we need to stick this axis onto the bottom of
        # whatever the map throws out
        if isinstance(multi_idx_collection, tensors.Map):
            temp_axis = self.prepend_map(multi_idx_collection, temp_axis)

        return temp_axis

    def make_gathers(self, temporaries, **kwargs):
        return tuple(self.make_gather(arg, temp, **kwargs) for arg, temp in temporaries.items())

    def make_gather(self, argument, temporary, **kwargs):
        # TODO cleanup the ids
        if argument.access in {exprs.READ, exprs.RW}:
            return tlang.Read(argument.tensor, temporary, id=self.name_generator.next("read"), **kwargs)
        elif argument.access in {exprs.WRITE, exprs.INC}:
            return tlang.Zero(argument.tensor, temporary, id=self.name_generator.next("write"), **kwargs)
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
        return tlang.FunctionCall(call.function, reads, writes,id=self.name_generator.next("func"), **kwargs)

    def make_scatters(self, temporaries, **kwargs):
        return tuple(
            filter(None, (self.make_scatter(arg, temp, **kwargs) for arg, temp in temporaries.items()))
        )

    def make_scatter(self, argument, temporary, **kwargs):
        if argument.access == exprs.READ:
            return None
        elif argument.access in {exprs.WRITE, exprs.RW}:
            return tlang.Write(argument.tensor, temporary, id=self.name_generator.next("write"), **kwargs)
        elif argument.access == exprs.INC:
            return tlang.Increment(argument.tensor, temporary, id=self.name_generator.next("inc"), **kwargs)
        else:
            raise AssertionError

import dataclasses
import numpy as np
import itertools
import functools
import pytools
import pyop3.utils
from pyop3 import exprs, tensors, tlang
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
        self._within_indices = []

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
        self._within_indices.extend(loop.indices)

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
        for _ in loop.indices:
            self._within_indices.pop()
        return new_expr


    @_inspect.register
    def _(self, expr: exprs.FunctionCall):
        # Important: we only need to construct temporaries if the tensor is indexed
        # (i.e. tensor.indices is not None)
        # therefore temporaries should not have any indices
        temporaries = {}
        for arg in expr.arguments:
            dims = self._construct_temp_dims(arg.tensor.axes, arg.tensor.indices)
            temporaries[arg] = tensors.MultiArray.new(dims, name=self._temp_name_generator(), dtype=arg.tensor.dtype,
                indices=None  # this must be the case for function arguments
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

        if map_.from_ in self._within_indices:
            return self.prepend_map(map_.from_, new_axis)
        else:
            return new_axis

    def _construct_temp_dims(self, axis, indices):
        """Return a multi-axis describing the temporary shape."""
        """
        Can have something like [:5, map2(map1(c))] which should return a temporary
        with shape (5, |map1|, |map2|, ...) where ... is whatever the bottom part of
        the tensor is.

        To do this we start with the first index (the slice) to generate the root axis.
        Then we encounter map2 so we first handle map1. Since c is a within_index we
        disregard it.
        """
        idx, *subidxs = indices

        # this bit is generic across maps and slices
        new_axis_parts = []
        for i, p in enumerate(idx.parts):
            new_axis_part = tensors.AxisPart(idx.sizes[i], layout=pyop3.codegen.AffineLayoutFunction(1))
            # recurse if needed
            if axis.parts[p].subaxis:
                subaxis = self._construct_temp_dims(axis.parts[p].subaxis, subidxs)
                new_axis_part.add_subaxis(subaxis)
            new_axis_parts.append(new_axis_part)

            new_axis = tensors.MultiAxis(new_axis_parts)

        if isinstance(idx, tensors.Map):
            new_axis = self.prepend_map(idx, new_axis)

        return new_axis

        # if it's a map then this needs to get stuck onto the bottom

        # if I encounter a map I should really create the remaining bottom of the tree and
        # then stick it onto something

        # if it's a map then do the map.from first to get tree above...
        # this gets super complicated if the maps above branch into multiple parts...
        # build a stack?
        if isinstance(idx, tensors.Map):
            new_axis = self._construct_temp_dims(axis, idx.from_index)
            new_axis.add_subaxis(tensors.MultiAxis(new_axis_parts))

        # how does calling recursively construct the right thing? multiaxis vs axispart


        # indices that are looped over do not contribute to the temporary's shape
        if idx in self._within_indices:
            self._construct_temp_dims(axis, subidxs)



        raise Exception("old code below")

        if idx in self._within_indices:
            if subidxs:
                return self._construct_temp_dims(subidxs)
            else:
                return None

        if isinstance(idx, tensors.NonAffineMap):
            extra_dims = self._construct_temp_dims(idx.input_indices) or []
        else:
            extra_dims = []

        dims = [tensors.MultiAxis(tensors.AxisPart(idx.size))]

        if subidxs:
            return extra_dims + dims + self._construct_temp_dims(subidxs)
        else:
            return extra_dims + dims

    def construct_temp_dims(self, tensor):
        flat_subdimss = [self._construct_temp_dims(idxs) for idxs in tensor.indicess]

        subdims = [self._nest_dims(sdims) for sdims in flat_subdimss]

        # N.B. each subdim at this point cannot branch (and have multiple parts)
        new_parts = tuple(sdim.part if sdim is not None else None for sdim in subdims)
        return tensors.MultiAxis(new_parts)

    def _nest_dims(self, flat_dims):
        if not flat_dims:
            return tensors.MultiAxis(tensors.ScalarAxisPart())
        d1, *rest = flat_dims
        if rest:
            return d1.copy(parts=d1.parts[0].copy(subaxis=self._nest_dims(rest)))
        else:
            return d1

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

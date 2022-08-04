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

    def build(self) -> MultiArrayLangKernel:
        new_expr = self._inspect(self._expr)
        return new_expr

    @functools.singledispatchmethod
    def _inspect(self, expr, **kwargs):
        raise AssertionError

    @_inspect.register
    def _(self, expr: exprs.Loop):
        # assert isinstance(expr.index, tensors.Indexed)

        if not len(expr.statements) == 1:
            raise NotImplementedError

        for stmt in expr.statements:
            if not isinstance(stmt, exprs.FunctionCall):
                # no support yet for nested loops as determining the
                # interleaving of the stencils is quite complicated.
                # It can maybe be resolved by inspecting Indexed nodes
                # but I think a better solution is to defer stencils until later.
                raise NotImplementedError

            stmts = self._inspect(stmt)
            return expr.copy(statements=stmts)

    @_inspect.register
    def _(self, expr: exprs.FunctionCall):
        temporaries = {}
        for arg in expr.arguments:
            dims = self.construct_temp_dims(arg.tensor)
            temporaries[arg] = tensors.MultiArray.new(dims, name=self._temp_name_generator(), dtype=arg.tensor.dtype)

        gathers = self.make_gathers(temporaries)
        call = self.make_function_call(
            expr, temporaries,
            depends_on=frozenset(gather.id for gather in gathers)
        )
        scatters = self.make_scatters(temporaries, depends_on=frozenset({call.id}))

        return (*gathers, call, *scatters)

    def _construct_temp_dims(self, indices):
        """Return a flattened list of dims that correspond to the provided indices.

        The result needs to be flattened (rather than nested) as it makes composing
        things easier.
        """
        idx, *subidxs = indices

        if idx.is_loop_index:
            if subidxs:
                return self._construct_temp_dims(subidxs)
            else:
                return None

        if isinstance(idx, tensors.NonAffineMap):
            extra_dims = self._construct_temp_dims(idx.input_indices) or []
        else:
            extra_dims = []

        dims = [tensors.MultiAxis(tensors.Axis(idx.size, label=idx.label))]

        if subidxs:
            return extra_dims + dims + self._construct_temp_dims(subidxs)
        else:
            return extra_dims + dims

    def construct_temp_dims(self, tensor):
        flat_subdimss = [self._construct_temp_dims(idxs) for idxs in tensor.indicess]

        subdims = [self._nest_dims(sdims) for sdims in flat_subdimss]

        # N.B. each subdim at this point cannot branch (and have multiple parts)
        new_parts = tuple(sdim.part if sdim else None for sdim in subdims)
        return tensors.MultiAxis(sections=new_parts)

    def _nest_dims(self, flat_dims):
        if not flat_dims:
            return tensors.MultiAxis(tensors.ScalarAxis())
        d1, *rest = flat_dims
        if rest:
            return d1.copy(sections=d1.parts[0].copy(subdim=self._nest_dims(rest)))
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

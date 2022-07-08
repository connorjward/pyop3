import dataclasses
import numpy as np
import itertools
import functools
import pytools
import pyop3.utils
from pyop3 import exprs, tensors, tlang
from typing import Any


def to_tlang(expr):
    return TensorLangKernelBuilder(expr).build()


@dataclasses.dataclass
class TensorLangKernel:
    instructions: Any


class TensorLangKernelBuilder:
    def __init__(self, expr):
        self._expr = expr
        self._instructions = []

        name_generator = pyop3.utils.MultiNameGenerator()
        self.name_generator = name_generator
        self._iname_generator = functools.partial(name_generator.next, "i")
        self._temp_name_generator = functools.partial(name_generator.next, "t")

    def build(self) -> TensorLangKernel:
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
            stencil, = arg.tensor.stencils
            indices, = stencil
            dims = pyop3.utils.Tree.from_nest(self.construct_temp_dims(arg.tensor, indices))
            # import pdb; pdb.set_trace()
            temporaries[arg] = tensors.Tensor(dims, name=self._temp_name_generator(), dtype=arg.tensor.dtype)["fill"]

        gathers = self.make_gathers(temporaries)
        call = self.make_function_call(
            expr, temporaries,
            depends_on=frozenset(gather.id for gather in gathers)
        )
        scatters = self.make_scatters(temporaries, depends_on=frozenset({call.id}))

        return (*gathers, call, *scatters)

    def construct_temp_dims(self, tensor, indices):
        # N.B. we should switch to a tree of stencils I think as otherwise making 'mixed' temporaries is hard.
        shape = tensor.indexed_shape_per_indices(indices)

        nest = None
        for extent in reversed(shape):
            if nest:
                nest = tensors.Dim(extent), [nest]
            else:
                nest = tensors.Dim(extent)
        return nest
        # (subdim_id, index), *subindices = indices
        #
        # subdims = dtree.get_children(dim)
        #
        # if index.within:
        #     if subdims:
        #         return self.construct_temp_dims(subindices, subdims[subdim_id], dtree)
        #     else:
        #         return ()
        # else:
        #     size = tensors.index_size(index, dim, subdim_id)
        #     if subdims:
        #         return tensors.Dim(size), self.construct_temp_dims(subindices, subdims[subdim_id], dtree)
        #     else:
        #         return (tensors.Dim(size), ())

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
            temporaries[arg] for arg in call.arguments
            if arg.access in {exprs.READ, exprs.INC, exprs.RW}
        )
        writes = tuple(
            temporaries[arg] for arg in call.arguments
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

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
            dims = self.construct_temp_dims(arg.tensor)
            # import pdb; pdb.set_trace()
            temporaries[arg] = tensors.Tensor.new(dims, name=self._temp_name_generator(), dtype=arg.tensor.dtype)

        gathers = self.make_gathers(temporaries)
        call = self.make_function_call(
            expr, temporaries,
            depends_on=frozenset(gather.id for gather in gathers)
        )
        scatters = self.make_scatters(temporaries, depends_on=frozenset({call.id}))

        return (*gathers, call, *scatters)

    def _construct_temp_dims(self, indices):
        # import pdb; pdb.set_trace()
        idx, *subidxs = indices

        if idx.within:
            if subidxs:
                return self._construct_temp_dims(subidxs)
            else:
                return None

        if isinstance(idx, tensors.NonAffineMap):
            dim = self._construct_temp_dims(idx.tensor.indices)
        else:
            sizes = (idx.size,)
            labels = (idx.label,)
            dim = tensors.Dim(sizes=sizes, labels=labels)

        if subidxs:
            return dim.copy(subdims=(self._construct_temp_dims(subidxs),))
        else:
            return dim

    def construct_temp_dims(self, tensor):
        subdims = [self._construct_temp_dims(idxs) for idxs in tensor.indicess]

        if len(subdims) == 1:
            # catch single-scalar case
            if subdims == [None]:
                return None
            # else non-mixed
            else:
                return subdims[0]

        sizes = tuple(subdim.size for subdim in subdims)
        labels = tuple(subdim.label for subdim in subdims)
        root = tensors.Dim(sizes=sizes, labels=labels, subdims=subdims)
        return root

    def _as_nest(self, it):
        item, *rest = it
        if rest:
             return [item, self._as_nest(rest)]
        else:
            return [item]

    def _getindexsize(self, index, subdim_id, dim):
        if isinstance(index, tensors.NonAffineMap):
            ans = []
            dim = index.tensor.dim.root
            for subdim_id, idx in index.tensor.indices:
                assert dim is not None

                ans.extend(self._getindexsize(idx, subdim_id, dim))

                if subdims := index.tensor.dim.get_children(dim):
                    dim = subdims[subdim_id]
            return ans
        else:
            # FIXME index_size returns an expression so this would break. Need dim.size
            # to also be an expression (IndexFunction?)
            # if isinstance(index.stop, tensors.Tensor):
            #     size = index.stop
            # else:
            #     size = tensor.index_shape(index, subdim_id, dim)
            size = index.stop or dim.sizes[subdim_id]
            return (dim, size),


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

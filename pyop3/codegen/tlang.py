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
        args = [dataclasses.replace(arg, tensor=self.preprocess_tensor(arg.tensor)) for arg in expr.arguments]
        for arg in args:
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

    def preprocess_tensor(self, tensor):
        # index tensor sizes!
        return tensor
        dims = self._preprocess_tensor(tensor, tensor.dim.root)
        # import pdb; pdb.set_trace()
        return tensor.copy(dim=dims)

    def _preprocess_tensor(self, dim, dims):
        new_tree = None
        for subdim, size in zip(dims.get_children(dim), dim.sizes):
            if isinstance(size, pym.primitives.Expression):
                if not isinstance(size, tensors.Tensor):
                    raise NotImplementedError

                idxs = tensor.indices[-dim.size.order-i:-i]
                new_size = dim.size[tensors.StencilGroup([tensors.Stencil([idxs])])]
                new_dim = dim.copy(sizes=(new_size,))
            else:
                new_dim = dim

            if not new_tree:
                new_tree = pyop3.utils.Tree(new_dim)
            else:
                new_tree = new_tree.add_child(curr_dim, new_dim)

        return new_tree

    def _construct_temp_dims(self, items):
        if any(idx.within for idx, _ in items):
            if len(items) > 1:
                raise NotImplementedError("needs more thought")

            item, = items
            _, children = item
            return self._construct_temp_dims(children)

        labels = tuple(idx.label for idx, _ in items)
        # FIXME this fails for maps!
        sizes = tuple(idx.size if not idx.within else 1 for idx, _ in items)
        # wont work - need clever recursion
        # sizes = tuple(tensors.index_shape(idx) for idx, _ in items)
        # import pdb; pdb.set_trace()

        return tensors.Dim(sizes=sizes, labels=labels), tuple(self._construct_temp_dims(children) for _, children in items if children)

    def construct_temp_dims(self, tensor):
        # FIXME This will fail if we start doing mixed (and hence need to think harder
        # about temp dims)
        # shape, = tensor.indexed_shapes
        # nest = None
        # for extent in reversed(shape):
        #     if nest:
        #         nest = tensors.Dim(extent), [nest]
        #     else:
        #         nest = [tensors.Dim(extent), []]
        # entries = []
        # for item in tensor.indices:
        #     entries.append(self._construct_temp_dims(item))
        dims = self._construct_temp_dims(tensor.indices)
        # import pdb; pdb.set_trace()
        return pyop3.utils.Tree.from_nest(dims)
        # new_dims = None
        # dim = tensor.dim.root
        # for subdim_id, index in tensor.indices:
        #     assert dim is not None
        #
        #     if index.within:
        #         if subdims := tensor.dim.get_children(dim):
        #             dim = subdims[subdim_id]
        #         continue
        #
        #     curr_dim = None
        #     for dim_, size in self._getindexsize(index, subdim_id, dim):
        #
        #         new_dim = tensors.Dim(size, name=dim_.name)
        #         if not new_dims:
        #             new_dims = pyop3.utils.Tree(new_dim)
        #         else:
        #             new_dims.add_child(curr_dim, new_dim)
        #         curr_dim = new_dim
        #
        #     if subdims := tensor.dim.get_children(dim):
        #         dim = subdims[subdim_id]
        #
        # return new_dims or pyop3.utils.Tree(None)

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

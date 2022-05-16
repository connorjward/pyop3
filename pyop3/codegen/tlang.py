import dataclasses
import itertools
import functools
import pytools
import pyop3.utils
from pyop3 import exprs, tensors, tlang
from typing import Any, Optional, List


def to_tlang(expr):
    return TensorLangKernelBuilder(expr).build()


@dataclasses.dataclass
class TensorLangKernel:
    instructions: Any


class TensorLangKernelBuilder:
    def __init__(self, expr):
        self._expr = expr
        self._instructions = []

        name_generator = pyop3.utils.UniqueNameGenerator()
        self.name_generator = name_generator
        self._iname_generator = functools.partial(name_generator.generate, "i")
        self._temp_name_generator = functools.partial(name_generator.generate, "t")

    def build(self) -> TensorLangKernel:
        self._inspect(self._expr, frozenset())
        return TensorLangKernel(self._instructions)

    @functools.singledispatchmethod
    def _inspect(self, expr, **kwargs):
        raise AssertionError

    @_inspect.register
    def _(self, expr: exprs.Loop, within_indices):
        for stmt in expr.statements:
            self._inspect(stmt, within_indices | {expr.index})

    @_inspect.register
    def _(self, expr: exprs.FunctionCall, within_indices):
        temporaries = {arg: self._temp_name_generator() for arg in expr.arguments}

        gathers = self.make_gathers(temporaries, within_indices=within_indices)
        call = self.make_function_call(expr, temporaries, depends_on=frozenset(gather.id for gather in gathers), within_indices=within_indices)
        # return needed in case later things depend on this...
        scatters = self.make_scatters(temporaries, depends_on=frozenset({call.id}), within_indices=within_indices)

        # TODO put somewhere nicer
        for insn in itertools.chain(gathers, [call], scatters):
            self._instructions.append(insn)

    def make_gathers(self, temporaries, **kwargs):
        return tuple(self.make_gather(arg, temp, **kwargs) for arg, temp in temporaries.items())

    def make_gather(self, argument, temporary, **kwargs):
        # TODO cleanup the ids
        if argument.access in {exprs.READ, exprs.RW}:
            return tlang.Read(argument.tensor, temporary, id=self.name_generator.generate("read"), **kwargs)
        elif argument.access in {exprs.WRITE, exprs.INC}:
            return tlang.Zero(argument.tensor, temporary, id=self.name_generator.generate("write"), **kwargs)
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
        return tlang.FunctionCall(call.function, reads, writes,id=self.name_generator.generate("func"), **kwargs)

    def make_scatters(self, temporaries, **kwargs):
        return tuple(
            filter(None, (self.make_scatter(arg, temp, **kwargs) for arg, temp in temporaries.items()))
        )

    def make_scatter(self, argument, temporary, **kwargs):
        if argument.access == exprs.READ:
            return None
        elif argument.access in {exprs.WRITE, exprs.RW}:
            return tlang.Write(argument.tensor, temporary, id=self.name_generator.generate("write"), **kwargs)
        elif argument.access == exprs.INC:
            return tlang.Increment(argument.tensor, temporary, id=self.name_generator.generate("inc"), **kwargs)
        else:
            raise AssertionError

import dataclasses
import functools
import textwrap
from typing import List

from pyop3 import arguments, domains, loops


# @dataclasses.dataclass
# class LoopyContext:
#     """Bag for collecting loopy kernel construction information."""
#     temporaries: List[lp.TemporaryVariable] = dataclasses.field(default_factory=list)


# TODO This is the next thing. I need this to get arguments and temporaries
# packed and unpacked correctly. Some thought will be needed re scoping.
class PseudoContext:
    """Bag for collecting kernel construction information."""

    def __init__(self):
        self.temporary_name_generator = domains.NameGenerator(prefix="t")
        self.temporaries = {}


def generate_pseudocode(op: loops.Statement):
    context = PseudoContext()
    return _generate_pseudocode(op, context)


@functools.singledispatch
def _generate_pseudocode(op: loops.Statement, context):
    raise NotImplementedError


@_generate_pseudocode.register
def _(op: loops.FunctionCall, context: PseudoContext):
    return f"{op.func.name}({', '.join(context.temporaries[arg].name for arg in op.arguments)})"


@_generate_pseudocode.register
def _(op: loops.Loop, context):
    # initialise any temporaries that do not yet exist in the context
    for arg in filter(lambda a: a not in context.temporaries.keys(), op.arguments):
        context.temporaries[arg] = arguments.Dat(next(context.temporary_name_generator))

    # initialise temporaries (including packing)
    if hasattr(op, "arguments"):
        code = [_generate_argument_pseudocode(arg, context) for arg in op.arguments]
    else:
        code = []

    # execute statements
    code += [_generate_pseudocode(stmt, context) for stmt in op.statements]

    # remove empty lines
    code = filter(None, code)

    return f"for {', '.join(index.name for index in op.indices)} ∊ {op.point_set}\n" + textwrap.indent("\n".join(code), "\t") + "\nend for"


@_generate_pseudocode.register
def _(op: loops.Assign, context):
    return f"{op.lhs} = {op.rhs}"


@functools.singledispatch
def _generate_argument_pseudocode(argument, context):
    raise NotImplementedError


@_generate_argument_pseudocode.register
def _(arg: arguments.Dat, context):
    return ""


index_name_generator = domains.NameGenerator(prefix="i")


@_generate_argument_pseudocode.register
def _(arg: arguments.DatSlice, ctx: PseudoContext):
    # check if indexed by a single index (and hence no loop is needed)
    if isinstance(arg.point_set, domains.Index):
        return ""

    # TODO I don't think that we can treat this the same as other loops since
    # we loop over pts AND integers.
    tmp = ctx.temporaries[arg]

    loop = loops.Loop(
        [i := arg.point_set.count_index, p := arg.point_set.point_index],
        tmp[i].assign(arg[p])
    )

    return _generate_pseudocode(loop, ctx)

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


def generate_pseudocode(expr):
    expr = replace_implicit_tensors(expr)
    expr = replace_restrictions_with_loops(expr)

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
def replace_implicit_tensors(expr: loops.Statement):
    """Replace non-temporary arguments to functions.

    (with temporaries and explicit packing instructions.)
    """
    raise AssertionError


@replace_implicit_tensors.register
def _(expr: loops.Loop):
    return type(expr)(expr.indices, chain(map(replace_implicit_tensors, expr.statements)))


@replace_implicit_tensors.register
def _(expr: loops.FunctionCall):
    statements = []
    arguments = []
    for arg in expr.arguments:
        if arg.is_temporary:
            arguments.append(arg)
        else:
            new_temp = "some_temporary"
            statements.append(loops.Restrict(arg.parent, arg.restriction, new_temp))
            arguments.append(new_temp)
    statements.append(type(expr)(expr.func, arguments))
    return tuple(statements)


@functools.singledispatch
def replace_restrictions_with_loops(expr):
    """Replace Restrict nodes (which correspond to linear transformations) with explicit loops."""
    raise AssertionError


@replace_restrictions_with_loops.register
def _(expr: loops.Restrict):
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

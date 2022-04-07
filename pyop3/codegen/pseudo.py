import functools
import textwrap

import dtl
import dtlpp.monads

import pyop3
import pyop3.arguments
import pyop3.transforms
import pyop3.utils


# TODO This is the next thing. I need this to get arguments and temporaries
# packed and unpacked correctly. Some thought will be needed re scoping.
class PseudoContext:
    """Bag for collecting kernel construction information."""

    def __init__(self):
        self.temporary_name_generator = pyop3.utils.NameGenerator(prefix="t")
        self.temporaries = {}  # TODO is this needed?


def preprocess(expr):
    # expr = pyop3.transforms.replace_restricted_tensors(expr)
    # expr = replace_restrictions_with_loops(expr)
    return expr


def lower(expr):
    """Generate pseudocode for the expression."""
    # TODO check expr status to avoid repeating myself
    expr = preprocess(expr)
    # expr = replace_restrictions_with_loops(expr)
    context = PseudoContext()
    return _lower(expr, context)


@functools.singledispatch
def _lower(op: dtl.Node, context):
    raise AssertionError


@_lower.register
def _(expr: dtlpp.monads.FunctionCall, context: PseudoContext):
    return str(expr)


@_lower.register
def _(op: pyop3.Loop, context):
    # execute statements
    code = [_lower(stmt, context) for stmt in op.statements]

    # remove empty lines
    code = filter(None, code)

    return (
        f"for {', '.join(index.name for index in op.indices)} ∊ {op.point_set}\n"
        + textwrap.indent("\n".join(code), "  ")
        + "\nend for"
    )

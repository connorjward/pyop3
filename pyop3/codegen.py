import dataclasses
import functools
from typing import List

from pyop3 import loops


# @dataclasses.dataclass
# class LoopyContext:
#     """Bag for collecting loopy kernel construction information."""
#     temporaries: List[lp.TemporaryVariable] = dataclasses.field(default_factory=list)


# TODO This is the next thing. I need this to get arguments and temporaries
# packed and unpacked correctly. Some thought will be needed re scoping.
class PseudoContext:
    """Bag for collecting kernel construction information."""


@functools.singledispatch
def generate_pseudocode(op: loops.Op):
    return str(op),


@generate_pseudocode.register
def _(op: loops.Loop):
    code = [f"for {op.domain_index} in {op.domain_index.domain}"]
    for stmt in op.statements:
        string = generate_pseudocode(stmt)
        for s in string:
            code.append("\t" + s)
    code += ["end for"]
    return code

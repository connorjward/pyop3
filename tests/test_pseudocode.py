import pyop3.codegen
import pyop3.loops
from pyop3.arguments import Dat
from pyop3.loops import Loop, Function
from pyop3.domains import FreePointSet, closure


def test_pseudocode():
    loop = pyop3.loops.Loop([], [], pyop3.loops.Function("myfunc"))

    assert pyop3.codegen.generate_pseudocode(loop) == "something that will fail"


def test_nested_domains():
    iterset = FreePointSet("P")
    func = Function("func")
    loop = Loop(
        p := iterset.point_index,
        statements=Loop(q := closure(p).point_index, func()))
    
    assert pyop3.codegen.generate_pseudocode(loop) == "something that fails"


def test_arguments():
    iterset = FreePointSet("P")
    dat = Dat("dat0")
    func = Function("myfunc")

    loop = Loop(
        p := iterset.point_index,
        statements=[func(dat[closure(p)])]
    )

    code = pyop3.codegen.generate_pseudocode(loop)
    assert code == "something that fails"


def test_multiple_statements():
    iterset = FreePointSet("P")
    dat = Dat("dat0")
    func1 = Function("func1")
    func2 = Function("func2")

    loop = Loop(
        p := iterset.point_index,
        statements=[func1(dat[p]), func2(dat[p])]
    )

    code = pyop3.codegen.generate_pseudocode(loop)
    assert code == "something that fails"


"""
An example of a pack

t0 = [0, ...]
for i, p' in closure(p)
    t0[i] = dat[p']
end for

loop(p' := closure(p).index,
     [t0, dat[p']]
     [Assign(t0, dat)])

loop(p' := closure(p).index,
     [Assign("t0", dat[p'])])

for something directly accessed:

    loop(p := P.index, [d := dat], [kernel(d)])


    loop(p := P.index, [kernel(dat[closure(p)])])
"""

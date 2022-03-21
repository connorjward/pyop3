import pyop3.codegen
import pyop3.loops
from pyop3.arguments import Dat
from pyop3.loops import Loop, Function
from pyop3.domains import FreeDomain, closure


def test_pseudocode():
    loop = pyop3.loops.Loop([], [], pyop3.loops.Function("myfunc"))

    assert pyop3.codegen.generate_pseudocode(loop) == "something that will fail"


def test_nested_domains():
    iterset = FreeDomain("extent")
    loop = Loop(
        i := iterset.index,
        arguments=[],
        temporaries=[],
        statements=Loop(j := closure(i).index, [], [], Function("myfunc"))
    )
    
    assert "\n".join(pyop3.codegen.generate_pseudocode(loop)) == "something that fails"


def test_arguments():
    iterset = FreeDomain(100)
    dat = Dat("dat0")
    func = Function("myfunc")

    loop = Loop(
        i := iterset.index,
        arguments=[d0 := dat[closure(i)]],
        statements=[func(d0)]
    )

    code = pyop3.codegen.generate_pseudocode(loop)
    assert "\n".join(code) == "something that fails"

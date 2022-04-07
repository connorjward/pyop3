import argparse

import dtlutils
from pyop3 import Dat, FreePointSet, Function, Loop, closure, star
from pyop3.codegen.pseudo import lower, preprocess

from pyop3.exprs import loop


FUNC_LOOKUP = {}


def register_func(func):
    global FUNC_LOOKUP
    FUNC_LOOKUP[func.__name__] = func
    return lambda *args, **kwargs: func(*args, **kwargs)


@register_func
def basic():
    iterset = FreePointSet("P")
    kernel = Function("kernel")
    return Loop(iterset.index, kernel())


@register_func
def with_nested_domains():
    iterset = FreePointSet("P")
    func = Function("func")
    return Loop(p := iterset.point_index, statements=Loop(closure(p).index, func()))


@register_func
def with_arguments():
    iterset = FreePointSet("P")
    dat = Dat("dat0")
    func = Function("myfunc")

    return Loop(p := iterset.point_index, statements=[func(dat[closure(p)])])


@register_func
def with_multiple_statements():
    iterset = FreePointSet("P")
    dat = Dat("dat0")
    func1 = Function("func1")
    func2 = Function("func2")

    return Loop(p := iterset.point_index, statements=[func1(dat[p]), func2(dat[p])])


@register_func
def with_temporaries():
    iterset = FreePointSet("P")
    dat0 = Dat("dat0")
    dat1 = Dat("dat1")
    func1 = Function("func1")
    func2 = Function("func2")

    return Loop(p := iterset.point_index, [func1(dat0[p], "t0"), func2("t0", dat1[p])])


@register_func
def pcpatch():
    iterset = FreePointSet("P")

    dat1 = Dat("dat1")
    dat2 = Dat("dat2")
    sol = Dat("sol")

    assemble_mat = Function("assemble_mat")
    assemble_vec = Function("assemble_vec")
    solve = Function("solve")

    return loop(
        p := iterset.point_index,
        [
            loop(q := star(p).index, assemble_mat(dat1[q], "mat")),
            loop(q := star(p).index, assemble_vec(dat2[q], "vec")),
            solve("mat", "vec", sol[p])
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str)
    args = parser.parse_args()

    func = FUNC_LOOKUP[args.function]
    expr = func()
    expr = preprocess(expr)

    print(func.__name__)
    # print(lower(expr))
    dtlutils.plot_dag(expr, name=func.__name__, view=True)


if __name__ == "__main__":
    main()

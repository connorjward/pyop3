import argparse

from pyop3 import (
    Dat, FreePointSet, Function, Loop, closure, star
)
from pyop3.codegen.pseudo import lower, preprocess
from pyop3.tools.visualize import visualize


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
    return Loop(
        p := iterset.point_index,
        statements=Loop(closure(p).index, func())
    )


@register_func
def with_arguments():
    iterset = FreePointSet("P")
    dat = Dat("dat0")
    func = Function("myfunc")

    return Loop(
        p := iterset.point_index,
        statements=[func(dat[closure(p)])]
    )


@register_func
def with_multiple_statements():
    iterset = FreePointSet("P")
    dat = Dat("dat0")
    func1 = Function("func1")
    func2 = Function("func2")

    return Loop(
        p := iterset.point_index,
        statements=[func1(dat[p]), func2(dat[p])]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("func", type=int)
    args = parser.parse_args()

    func = FUNC_LOOKUP[args.func]
    expr = func()
    expr = preprocess(expr)

    print(func.__name__)
    print(lower(expr))
    visualize(expr, name=func.__name__, view=True)


if __name__ == "__main__":
    main()

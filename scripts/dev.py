import argparse

import dtl
import dtlutils
from pyop3 import Dat, Mat, FreePointSet, Loop, closure, star
import dtlpp
from dtlpp.functions import Function, ArgumentSpec
from dtlc.backends.pseudo import lower
from dtlpp import READ, WRITE, INC, RealVectorSpace

from pyop3.exprs import loop


DEMOS = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=str, choices=DEMOS.keys())
    parser.add_argument("--show")
    args = parser.parse_args()

    expr = DEMOS[args.demo]()

    print(args.demo)
    print(lower(expr))

    if args.show:
        dtlutils.plot_dag(expr, name=args.demo, view=True)


NPOINTS = 25
ITERSET = FreePointSet("P", NPOINTS)
ITER_SPACE = RealVectorSpace(NPOINTS)

CLOSURE_ARITY = 2
STAR_ARITY = 5

CLOSURE_SPACE = RealVectorSpace(CLOSURE_ARITY)
STAR_SPACE = RealVectorSpace(STAR_ARITY)

dat1 = Dat(ITER_SPACE**1, "dat1")
dat2 = Dat(ITER_SPACE**1, "dat2")
dat3 = Dat(ITER_SPACE**1, "dat3")

mat1 = Mat(ITER_SPACE**2, "mat1")


def register_demo(func):
    global DEMOS
    DEMOS[func.__name__] = func
    return lambda *args, **kwargs: func(*args, **kwargs)


@register_demo
def matvec():
    i, j = dtl.indices("i", "j")
    return (mat1[i, j] * dat1[i]).forall(j)


@register_demo
def trace():
    i, j = dtl.indices("i", "j")
    return mat1[i, j].forall()


@register_demo
def basic_parloop():
    result = Dat(ITER_SPACE**1, "result")
    kernel = Function(
        "kernel",
        [
            ArgumentSpec(READ, CLOSURE_SPACE**1),
            ArgumentSpec(READ, CLOSURE_SPACE**1),
            ArgumentSpec(INC, CLOSURE_SPACE**1)
        ]
    )
    return loop(
        p := ITERSET.index,
        kernel(
            dat1[closure(p, CLOSURE_ARITY)],
            dat2[closure(p, CLOSURE_ARITY)],
            result[closure(p, CLOSURE_ARITY)]
        )
    )


@register_demo
def multi_kernel():
    result = Dat(ITER_SPACE**1, "result")
    kernel1 = Function(
        "kernel1",
        [ArgumentSpec(READ, CLOSURE_SPACE**1), ArgumentSpec(WRITE, CLOSURE_SPACE**1)]
    )
    kernel2 = Function(
        "kernel2",
        [ArgumentSpec(READ, CLOSURE_SPACE**1), ArgumentSpec(INC, CLOSURE_SPACE**1)]
    )

    return Loop(p := ITERSET.index,
                [kernel1(dat1[closure(p)], "tmp"), kernel2("tmp", result[closure(p)])])


@register_demo
def pcpatch():
    result = Dat(ITER_SPACE**1, "result")

    assemble_mat = Function(
        "assemble_mat",
        [ArgumentSpec(READ, dtl.TensorSpace([])), ArgumentSpec(WRITE, STAR_SPACE**2)]
    )
    assemble_vec = Function(
        "assemble_vec",
        [ArgumentSpec(READ, dtl.TensorSpace([])), ArgumentSpec(WRITE, STAR_SPACE**1)]
    )
    solve = Function(
        "solve",
        [
            ArgumentSpec(READ, STAR_SPACE**2),
            ArgumentSpec(READ, STAR_SPACE**1),
            ArgumentSpec(WRITE, dtl.TensorSpace([]))
        ]
    )

    return loop(
        p := ITERSET.index,
        [
            loop(q := star(p, STAR_ARITY).index, assemble_mat(dat1[q], "mat")),
            loop(q := star(p, STAR_ARITY).index, assemble_vec(dat2[q], "vec")),
            solve("mat", "vec", result[p])
        ]
    )


if __name__ == "__main__":
    main()

import argparse
import pdb

import loopy as lp
import numpy as np
import pyop3
import pyop3.codegen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=str, choices=DEMOS.keys())
    parser.add_argument(
        "-t", "--target", type=str, required=True, choices={"dag", "c", "loopy"}
    )
    parser.add_argument("--pdb", action="store_true", dest="breakpoint")
    args = parser.parse_args()

    expr = DEMOS[args.demo]()

    if args.target == "dag":
        pyop3.visualize.plot_expression_dag(expr, name=args.demo, view=True)
        exit()
    elif args.target == "c":
        program = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    elif args.target == "loopy":
        program = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.LOOPY)
    else:
        raise AssertionError

    print(program)

    if args.breakpoint:
        pdb.set_trace()


class DemoMesh:

    CLOSURE_ARITY = 2
    STAR_ARITY = 5

    def __init__(self, ncells):
        self.ncells = ncells

    @property
    def cells(self):
        return pyop3.Domain(self.ncells, self)

    def closure(self, index):
        return pyop3.Domain(self.CLOSURE_ARITY, self, index)


NPOINTS = 25
MESH = DemoMesh(NPOINTS)
ITERSET = MESH.cells

dat1 = pyop3.Dat(NPOINTS, "dat1")
dat2 = pyop3.Dat(NPOINTS, "dat2")
dat3 = pyop3.Dat(NPOINTS, "dat3")

mat1 = pyop3.Mat((NPOINTS, NPOINTS), "mat1")

DEMOS = {}


def register_demo(func):
    global DEMOS
    DEMOS[func.__name__] = func
    return lambda *args, **kwargs: func(*args, **kwargs)


@register_demo
def basic_parloop():
    result = pyop3.Dat(NPOINTS, "result")
    loopy_kernel = lp.make_kernel(
        f"{{ [i]: 0 <= i < {MESH.CLOSURE_ARITY} }}",
        ["z[i] = z[i] + x[i] * y[i]"],
        [
            lp.GlobalArg(
                "x", np.float64, (MESH.CLOSURE_ARITY,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (MESH.CLOSURE_ARITY,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (MESH.CLOSURE_ARITY,), is_input=True, is_output=True
            ),
        ],
        target=lp.CTarget(),
        name="local_kernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, MESH.CLOSURE_ARITY),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, MESH.CLOSURE_ARITY),
            pyop3.ArgumentSpec(pyop3.INC, np.float64, MESH.CLOSURE_ARITY),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := ITERSET.index,
        [
            kernel(
                dat1[pyop3.closure(p)], dat2[pyop3.closure(p)], result[pyop3.closure(p)]
            )
        ],
    )


@register_demo
def multi_kernel():
    result = Dat(ITER_SPACE**1, "result")
    kernel1 = Function(
        "kernel1",
        [
            ArgumentSpec(READ, CLOSURE_SPACE**1),
            ArgumentSpec(WRITE, CLOSURE_SPACE**1),
        ],
    )
    kernel2 = Function(
        "kernel2",
        [ArgumentSpec(READ, CLOSURE_SPACE**1), ArgumentSpec(INC, CLOSURE_SPACE**1)],
    )

    return Loop(
        p := ITERSET.index,
        [kernel1(dat1[closure(p)], "tmp"), kernel2("tmp", result[closure(p)])],
    )


@register_demo
def pcpatch():
    result = Dat(ITER_SPACE**1, "result")

    assemble_mat = Function(
        "assemble_mat",
        [ArgumentSpec(READ, dtl.TensorSpace([])), ArgumentSpec(WRITE, STAR_SPACE**2)],
    )
    assemble_vec = Function(
        "assemble_vec",
        [ArgumentSpec(READ, dtl.TensorSpace([])), ArgumentSpec(WRITE, STAR_SPACE**1)],
    )
    solve = Function(
        "solve",
        [
            ArgumentSpec(READ, STAR_SPACE**2),
            ArgumentSpec(READ, STAR_SPACE**1),
            ArgumentSpec(WRITE, dtl.TensorSpace([])),
        ],
    )

    return Loop(
        p := ITERSET.index,
        [
            Loop(q := star(p, STAR_ARITY).index, [assemble_mat(dat1[q], "mat")]),
            Loop(q := star(p, STAR_ARITY).index, [assemble_vec(dat2[q], "vec")]),
            solve("mat", "vec", result[p]),
        ],
    )


if __name__ == "__main__":
    main()

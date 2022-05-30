import argparse
import pdb

import loopy as lp
import numpy as np

import pyop3
import pyop3.codegen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", type=str, choices=[*DEMOS.keys(), "check"])
    parser.add_argument(
        "-t", "--target", type=str, required=True, choices={"dag", "c", "loopy"}
    )
    parser.add_argument("--pdb", action="store_true", dest="breakpoint")
    args = parser.parse_args()

    if args.demo == "check":
        for name, demo in DEMOS.items():
            print(name, end="")
            try:
                expr = demo()
                pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
                print(" SUCCESS")
            except:
                print(" FAIL")
        exit()

    # if args.breakpoint:
    #     pdb.set_trace()

    expr = DEMOS[args.demo]()

    if args.target == "c":
        program = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    elif args.target == "loopy":
        program = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.LOOPY)
    elif args.target == "pyop3":
        program = expr
    elif args.target == "tlang":
        program = pyop3.codegen.to_tlang(expr)
    else:
        raise AssertionError

    print(program)


NBASE_EDGES = 25
NBASE_VERTS = 26
EXTRUDED_MESH = pyop3.ExtrudedMesh((NBASE_EDGES, NBASE_VERTS))

NCELLS = 25
NEDGES = 40
NVERTS = 30
MESH = pyop3.UnstructuredMesh((NCELLS, NEDGES, NVERTS))
ITERSET = MESH.cells

section = pyop3.Section([6, 7, 8])  # 6 dof per cell, 7 per edge and 8 per vert

glob1 = pyop3.Global(name="glob1")

dat1 = pyop3.Dat(MESH, section, name="dat1")
dat2 = pyop3.Dat(MESH, section, name="dat2")
dat3 = pyop3.Dat(MESH, section, name="dat3")

edat1 = pyop3.ExtrudedDat(EXTRUDED_MESH, section, name="edat1")
edat2 = pyop3.ExtrudedDat(EXTRUDED_MESH, section, name="edat2")
edat3 = pyop3.ExtrudedDat(EXTRUDED_MESH, section, name="edat3")

# vdat1 = pyop3.VectorDat(MESH, section, 3, name="vdat1")
# vdat2 = pyop3.VectorDat(MESH, section, 3, name="vdat2")
# vdat3 = pyop3.VectorDat(MESH, section, 3, name="vdat3")

# mat1 = pyop3.Mat((MESH.points, MESH.points), name="mat1")

DEMOS = {}


def register_demo(func):
    global DEMOS
    DEMOS[func.__name__] = func
    return lambda *args, **kwargs: func(*args, **kwargs)


@register_demo
def direct():
    result = pyop3.Dat(MESH, section, name="result")
    loopy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        ["z[i] = z[i] + x[i] * y[i]"],
        [
            lp.GlobalArg(
                "x", np.float64, (6,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (6,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (6,), is_input=True, is_output=True
            ),
        ],
        target=lp.CTarget(),
        name="local_kernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, ()),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, ()),
            pyop3.ArgumentSpec(pyop3.INC, np.float64, ()),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := ITERSET.index,
        [kernel(dat1[p], dat2[p], result[p])],
    )


@register_demo
def inc():
    result = pyop3.Dat(MESH, section, name="result")
    # FIXME since we don't do broadcasting currently
    # size = 6 + MESH.NEDGES_IN_CELL_CLOSURE * 7 + MESH.NVERTS_IN_CELL_CLOSURE * 8
    size = 1 + MESH.NEDGES_IN_CELL_CLOSURE + MESH.NVERTS_IN_CELL_CLOSURE
    loopy_kernel = lp.make_kernel(
        f"{{ [i]: 0 <= i < {size} }}",
        ["z[i] = z[i] + x[i] * y[i]"],
        [
            lp.GlobalArg(
                "x", np.float64, (size,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (size,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (size,), is_input=True, is_output=True
            ),
        ],
        target=lp.CTarget(),
        name="local_kernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, size),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, size),
            pyop3.ArgumentSpec(pyop3.INC, np.float64, size),
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
def global_parloop():
    result = pyop3.Dat(NPOINTS, name="result")
    loopy_kernel = lp.make_kernel(
        [f"{{ [i]: 0 <= i < {MESH.CLOSURE_ARITY} }}"],
        ["z[i] = g"],
        [
            lp.GlobalArg("g", np.float64, (), is_input=True, is_output=False),
            lp.GlobalArg(
                "z", np.float64, (MESH.CLOSURE_ARITY,), is_input=False, is_output=True
            ),
        ],
        target=pyop3.codegen.LOOPY_TARGET,
        name="local_kernel",
        lang_version=pyop3.codegen.LOOPY_LANG_VERSION,
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, 1),
            pyop3.ArgumentSpec(pyop3.WRITE, np.float64, MESH.CLOSURE_ARITY),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := ITERSET.index,
        [kernel(glob1, result[pyop3.closure(p)])],
    )


@register_demo
def vdat():
    result = pyop3.Dat(NPOINTS, name="result")
    loopy_kernel = lp.make_kernel(
        [f"{{ [i]: 0 <= i < {MESH.CLOSURE_ARITY} }}", "{ [j]: 0<= j < 3}"],
        ["z[i] = z[i] + x[i, j] * y[i, j]"],
        [
            lp.GlobalArg(
                "x", np.float64, (MESH.CLOSURE_ARITY, 3), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (MESH.CLOSURE_ARITY, 3), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (MESH.CLOSURE_ARITY,), is_input=False, is_output=True
            ),
        ],
        target=lp.CTarget(),
        name="local_kernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, (MESH.CLOSURE_ARITY, 3)),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, (MESH.CLOSURE_ARITY, 3)),
            pyop3.ArgumentSpec(pyop3.WRITE, np.float64, MESH.CLOSURE_ARITY),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := ITERSET.index,
        [
            kernel(
                vdat1[pyop3.closure(p)],
                vdat2[pyop3.closure(p)],
                result[pyop3.closure(p)],
            )
        ],
    )


@register_demo
def extruded():
    dat1 = pyop3.Dat((EXTRUDED_MESH.NBASECELLS, 5), name="dat1")
    dat2 = pyop3.Dat((EXTRUDED_MESH.NBASECELLS, 5), name="dat2")
    result = pyop3.Dat((EXTRUDED_MESH.NBASECELLS, 5), name="result")
    loopy_kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        ["z = x + y"],
        [
            lp.GlobalArg(
                "x", np.float64, (), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (), is_input=False, is_output=True
            ),
        ],
        target=pyop3.codegen.LOOPY_TARGET,
        name="local_kernel",
        lang_version=pyop3.codegen.LOOPY_LANG_VERSION,
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, ()),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, ()),
            pyop3.ArgumentSpec(pyop3.WRITE, np.float64, ()),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := EXTRUDED_MESH.cells.index,
        [
            kernel(
                dat1[p], dat2[p], result[p]
            )
        ],
    )


@register_demo
def extr_inc():
    size = 6 + EXTRUDED_MESH.NEDGES_IN_CELL_CLOSURE * 7 + EXTRUDED_MESH.NVERTS_IN_CELL_CLOSURE * 8
    loopy_kernel = lp.make_kernel(
        f"{{ [i]: 0 <= i < {size} }}",
        ["z[i] = z[i] + x[i] * y[i]"],
        [
            lp.GlobalArg(
                "x", np.float64, (size,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (size,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (size,), is_input=True, is_output=True
            ),
        ],
        target=lp.CTarget(),
        name="local_kernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, size),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, size),
            pyop3.ArgumentSpec(pyop3.INC, np.float64, size),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := EXTRUDED_MESH.cells.index,
        [
            kernel(
                edat1[pyop3.closure(p)], edat2[pyop3.closure(p)], edat3[pyop3.closure(p)]
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
        p := ITERSET,
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
        p := ITERSET,
        [
            Loop(q := star(p, STAR_ARITY), [assemble_mat(dat1[q], "mat")]),
            Loop(q := star(p, STAR_ARITY), [assemble_vec(dat2[q], "vec")]),
            solve("mat", "vec", result[p]),
        ],
    )


if __name__ == "__main__":
    main()

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
                demo()
                print(" SUCCESS")
            except:
                print(" FAIL")
        exit()

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

    def __init__(self, npoints, ncells):
        self.npoints = "npoints"
        self.ncells = "ncells"
        self.map_cache = {}

    @property
    def points(self):
        return pyop3.Range(self.npoints, mesh=self)

    @property
    def cells(self):
        return pyop3.Range(self.ncells, mesh=self)

    def closure(self, index):
        key = (self.closure.__name__, index)
        try:
            return self.map_cache[key]
        except KeyError:
            map_ = pyop3.Tensor(self.CLOSURE_ARITY, index, mesh=self, prefix="map")
            return self.map_cache.setdefault(key, map_)


class DemoExtrudedMesh:

    NPOINTS = 68
    NBASECELLS = 12
    BASE_CLOSURE_ARITY = 5

    def __init__(self):
        base = pyop3.Range(self.NBASECELLS)
        self.basecells = base
        # self.offsets = pyop3.Dat(base, name="offsets")
        self.layer_count = pyop3.Dat(self.NBASECELLS, name="layer_count")

    def closure(self, index):
        """

        Something like:
        for cell
            for layer
                for i
                    dat[map[i], offset[i]+layer*layersize]
        """
        raise NotImplementedError

        # map from the cell to the starting points for the extruded traversal
        # cell -> other base cells
        base_domain = pyop3.SparseDomain(index.domain.parent, self.BASE_CLOSURE_ARITY)

        # strided access to the layers
        # should be:
        # layer_domain = DenseDomain(0, self.layer_count[index]*self.offsets[index], self.offsets[index])

        # FIXME haven't implemented __mul__: is this even right?
        return pyop3.CompositeDomain(base_domain, index.domain*self.offsets[index])

    @property
    def cells(self):
        layers = pyop3.Range(self.layer_count[self.basecells.index])
        return IndexBag(self.basecells, layers)


class IndexBag:
    def __init__(self, *indices):
        self.indices = indices

    @property
    def index(self):
        return tuple(idx.index for idx in self.indices)


EXTRUDED_MESH = DemoExtrudedMesh()


NPOINTS = 120
NCELLS = 25
MESH = DemoMesh(NPOINTS, NCELLS)
ITERSET = MESH.cells

glob1 = pyop3.Global(name="glob1")

dat1 = pyop3.Dat(NPOINTS, name="dat1")
dat2 = pyop3.Dat(NPOINTS, name="dat2")
dat3 = pyop3.Dat(NPOINTS, name="dat3")

vdat1 = pyop3.Dat((NPOINTS, 3), name="vdat1")
vdat2 = pyop3.Dat((NPOINTS, 3), name="vdat2")

# mat1 = pyop3.Mat((MESH.points, MESH.points), name="mat1")

DEMOS = {}


def register_demo(func):
    global DEMOS
    DEMOS[func.__name__] = func
    return lambda *args, **kwargs: func(*args, **kwargs)


@register_demo
def basic_parloop():
    result = pyop3.Dat(NPOINTS, name="result")
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
def vdat_parloop():
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
def extruded_direct():
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
def extruded_parloop():
    dat1 = pyop3.Dat(EXTRUDED_MESH.points, name="dat1")
    dat2 = pyop3.Dat(EXTRUDED_MESH.points, name="dat2")
    result = pyop3.Dat(EXTRUDED_MESH.points, name="result")
    loopy_kernel = lp.make_kernel(
        f"{{ [i]: 0 <= i < {EXTRUDED_MESH.CLOSURE_ARITY} }}",
        ["z[i] = z[i] + x[i] * y[i]"],
        [
            lp.GlobalArg(
                "x", np.float64, (EXTRUDED_MESH.CLOSURE_ARITY,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (EXTRUDED_MESH.CLOSURE_ARITY,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (EXTRUDED_MESH.CLOSURE_ARITY,), is_input=True, is_output=True
            ),
        ],
        target=lp.CTarget(),
        name="local_kernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, EXTRUDED_MESH.CLOSURE_ARITY),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, EXTRUDED_MESH.CLOSURE_ARITY),
            pyop3.ArgumentSpec(pyop3.INC, np.float64, EXTRUDED_MESH.CLOSURE_ARITY),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := EXTRUDED_ITERSET.index,
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

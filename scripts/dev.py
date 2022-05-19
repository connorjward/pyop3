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



class DemoMesh(pyop3.UnstructuredMesh):

    NCELLS_IN_CELL_CLOSURE = 1
    NEDGES_IN_CELL_CLOSURE = 3
    NVERTS_IN_CELL_CLOSURE = 3
    CLOSURE_SIZE = NCELLS_IN_CELL_CLOSURE + NEDGES_IN_CELL_CLOSURE + NVERTS_IN_CELL_CLOSURE

    map_cache = {}

    # def cone(self, itree):
    #     (stratum, subtree), = itree.indices.items()
    #     range_, = subtree.indices.keys()


    def closure(self, itree):
        key = (self.closure.__name__, itree)
        try:
            return self.map_cache[key]
        except KeyError:
            subtree, _, _ = itree.children
            range_ = subtree.index

            new_itree = pyop3.IndexTree(None,
                children=(subtree,
                          pyop3.IndexTree(pyop3.NonAffineMap(range_, pyop3.Range(self.NEDGES_IN_CELL_CLOSURE))),
                          pyop3.IndexTree(pyop3.NonAffineMap(range_, pyop3.Range(self.NVERTS_IN_CELL_CLOSURE)))),
            mesh=self)
            return self.map_cache.setdefault(key, new_itree)


class DemoExtrudedMesh:

    NPOINTS = 68
    NBASECELLS = 12
    BASE_CLOSURE_ARITY = 5

    def __init__(self):
        base = pyop3.Slice(self.NBASECELLS)
        self.basecells = base
        # self.offsets = pyop3.Dat(base, name="offsets")
        self.layer_count = pyop3.Dat(self.NBASECELLS, name="layer_count")

    def closure(self, index):
        """
        For semi-structured meshes we return a tuple of slices for plex ops. The
        first index is for the base mesh entities touched by the stencil. This can have variable arity depending
        on the number of incident base entities (for example a cubed-sphere mesh can touch
        just faces, faces and edges or vertices). The inner map is simply an affine start and offset
        such that the correct memory address can be retrieved given an input point. Like the map, this can have
        non-unity arity (e.g. if there are 5 DoFs in the cell (edge of base mesh) column but only 2 in the facet column
        (vertex in base mesh)). The initial start point of the map can also depend on the outer dimension. This
        is needed for variable layers in extrusion.
        """
        # TODO clean up the semantics of slices versus index tuples and maps. A slice implies an extent but what we
        # really need is a map from p -> qs and the extent is merely provided by p. Maybe 'range' is clearer (and our
        # current slice and map no longer inherit from it). Possibly call the current slice an 'affine map'.

        """For this example, let's assume that we have a cell that looks like this (+'s are DoFs):

        +----+----+
        |         |
        +   +++   +
        |         |
        +----+----+

        If we take the closure of the cell then we touch 3 base entities - 2 vertices and 1 edge - so the base map
        has arity 3. Due to the differing DoF counts for each entity, the arities of the column maps is different
        for each: 3, 5 and 3 respectively with offsets (0, 1, 2), (0, 1, 2, 3, 4) and (0, 1, 2).

        The base map points to the first entity in each column from which the data is recovered by applying a fixed offset and stride.
        This resolves problems with variable layers in extruded meshes where we might have a mesh that looks something like:

        +---+
        | 4 |
        +---+---+
        | 3 | 2 |
        z-a-y---+
            | 1 |
            x---+

        We can get around any problems by having base_map(3) return (a, y, z) rather than (a, x, z). If we do that
        then we can forget about it as an issue later on during code generation.
        """
        base_map = pyop3.Map(index, 3, mesh=self)
        column_maps = (
            pyop3.Slice(0, 3),
            pyop3.Slice(0, 5),
            pyop3.Slice(0, 3),
        )
        return base_map, column_maps

    @property
    def cells(self):
        layers = pyop3.Slice(self.layer_count[self.basecells.index])
        return IndexBag(self.basecells, layers)


# EXTRUDED_MESH = DemoExtrudedMesh()


NCELLS = 25
NEDGES = 40
NVERTS = 30
MESH = DemoMesh((NCELLS, NEDGES, NVERTS))
ITERSET = MESH.cells

section = pyop3.Section([6, 7, 8])  # 6 dof per cell, 7 per edge and 8 per vert

glob1 = pyop3.Global(name="glob1")

dat1 = pyop3.Dat(MESH, section, name="dat1")
dat2 = pyop3.Dat(MESH, section, name="dat2")
dat3 = pyop3.Dat(MESH, section, name="dat3")

# won't work as dat expects two sections when it needs 3
# vdat1 = pyop3.Dat((MESH, section, vsection), name="vdat1")
# vdat2 = pyop3.Dat((MESH, section, vsection), name="vdat2")

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
                "x", np.float64, (), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (), is_input=True, is_output=True
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
        p := ITERSET.loop_index,
        [kernel(dat1[p], dat2[p], result[p])],
    )


@register_demo
def inc():
    result = pyop3.Dat(MESH, section, name="result")
    loopy_kernel = lp.make_kernel(
        f"{{ [i]: 0 <= i < {MESH.CLOSURE_SIZE} }}",
        ["z[i] = z[i] + x[i] * y[i]"],
        [
            lp.GlobalArg(
                "x", np.float64, (MESH.CLOSURE_SIZE,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "y", np.float64, (MESH.CLOSURE_SIZE,), is_input=True, is_output=False
            ),
            lp.GlobalArg(
                "z", np.float64, (MESH.CLOSURE_SIZE,), is_input=True, is_output=True
            ),
        ],
        target=lp.CTarget(),
        name="local_kernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.Function(
        "local_kernel",
        [
            pyop3.ArgumentSpec(pyop3.READ, np.float64, MESH.CLOSURE_SIZE),
            pyop3.ArgumentSpec(pyop3.READ, np.float64, MESH.CLOSURE_SIZE),
            pyop3.ArgumentSpec(pyop3.INC, np.float64, MESH.CLOSURE_SIZE),
        ],
        loopy_kernel,
    )
    return pyop3.Loop(
        p := ITERSET.loop_index,
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

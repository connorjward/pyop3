import numpy as np
from petsc4py import PETSc


def create_mesh(npoints, edges):
    dm = PETSc.DMPlex()
    dm.create()
    dm.setChart(0, npoints)
    for start, stops in edges.items():
        dm.setConeSize(start, len(stops))
    dm.setUp()
    for start, stops in edges.items():
        dm.setCone(start, stops)
    dm.setDimension(2)
    dm.symmetrize()
    dm.stratify()
    return dm


def mesh1():
    """A PETSc DMPlex object representing the following (extruded) mesh:

          18-11-21
          |     |
          8  3  13
          |     |
    15-6--17-10-20
    |     |     |
    4  1  7  2  12
    |     |     |
    14-5--16-9--19

    Note that it is important that the numbering is done *up* the columns.
    """
    npoints = 21
    edges = {
        1: [4, 5, 6, 7],
        2: [7, 9, 10, 12],
        3: [8, 10, 11, 13],
        4: [14, 15],
        5: [14, 16],
        6: [15, 17],
        7: [16, 17],
        8: [17, 18],
        9: [16, 19],
        10: [17, 20],
        11: [18, 21],
        12: [19, 20],
        13: [20, 21],
    }
    return create_mesh(npoints, edges)


def mesh2():
    """A PETSc DMPlex object representing the following (extruded) mesh:

          29-18-33
          |     |
          14 5  21
          |     |
          28-17-32
          |     |
          13 4  20
          |     |
    24-10-27-16-31
    |     |     |
    7  2  12 3  19
    |     |     |
    23-9--26-15-30
    |     |
    6  1  11
    |     |
    22-8--25

    Note that it is important that the numbering is done *up* the columns.
    """
    npoints = 33
    edges = {
        1: [6, 8, 9, 11],
        2: [7, 9, 10, 12],
        3: [12, 15, 16, 19],
        4: [13, 16, 17, 20],
        5: [14, 17, 18, 21],
        6: [22, 23],
        7: [23, 24],
        8: [22, 25],
        9: [23, 26],
        10: [24, 27],
        11: [25, 26],
        12: [26, 27],
        13: [27, 28],
        14: [28, 29],
        15: [26, 30],
        16: [27, 31],
        17: [28, 32],
        18: [29, 33],
        19: [30, 31],
        20: [31, 32],
        21: [32, 33],
    }
    return create_mesh(npoints, edges)


def closure(dm, point):
    return dm.getTransitiveClosure(point)[0]


def star(dm, point):
    return dm.getTransitiveClosure(point, useCone=False)[0]


def cone(dm, point):
    return dm.getCone(point)


def support(dm, point):
    return dm.getSupport(point)


def compute_covering(dm, points, fns):
    coverings = {}
    for point in points:
        fn, *others = fns

        covering = fn(dm, point)
        if others:
            covering = compute_covering(dm, covering, others)
            covering = np.array([c for cover in covering.values() for c in cover])

        coverings[point] = covering
    return coverings


def collect_adjacent(coverings):
    starts = []
    counts = []
    for covering in coverings.values():
        if starts and all(covering == starts[-1] + counts[-1]):
            counts[-1] += 1
        else:
            starts.append(covering)
            counts.append(1)
    return starts, counts


def get_interior_facets(dm):
    edges = range(*dm.getHeightStratum(1))
    facet_covering = compute_covering(dm, edges, [support])
    return [pt for pt, cover in facet_covering.items() if len(cover) == 2]


def get_exterior_facets(dm):
    edges = range(*dm.getHeightStratum(1))
    facet_covering = compute_covering(dm, edges, [support])
    return [pt for pt, cover in facet_covering.items() if len(cover) == 1]


def test_facets(dm):
    interior_facets = get_interior_facets(dm)
    exterior_facets = get_exterior_facets(dm)

    breakpoint()
    pass


def test_partitioning(dm):
    cells = range(*dm.getHeightStratum(0))
    closures = compute_covering(dm, cells, [closure])
    starts, counts = collect_adjacent(closures)

    breakpoint()
    pass


def test_int_facet_partitioning(dm):
    interior_facets = get_interior_facets(dm)
    closures = compute_covering(dm, interior_facets, [support, closure])
    starts, counts = collect_adjacent(closures)

    breakpoint()
    pass


if __name__ == "__main__":
    mesh = mesh2()
    # test_partitioning(mesh)
    # test_facets(mesh)
    test_int_facet_partitioning(mesh)

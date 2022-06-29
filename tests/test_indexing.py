import loopy as lp
import numpy as np
import pytest

import pyop3


@pytest.fixture
def dtype():
    return np.float64


@pytest.fixture
def ncells():
    return 48


@pytest.fixture
def nedges():
    return 57


@pytest.fixture
def nverts():
    return 49


@pytest.fixture
def inc(dtype):
    kernel = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "x[i] = x[i] + 1",
        [lp.GlobalArg(name="x", shape=(), dtype=dtype)],
        name="inc"
    )
    return pyop3.Kernel("inc",
        [
            pyop3.ArgumentSpec(pyop3.INC, np.float64, 1),
        ],
        kernel)


@pytest.fixture
def mesh(ncells, nedges, nverts):
    sizes = ncells, nedges, nverts
    return pyop3.UnstructuredMesh(sizes)


@pytest.fixture
def dofs():
    return pyop3.Section([6, 7, 8])  # 6 dof per cell, 7 per edge and 8 per vert


@pytest.fixture
def dat(mesh, dofs, dtype):
    return pyop3.Dat(mesh, dofs, name="dat")


def test_stencils(dat):
    example1 = pyop3.StencilGroup([
        pyop3.Stencil([
            (0,)
        ])
    ])
    assert dat[0].stencils == example1

    example2 = pyop3.StencilGroup([
        pyop3.Stencil([
            (0, slice(None), slice(1))
        ])
    ])
    assert dat[dat.mesh.cells, :1].stencils == example2


def test_stencil_product():
    s1 = pyop3.StencilGroup([
        pyop3.Stencil([
            (1, 2, 3)
        ])
    ])
    s2 = pyop3.StencilGroup([
        pyop3.Stencil([
            (4, 5)
        ])
    ])

    assert s1 * s2 == (((1, 4, 2, 5, 3),),)

def test_range(inc, dat):
    pyop3.do_loop(
        p := pyop3.index(range(10)),
        inc(dat[0, p])
    )
    assert all(dat.data[:10*6] == 1) and all(dat.data[10*6:] == 0)


def test_range_with_plex_op(inc, dat):
    pyop3.do_loop(
        p := pyop3.index(range(10)),
        inc(dat[0, pyop3.cone(p)])
    )

    ncells, nedges, _ = dat.mesh.strata_sizes
    assert sum(dat.data[:ncells]) == 0
    assert sum(dat.data[ncells:ncells+nedges]) == 30
    assert sum(dat.data[ncells+nedges:]) == 0


def test_vdat_closure(inc, vdat):
    pyop3.do_loop(
        p := pyop3.index(vdat.mesh.cells),
        inc(dat[pyop3.closure(p)])
    )
    assert all(dat.data >= 1)


def test_indexed_vdat(inc, vdat):
    pyop3.do_loop(
        p := pyop3.index(vdat.mesh.cells),
        inc(dat[pyop3.closure(p), 1])
    )
    # assert all(dat.data >= 1)


def test_vdat_with_inner_slice(inc, vdat):
    pyop3.do_loop(
        p := pyop3.index(vdat.mesh.cells),
        inc(dat[pyop3.closure(p), 1:])
    )
    # assert all(dat.data >= 1)


def test_inc_mixed_dat(inc, mdat):
    pyop3.do_loop(
        p := pyop3.index(mdat.mesh.cells),
        inc(mdat[p])
    )


def test_inc_mixed_dat_slice(inc, mdat):
    pyop3.do_loop(
        p := pyop3.index(mdat[0].mesh.cells),
        inc(mdat[:, p])
    )


def test_inc_mixed_dat_part(inc, mdat):
    pyop3.do_loop(
        p := pyop3.index(mdat[1].mesh.cells),
        inc(mdat[1, p])
    )


def test_closure(inc, dat):
    pyop3.do_loop(
        p := pyop3.index(dat.mesh.cells),
        inc(dat[pyop3.closure(p)])
    )
    assert all(dat.data >= 1)


def test_cone(inc, dat):
    pyop3.do_loop(
        p := pyop3.index(dat.mesh.cells),
        inc(dat[pyop3.cone(p)])
    )
    assert all(dat[1].data == 1)


def test_closure_slice_cone_equivalence(inc, dat):
    d1 = dat.copy()
    cone = pyop3.loop(
        p := pyop3.index(dat.mesh.cells),
        inc(dat[pyop3.cone(p)])
    )
    cone(d1)

    d2 = dat.copy()
    closure = pyop3.loop(
        p := pyop3.index(dat.mesh.cells),
        inc(dat[pyop3.closure(p)[1:4]])
    )
    closure(d2)

    assert d1 == d2


def test_mask(inc, dat, mask):
    pyop3.do_loop(
        p := pyop3.index(dat.mesh.cells),
        inc(dat[mask][p])
    )
    # assert dat...

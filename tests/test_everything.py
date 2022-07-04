import ctypes
import loopy as lp
import dataclasses
import numpy as np
import pyop2.compilation

import pyop3
import pyop3.codegen
from pyop3.tensors import *
from pyop3.utils import Tree

from petsc4py import PETSc
PETSc.Sys.popErrorHandler()


pyop2.compilation.set_default_compiler("gcc")


def make_offset_map(dim, dtree, points=None, imap=None):
    if imap is None:
        imap = {}

    if points is None:
        if isinstance(dim.sizes[0], Tensor):
            points = list(range(read_tensor(dim.sizes[0], imap)))
        else:
            points = list(range(sum(dim.sizes)))


    ptr = 0
    offsets = {}
    # breakpoint()
    for point in points:
        if dim.permutation:
            point = dim.permutation[point]

        offsets[point] = ptr

        subdim = get_subdim(dim, dtree, point)
        if subdim:
            ptr += make_offset_map(subdim, dtree, imap=imap|{dim: point})[1]
        else:
            ptr += 1

    offsets = np.array([offsets[i] for i in range(len(offsets))], dtype=np.int32)
    return offsets, ptr


def get_subdim(dim, dtree, point):
    subdims = dtree.get_children(dim)

    if not subdims:
        return None

    bounds = list(np.cumsum(dim.sizes))
    for i, (start, stop) in enumerate(zip([0]+bounds, bounds)):
        if start <= point < stop:
            stratum = i
            break
    return subdims[stratum]


def read_tensor(tensor, imap):
    # breakpoint()
    # assume a flat tensor for now
    assert not tensor.dim.get_children(tensor.dim.root)

    ptr = imap[tensor.dim.root] - tensor.dim.root.offset
    return tensor.data[ptr]


def test_single_loop():
    dims = Tree(Dim(10))
    offsets = make_offset_map(dims.root, dims)
    # assert False


def test_double_loop():
    dims = Tree.from_nest([Dim(10), [Dim(3)]])
    offsets = make_offset_map(dims.root, dims)
    # assert False


def test_double_mixed_loop():
    dims = Tree.from_nest([
        Dim((10, 6)),
        [Dim(2), Dim(3)]
    ])
    offsets = make_offset_map(dims.root, dims)[0]
    assert all(offsets == np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35]))

def test_permuted_loop():
    perm = (1, 4, 0, 3, 2)  # i.e. first elem 1, then elem 4, then elem 0...
    # start = [a1, a2, a3, a4, a5]
    # resulting data layout: [a2, a5, a1, a4, a3]
    # so the offsets must be [5, 0, 10, 7, 2]
    dims = Tree.from_nest([
        Dim((3, 2), permutation=perm),
        [Dim(2), Dim(3)]
    ])
    offsets, size = make_offset_map(dims.root, dims)
    ans = np.array([5, 0, 10, 7, 2])
    assert all(offsets == ans)


def test_ragged_loop():
    root = Dim(5)
    steps = np.array([3, 2, 1, 3, 2])
    nnz = Tensor(Tree(root), data=steps, dtype=np.int32)
    dims = Tree.from_nest([
        root,
        [Dim(nnz)]
    ])
    offsets = make_offset_map(dims.root, dims)[0]
    ans = [0, 3, 5, 6, 9]
    assert all(offsets == ans)


def test_read_single_dim():
    root = Dim(10)
    dims = Tree(root)
    dat1 = Tensor(dims, name="dat1", data=np.arange(10, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 0),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")
    fn.argtypes = (ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp)

    map1 = make_offset_map(root, dims)[0]
    map2 = make_offset_map(root, dims)[0]

    fn(map1.ctypes.data, dat1.data.ctypes.data, dat2.data.ctypes.data,
            map2.ctypes.data)

    assert all(dat2.data == dat1.data + 1)
    print("read_single_dim PASSED")


def test_compute_double_loop():
    root = Dim(10)
    subdim = Dim(3)
    dims = Tree.from_nest([root, [subdim]])
    dat1 = Tensor(dims, name="dat1", data=np.arange(30, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(30, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 0),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (3,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")
    # sig: ???

    """
      for (int32_t i0 = 0; i0 <= 9; ++i0)
      {
        for (int32_t i2 = 0; i2 <= 2; ++i2)
          t1[map3[i2]] = 0.0;
        for (int32_t i1 = 0; i1 <= 2; ++i1)
          t0[map1[i1]] = dat1[map0[i0] + map2[i1]];
        mylocalkernel(&(t0[0]), &(t1[0]));
        for (int32_t i5 = 0; i5 <= 2; ++i5)
          dat2[map4[i0] + map6[i5]] = t1[map5[i5]];
      }
    """

    map0 = make_offset_map(root, dims)[0]
    map1 = make_offset_map(subdim, dims)[0]
    map2 = map1.copy()
    map3 = map1.copy()
    map4 = map0.copy()
    map5 = map1.copy()
    map6 = map1.copy()

    args = [map0, map1, map2, dat1.data, map3, dat2.data, map4, map5, map6]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)
    print("compute_double_loop PASSED")


def test_compute_double_loop_mixed():
    root = Dim((10, 12))
    subdims = [Dim(3), Dim(2)]
    dims = Tree.from_nest([root, [*subdims]])
    dat1 = Tensor(dims, name="dat1", data=np.arange(50, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(50, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 1),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 10; i0 <= 21; ++i0)
  {
    for (int32_t i2 = 0; i2 <= 1; ++i2)
      t1[map3[i2]] = 0.0;
    for (int32_t i1 = 0; i1 <= 1; ++i1)
      t0[map1[i1]] = dat1[map0[i0] + map2[i1]];
    mylocalkernel(&(t0[0]), &(t1[0]));
    for (int32_t i5 = 0; i5 <= 1; ++i5)
      dat2[map4[i0] + map6[i5]] = t1[map5[i5]];
  }
    """

    map0 = make_offset_map(root, dims)[0]
    map1 = make_offset_map(subdims[1], dims)[0]
    map2 = map1.copy()
    map3 = map1.copy()
    map4 = map0.copy()
    map5 = map1.copy()
    map6 = map1.copy()

    args = [map0, map1, map2, dat1.data, map3, dat2.data, map4, map5, map6]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:30] == 0)
    assert all(dat2.data[30:] == dat1.data[30:] + 1)
    print("compute_double_loop_mixed PASSED")


def test_compute_double_loop_scalar():
    """As in the temporary lives within both of the loops"""
    root = Dim((6, 4))
    subdims = [Dim(3), Dim(2)]
    dims = Tree.from_nest([root, subdims])
    dat1 = Tensor(dims, name="dat1", data=np.arange(18+8, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(18+8, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 1), Slice(subdims[1], 0))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i1 = 0; i1 <= 1; ++i1)
    for (int32_t i0 = 6; i0 <= 9; ++i0)
    {
      t1[0] = 0.0;
      t0[0] = dat1[map0[i0] + map1[i1]];
      mylocalkernel(&(t0[0]), &(t1[0]));
      dat2[map2[i0] + map3[i1]] = t1[0];
    }
    """

    map0 = make_offset_map(root, dims)[0]
    map1 = make_offset_map(subdims[1], dims)[0]
    map2 = map0.copy()
    map3 = map1.copy()

    args = [map0, map1, dat1.data, dat2.data, map2, map3]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:18] == 0)
    assert all(dat2.data[18:] == dat1.data[18:] + 1)
    print("compute_double_loop_scalar PASSED")



def test_compute_double_loop_permuted():
    root = Dim(6, permutation=(3, 2, 5, 0, 4, 1))
    subdim = Dim(3)
    dims = Tree.from_nest([root, [subdim]])
    dat1 = Tensor(dims, name="dat1", data=np.arange(18, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(18, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 0),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (3,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 0; i0 <= 5; ++i0)
  {
    for (int32_t i2 = 0; i2 <= 2; ++i2)
      t1[map3[i2]] = 0.0;
    for (int32_t i1 = 0; i1 <= 2; ++i1)
      t0[map1[i1]] = dat1[map0[i0] + map2[i1]];
    mylocalkernel(&(t0[0]), &(t1[0]));
    for (int32_t i5 = 0; i5 <= 2; ++i5)
      dat2[map4[i0] + map6[i5]] = t1[map5[i5]];
  }
    """

    map0 = make_offset_map(root, dims)[0]
    map1 = make_offset_map(subdim, dims)[0]
    map2 = map1.copy()
    map3 = map1.copy()
    map4 = map0.copy()
    map5 = map1.copy()
    map6 = map1.copy()

    assert any(map0 != np.arange(len(map0)))

    args = [map0, map1, map2, dat1.data, map3, dat2.data, map4, map5, map6]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_permuted_mixed():
    root = Dim((4, 3), permutation=(3, 6, 2, 5, 0, 4, 1))
    subdims = [Dim(1), Dim(2)]
    dims = Tree.from_nest([root, subdims])
    dat1 = Tensor(dims, name="dat1", data=np.arange(10, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 1),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 4; i0 <= 6; ++i0)
  {
    for (int32_t i2 = 0; i2 <= 1; ++i2)
      t1[map3[i2]] = 0.0;
    for (int32_t i1 = 0; i1 <= 1; ++i1)
      t0[map1[i1]] = dat1[map0[i0] + map2[i1]];
    mylocalkernel(&(t0[0]), &(t1[0]));
    for (int32_t i5 = 0; i5 <= 1; ++i5)
      dat2[map4[i0] + map6[i5]] = t1[map5[i5]];
  }
    """

    map0 = make_offset_map(root, dims)[0]
    map1 = make_offset_map(subdims[1], dims)[0]
    map2 = map1.copy()
    map3 = map1.copy()
    map4 = map0.copy()
    map5 = map1.copy()
    map6 = map1.copy()

    assert any(map0 != np.arange(len(map0)))

    args = [map0, map1, map2, dat1.data, map3, dat2.data, map4, map5, map6]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == [0., 2., 3., 0., 5., 6., 0., 8., 9., 0.])
    print("compute_double_loop_permuted_mixed PASSED")


def test_compute_double_loop_ragged():
    root = Dim(5)
    steps = np.array([3, 2, 1, 3, 2], dtype=np.int32)
    nnz = Tensor(Tree(root), data=steps, name="nnz", dtype=np.int32)
    subdim = Dim(nnz)
    dims = Tree.from_nest([
        root,
        [subdim]
    ])
    dat1 = Tensor(dims, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 0), Slice(subdim, 0))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 0; i0 <= 4; ++i0)
  {
    p0[0] = nnz[map0[i0]];
    for (int32_t i1 = 0; i1 <= -1 + p0; ++i1)
    {
      t1[0] = 0.0;
      t0[0] = dat1[map1[i0] + map2[i1]];
      mylocalkernel(&(t0[0]), &(t1[0]));
      dat2[map3[i0] + map4[i1]] = t1[0];
    }
  }
    """

    # breakpoint()
    map0 = make_offset_map(nnz.dim.root, nnz.dim)[0]
    map1 = make_offset_map(dims.root, dims)[0]
    # map2 = make_offset_map(subdim, dims)[0]
    map2 = np.arange(11, dtype=np.int32)
    map3 = map1.copy()
    map4 = map2.copy()

    args = [map0, nnz.data, map1, map2, dat1.data, dat2.data, map3, map4]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)
    print("compute_double_loop_ragged PASSED")


def test_compute_double_loop_ragged_mixed():
    # I don't think this is possible - the nnz array needs to be as big as the outer index.
    root = Dim((4, 5, 4))
    nnz_data = np.array([3, 2, 0, 0, 1], dtype=np.int32)
    nnz = Tensor(Tree(root.copy(offset=4)), data=nnz_data, name="nnz", dtype=np.int32)
    subdims = [Dim(1), Dim(nnz), Dim(2)]
    dims = Tree.from_nest([root, subdims])

    dat1 = Tensor(dims, name="dat1", data=np.arange(4+6+8, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(4+6+8, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 1), Slice(subdims[1], 0))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 4; i0 <= 8; ++i0)
  {
    p0 = nnz[map0[i0] + -4];
    for (int32_t i1 = 0; i1 <= -1 + p0; ++i1)
    {
      t1[0] = 0.0;
      t0[0] = dat1[map1[i0] + map2[i1]];
      mylocalkernel(&(t0[0]), &(t1[0]));
      dat2[map3[i0] + map4[i1]] = t1[0];
    }
  }
    """

    # breakpoint()
    map0 = make_offset_map(nnz.dim.root, nnz.dim)[0]
    map1 = make_offset_map(dims.root, dims)[0]
    # map2 = make_offset_map(subdims[0], dims)[0]
    # FIXME
    map2 = np.arange(3, dtype=np.int32)
    map3 = map1.copy()
    map4 = map2.copy()

    args = [map0, nnz.data, map1, map2, dat1.data, dat2.data, map3, map4]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # root = Dim((4, 5, 4))
    # nnz_data = np.array([3, 2, 0, 0, 1], dtype=np.int32)
    # nnz = Tensor(Tree(root.copy(offset=4)), data=nnz_data, name="nnz", dtype=np.int32)
    # subdims = [Dim(1), Dim(nnz), Dim(2)]
    assert all(dat2.data[:4] == 0)
    assert all(dat2.data[4:10] == dat1.data[4:10] + 1)
    assert all(dat2.data[10:] == 0)
    print("compute_double_loop_ragged_mixed PASSED")


def test_compute_ragged_permuted():
    root = Dim(6, permutation=(3, 2, 5, 0, 4, 1))
    # the nnz array doesn't need to be permuted
    nnz_ = np.array([3, 2, 0, 1, 3, 2], dtype=np.int32)
    nnz = Tensor(Tree(root.copy(permutation=None)), data=nnz_, name="nnz", dtype=np.int32)
    subdim = Dim(nnz)
    dims = Tree.from_nest([root, [subdim]])

    dat1 = Tensor(dims, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([(Slice(root, 0), Slice(subdim, 0))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 0; i0 <= 5; ++i0)
  {
    p0 = nnz[map0[i0]];
    for (int32_t i1 = 0; i1 <= -1 + p0; ++i1)
    {
      t1[0] = 0.0;
      t0[0] = dat1[map1[i0] + map2[i1]];
      mylocalkernel(&(t0[0]), &(t1[0]));
      dat2[map3[i0] + map4[i1]] = t1[0];
    }
  }
    """

    map0 = make_offset_map(nnz.dim.root, nnz.dim)[0]
    map1 = make_offset_map(dims.root, dims)[0]
    # map2 = make_offset_map(subdim, dims)[0]
    map2 = np.arange(11, dtype=np.int32)
    map3 = map1.copy()
    map4 = map2.copy()

    args = [map0, nnz.data, map1, map2, dat1.data, dat2.data, map3, map4]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # root = Dim(6, permutation=(3, 2, 5, 0, 4, 1))
    # nnz_ = np.array([3, 2, 0, 1, 3, 2], dtype=np.int32)
    # FIXME
    assert all(map0 == np.arange(6))
    assert all(map1 == np.array([3, 9, 1, 0, 6, 1]))

    assert all(dat2.data == dat1.data + 1)
    print("compute_ragged_permuted PASSED")


def test_subset():
    root = Dim(6)
    dims = Tree(root)

    dat1 = Tensor(dims, name="dat1", data=np.arange(6, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(6, dtype=np.float64), dtype=np.float64)

    subset_dim = Dim(4)
    subset_tensor = Tensor(Tree.from_nest([subset_dim, [Dim(1)]]),
            data=np.array([2, 3, 5, 0], dtype=np.int32), dtype=np.int32, prefix="subset")
    subset = NonAffineMap(root, 0, subset_tensor)

    iterset = StencilGroup([Stencil([(subset,)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
    for (int32_t i1 = 0; i1 <= 3; ++i1)
    {
      t1[0] = 0.0;
      i0 = subset0[map0[i1] + map1[0]];
      t0[0] = dat1[map2[i0]];
      mylocalkernel(&(t0[0]), &(t1[0]));
      dat2[map3[i0]] = t1[0];
    }
    """

    map0 = make_offset_map(subset_tensor.dim.root, subset.tensor.dim)[0]
    map1 = np.array([0], dtype=np.int32)
    map2 = make_offset_map(root, dims)[0]
    map3 = map2.copy()

    args = [map0, map1, subset_tensor.data, map2, dat1.data, dat2.data, map3]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[[2, 3, 5, 0]] == dat1.data[[2, 3, 5, 0]] + 1)
    assert all(dat2.data[[1, 4]] == 0)
    print("test_subset PASSED")


def test_map():
    root = Dim(5)
    dims = Tree(root)

    dat1 = Tensor(dims, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map_tensor = Tensor(Tree.from_nest([root, [Dim(2)]]),
            data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32), dtype=np.int32, prefix="imap")
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.ExecutableCTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    slice = Slice(root, 0)
    i1 = pyop3.index(StencilGroup([Stencil([(slice,)])]))
    map = NonAffineMap(root, 0, map_tensor[i1])
    i2 = StencilGroup([Stencil([(map,)])])
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = pyop2.compilation._compiler().get_so(jitmodule, "c")
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 0; i0 <= 4; ++i0)
  {
    t1[0] = 0.0;
    j0 = i0;
    for (int32_t i1 = 0; i1 <= 1; ++i1)
    {
      j2 = i1;
      j0 = i0;
      j1 = imap0[map0[j0] + map1[j2]];
      t0[i1] = dat1[map2[j1]];
    }
    mylocalkernel(&(t0[0]), &(t1[0]));
    j0 = i0;
    dat2[map3[j0]] = t1[0];
  }
    """

    map0 = make_offset_map(map_tensor.dim.root, map_tensor.dim)[0]
    map1 = make_offset_map(map_tensor.dim.get_child(map_tensor.dim.root), map_tensor.dim)[0]
    map2 = make_offset_map(root, dims)[0]
    map3 = map2.copy()

    args = [map0, map1, map_tensor.data, map2, dat1.data, dat2.data, map3]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    assert all(dat2.data == np.array([1+2, 0+2, 0+1, 3+4, 2+1], dtype=np.int32))
    print("test_map PASSED")


@dataclasses.dataclass
class JITModule:
    code_to_compile: str
    cache_key: str



if __name__ == "__main__":
    # test_subset()
    test_map()
    # test_single_loop()
    # test_double_loop()
    # test_double_mixed_loop()
    # test_permuted_loop()
    # test_ragged_loop()
    # test_read_single_dim()
    # test_compute_double_loop()
    # test_compute_double_loop_mixed()
    # test_compute_double_loop_permuted()
    # test_compute_double_loop_permuted_mixed()
    # test_compute_double_loop_scalar()
    # test_compute_double_loop_ragged()
    # test_compute_ragged_permuted()
    # test_compute_double_loop_ragged_mixed()

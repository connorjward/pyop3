import copy
import time
import subprocess
import os
from hashlib import md5
import ctypes
import loopy as lp
import dataclasses
import numpy as np

import pyop3
import pyop3.codegen
from pyop3.tensors import *
from pyop3.utils import Tree


def compilemythings(jitmodule):
        """Build a shared library and load it

        :arg jitmodule: The JIT Module which can generate the code to compile.
        :arg extension: extension of the source file (c, cpp).
        Returns a :class:`ctypes.CDLL` object of the resulting shared
        library."""

        compiler = "gcc"
        compiler_flags = ("-fPIC", "-Wall", "-std=gnu11","-shared","-O0", "-g")

        extension="c"

        # Determine cache key
        hsh = md5(str(jitmodule.cache_key).encode())

        basename = hsh.hexdigest()

        cachedir = "mycache"
        dirpart, basename = basename[:2], basename[2:]
        cachedir = os.path.join(cachedir, dirpart)
        pid = os.getpid()
        cname = os.path.join(cachedir, "%s_p%d.%s" % (basename, pid, extension))
        oname = os.path.join(cachedir, "%s_p%d.o" % (basename, pid))
        soname = os.path.join(cachedir, "%s.so" % basename)
        # Link into temporary file, then rename to shared library
        # atomically (avoiding races).
        tmpname = os.path.join(cachedir, "%s_p%d.so.tmp" % (basename, pid))

        try:
            # Are we in the cache?
            return ctypes.CDLL(soname)
        except OSError:
            # No need to do this on all ranks
            os.makedirs(cachedir, exist_ok=True)
            logfile = os.path.join(cachedir, "%s_p%d.log" % (basename, pid))
            errfile = os.path.join(cachedir, "%s_p%d.err" % (basename, pid))
            with open(cname, "w") as f:
                f.write(jitmodule.code_to_compile)
            # Compiler also links
            cc = (compiler,) \
                + compiler_flags \
                + ('-o', tmpname, cname)
            with open(logfile, "w") as log, open(errfile, "w") as err:
                log.write("Compilation command:\n")
                log.write(" ".join(cc))
                log.write("\n\n")
                try:
                    subprocess.check_call(cc, stderr=err, stdout=log)
                except subprocess.CalledProcessError as e:
                    raise Exception(
                        """Command "%s" return error status %d.
Unable to compile code
Compile log in %s
Compile errors in %s""" % (e.cmd, e.returncode, logfile, errfile))
            # Atomically ensure soname exists
            os.rename(tmpname, soname)
            # Load resulting library
            return ctypes.CDLL(soname)


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

    iterset = StencilGroup([Stencil([((0, Slice()),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
    fn = getattr(dll, "mykernel")
    # import pdb; pdb.set_trace()

    sec0 = make_offset_map(root, dims)[0]
    sec2 = make_offset_map(root, dims)[0]
    # breakpoint()

    args = [sec0, dat1.data, sec0, dat2.data, sec2]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)
    print("read_single_dim PASSED", flush=True)


def test_compute_double_loop():
    print("compute_double_loop START", flush=True)
    root = Dim(10)
    subdim = Dim(3)
    dims = Tree.from_nest([root, [subdim]])
    dat1 = Tensor(dims, name="dat1", data=np.arange(30, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(30, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([((0, Slice()),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (3,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
    fn = getattr(dll, "mykernel")

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
    # import pdb; pdb.set_trace()

    sec0 = np.arange(3, dtype=np.int32)
    sec1 = make_offset_map(root, dims)[0]
    sec2 = make_offset_map(subdim, dims)[0]
    sec3 = sec0.copy()
    sec4 = sec0.copy()  # skipped
    sec5 = sec0.copy()  # skipped
    sec6 = sec1.copy()
    sec7 = sec2.copy()
    sec8 = sec0.copy()

    args = [sec0, sec1, sec2, dat1.data, sec3, sec4, sec5, dat2.data, sec6, sec7, sec8]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)
    print("compute_double_loop PASSED", flush=True)


def test_compute_double_loop_mixed():
    print("compute_double_loop_mixed START", flush=True)
    root = Dim((10, 12))
    subdims = [Dim(3), Dim(2)]
    dims = Tree.from_nest([root, subdims])
    dat1 = Tensor(dims, name="dat1", data=np.arange(54, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(54, dtype=np.float64), dtype=np.float64)

    # import pdb; pdb.set_trace()
    iterset = StencilGroup([Stencil([((1, Slice()),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    # breakpoint()

    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
    fn = getattr(dll, "mykernel")

    """
  for (int32_t i0 = 0; i0 <= 11; ++i0)
  {
    j0 = i0 + 10l;
    for (int32_t i2 = 0; i2 <= 1; ++i2)
    {
      t1[map3[i2]] = 0.0;
    }
    for (int32_t i1 = 0; i1 <= 1; ++i1)
    {
      j1 = i1;
      t0[map1[i1]] = dat1[map0[j0] + map2[j1]];
    }
    mylocalkernel(&(t0[0]), &(t1[0]));
    for (int32_t i5 = 0; i5 <= 1; ++i5)
    {
      j3 = i5;
      dat2[map4[j0] + map6[j3]] = t1[map5[i5]];
    }
  }
    """

    # import pdb; pdb.set_trace()
    sec0 = np.arange(2, dtype=np.int32)
    sec1 = make_offset_map(root, dims)[0]
    sec2 = make_offset_map(subdims[1], dims)[0]
    sec3 = sec0.copy()
    sec4 = np.empty(1)  # missing
    sec5 = np.empty(1)  # missing
    sec6 = sec1.copy()
    sec7 = sec2.copy()
    sec8 = sec0.copy()

    args = [sec0, sec1, sec2, dat1.data, sec3, sec4, sec5, dat2.data, sec6, sec7, sec8]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn.restype = ctypes.c_int

    myargs = [d.ctypes.data for d in args]
    fn(*myargs)

    assert all(dat2.data[:30] == 0)
    assert all(dat2.data[30:] == dat1.data[30:] + 1)
    print("compute_double_loop_mixed PASSED", flush=True)


def test_compute_double_loop_scalar():
    print("compute_double_loop_scalar START", flush=True)
    """As in the temporary lives within both of the loops"""
    root = Dim((6, 4))
    subdims = [Dim(3), Dim(2)]
    dims = Tree.from_nest([root, subdims])
    dat1 = Tensor(dims, name="dat1", data=np.arange(18+8, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(18+8, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([((1, Slice()), (0, Slice()))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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

    # import pdb; pdb.set_trace()
    sec0 = make_offset_map(root, dims)[0]
    sec1 = make_offset_map(subdims[1], dims)[0]
    sec2 = np.empty(1)  # missing
    sec3 = np.empty(1)  # missing
    sec4 = sec0.copy()
    sec5 = sec1.copy()

    args = [sec0, sec1, dat1.data, sec2, sec3, dat2.data, sec4, sec5]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:18] == 0)
    assert all(dat2.data[18:] == dat1.data[18:] + 1)
    print("compute_double_loop_scalar PASSED", flush=True)



def test_compute_double_loop_permuted():
    print("test_compute_double_loop_permuted START", flush=True)
    root = Dim(6, permutation=(3, 2, 5, 0, 4, 1))
    subdim = Dim(3)
    dims = Tree.from_nest([root, [subdim]])
    dat1 = Tensor(dims, name="dat1", data=np.arange(18, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(18, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([((0, Slice()),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (3,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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

    # import pdb; pdb.set_trace()
    sec0 = np.arange(3, dtype=np.int32)
    sec1 = make_offset_map(root, dims)[0]
    sec2 = make_offset_map(subdim, dims)[0]
    sec3 = sec0.copy()
    sec4 = np.empty(1)  # missing
    sec5 = np.empty(1)  # missing
    sec6 = sec1.copy()
    sec7 = sec2.copy()
    sec8 = sec0.copy()

    args = [sec0, sec1, sec2, dat1.data, sec3, sec4, sec5, dat2.data, sec6, sec7, sec8]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)
    print("test_compute_double_loop_permuted SUCCESS", flush=True)


def test_compute_double_loop_permuted_mixed():
    print("test_compute_double_loop_permuted_mixed START", flush=True)
    root = Dim((4, 3), permutation=(3, 6, 2, 5, 0, 4, 1))
    subdims = [Dim(1), Dim(2)]
    dims = Tree.from_nest([root, subdims])
    dat1 = Tensor(dims, name="dat1", data=np.arange(10, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([((1, Slice()),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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

    # import pdb; pdb.set_trace()
    sec0 = np.arange(3, dtype=np.int32)
    sec1 = make_offset_map(root, dims)[0]
    sec2 = make_offset_map(subdims[1], dims)[0]
    sec3 = sec0.copy()
    sec4 = np.empty(1)  # missing
    sec5 = np.empty(1)  # missing
    sec6 = sec1.copy()
    sec7 = sec2.copy()
    sec8 = sec0.copy()

    args = [sec0, sec1, sec2, dat1.data, sec3, sec4, sec5, dat2.data, sec6, sec7, sec8]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == [0., 2., 3., 0., 5., 6., 0., 8., 9., 0.])
    print("compute_double_loop_permuted_mixed PASSED", flush=True)


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

    iterset = StencilGroup([Stencil([((0, Slice()), (0, Slice()))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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

    # import pdb; pdb.set_trace()
    sec0 = make_offset_map(nnz.dim.root, nnz.dim)[0]
    sec1 = make_offset_map(dims.root, dims)[0]
    # sec2 = make_offset_map(subdim, dims)[0]
    sec2 = np.arange(3, dtype=np.int32)
    sec3 = np.empty(1, dtype=np.int32)  # missing
    sec4 = np.empty(1, dtype=np.int32)  # missing
    sec5 = sec1.copy()
    sec6 = sec2.copy()

    args = [sec0, nnz.data, sec1, sec2, dat1.data, sec3, sec4, dat2.data, sec5, sec6]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)
    print("compute_double_loop_ragged PASSED", flush=True)


def test_compute_double_loop_ragged_mixed():
    root = Dim((4, 5, 4))
    nnz_data = np.array([3, 2, 0, 0, 1], dtype=np.int32)
    nnz = Tensor(Tree(root.copy(offset=4)), data=nnz_data, name="nnz", dtype=np.int32)
    subdims = [Dim(1), Dim(nnz), Dim(2)]
    dims = Tree.from_nest([root, subdims])

    dat1 = Tensor(dims, name="dat1", data=np.arange(4+6+8, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(4+6+8, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([((1, Slice()), (0, Slice()))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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

    # import pdb; pdb.set_trace()
    # sec0 = make_offset_map(nnz.dim.root, nnz.dim)[0]
    sec0 = np.arange(10, dtype=np.int32)
    sec1 = make_offset_map(dims.root, dims)[0]
    # FIXME
    # sec2 = make_offset_map(subdims[0], dims)[0]
    sec2 = np.arange(3, dtype=np.int32)
    sec3 = np.empty(1, dtype=np.int32)  # missing
    sec4 = np.empty(1, dtype=np.int32)  # missing
    sec5 = sec1.copy()
    sec6 = sec2.copy()

    args = [sec0, nnz.data, sec1, sec2, dat1.data, sec3, sec4, dat2.data, sec5, sec6]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # root = Dim((4, 5, 4))
    # nnz_data = np.array([3, 2, 0, 0, 1], dtype=np.int32)
    # nnz = Tensor(Tree(root.copy(offset=4)), data=nnz_data, name="nnz", dtype=np.int32)
    # subdims = [Dim(1), Dim(nnz), Dim(2)]
    assert all(dat2.data[:4] == 0)
    assert all(dat2.data[4:10] == dat1.data[4:10] + 1)
    assert all(dat2.data[10:] == 0)
    print("compute_double_loop_ragged_mixed PASSED", flush=True)


def test_compute_ragged_permuted():
    root = Dim(6, permutation=(3, 2, 5, 0, 4, 1))
    # the nnz array doesn't need to be permuted
    nnz_ = np.array([3, 2, 0, 1, 3, 2], dtype=np.int32)
    nnz = Tensor(Tree(root.copy(permutation=None)), data=nnz_, name="nnz", dtype=np.int32)
    subdim = Dim(nnz)
    dims = Tree.from_nest([root, [subdim]])

    dat1 = Tensor(dims, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

    iterset = StencilGroup([Stencil([((0, Slice()), (0, Slice()))])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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

    sec0 = make_offset_map(nnz.dim.root, nnz.dim)[0]
    sec1 = make_offset_map(dims.root, dims)[0]
    # FIXME
    # map2 = make_offset_map(subdim, dims)[0]
    sec2 = np.arange(11, dtype=np.int32)
    sec3 = np.empty(1, dtype=np.int32)  # missing
    sec4 = np.empty(1, dtype=np.int32)  # missing
    sec5 = sec1.copy()
    sec6 = sec2.copy()

    args = [sec0, nnz.data, sec1, sec2, dat1.data, sec3, sec4, dat2.data, sec5, sec6]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # root = Dim(6, permutation=(3, 2, 5, 0, 4, 1))
    # nnz_ = np.array([3, 2, 0, 1, 3, 2], dtype=np.int32)
    # FIXME
    assert all(sec0 == np.arange(6))
    assert all(sec1 == np.array([3, 9, 1, 0, 6, 1]))

    assert all(dat2.data == dat1.data + 1)
    print("compute_ragged_permuted PASSED", flush=True)


def test_subset():
    root = Dim(6)
    dims = Tree(root)

    dat1 = Tensor(dims, name="dat1", data=np.arange(6, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(6, dtype=np.float64), dtype=np.float64)

    subset_dim = Dim(4)
    subset_tensor = Tensor(Tree(subset_dim),
            data=np.array([2, 3, 5, 0], dtype=np.int32), dtype=np.int32, prefix="subset")
    subset = NonAffineMap(subset_tensor)

    iterset = StencilGroup([Stencil([((0, subset),)])])
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    import time
    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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


    sec0 = make_offset_map(subset_tensor.dim.root, subset.tensor.dim)[0]
    sec1 = make_offset_map(root, dims)[0]
    sec2 = np.empty(1, dtype=np.int32)  # missing
    sec3 = sec1.copy()

    args = [sec0, subset_tensor.data, sec1, dat1.data, sec2, dat2.data, sec3]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[[2, 3, 5, 0]] == dat1.data[[2, 3, 5, 0]] + 1)
    assert all(dat2.data[[1, 4]] == 0)
    print("test_subset PASSED", flush=True)


def test_map():
    root = Dim(5)
    dims = Tree(root)

    dat1 = Tensor(dims, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map_tensor = Tensor(Tree.from_nest([root, [Dim(2)]]),
            data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32), dtype=np.int32, prefix="map")
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = pyop3.index(StencilGroup([Stencil([((0, Slice()),)])]))
    map = NonAffineMap(map_tensor[i1])
    i2 = StencilGroup([Stencil([((0, map),)])])
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
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
      t0[map2[i1]] = dat1[map3[j1]];
    }
    mylocalkernel(&(t0[0]), &(t1[0]));
    j0 = i0;
    dat2[map4[j0]] = t1[0];
  }
    """

    sec0 = np.arange(2, dtype=np.int32)
    sec1 = make_offset_map(map_tensor.dim.root, map_tensor.dim)[0]
    sec2 = make_offset_map(map_tensor.dim.get_child(map_tensor.dim.root), map_tensor.dim)[0]
    sec3 = make_offset_map(root, dims)[0]
    sec4 = np.empty(1, dtype=np.int32)
    sec5 = sec3.copy()

    import pdb; pdb.set_trace()
    args = [sec0, sec1, sec2, map_tensor.data, sec3, dat1.data, sec4, dat2.data, sec5]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    assert all(dat2.data == np.array([1+2, 0+2, 0+1, 3+4, 2+1], dtype=np.int32))
    print("test_map PASSED", flush=True)


def test_map_composition():
    root = Dim(5)
    dims = Tree(root)

    dat1 = Tensor(dims, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = Tensor(dims, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0_tensor = Tensor(Tree.from_nest([root, [Dim(2)]]),
                         data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
                         dtype=np.int32, prefix="map")
    map1_tensor = Tensor(Tree.from_nest([root, [Dim(2)]]),
                         data=np.array([3, 2, 4, 1, 0, 2, 4, 2, 1, 3], dtype=np.int32),
                         dtype=np.int32, prefix="map")

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 4 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (4,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = pyop3.index(StencilGroup([Stencil([((0, Slice()),)])]))
    map0 = NonAffineMap(map0_tensor[i1])
    i2 = StencilGroup([Stencil([((0, map0),)])])
    map1 = NonAffineMap(map1_tensor[i2])
    i3 = StencilGroup([Stencil([((0, map1),)])])
    expr = pyop3.Loop(i1, kernel(dat1[i3], dat2[i1]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    cache_key = str(time.time())
    jitmodule = JITModule(exe, cache_key)
    dll = compilemythings(jitmodule)
    fn = getattr(dll, "mykernel")

    sec0 = np.arange(0, 4, 2, dtype=np.int32)
    sec1 = np.arange(2, dtype=np.int32)

    sec2 = make_offset_map(map0_tensor.dim.root, map0_tensor.dim)[0]
    sec3 = make_offset_map(map0_tensor.dim.get_child(map0_tensor.dim.root), map0_tensor.dim)[0]

    sec4 = make_offset_map(map1_tensor.dim.root, map1_tensor.dim)[0]
    sec5 = make_offset_map(map1_tensor.dim.get_child(map1_tensor.dim.root), map1_tensor.dim)[0]

    sec6 = make_offset_map(root, dims)[0]

    sec7 = np.empty(1, dtype=np.int32)  # missing

    sec8 = sec6.copy()

    # import pdb; pdb.set_trace()
    args = [sec0, sec1, sec2, sec3, map0_tensor.data, sec4, sec5, map1_tensor.data, sec6, dat1.data, sec7, dat2.data, sec8]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    ans = [4+1+0+2, 3+2+0+2, 3+2+4+1, 4+2+1+3, 0+2+4+1]
    assert all(dat2.data == np.array(ans, dtype=np.int32))
    print("test_map_composition PASSED", flush=True)


@dataclasses.dataclass
class JITModule:
    code_to_compile: str
    cache_key: str



if __name__ == "__main__":
    # test_subset()
    # test_map()
    # test_single_loop()
    # test_double_loop()
    # test_double_mixed_loop()
    # test_permuted_loop()
    # test_ragged_loop()
    # test_read_single_dim()
    # test_compute_double_loop()
    # test_compute_double_loop_mixed()
    # import gc; gc.collect()
    # test_compute_double_loop_permuted()
    # test_compute_double_loop_permuted_mixed()
    # test_compute_double_loop_scalar()
    # test_compute_double_loop_ragged()
    # test_compute_ragged_permuted()
    # test_compute_double_loop_ragged_mixed()
    # mfe()
    test_map_composition()

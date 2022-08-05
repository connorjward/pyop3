import copy
import pytest
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


def compilemythings(code):
        """Build a shared library and load it

        :arg jitmodule: The JIT Module which can generate the code to compile.
        :arg extension: extension of the source file (c, cpp).
        Returns a :class:`ctypes.CDLL` object of the resulting shared
        library."""

        compiler = "gcc"
        compiler_flags = ("-fPIC", "-Wall", "-std=gnu11","-shared","-O0", "-g")

        extension="c"

        # to avoid lots of recompilation, just hash the source code and use as the cache key
        hsh = md5(code.encode())

        # Determine cache key
        # hsh = md5(str(jitmodule.cache_key).encode())

        basename = hsh.hexdigest()

        cachedir = "mycache"
        dirpart, basename = basename[:2], basename[2:]
        cachedir = os.path.join(cachedir, dirpart)
        pid = os.getpid()
        cname = os.path.join(cachedir, "%s_p%d.%s" % (basename, pid, extension))
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
                f.write(code)
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


@pytest.mark.skip
def test_single_loop():
    dims = MultiAxis(10)
    offsets = MultiArray._make_offset_map(dims, dims.label)
    # assert False


@pytest.mark.skip
def test_double_loop():
    dims = MultiAxis(10, subdims=(MultiAxis(3),))
    offsets = MultiArray._make_offset_map(dims, dims.label)
    # assert False


@pytest.mark.skip
def test_double_mixed_loop():
    dims = MultiAxis((10, 6), subdims=(MultiAxis(2), MultiAxis(3)))
    o1 = MultiArray._make_offset_map(dims, dims.labels[0])[0]
    o2 = MultiArray._make_offset_map(dims, dims.labels[1])[0]
    assert all(o1.data == np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]))
    assert all(o2.data == np.array([20, 23, 26, 29, 32, 35]))


@pytest.mark.skip
def test_permuted_loop():
    perm = (1, 4, 0, 3, 2)  # i.e. first elem 1, then elem 4, then elem 0...
    # start = [a1, a2, a3, a4, a5]
    # resulting data layout: [a2, a5, a1, a4, a3]
    # so the offsets must be [5, 0, 10, 7, 2]
    dims = MultiAxis((3, 2), permutation=perm, subdims=(MultiAxis(2), MultiAxis(3)))
    offsets, size = MultiArray._make_offset_map(dims, "myname")
    ans = np.array([5, 0, 10, 7, 2])
    assert all(offsets.data == ans)


@pytest.mark.skip
def test_ragged_loop():
    root = MultiAxis(5)
    steps = np.array([3, 2, 1, 3, 2])
    nnz = MultiArray.new(root, data=steps, dtype=np.int32)
    dims = root.copy(subdims=(MultiAxis(nnz),))
    offsets = MultiArray._make_offset_map(dims, "myname")[0]
    ans = [0, 3, 5, 6, 9]
    assert all(offsets.data == ans)


def test_read_single_dim():
    axes = Axis(10)
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(10, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64)

    iterset = [Slice.from_dim(axes, 0)]
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

    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop():
    dims = Axis(10, subdim=Axis(3))
    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(30, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(30, dtype=np.float64), dtype=np.float64)

    iterset = [Slice.from_dim(dims, 0)]
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

    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_mixed():
    axes = MultiAxis((Axis(10, subdim=Axis(3)), Axis(12, subdim=Axis(2))))
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(54, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(54, dtype=np.float64), dtype=np.float64)

    iterset = [Slice.from_dim(axes, 1)]
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

    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn.restype = ctypes.c_int

    myargs = [d.ctypes.data for d in args]
    fn(*myargs)

    assert all(dat2.data[:30] == 0)
    assert all(dat2.data[30:] == dat1.data[30:] + 1)


def test_compute_double_loop_scalar():
    """As in the temporary lives within both of the loops"""
    axes = MultiAxis((Axis(6, subdim=Axis(3)), Axis(4, subdim=Axis(2))))
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(18+8, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(18+8, dtype=np.float64), dtype=np.float64)

    iterset = [Slice.from_dim(axes, 1), Slice.from_dim(axes.parts[1].subdim, 0)]
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
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:18] == 0)
    assert all(dat2.data[18:] == dat1.data[18:] + 1)



def test_compute_double_loop_permuted():
    axes = MultiAxis(Axis(6, subdim=MultiAxis(Axis(3))), permutation=(3, 2, 5, 0, 4, 1))
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(18, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(18, dtype=np.float64), dtype=np.float64)

    iterset = [Slice.from_dim(axes, 0)]
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
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")
    args = [dat1.data, dat2.data, dat1.dim.parts[0].layout.data, dat2.dim.parts[0].layout.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_permuted_mixed():
    axes = MultiAxis((Axis(4, subdim=MultiAxis(Axis(1))), Axis(3, subdim=MultiAxis(Axis(2)))), permutation=(3, 6, 2, 5, 0, 4, 1))
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(10, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64)

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
    iterset = [Slice.from_dim(axes, 1)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")
    args = [dat1.data, dat2.data, dat1.dim.parts[1].layout.data, dat2.dim.parts[1].layout.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == [0., 2., 3., 0., 5., 6., 0., 8., 9., 0.])


def test_compute_double_loop_ragged():
    root = Axis(5)
    nnz = MultiArray.new(
        root, name="nnz", dtype=np.int32, data=np.array([3, 2, 1, 3, 2], dtype=np.int32)
    )
    dims = root.copy(subdim=Axis(nnz))

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

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

    iterset = [Slice.from_dim(dims, 0), Slice.from_dim(dims.subdim, 0)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [nnz.data, dat1.data, dat2.data, dat1.dim.parts[0].layout.data, dat2.dim.parts[0].layout.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


@pytest.mark.skip
def test_doubly_ragged():
    nnz1 = MultiArray.new(root, indicess=iterset, data=steps, name="nnz", dtype=np.int32, max_value=max(steps))

    root = MultiAxis(Axis(3))
    iterset = [Slice.from_dim(root, 0, is_loop_index=True)]

    steps = np.array([3, 2, 1, 3, 2], dtype=np.int32)
    subdim = MultiAxis(nnz)
    dims = root.copy(subdims=(subdim,))


    dat1 = MultiArray.new(dims, indicess=iterset, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, indicess=iterset, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        lp.ValueArg("n", dtype=np.int32)],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    expr = pyop3.Loop(iterset, kernel(dat1, dat2))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.parts[0].layout.data
    sec1 = dat2.dim.parts[0].layout.data

    args = [nnz.data, dat1.data, dat2.data, sec0, sec1]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_ragged_inner():
    root = MultiAxis(Axis(5))
    nnz = MultiArray.new(
        root, name="nnz", dtype=np.int32, max_value=3,
        data=np.array([3, 2, 1, 3, 2], dtype=np.int32)
    )
    subdim = MultiAxis(Axis(nnz))
    dims = root.copy(sections=root.part.copy(subdim=subdim))

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        lp.ValueArg("n", dtype=np.int32)],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    iterset = [Slice.from_dim(root, 0)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.parts[0].layout.data
    sec1 = dat2.dim.parts[0].layout.data

    args = [nnz.data, dat1.data, dat2.data, sec0, sec1]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_ragged_mixed():
    root = MultiAxis((Axis(4), Axis(5), Axis(4)))
    nnz_data = np.array([3, 2, 0, 0, 1], dtype=np.int32)
    nnz = MultiArray.new(MultiAxis(Axis(5, label=root.parts[1].label)), data=nnz_data, name="nnz", dtype=np.int32)

    axes = MultiAxis((
        root.parts[0].copy(subdim=MultiAxis(Axis(1))),
        root.parts[1].copy(subdim=MultiAxis(Axis(nnz))),
        root.parts[2].copy(subdim=MultiAxis(Axis(2))),
    ))

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(4+6+8, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(4+6+8, dtype=np.float64), dtype=np.float64)

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
    iterset = [Slice.from_dim(axes, 1), Slice.from_dim(axes.parts[1].subdim, 0)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.parts[1].layout
    sec1 = dat2.dim.parts[1].layout

    args = [nnz.data, dat1.data, dat2.data, sec0.data, sec1.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:4] == 0)
    assert all(dat2.data[4:10] == dat1.data[4:10] + 1)
    assert all(dat2.data[10:] == 0)


def test_compute_ragged_permuted():
    root = MultiAxis(Axis(6), permutation=(3, 2, 5, 0, 4, 1))
    # the nnz array doesn't need to be permuted
    nnz_ = np.array([3, 2, 0, 1, 3, 2], dtype=np.int32)
    nnz = MultiArray.new(root.copy(permutation=None), data=nnz_, name="nnz", dtype=np.int32)
    subdim = MultiAxis(Axis(nnz))
    axes = root.copy(sections=root.part.copy(subdim=subdim))

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

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
    iterset = [Slice.from_dim(axes, 0), Slice.from_dim(axes.part.subdim, 0)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.parts[0].layout
    sec1 = dat2.dim.parts[0].layout

    args = [nnz.data, dat1.data, dat2.data, sec0.data, sec1.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_subset():
    root = MultiAxis(Axis(6))
    dims = root

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(6, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(6, dtype=np.float64), dtype=np.float64)

    subset_dim = MultiAxis(Axis(4, label=root.part.label))
    subset_tensor = MultiArray.new(subset_dim,
            data=np.array([2, 3, 5, 0], dtype=np.int32), dtype=np.int32, prefix="subset")

    i1 = pyop3.index([[Slice.from_dim(subset_dim, 0)]])
    subset = NonAffineMap(subset_tensor[i1], subdim_id=0)

    iterset = [subset]
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

    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")
    args = [subset_tensor.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[[2, 3, 5, 0]] == dat1.data[[2, 3, 5, 0]] + 1)
    assert all(dat2.data[[1, 4]] == 0)


def test_map():
    root = MultiAxis(Axis(5))
    dims = root

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map_tensor = MultiArray.new(root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.part.label)))),
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

    i1 = pyop3.index([Slice.from_dim(root, 0)])
    map = NonAffineMap(map_tensor[i1], subdim_id=0)
    i2 = [[map]]
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map_tensor.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    assert all(dat2.data == np.array([1+2, 0+2, 0+1, 3+4, 2+1], dtype=np.int32))


def test_closure_ish():
    root = MultiAxis((Axis(3), Axis(4)))

    dat1 = MultiArray.new(root, name="dat1", data=np.arange(7, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(MultiAxis(Axis(3, label=root.parts[0].label)), name="dat2", data=np.zeros(3, dtype=np.float64), dtype=np.float64)

    map0 = MultiArray.new(MultiAxis(Axis(3, label=root.parts[0].label, subdim=MultiAxis(Axis(2, label=root.parts[1].label)))),
            data=np.array([1, 2, 0, 1, 3, 2], dtype=np.int32), dtype=np.int32, prefix="map")

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = pyop3.index([Slice.from_dim(root, 0)]) # loop over 'cells'
    i2 = [i1, [NonAffineMap(map0[i1], subdim_id=1)]]  # access 'cell' and 'edge' data
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [map0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 1, 3, 2] (-> [4, 5, 3, 4, 6, 5]) and [0, 1, 2]
    assert all(dat2.data == np.array([4+5+0, 3+4+1, 6+5+2], dtype=np.int32))


def test_index_function():
    """Imagine an interval mesh:

        3 0 4 1 5 2 6
        x---x---x---x
    """
    root = MultiAxis((Axis(3), Axis(4)))

    dat1 = MultiArray.new(root, name="dat1", data=np.arange(7, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(MultiAxis(Axis(3, label=root.parts[0].label)), name="dat2", data=np.zeros(3, dtype=np.float64), dtype=np.float64)

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    # an IndexFunction contains an expression and the corresponding dim labels
    x0, x1 = pym.variables("x0 x1")
    map = IndexFunction(x0 + x1, arity=2, vardims=[(x0, root.parts[0].label), (x1, root.parts[1].label)], subdim_id=1)

    i1 = pyop3.index([Slice.from_dim(root, 0)]) # loop over 'cells'
    i2 = [i1, [map]]  # access 'cell' and 'edge' data
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # [0, 1, 2] + [3+4, 4+5, 5+6]
    assert all(dat2.data == np.array([0+3+4, 1+4+5, 2+5+6], dtype=np.int32))


def test_multimap():
    root = MultiAxis(Axis(5))

    dat1 = MultiArray.new(
        root, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(
        root, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0 = MultiArray.new(
        root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.part.label)))),
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
        dtype=np.int32, prefix="map")

    map1 = MultiArray.new(
        root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.part.label)))),
        data=np.array([1, 1, 3, 0, 2, 1, 4, 3, 0, 1], dtype=np.int32),
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

    i1 = pyop3.index([Slice.from_dim(root, 0)])
    i2 = [[NonAffineMap(map0[i1], subdim_id=0)], [NonAffineMap(map1[i1], subdim_id=0)]]
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map0.data, map1.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    # and [1, 1, 3, 0, 2, 1, 4, 3, 0, 1]
    assert all(dat2.data == np.array([1+2+1+1, 0+2+3+0, 0+1+2+1, 3+4+4+3, 2+1+0+1],
                                     dtype=np.int32))


def test_multimap_with_scalar():
    root = MultiAxis(Axis(5))

    dat1 = MultiArray.new(
        root, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(
        root, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0 = MultiArray.new(
        root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.parts[0].label)))),
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
        dtype=np.int32, prefix="map")

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = pyop3.index([Slice.from_dim(root, 0)])
    i2 = [i1, [NonAffineMap(map0[i1], subdim_id=0)]]
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1] and [0, 1, 2, 3, 4]
    assert all(dat2.data == np.array([1+2+0, 0+2+1, 0+1+2, 3+4+3, 2+1+4],
                                     dtype=np.int32))

def test_map_composition():
    root = MultiAxis(Axis(5))
    dims = root

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0_tensor = MultiArray.new(root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.part.label)))),
                         data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
                         dtype=np.int32, prefix="map")
    map1_tensor = MultiArray.new(root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.part.label)))),
                         data=np.array([3, 2, 4, 1, 0, 2, 4, 2, 1, 3], dtype=np.int32),
                         dtype=np.int32, prefix="map")

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 4 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, shape=(4,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, shape=(1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = pyop3.index([[Slice.from_dim(root, 0)]])
    map0 = NonAffineMap(map0_tensor[i1], subdim_id=0)
    i2 = [[map0]]
    map1 = NonAffineMap(map1_tensor[i2], subdim_id=0)
    i3 = [[map1]]
    expr = pyop3.Loop(i1, kernel(dat1[i3], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map0_tensor.data, map1_tensor.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    ans = [4+1+0+2, 3+2+0+2, 3+2+4+1, 4+2+1+3, 0+2+4+1]
    assert all(dat2.data == np.array(ans, dtype=np.int32))


def test_mixed_arity_map():
    root = MultiAxis(Axis(3))
    dims = root

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(1, 4, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(3, dtype=np.float64), dtype=np.float64)

    nnz_ = np.array([3, 2, 1], dtype=np.int32)
    nnz = MultiArray.new(root, data=nnz_, name="nnz", dtype=np.int32, max_value=3)

    map_data = np.array([2, 1, 0, 2, 1, 2], dtype=np.int32)
    map_tensor = MultiArray.new(root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(nnz, label=root.part.label)))),
            data=map_data, dtype=np.int32, prefix="map")

    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        lp.ValueArg("n", dtype=np.int32)],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = pyop3.index([[Slice.from_dim(root, 0)]])
    map = NonAffineMap(map_tensor[i1], subdim_id=0)
    i2 = [[map]]
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [nnz.data, map_tensor.data, dat1.data, dat2.data, map_tensor.dim.parts[0].layout.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == np.array([1+2+3, 2+3, 3], dtype=np.int32))
    print("test_mixed_arity_map PASSED", flush=True)

def test_iter_map_composition():
    root = MultiAxis(Axis(5))
    dims = root

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0_tensor = MultiArray.new(root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.part.label)))),
                         data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
                         dtype=np.int32, prefix="map")
    map1_tensor = MultiArray.new(root.copy(sections=root.part.copy(subdim=MultiAxis(Axis(2, label=root.part.label)))),
                         data=np.array([3, 2, 2, 3, 0, 2, 1, 2, 1, 3], dtype=np.int32),
                         dtype=np.int32, prefix="map")

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = y[i] + x[i]",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = pyop3.index([[Slice.from_dim(root, 0)]])
    map0 = NonAffineMap(map0_tensor[i1], subdim_id=0)
    i2 = [[map0]]
    map1 = NonAffineMap(map1_tensor[i2], subdim_id=0)
    i3 = [[map1]]
    expr = pyop3.Loop(p := pyop3.index(i3), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map0_tensor.data, map1_tensor.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # import pdb; pdb.set_trace()
    # data is just written to itself (but not the final one because it's not in map1)
    ans = [0, 1, 2, 3, 0]
    assert all(dat2.data == np.array(ans, dtype=np.int32))
    print("test_iter_map_composition PASSED", flush=True)


if __name__ == "__main__":
    # test_subset()
    # test_map()
    # test_single_loop()
    # test_double_loop()
    # test_double_mixed_loop()
    # test_permuted_loop()
    # test_ragged_loop()
    test_read_single_dim()
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
    # test_map_composition()
    # test_iter_map_composition()
    # test_compute_double_loop_ragged_inner()
    # test_mixed_arity_map()

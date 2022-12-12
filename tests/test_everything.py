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
from pyop3.mesh import *
from pyop3.tensors import *


"""
COMMON ERRORS
-------------

If you see the message:

corrupted size vs. prev_size
Aborted (core dumped)

then this usually means that the arrays you are passing in are too small.

This happens usually when you copy and paste things and forget.
"""


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


def test_read_single_dim():
    axes = MultiAxis(AxisPart(10)).set_up()
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(10, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64)

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [
            lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    p = MultiIndexCollection([
        MultiIndex([
            TypedIndex(0, IndexSet(10))
        ])
    ])

    # use [p] instead of p to have a list of multi-index collections. this is
    # needed if we have dat[cone(p0), cone(p1)] for example (i.e. each cone(...)
    # produces a multi-index collection).
    expr = pyop3.Loop(p, kernel(dat1[[p]], dat2[[p]]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    # import pdb; pdb.set_trace()

    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop():
    axes = MultiAxis(AxisPart(10, id="ax1"))
    axes = axes.add_subaxis("ax1", MultiAxis(AxisPart(3))).set_up()

    dat1 = MultiArray.new(
        axes, name="dat1", data=np.arange(30, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray.new(
            axes, name="dat2", data=np.zeros(30, dtype=np.float64),
            dtype=np.float64)

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[i] = x[i] + 1",
        [
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (3,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    p = MultiIndexCollection([
        MultiIndex([
            TypedIndex(0, IndexSet(10))
        ])
    ])
    expr = pyop3.Loop(p, kernel(dat1[[p]], dat2[[p]]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_mixed():
    axes = (
        MultiAxis([
            AxisPart(10, id="ax1"),
            AxisPart(12, id="ax2"),
        ])
        .add_subaxis("ax1", MultiAxis([AxisPart(3)]))
        .add_subaxis("ax2", MultiAxis([AxisPart(2)]))
    ).set_up()

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(54, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(54, dtype=np.float64), dtype=np.float64)

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
    p = MultiIndexCollection([
        MultiIndex([
            TypedIndex(1, IndexSet(12))
        ])
    ])
    expr = pyop3.Loop(p, kernel(dat1[[p]], dat2[[p]]))

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
    axes = (
        MultiAxis([
            AxisPart(6, id="ax1"),
            AxisPart(4, id="ax2"),
        ])
        .add_subaxis("ax1", MultiAxis([AxisPart(3)]))
        .add_subaxis("ax2", MultiAxis([AxisPart(2)]))
    ).set_up()
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(18+8, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(18+8, dtype=np.float64), dtype=np.float64)

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
    p = MultiIndexCollection([
        MultiIndex([
            TypedIndex(1, IndexSet(4)),
            TypedIndex(0, IndexSet(2)),
        ])
    ])
    expr = pyop3.Loop(p, kernel(dat1[[p]], dat2[[p]]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:18] == 0)
    assert all(dat2.data[18:] == dat1.data[18:] + 1)



def test_compute_double_loop_permuted():
    axes = MultiAxis(AxisPart(6, id="sax1", numbering=np.array([3, 2, 5, 0, 4, 1])))
    axes = axes.add_subaxis("sax1", MultiAxis([AxisPart(3)]))
    axes = axes.set_up()

    dat1 = MultiArray.new(
            axes, name="dat1", data=np.arange(18, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(
            axes, name="dat2", data=np.zeros(18, dtype=np.float64), dtype=np.float64)

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
    p = MultiIndexCollection([
        MultiIndex([
            TypedIndex(0, IndexSet(6)),
        ])
    ])
    expr = pyop3.Loop(p, kernel(dat1[[p]], dat2[[p]]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    layout0 = dat1.dim.part.layout_fn.data

    args = [layout0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_permuted_twice():
    axes = (
        MultiAxis(AxisPart(3, id="ax1"), numbering=(2, 1, 0))
        .add_subaxis("ax1", MultiAxis(AxisPart(3, id="ax2"), numbering=(2, 0, 1)))
        .add_subaxis("ax2", 2)
    )

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(18, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(18, dtype=np.float64), dtype=np.float64)

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i]",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    iterset = [Slice(3), Slice(3)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.part.layout[0]
    sec1 = dat1.dim.part.subaxis.part.layout[0]
    sec2 = sec0.copy()
    sec3 = sec1.copy()

    args = [dat1.data, dat2.data, sec0.data, sec1.data, sec2.data, sec3.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data)


def test_somewhat_permuted():
    axes = (
        MultiAxis(AxisPart(2, id="ax1"))
        .add_subaxis("ax1", MultiAxis(AxisPart(3, id="ax2"), permutation=(2, 0, 1)))
        .add_subaxis("ax2", 2)
    )

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(12, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(12, dtype=np.float64), dtype=np.float64)

    iterset = [Slice(2), Slice(3)]
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i]",
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

    sec2 = dat1.dim.part.subaxis.part.layout[0]
    sec3 = dat2.dim.part.subaxis.part.layout[0]

    args = [dat1.data, dat2.data, sec2.data, sec3.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data)


def test_compute_double_loop_permuted_mixed():
    axes = (
        MultiAxis([AxisPart(4, id="ax1"), AxisPart(3, id="ax2")],
                  permutation=(3, 6, 2, 5, 0, 4, 1))
        .add_subaxis("ax1", 1)
        .add_subaxis("ax2", 2)
    )
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
    iterset = [Slice(3, npart=1)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec5 = dat1.dim.parts[1].layout[0]
    sec7 = dat2.dim.parts[1].layout[0]

    args = [dat1.data, dat2.data, sec5.data, sec7.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == [0., 2., 3., 0., 5., 6., 0., 8., 9., 0.])


def test_compute_double_loop_ragged():
    axes1 = MultiAxis(AxisPart(5, id="ax1"))
    nnz = MultiArray.new(
        axes1, name="nnz", dtype=np.int32, data=np.array([3, 2, 1, 3, 2], dtype=np.int32)
    )

    axes2 = axes1.add_subaxis("ax1", nnz)

    dat1 = MultiArray.new(axes2, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes2, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

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

    iterset = [Slice(5), Slice(nnz)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [nnz.data, dat1.data, dat2.data, dat1.dim.part.subaxis.part.layout[0].data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_doubly_ragged():
    ax1 = MultiAxis(AxisPart(3, id="ax1"))
    nnz1 = MultiArray.new(
        ax1, name="nnz1", dtype=np.int32, max_value=3,
        data = np.array([3, 0, 2], dtype=np.int32)
    )

    ax2 = ax1.add_subaxis("ax1", MultiAxis(AxisPart(nnz1, id="ax2")))
    nnz2 = MultiArray.new(
        ax2, name="nnz2", dtype=np.int32, max_value=5,
        data = np.array([1, 0, 5, 2, 3], dtype=np.int32)
    )


    ax3 = ax2.add_subaxis("ax2", nnz2)
    dat1 = MultiArray.new(
        ax3, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray.new(
        ax3, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True)],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    iterset = [Slice(3), Slice(nnz1), Slice(nnz2)]

    expr = pyop3.Loop(p := index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    nnz1c, _ = dat1.dim.part.subaxis.part.layout
    nnz2c, _ = dat1.dim.part.subaxis.part.subaxis.part.layout

    # import pdb; pdb.set_trace()

    args = [nnz1.data, nnz2.data, dat1.data, dat2.data, nnz1c.data, nnz2c.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_ragged_inside_two_standard_loops():
    ax1 = MultiAxis(AxisPart(2, id="ax1"))
    ax2 = ax1.add_subaxis("ax1", AxisPart(2, id="ax2"))

    nnz = MultiArray.new(
        ax2, name="nnz", dtype=np.int32, max_value=2,
        data=np.array([1, 2, 1, 2], dtype=np.int32)
    )

    ax3 = ax2.add_subaxis("ax2", nnz)
    dat1 = MultiArray.new(
        ax3, name="dat1", data=np.arange(6, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray.new(
        ax3, name="dat2", data=np.zeros(6, dtype=np.float64), dtype=np.float64
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True)],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    iterset = [Slice(2), Slice(2), Slice(nnz)]

    expr = pyop3.Loop(p := index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    nnzc, _ = dat1.dim.part.subaxis.part.subaxis.part.layout

    args = [nnz.data, dat1.data, dat2.data, nnzc.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    # import pdb; pdb.set_trace()

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_ragged_inner():
    axes1 = MultiAxis(AxisPart(5, id="ax1"))

    nnz = MultiArray.new(
        axes1, name="nnz", dtype=np.int32, max_value=3,
        data=np.array([3, 2, 1, 3, 2], dtype=np.int32)
    )

    axes2 = axes1.add_subaxis("ax1", nnz)

    dat1 = MultiArray.new(axes2, name="dat1", data=np.arange(11, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes2, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64)

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
    iterset = [Slice(5)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    nnzc, _ = dat1.dim.part.subaxis.part.layout

    args = [nnz.data, dat1.data, dat2.data, nnzc.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_ragged_mixed():
    nnz = MultiArray.new(
        MultiAxis(5), name="nnz", dtype=np.int32,
        data=np.array([3, 2, 0, 0, 1], dtype=np.int32)
    )

    axes = (
        MultiAxis([
            AxisPart(4, id="ax1"), AxisPart(5, id="ax2"), AxisPart(4, id="ax3")
        ])
        .add_subaxis("ax1", 1)
        .add_subaxis("ax2", nnz)
        .add_subaxis("ax3", 2)
    )

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
    iterset = [Slice(5, npart=1), Slice(nnz)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    nnzc, _ = dat1.dim.parts[1].subaxis.part.layout

    args = [nnz.data, dat1.data, dat2.data, nnzc.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:4] == 0)
    assert all(dat2.data[4:10] == dat1.data[4:10] + 1)
    assert all(dat2.data[10:] == 0)


def test_compute_ragged_permuted():
    nnz = MultiArray.new(
        MultiAxis(6), name="nnz", dtype=np.int32,
        data=np.array([3, 2, 0, 1, 3, 2], dtype=np.int32)
    )

    axes = (
        MultiAxis(AxisPart(6, id="ax1"), permutation=(3, 2, 5, 0, 4, 1))
        .add_subaxis("ax1", nnz)
    )

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
    iterset = [Slice(6), Slice(nnz)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.parts[0].layout[0]
    sec1 = dat2.dim.parts[0].layout[0]

    args = [nnz.data, dat1.data, dat2.data, sec0.data, sec1.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_permuted_ragged_permuted():
    nnz = MultiArray.new(
        MultiAxis(6), name="nnz", dtype=np.int32,
        data=np.array([3, 2, 0, 1, 3, 2], dtype=np.int32)
    )

    axes = (
        MultiAxis(AxisPart(6, id="ax1"), permutation=(3, 2, 5, 0, 4, 1))
        .add_subaxis("ax1", AxisPart(nnz, id="ax2"))
        .add_subaxis("ax2", MultiAxis(2, permutation=(1, 0)))
    )

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(22, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(22, dtype=np.float64), dtype=np.float64)

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
    iterset = [Slice(6), Slice(nnz), Slice(2)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.part.layout[0]
    sec1 = dat2.dim.part.subaxis.part.subaxis.part.layout[0]
    sec2 = sec0.copy()
    sec3 = sec1.copy()

    args = [nnz.data, dat1.data, dat2.data, sec0.data, sec1.data, sec2.data, sec3.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


@pytest.mark.skip
def test_permuted_inner_and_ragged():
    axes = (
        MultiAxis(AxisPart(2, id="ax1"))
        .add_subaxis("ax1", MultiAxis(AxisPart(2, id="ax2"), permutation=(1, 0)))
    )

    nnz = MultiArray.new(
        axes, name="nnz", dtype=np.int32,
        data=np.array([3, 2, 0, 1], dtype=np.int32)
    )

    axes = axes.add_subaxis("ax2", nnz)

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(6, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(6, dtype=np.float64), dtype=np.float64)

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
    iterset = [Slice(3), Slice(2), Slice(nnz)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.part.subaxis.part.layout[0]
    sec1 = sec0.copy()

    args = [dat1.data, dat2.data, sec0.data, sec1.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)



def test_permuted_inner():
    axes = (
        MultiAxis(AxisPart(4, id="ax1"))
        .add_subaxis("ax1", MultiAxis(3, permutation=(2, 0, 1)))
    )

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(12, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(12, dtype=np.float64), dtype=np.float64)

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
    iterset = [Slice(4), Slice(3)]
    expr = pyop3.Loop(p := pyop3.index(iterset), kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.part.subaxis.part.layout[0]
    sec1 = sec0.copy()

    args = [dat1.data, dat2.data, sec0.data, sec1.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_subset():
    dat1 = MultiArray.new(
        MultiAxis(6), name="dat1", data=np.arange(6, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray.new(MultiAxis(6), name="dat2", data=np.zeros(6, dtype=np.float64), dtype=np.float64)

    subset_tensor = MultiArray.new(
        MultiAxis(4), dtype=np.int32, prefix="subset",
        data=np.array([2, 3, 5, 0], dtype=np.int32)
    )

    i1 = pyop3.index([Slice(4)])
    subset = NonAffineMap(subset_tensor[i1], npart=0)

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
    axes = MultiAxis(AxisPart(5, id="ax1"))

    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map_array = MultiArray.new(
        axes.add_subaxis("ax1", 2),
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
        dtype=np.int32, prefix="map"
    )

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

    def to_func(inames, types_str, idxs_str, off_str):
        iname = inames[-1]
        return [
            # read the map array entry
            *pyop3.codegen.emit_offset_insns(map_array, types_str, idxs_str, off_str),
            f"{types_str}[{iname}] = 0",
            f"{idxs_str}[{iname}] = {map_array.name}[{off_str}]",
        ]

    i1 = Slice(5)  # or axes.index
    i2 = Map(
        i1,
        size=1,  # consumes one index
        arity=lambda _: 2,  # function returning arity of 2
        to=to_func,
        # N.B. arity returns an expression whereas to_func returns a set of instructions.
    )

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
    axes = MultiAxis([3, 4])
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(7, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(MultiAxis(3), name="dat2", data=np.zeros(3, dtype=np.float64), dtype=np.float64)

    map_axes = (
        MultiAxis(AxisPart(3, id="ax1"))
        .add_subaxis("ax1", 2)
    )
    map0 = MultiArray.new(
        map_axes, dtype=np.int32, prefix="map",
        data=np.array([1, 2, 0, 1, 3, 2], dtype=np.int32)
    )

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

    i1 = pyop3.index([Slice(3, npart=0)]) # loop over 'cells'
    i2 = [i1, [NonAffineMap(map0[i1], npart=1)]]  # access 'cell' and 'edge' data
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
    root = MultiAxis([3, 4])

    dat1 = MultiArray.new(root, name="dat1", data=np.arange(7, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(MultiAxis(3), name="dat2", data=np.zeros(3, dtype=np.float64), dtype=np.float64)

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
    map = IndexFunction(x0 + x1, arity=2, vars=[x0, x1], npart=1)

    i1 = pyop3.index([Slice(3, npart=0)]) # loop over 'cells'
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
    root = MultiAxis(AxisPart(5, id="ax1"))

    dat1 = MultiArray.new(
        root, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(
        root, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0 = MultiArray.new(
        root.add_subaxis("ax1", 2),
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
        dtype=np.int32, name="map0")

    map1 = MultiArray.new(
        root.add_subaxis("ax1", 2),
        data=np.array([1, 1, 3, 0, 2, 1, 4, 3, 0, 1], dtype=np.int32),
        dtype=np.int32, name="map1")

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

    i1 = pyop3.index([Slice(5)])
    i2 = [[NonAffineMap(map0[i1], npart=0)], [NonAffineMap(map1[i1], npart=0)]]
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
    root = MultiAxis(AxisPart(5, id="ax1"))

    dat1 = MultiArray.new(
        root, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(
        root, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0 = MultiArray.new(
        root.add_subaxis("ax1", 2),
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

    i1 = pyop3.index([Slice(5)])
    i2 = [i1, [NonAffineMap(map0[i1], npart=0)]]
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
    axes = MultiAxis(AxisPart(5, id="ax1"))
    dat1 = MultiArray.new(axes, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(axes, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0_tensor = MultiArray.new(axes.add_subaxis("ax1", 2),
                         data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
                         dtype=np.int32, prefix="map")
    map1_tensor = MultiArray.new(axes.add_subaxis("ax1", 2),
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

    i1 = pyop3.index([[Slice(5)]])
    map0 = NonAffineMap(map0_tensor[i1], npart=0)
    i2 = [[map0]]
    map1 = NonAffineMap(map1_tensor[i2], npart=0)
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
    root = MultiAxis(AxisPart(3, id="ax1"))
    dims = root

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(1, 4, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(3, dtype=np.float64), dtype=np.float64)

    nnz_ = np.array([3, 2, 1], dtype=np.int32)
    nnz = MultiArray.new(root, data=nnz_, name="nnz", dtype=np.int32, max_value=3)

    map_data = np.array([2, 1, 0, 2, 1, 2], dtype=np.int32)
    map_tensor = MultiArray.new(root.add_subaxis("ax1", nnz),
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

    i1 = pyop3.index([[Slice(3)]])
    map = NonAffineMap(map_tensor[i1], npart=0)
    i2 = [[map]]
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    nnzc, _ = map_tensor.dim.part.subaxis.part.layout

    args = [nnz.data, map_tensor.data, dat1.data, dat2.data, nnzc.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == np.array([1+2+3, 2+3, 3], dtype=np.int32))

def test_iter_map_composition():
    root = MultiAxis(AxisPart(5, id="ax1"))
    dims = root

    dat1 = MultiArray.new(dims, name="dat1", data=np.arange(5, dtype=np.float64), dtype=np.float64)
    dat2 = MultiArray.new(dims, name="dat2", data=np.zeros(5, dtype=np.float64), dtype=np.float64)

    map0_tensor = MultiArray.new(root.add_subaxis("ax1", 2),
                         data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
                         dtype=np.int32, prefix="map")
    map1_tensor = MultiArray.new(root.add_subaxis("ax1", 2),
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

    i1 = pyop3.index([Slice(5)])
    map0 = NonAffineMap(map0_tensor[i1], npart=0)
    i2 = [[map0]]
    map1 = NonAffineMap(map1_tensor[i2], npart=0)
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


def test_cone():
    mesh = Mesh.create_square(2, 2, 1)

    dofs = {(0,): 1, (1,): 2, (2,): 0}
    axes = mesh.axis
    for part_id, subaxis in dofs.items():
        axes = axes.add_subaxis(part_id, subaxis)

    dat1 = MultiArray.new(
        axes, name="dat1",
        data=np.ones(MultiArray._compute_full_axis_size(axes), dtype=np.float64),
        dtype=np.float64
    )
    dat2 = MultiArray.new(
        MultiAxis(mesh.ncells), name="dat2", dtype=np.float64,
        data=np.zeros(mesh.ncells, dtype=np.float64)
    )

    loopy_knl = lp.make_kernel(
        "{ [i]: 0 <= i < 6 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (6,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(loopy_knl, [pyop3.READ, pyop3.WRITE])

    expr = pyop3.Loop(
        p := pyop3.index(mesh.cells),
        [
            kernel(dat1[cone(p)], dat2[p])
        ]
    )

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    map0 = mesh.cone(p[0])

    args = [map0.tensor.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert (dat2.data == 6).all()


def test_extruded_mesh():
    # create an extruded hexahedral mesh of size (2*2)*3
    base_mesh = Mesh.create_square(2, 2, 2, quadrilateral=True)
    interval = Mesh.create_interval(3, 1)
    mesh = base_mesh * interval

    dofs = {
        (0, 0): 2,  # cells
        (0, 1): 1,  # horiz facets
        (1, 0): 0,  # vert facets
        (1, 1): 2,  # horiz edges
        (2, 0): 1,  # vert edges
        (2, 1): 3,  # vertices
    }

    axes = mesh.axis
    for part_id, subaxis in dofs.items():
        axes = axes.add_subaxis(part_id, subaxis)

    dat1 = MultiArray.new(
        axes, name="dat1",
        data=np.ones(MultiArray._compute_full_axis_size(axes), dtype=np.float64),
        dtype=np.float64
    )
    # import pdb; pdb.set_trace()

    cells = mesh.axis.parts[0].subaxis.parts[0]
    dat2 = MultiArray.new(
        MultiAxis(cells), name="dat2", dtype=np.float64,
        data=np.zeros(MultiArray._compute_full_part_size(cells), dtype=np.float64)
    )

    loopy_knl = lp.make_kernel(
        "{ [i]: 0 <= i < 6 }",
        "y[0] = y[0] + x[i]",
        [lp.GlobalArg("x", np.float64, (6,), is_input=True, is_output=False),
        lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(loopy_knl, [pyop3.READ, pyop3.WRITE])

    expr = pyop3.Loop(
        p := pyop3.index([Slice(mesh.axis.parts[0].size, npart=0, mesh=base_mesh), Slice(mesh.axis.parts[0].subaxis.parts[0].size, npart=0, mesh=interval)]),
        [
            kernel(dat1[cone(p)], dat2[p])
        ]
    )

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    map0 = base_mesh.cone(p[0])

    # TODO Note that we use an indirection map for the interval mesh even though
    # we could use a functional alternative.
    map1 = interval.cone(p[1])

    args = [map0.tensor.data, map1.tensor.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert (dat2.data == 6).all()


if __name__ == "__main__":
    test_subset()

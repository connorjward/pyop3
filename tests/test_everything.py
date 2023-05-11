import copy
import ctypes
import dataclasses
import os
import subprocess
import time
from hashlib import md5

import loopy as lp
import numpy as np
import pytest

import pyop3
import pyop3.codegen
from pyop3.distarray.multiarray import *
from pyop3.mesh import *
from pyop3.multiaxis import *

"""
COMMON ERRORS
-------------

If you see the message:

corrupted size vs. prev_size
Aborted (core dumped)

then this usually means that the arrays you are passing in are too small.

This happens usually when you copy and paste things and forget.
"""


def compilemythings(code, basename=None, recompile_existing_code=False):
    """Build a shared library and load it

    :arg jitmodule: The JIT Module which can generate the code to compile.
    :arg extension: extension of the source file (c, cpp).
    Returns a :class:`ctypes.CDLL` object of the resulting shared
    library."""

    compiler = "gcc"
    compiler_flags = ("-fPIC", "-Wall", "-std=gnu11", "-shared", "-O0", "-g")

    extension = "c"

    if not basename:
        hsh = md5(code.encode())
        basename = hsh.hexdigest()

    cachedir = ".cache"
    dirpart, basename = basename[:2], basename[2:]
    cachedir = os.path.join(cachedir, dirpart)
    cname = os.path.join(cachedir, "%s.%s" % (basename, extension))
    soname = os.path.join(cachedir, "%s.so" % basename)
    # Link into temporary file, then rename to shared library
    # atomically (avoiding races).
    tmpname = os.path.join(cachedir, "%s.so.tmp" % (basename))

    try:
        if recompile_existing_code:
            raise OSError
        # Are we in the cache?
        return ctypes.CDLL(soname)
    except OSError:
        # No need to do this on all ranks
        os.makedirs(cachedir, exist_ok=True)
        logfile = os.path.join(cachedir, "%s.log" % (basename))
        errfile = os.path.join(cachedir, "%s.err" % (basename))

        if not recompile_existing_code or os._exists(cname):
            with open(cname, "w") as f:
                f.write(code)
        # Compiler also links
        cc = (compiler,) + compiler_flags + ("-o", tmpname, cname)
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
Compile errors in %s"""
                    % (e.cmd, e.returncode, logfile, errfile)
                )
        # Atomically ensure soname exists
        os.rename(tmpname, soname)
        # Load resulting library
        return ctypes.CDLL(soname)


@pytest.fixture
def scalar_inc_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i] + y[i] + 1",
        [
            lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    return pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])


@pytest.fixture
def scalar_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="scalar_copy",
        lang_version=(2018, 2),
    )
    return pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])


@pytest.fixture
def ragged_copy_kernel():
    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, shape=None, is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, shape=None, is_input=False, is_output=True),
            lp.ValueArg("n", dtype=np.int32),
        ],
        assumptions="n <= 3",
        target=lp.CTarget(),
        name="ragged_copy",
        lang_version=(2018, 2),
    )
    return pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])


def test_read_single_dim(scalar_copy_kernel):
    axes = MultiAxis([AxisPart(10, label="l1")]).set_up()
    dat1 = MultiArray(
        axes, name="dat1", data=np.ones(10, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("l1", 10)])
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    # import pdb; pdb.set_trace()

    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_compute_double_loop():
    axes = MultiAxis([AxisPart(10, id="ax1", label="x")])
    axes.add_subaxis("ax1", [AxisPart(3, label="y")]).set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.arange(30, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(30, dtype=np.float64), dtype=np.float64
    )

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

    p = IndexTree([RangeNode("x", 10)])
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")
    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_mixed():
    axes = (
        MultiAxis(
            [
                AxisPart(10, id="p1", label="p1"),
                AxisPart(12, id="p2", label="p2"),
            ]
        )
        .add_subaxis("p1", [AxisPart(3)])
        .add_subaxis("p2", [AxisPart(2)])
    ).set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.arange(54, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(54, dtype=np.float64), dtype=np.float64
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i] + 1",
        [
            lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    p = IndexTree([RangeNode("p2", 12)])
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

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
        MultiAxis(
            [
                AxisPart(6, id="ax1", label="a"),
                AxisPart(4, id="ax2", label="b"),
            ]
        )
        .add_subaxis("ax1", [AxisPart(3, label="c")])
        .add_subaxis("ax2", [AxisPart(2, label="c")])
    ).set_up()
    dat1 = MultiArray(
        axes, name="dat1", data=np.arange(18 + 8, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(18 + 8, dtype=np.float64), dtype=np.float64
    )

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

    p = IndexTree.from_dict(
        {
            IndexTree.ROOT: ("x",),
            RangeNode("b", 4, id="x"): ("y",),
            RangeNode("c", 2, id="y"): (),
        }
    )
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data[:18] == 0)
    assert all(dat2.data[18:] == dat1.data[18:] + 1)


def test_compute_double_loop_permuted():
    ax1 = MultiAxis(
        [AxisPart(6, id="ax1", label="ax1", numbering=np.array([3, 2, 5, 0, 4, 1]))]
    )
    ax2 = ax1.add_subaxis("ax1", [AxisPart(3)]).set_up()

    dat1 = MultiArray(
        ax2, name="dat1", data=np.arange(18, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        ax2, name="dat2", data=np.zeros(18, dtype=np.float64), dtype=np.float64
    )

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
    p = IndexTree([RangeNode("ax1", 6)])
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    layout0 = dat1.dim.node("ax1").layout_fn.data

    args = [layout0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_permuted_twice():
    ax1 = MultiAxis([AxisPart(3, id="p1", label=0, numbering=[1, 0, 2])])
    ax2 = ax1.add_subaxis("p1", [AxisPart(3, id="p2", label=1, numbering=[2, 0, 1])])
    ax3 = ax2.add_subaxis("p2", [AxisPart(2)])
    ax3.set_up()

    dat1 = MultiArray(
        ax3, name="dat1", data=np.ones(18, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        ax3, name="dat2", data=np.zeros(18, dtype=np.float64), dtype=np.float64
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i] + y[i]",
        [
            lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (2,), is_input=True, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.INC])
    p = IndexTree([RangeNode(0, 3, id="x")])
    p.add_node(RangeNode(1, 3), parent="x")
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.node("p1").layout_fn.data
    sec1 = dat1.dim.node("p2").layout_fn.data

    args = [sec0.data, sec1.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data)


def test_somewhat_permuted():
    ax1 = MultiAxis([AxisPart(2, "a", id="ax1")])
    ax2 = ax1.add_subaxis("ax1", [AxisPart(3, "b", id="ax2", numbering=[2, 0, 1])])
    ax3 = ax2.add_subaxis("ax2", [AxisPart(2)]).set_up()

    dat1 = MultiArray(
        ax3, name="dat1", data=np.arange(12, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        ax3, name="dat2", data=np.zeros(12, dtype=np.float64), dtype=np.float64
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    p = IndexTree.from_dict(
        {
            IndexTree.ROOT: ("x",),
            RangeNode("a", 2, id="x"): ("y",),
            RangeNode("b", 3, id="y"): (),
        }
    )
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    sec0 = dat1.dim.node("ax2").layout_fn.data

    args = [sec0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data)


def test_compute_double_loop_permuted_mixed():
    axes = (
        MultiAxis(
            [
                AxisPart(4, id="p1", label=0, numbering=[4, 6, 2, 0]),
                AxisPart(3, id="p2", label=1, numbering=[5, 3, 1]),
            ]
        )
        .add_subaxis("p1", [AxisPart(1)])
        .add_subaxis("p2", [AxisPart(2)])
    ).set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.arange(10, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(10, dtype=np.float64), dtype=np.float64
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[i] = x[i] + 1",
        [
            lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (2,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])
    p = IndexTree([RangeNode(1, 3)])
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # import pdb; pdb.set_trace()
    sec1 = dat1.dim.children("root")[1].layout_fn.data

    args = [sec1.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == [0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0])


def test_compute_double_loop_ragged(scalar_copy_kernel):
    ax1 = MultiAxis([AxisPart(5, id="p1", label="a")])
    ax1.set_up()

    nnz = MultiArray(
        ax1, name="nnz", dtype=np.int32, data=np.array([3, 2, 1, 3, 2], dtype=np.int32)
    )

    ax2 = ax1.copy()
    ax2.add_subaxis("p1", [AxisPart(nnz, max_count=3, id="p2", label="b")])
    ax2.set_up()

    dat1 = MultiArray(
        ax2, name="dat1", data=np.ones(11, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        ax2, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("a", 5, id="p0")])
    p.add_node(RangeNode("b", nnz[p.copy()]), "p0")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [nnz.data, dat1.axes.node("p2").layout_fn.start.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_doubly_ragged():
    ax1 = MultiAxis([AxisPart(3, "a", id="p1")])
    nnz1 = MultiArray(
        ax1.set_up(),
        name="nnz1",
        dtype=np.int32,
        max_value=3,
        data=np.array([3, 1, 2], dtype=np.int32),
    )

    ax2 = ax1.copy()
    ax2.add_subaxis("p1", [AxisPart(nnz1, "b", id="p2", max_count=3)])
    nnz2 = MultiArray(
        ax2.set_up(),
        name="nnz2",
        dtype=np.int32,
        max_value=5,
        data=np.array([1, 1, 5, 4, 2, 3], dtype=np.int32),
    )

    ax3 = ax2.copy()
    ax3 = ax3.add_subaxis("p2", [AxisPart(nnz2, "c", max_count=5, id="p3")]).set_up()
    dat1 = MultiArray(
        ax3, name="dat1", data=np.arange(16, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        ax3, name="dat2", data=np.zeros(16, dtype=np.float64), dtype=np.float64
    )

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

    p = IndexTree([RangeNode("a", 3, id="p0")])
    p.add_node(RangeNode("b", nnz1[p.copy()], id="p1"), "p0")
    p.add_node(RangeNode("c", nnz2[p.copy()]), "p1")

    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz1, layout_0_0, nnz2, layout_2_0, layout_1_0, dat1, dat2)
    layout0_0 = nnz2.root.node("p2").layout_fn.start
    layout1_0 = dat1.root.node("p3").layout_fn.start
    layout2_0 = layout1_0.dim.leaves[0].layout_fn.start
    args = [
        nnz1.data,
        layout0_0.data,
        nnz2.data,
        layout2_0.data,
        layout1_0.data,
        dat1.data,
        dat2.data,
    ]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_interleaved_ragged(scalar_copy_kernel):
    """Two data layout types are possible: constant or consistent-ragged.

    The latter means that any ragged bits inside *must* obey all of the shape stuff outside of it.

    For instance it doesn't make sense to have 3 -> [1, 2, 2, 3] so neither does it
    make sense to have 2 -> [1, 2] -> [2, 3] (instead you want [[2], [3, 1]] or something).

    Therefore this test makes sure that we can have

    2 -> [1, 2] -> 2 -> [[[a, b]], [[c, d], [e, f]]]
    """
    ax1 = MultiAxis([AxisPart(3, "a", id="p1")])
    nnz1 = MultiArray(
        ax1.set_up(),
        name="nnz1",
        dtype=np.int32,
        max_value=3,
        data=np.array([1, 3, 2], dtype=np.int32),
    )
    ax2 = ax1.copy().add_subaxis("p1", [AxisPart(nnz1, "b", id="p2")])
    ax3 = ax2.add_subaxis("p2", [AxisPart(2, "c", id="p3")])
    nnz2 = MultiArray(
        ax3.set_up(),
        name="nnz2",
        dtype=np.int32,
        max_value=3,
        data=np.array(
            utils.flatten([[[1, 2]], [[2, 1], [1, 1], [1, 1]], [[2, 3], [3, 1]]]),
            dtype=np.int32,
        ),
    )
    ax4 = ax3.copy().add_subaxis("p3", [AxisPart(nnz2, "d", id="p4")])

    root = ax4.set_up()

    dat1 = MultiArray(
        root, name="dat1", data=np.ones(19, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        root, name="dat2", data=np.zeros(19, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("a", 3, id="i0")])
    p.add_node(RangeNode("b", nnz1[p.copy()], id="i1"), "i0")
    p.add_node(RangeNode("c", 2, id="i2"), "i1")
    p.add_node(RangeNode("d", nnz2[p.copy()]), "i2")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)

    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz1, layout0_0, nnz2, layout4_0, layout3_0, layout1_0, dat1, dat2)

    layout0_0 = nnz2.root.node("p2").layout_fn.start
    # yes, I'm well aware this is insane
    layout1_0 = dat1.root.node("p4").layout_fn.start
    layout3_0 = layout1_0.dim.leaf.layout_fn.start
    layout4_0 = layout3_0.root.leaf.layout_fn.start

    args = [
        nnz1.data,
        layout0_0.data,
        nnz2.data,
        layout4_0.data,
        layout3_0.data,
        layout1_0.data,
        dat1.data,
        dat2.data,
    ]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_ragged_inside_two_standard_loops(scalar_inc_kernel):
    ax1 = MultiAxis([AxisPart(2, "a", id="p1")])
    ax2 = ax1.add_subaxis("p1", [AxisPart(2, "b", id="p2")])
    nnz = MultiArray(
        ax2.set_up(),
        name="nnz",
        dtype=np.int32,
        max_value=2,
        data=np.array([1, 2, 1, 2], dtype=np.int32),
    )
    ax3 = ax2.copy().add_subaxis("p2", [AxisPart(nnz, "c", id="p3")])

    root = ax3.set_up()
    dat1 = MultiArray(
        root, name="dat1", data=np.ones(6, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        root, name="dat2", data=np.zeros(6, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("a", 2, id="a")])
    p.add_node(RangeNode("b", 2, id="b"), "a")
    p.add_node(RangeNode("c", nnz[p.copy()]), "b")

    expr = pyop3.Loop(p, scalar_inc_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout1_0, layout0_0, dat1, dat2)
    layout0_0 = root.node("p3").layout_fn.start

    # TODO: this is affine here, should it generally be?
    layout1_0 = layout0_0.dim.leaf.layout_fn.start

    args = [nnz.data, layout1_0.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == dat1.data + 1)


def test_compute_double_loop_ragged_inner(ragged_copy_kernel):
    ax1 = MultiAxis([AxisPart(5, label="a", id="p1")])
    nnz = MultiArray(
        ax1.set_up(),
        name="nnz",
        dtype=np.int32,
        max_value=3,
        data=np.array([3, 2, 1, 3, 2], dtype=np.int32),
    )
    ax2 = ax1.copy().add_subaxis("p1", [AxisPart(nnz, label="b", id="p2")])

    root = ax2.set_up()
    dat1 = MultiArray(
        root, name="dat1", data=np.ones(11, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        root, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("a", 5)])
    expr = pyop3.Loop(p, ragged_copy_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, dat1, dat2)
    layout0_0 = root.node("p2").layout_fn.start
    args = [nnz.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_compute_double_loop_ragged_mixed(scalar_copy_kernel):
    ax1 = MultiAxis([AxisPart(5, label=1, id="p1")])
    nnz = MultiArray(
        ax1.set_up(),
        name="nnz",
        dtype=np.int32,
        data=np.array([3, 2, 1, 2, 1], dtype=np.int32),
    )

    axes = (
        MultiAxis(
            [AxisPart(4, id="p1"), AxisPart(5, label=1, id="p2"), AxisPart(4, id="p3")]
        )
        .add_subaxis("p1", [AxisPart(1)])
        .add_subaxis("p2", [AxisPart(nnz, label=0, id="p4")])
        .add_subaxis("p3", [AxisPart(2)])
    ).set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.ones(4 + 9 + 8, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(4 + 9 + 8, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode(1, 5, id="i0")])
    p.add_node(RangeNode(0, nnz[p.copy()]), "i0")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, dat1, dat2)
    layout0_0 = dat1.root.node("p4").layout_fn.start

    args = [nnz.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat2.data[:4], 0)
    assert np.allclose(dat1.data[4:13], dat2.data[4:13])
    assert np.allclose(dat2.data[13:], 0)


def test_compute_ragged_permuted(scalar_copy_kernel):
    nnz = MultiArray(
        MultiAxis([AxisPart(6, "a")]).set_up(),
        name="nnz",
        dtype=np.int32,
        data=np.array([3, 2, 0, 1, 3, 2], dtype=np.int32),
    )

    axes = (
        MultiAxis(
            [AxisPart(6, id="p1", label="a", numbering=[3, 2, 5, 0, 4, 1])]
        ).add_subaxis("p1", [AxisPart(nnz, label="b")])
    ).set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.ones(11, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(11, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("a", 6, id="i0")])
    p.add_node(RangeNode("b", nnz[p.copy()]), "i0")

    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))
    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, dat1, dat2)
    layout0_0 = dat1.root.leaf.layout_fn.start
    args = [nnz.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_permuted_ragged_permuted(scalar_copy_kernel):
    nnz = MultiArray(
        MultiAxis([AxisPart(6, label="a")]).set_up(),
        name="nnz",
        dtype=np.int32,
        data=np.array([3, 2, 0, 1, 3, 2], dtype=np.int32),
    )

    axes = (
        MultiAxis([AxisPart(6, id="p1", label="a", numbering=[3, 2, 5, 0, 4, 1])])
        .add_subaxis("p1", [AxisPart(nnz, id="p2", label="b")])
        .add_subaxis("p2", [AxisPart(2, numbering=[1, 0], id="p3", label="c")])
    ).set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.ones(22, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(22, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("a", 6, id="i0")])
    p.add_node(RangeNode("b", nnz[p.copy()], id="i1"), "i0")
    p.add_node(RangeNode("c", 2), "i1")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # void mykernel(nnz, layout0_0, layout1_0, dat1, dat2)
    layout0_0 = axes.node("p2").layout_fn.start
    layout1_0 = axes.node("p3").layout_fn.data

    args = [nnz.data, layout0_0.data, layout1_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_permuted_inner_and_ragged(scalar_copy_kernel):
    axes = MultiAxisTree.from_dict(
        {
            MultiAxisNode([MultiAxisComponent(2, "x")], id="a"): None,
            MultiAxisNode([MultiAxisComponent(2, "y")]): ("a", "x"),
        }
    )
    # breakpoint()
    nnz = MultiArray(
        axes.copy().set_up(),
        name="nnz",
        dtype=np.int32,
        data=np.array([3, 2, 1, 1], dtype=np.int32),
    )

    # we currently need to do this because ragged things admit no numbering
    # probably want a .without_numbering() method or similar
    # also, we might want to store the numbering per MultiAxisNode instead of per
    # component. That would then match DMPlex.
    axes = MultiAxisTree.from_dict(
        {
            MultiAxisNode([MultiAxisComponent(2, "x")], id="id0"): None,
            MultiAxisNode([MultiAxisComponent(2, "y", numbering=[1, 0])], id="id1"): (
                "id0",
                "x",
            ),
            MultiAxisNode([MultiAxisComponent(nnz, "z")]): ("id1", "y"),
        }
    )
    axes.set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.ones(7, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(7, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("x", 2, id="i0")])
    p.add_node(RangeNode("y", 2, id="i1"), "i0")
    p.add_node(RangeNode("z", nnz[p.copy()]), "i1")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    layout0_0 = dat1.dim.leaf.components[0].layout_fn.start
    layout1_0 = layout0_0.root.leaf.components[0].layout_fn.start
    args = [nnz.data, layout1_0.data, layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_permuted_inner(scalar_copy_kernel):
    axes = (
        MultiAxis([AxisPart(4, "a", id="p1")]).add_subaxis(
            "p1", [AxisPart(3, "b", numbering=[2, 0, 1])]
        )
    ).set_up()

    dat1 = MultiArray(
        axes, name="dat1", data=np.ones(12, dtype=np.float64), dtype=np.float64
    )
    dat2 = MultiArray(
        axes, name="dat2", data=np.zeros(12, dtype=np.float64), dtype=np.float64
    )

    p = IndexTree([RangeNode("a", 4, id="i0")])
    p.add_node(RangeNode("b", 3), "i0")
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    layout0_0 = dat1.root.leaf.layout_fn.data
    args = [layout0_0.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_subset(scalar_copy_kernel):
    axes = MultiAxis([AxisPart(6, "a")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.ones(6, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(6, dtype=np.float64))

    # a subset is really a map
    subset_axes = MultiAxis([AxisPart(4, "b", id="p1")])
    subset_axes.add_node(AxisPart(1, "c"), "p1")
    subset_axes.set_up()
    subset_array = MultiArray(
        subset_axes, prefix="subset", data=np.array([2, 3, 5, 0], dtype=np.int32)
    )

    p = IndexTree([RangeNode("b", 4, id="i0")])
    p.add_node(
        TabulatedMapNode(("b",), ("a",), arity=1, data=subset_array[p.copy()]), "i0"
    )
    expr = pyop3.Loop(p, scalar_copy_kernel(dat1[p], dat2[p]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [subset_array.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat2.data[[2, 3, 5, 0]], 1)
    assert np.allclose(dat2.data[[1, 4]], 0)


def test_map():
    axes = MultiAxis([AxisPart(5, "a", id="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    map_axes = axes.copy().add_subaxis("p1", [AxisPart(2, "b")]).set_up()
    map_array = MultiArray(
        map_axes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 2 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=True, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.INC])

    p0 = IndexTree([RangeNode("a", 5, id="i0")])
    p1 = p0.copy()
    p1.add_node(TabulatedMapNode(("a",), ("a",), arity=2, data=map_array[p0]), "i0")

    expr = pyop3.Loop(p0, kernel(dat1[p1], dat2[p0]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map_array.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    assert all(
        dat2.data == np.array([1 + 2, 0 + 2, 0 + 1, 3 + 4, 2 + 1], dtype=np.int32)
    )


def test_closure_ish():
    axes1 = MultiAxis([AxisPart(3, label="p1"), AxisPart(4, label="p2")]).set_up()
    dat1 = MultiArray(axes1, name="dat1", data=np.arange(7, dtype=np.float64))
    axes2 = MultiAxis([AxisPart(3, label="p1")]).set_up()
    dat2 = MultiArray(axes2, name="dat2", data=np.zeros(3, dtype=np.float64))

    # create a map from each cell to 2 edges
    axes3 = (
        MultiAxis([AxisPart(3, id="p1", label="p1")])
        .add_subaxis("p1", [AxisPart(2)])
        .set_up()
    )
    map1 = MultiArray(
        axes3, name="map1", data=np.array([1, 2, 0, 1, 3, 2], dtype=np.int32)
    )

    # we have a loop of size 3 here because the temporary has 1 cell DoF and 2 edge DoFs
    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=True, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.INC])

    p0 = IndexTree([RangeNode("p1", 3, id="i0")])
    p1 = p0.copy()
    p1.add_node(IdentityMapNode(("p1",), ("p1",), arity=1), "i0")
    p1.add_node(TabulatedMapNode(("p1",), ("p2",), arity=2, data=map1[p0]), "i0")

    expr = pyop3.Loop(p0, kernel(dat1[p1], dat2[p0]))

    exe = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(exe)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, map1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 1, 3, 2] (-> [4, 5, 3, 4, 6, 5]) and [0, 1, 2]
    assert all(dat2.data == np.array([4 + 5 + 0, 3 + 4 + 1, 6 + 5 + 2], dtype=np.int32))


def test_multipart_inner():
    axes = MultiAxis([AxisPart(5, label="p1", id="p1")])
    axes.add_nodes([AxisPart(3, label="p2_0"), AxisPart(2, label="p2_1")], "p1")

    axes.set_up()

    dat1 = MultiArray(axes, name="dat1", data=np.ones(25, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(25, dtype=np.float64))

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 5 }",
        "y[i] = x[i]",
        [
            lp.GlobalArg("x", np.float64, (5,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (5,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    p = IndexTree([RangeNode("p1", 5)])
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, dat2.data)


def test_index_function():
    """Imagine an interval mesh:

    3 0 4 1 5 2 6
    x---x---x---x
    """
    axes1 = MultiAxis([AxisPart(3, label="p1"), AxisPart(4, label="p2")]).set_up()
    dat1 = MultiArray(axes1, name="dat1", data=np.arange(7, dtype=np.float64))
    axes2 = MultiAxis([AxisPart(3, label="p1")]).set_up()
    dat2 = MultiArray(axes2, name="dat2", data=np.zeros(3, dtype=np.float64))

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    # import pdb; pdb.set_trace()

    # basically here we have "target multi-index is self and self+1"
    mapexpr = ((j0 := pym.var("j0"), j1 := pym.var("j1")), j0 + j1)

    i1 = IndexTree([RangeNode("p1", 3, id="i0")])  # loop over "cells"
    i2 = i1.copy()
    i2.add_nodes(
        [
            IdentityMapNode(("p1",), ("p1",), arity=1),  # "cell" data
            AffineMapNode(("p1",), ("p2",), arity=2, expr=mapexpr),  # "vert" data
        ],
        "i0",
    )

    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    # [0, 1, 2] + [3+4, 4+5, 5+6]
    assert np.allclose(dat2.data, [0 + 3 + 4, 1 + 4 + 5, 2 + 5 + 6])


def test_multimap():
    axes = MultiAxis([AxisPart(5, label="p1", id="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [AxisPart(2)]).set_up()
    map0 = MultiArray(
        mapaxes,
        name="map0",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 1, 3, 0, 2, 1, 4, 3, 0, 1], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 4 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (4,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 5, id="i1")])
    i2 = i1.copy()
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map0[i1]), "i1")
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[i1]), "i1")
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map0.data, dat1.data, map1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1]
    # and [1, 1, 3, 0, 2, 1, 4, 3, 0, 1]
    assert all(
        dat2.data
        == np.array(
            [1 + 2 + 1 + 1, 0 + 2 + 3 + 0, 0 + 1 + 2 + 1, 3 + 4 + 4 + 3, 2 + 1 + 0 + 1],
            dtype=np.int32,
        )
    )


def test_multimap_with_scalar():
    axes = MultiAxis([AxisPart(5, id="p1", label="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [AxisPart(2)]).set_up()
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (3,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 5, id="i1")])
    i2 = i1.copy()
    i2.add_node(IdentityMapNode(("p1",), ("p1",), arity=1), "i1")
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[i1]), "i1")
    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data, map1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    # from [1, 2, 0, 2, 0, 1, 3, 4, 2, 1] and [0, 1, 2, 3, 4]
    assert all(
        dat2.data
        == np.array(
            [1 + 2 + 0, 0 + 2 + 1, 0 + 1 + 2, 3 + 4 + 3, 2 + 1 + 4], dtype=np.int32
        )
    )


def test_map_composition():
    axes = MultiAxis([AxisPart(5, id="p1", label="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [AxisPart(2)]).set_up()
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )
    map2 = MultiArray(
        mapaxes,
        name="map2",
        data=np.array([3, 2, 4, 1, 0, 2, 4, 2, 1, 3], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 4 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, shape=(4,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, shape=(1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 5, id="i1")])
    i2 = i1.copy()
    i2.add_node(
        TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[i1], id="i2"), "i1"
    )
    i3 = i2.copy()
    i3.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map2[i2]), "i2")

    expr = pyop3.Loop(i1, kernel(dat1[i3], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map1.data, map2.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    ans = [4 + 1 + 0 + 2, 3 + 2 + 0 + 2, 3 + 2 + 4 + 1, 4 + 2 + 1 + 3, 0 + 2 + 4 + 1]
    assert all(dat2.data == np.array(ans, dtype=np.int32))


def test_mixed_arity_map():
    axes = MultiAxis([AxisPart(3, id="p1", label="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(1, 4, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(3, dtype=np.float64))

    nnz = MultiArray(
        axes, name="nnz", data=np.array([3, 2, 1], dtype=np.int32), max_value=3
    )

    mapaxes = axes.copy().add_subaxis("p1", [AxisPart(nnz)]).set_up()
    map1 = MultiArray(
        mapaxes, name="map1", data=np.array([2, 1, 0, 2, 1, 2], dtype=np.int32)
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < n }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, shape=None, is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, shape=None, is_input=False, is_output=True),
            lp.ValueArg("n", dtype=np.int32),
        ],
        assumptions="n <= 3",
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    i1 = IndexTree([RangeNode("p1", 3, id="i1")])
    i2 = i1.copy()
    i2.add_node(TabulatedMapNode(("p1",), ("p1",), arity=nnz[i1], data=map1[i1]), "i1")

    expr = pyop3.Loop(i1, kernel(dat1[i2], dat2[i1]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    # import pdb; pdb.set_trace()

    layout0_0 = map1.axes.leaf.layout_fn.start
    args = [nnz.data, layout0_0.data, map1.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert all(dat2.data == np.array([1 + 2 + 3, 2 + 3, 3], dtype=np.int32))


def test_iter_map_composition():
    axes = MultiAxis([AxisPart(5, label="p1", id="p1")]).set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.arange(5, dtype=np.float64))
    dat2 = MultiArray(axes, name="dat2", data=np.zeros(5, dtype=np.float64))

    mapaxes = axes.copy().add_subaxis("p1", [AxisPart(2)]).set_up()
    map1 = MultiArray(
        mapaxes,
        name="map1",
        data=np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 1], dtype=np.int32),
    )
    map2 = MultiArray(
        mapaxes,
        name="map2",
        data=np.array([3, 2, 2, 3, 0, 2, 1, 2, 1, 3], dtype=np.int32),
    )

    code = lp.make_kernel(
        "{ [i]: 0 <= i < 1 }",
        "y[i] = y[i] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (1,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(code, [pyop3.READ, pyop3.WRITE])

    p = IndexTree([RangeNode("p1", 5, id="i1")])
    p.add_node(
        TabulatedMapNode(("p1",), ("p1",), arity=2, data=map1[p.copy()], id="i2"), "i1"
    )
    p.add_node(TabulatedMapNode(("p1",), ("p1",), arity=2, data=map2[p.copy()]), "i2")
    expr = pyop3.Loop(p, kernel(dat1[p], dat2[p]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [map1.data, map2.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    # data is just written to itself (but not the final one because it's not in map1)
    ans = [0, 1, 2, 3, 0]
    assert all(dat2.data == np.array(ans, dtype=np.int32))


def test_mixed_real_loop():
    axes = MultiAxis(
        [
            AxisPart(3, label="p1", id="p1"),  # regular part
            AxisPart(1, label="p2"),  # "real" part
        ]
    )
    axes.add_node(AxisPart(2), "p1")

    axes.set_up()
    dat1 = MultiArray(axes, name="dat1", data=np.zeros(7))

    lpknl = lp.make_kernel(
        "{ [i]: 0 <= i < 3 }",
        "x[i]  = x[i] + 1",
        [lp.GlobalArg("x", np.float64, (2,), is_input=True, is_output=True)],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(lpknl, [pyop3.INC])

    i1 = IndexTree([RangeNode("p1", 3, id="i1")])
    i2 = i1.copy()
    i2.add_node(IdentityMapNode(("p1",), ("p1",), arity=1), "i1")
    # it's a map from everything to zero
    i2.add_node(
        AffineMapNode(("p1",), ("p2",), arity=1, expr=(pym.variables("x y"), 0)), "i1"
    )

    expr = pyop3.Loop(i1, kernel(dat1[i2]))

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    args = [dat1.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)
    fn(*(d.ctypes.data for d in args))

    assert np.allclose(dat1.data, [1, 1, 1, 1, 1, 1, 3])


@pytest.mark.skip
def test_cone():
    mesh = Mesh.create_square(2, 2, 1)

    dofs = {(0,): 1, (1,): 2, (2,): 0}
    axes = mesh.axis
    for part_id, subaxis in dofs.items():
        axes = axes.add_subaxis(part_id, subaxis)

    dat1 = MultiArray(
        axes,
        name="dat1",
        data=np.ones(MultiArray._compute_full_axis_size(axes), dtype=np.float64),
        dtype=np.float64,
    )
    dat2 = MultiArray(
        MultiAxis(mesh.ncells),
        name="dat2",
        dtype=np.float64,
        data=np.zeros(mesh.ncells, dtype=np.float64),
    )

    loopy_knl = lp.make_kernel(
        "{ [i]: 0 <= i < 6 }",
        "y[0] = y[0] + x[i]",
        [
            lp.GlobalArg("x", np.float64, (6,), is_input=True, is_output=False),
            lp.GlobalArg("y", np.float64, (1,), is_input=False, is_output=True),
        ],
        target=lp.CTarget(),
        name="mylocalkernel",
        lang_version=(2018, 2),
    )
    kernel = pyop3.LoopyKernel(loopy_knl, [pyop3.READ, pyop3.WRITE])

    expr = pyop3.Loop(p := pyop3.index(mesh.cells), [kernel(dat1[cone(p)], dat2[p])])

    code = pyop3.codegen.compile(expr, target=pyop3.codegen.CodegenTarget.C)
    dll = compilemythings(code)
    fn = getattr(dll, "mykernel")

    map0 = mesh.cone(p[0])

    args = [map0.tensor.data, dat1.data, dat2.data]
    fn.argtypes = (ctypes.c_voidp,) * len(args)

    fn(*(d.ctypes.data for d in args))

    assert (dat2.data == 6).all()


if __name__ == "__main__":
    test_compute_double_loop_ragged()

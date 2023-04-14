from pyop3.darray import *  # noqa: F401
from pyop3.exprs import *  # noqa: F401
from pyop3.mesh import *  # noqa: F401
from pyop3.tensors import *  # noqa: F401


# TODO put these in the right place (probably relate to mesh somehow)
def make_global(*args, **kwargs):
    # not related to a mesh
    raise NotImplementedError
    return MultiArray(...)


def make_literal(*args, **kwargs):
    # a literal is a global but with no parallel semantics. it is useful for
    # passing parametrised arguments to a kernel
    raise NotImplementedError
    return MultiArray(...)


def make_dat(*args, **kwargs):
    # related to one mesh

    # TODO Excellent candidate for pattern matching in 3.10
    if vec_type == "pyop3":
        return MultiArray(...)
    else:
        return PetscVec(...)


def make_mat(*args, **kwargs):
    # We currently only support PETSc matrices
    return PetscMat(*args, **kwargs)

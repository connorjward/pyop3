import os

import numpy as np
import petsc4py
from setuptools import Extension, setup


def get_petsc_dirs():
    try:
        petsc_dir = os.environ["PETSC_DIR"]
        petsc_arch = os.environ["PETSC_ARCH"]
    except KeyError:
        raise RuntimeError("PETSC_DIR and PETSC_ARCH variables not defined")
    return (petsc_dir, f"{petsc_dir}/{petsc_arch}")


def make_sparsity_extension():
    petsc_dirs = get_petsc_dirs()
    include_dirs = ([np.get_include(), petsc4py.get_include()]
                    + [f"{dir}/include" for dir in petsc_dirs])
    extra_link_args = ([f"-L{dir}/lib" for dir in petsc_dirs]
                       + [f"-Wl,-rpath,{dir}/lib" for dir in petsc_dirs])

    return Extension(name="pyop3.sparsity", sources=["pyop3/sparsity.pyx"],
                             include_dirs=include_dirs, language="c",
                             libraries=["petsc"],
                             extra_link_args=extra_link_args)


if __name__ == "__main__":
    sparsity_ext = make_sparsity_extension()

    setup(ext_modules=[sparsity_ext])

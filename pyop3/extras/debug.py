from typing import Optional, Union

from mpi4py import MPI
from petsc4py import PETSc


def print_with_rank(*args, comm: Optional[Union[PETSc.Comm, MPI.Comm]] = None) -> None:
    comm = comm or PETSc.Sys.getDefaultComm()
    print(f"[rank {comm.rank}] : ", *args, sep="", flush=True)


def print_if_rank(
    rank: int, *args, comm: Optional[Union[PETSc.Comm, MPI.Comm]] = None
) -> None:
    comm = comm or PETSc.Sys.getDefaultComm()
    if rank == comm.rank:
        print(*args, flush=True)

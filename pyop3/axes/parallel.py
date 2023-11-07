from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import pmap

from pyop3.dtypes import IntType, get_mpi_dtype
from pyop3.extras.debug import print_with_rank
from pyop3.utils import just_one, strict_int


def mysize(axes, axis, component):
    if subaxis := axes.child(axis, component):
        return _axis_size(axes, subaxis)
    else:
        return 1


def partition_ghost_points(axis, sf):
    npoints = sum(strict_int(c.count) for c in axis.components)
    nroots, ilocal, iremote = sf.getGraph()

    is_owned = np.full(npoints, True, dtype=bool)
    is_owned[ilocal] = False

    numbering = np.empty(npoints, dtype=IntType)
    owned_ptr = 0
    ghost_ptr = npoints - len(ilocal)
    for pt in axis.numbering or range(npoints):
        if is_owned[pt]:
            numbering[owned_ptr] = pt
            owned_ptr += 1
        else:
            numbering[ghost_ptr] = pt
            ghost_ptr += 1

    assert owned_ptr == npoints - len(ilocal)
    assert ghost_ptr == npoints
    return numbering


# stolen from stackoverflow
# https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
def invert(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p)  # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def renumber_sf(sf, numbering):
    """Create a new point SF."""
    # I think that this might be able to be much simpler, since we guarantee storing
    # ghost entities at the end

    to_numbering = np.zeros_like(numbering)
    inv = invert(numbering)

    cdim = 1
    dtype, _ = get_mpi_dtype(np.dtype(IntType), cdim)
    # bcast_args = dtype, numbering, to_numbering, MPI.REPLACE
    bcast_args = dtype, inv, to_numbering, MPI.REPLACE
    sf.bcastBegin(*bcast_args)
    sf.bcastEnd(*bcast_args)

    # construct a new SF with these offsets
    nroots, ilocal, iremote = sf.getGraph()

    local_offsets = []
    remote_offsets = []

    for i, (rank, _) in zip(ilocal, iremote):
        local_offsets.append(inv[i])
        remote_offsets.append((rank, to_numbering[i]))

    local_offsets = np.array(local_offsets, dtype=IntType)
    remote_offsets = np.array(remote_offsets, dtype=IntType)

    new_sf = PETSc.SF().create(sf.comm)
    new_sf.setGraph(nroots, local_offsets, remote_offsets)
    return new_sf


def collect_sf_graphs(axes, axis=None, path=pmap(), indices=pmap()):
    # NOTE: This function does not check for nested SFs (which should error)
    from pyop3.axes.tree import _as_int, _axis_component_size

    axis = axis or axes.root

    if axis.sf is not None:
        return (grow_dof_sf(axes, axis, path, indices),)
    else:
        graphs = []
        for component in axis.components:
            subaxis = axes.child(axis, component)
            if subaxis is not None:
                for pt in range(_as_int(component.count, path, indices)):
                    graphs.extend(
                        collect_sf_graphs(
                            axes,
                            subaxis,
                            path | {axis.label: component.label},
                            indices | {axis.label: pt},
                        )
                    )
        return tuple(graphs)


# perhaps I can defer renumbering the SF to here?
def grow_dof_sf(axes: FrozenAxisTree, axis, path, indices):
    point_sf = axis.sf
    nroots, ilocal, iremote = point_sf.getGraph()

    component_counts = tuple(c.count for c in axis.components)
    *component_offsets, npoints = [0] + list(np.cumsum(component_counts))

    # should be a property really, get the root indices so we can compute the offset for them
    buffer = np.full(npoints, False, dtype=bool)
    buffer[-len(ilocal) :] = True
    dtype, _ = get_mpi_dtype(buffer.dtype)
    bcast_args = dtype, buffer, buffer, MPI.REPLACE
    point_sf.reduceBegin(*bcast_args)
    point_sf.reduceEnd(*bcast_args)
    buffer[-len(ilocal) :] = False

    iroot = just_one(np.nonzero(buffer))

    print_with_rank("iroots: ", iroot)

    # effectively build the section
    # TODO this is overkill since we only need to broadcast the roots
    offsets = np.full(npoints, -1, IntType)
    ndofs = np.copy(offsets)
    for pt in iroot:
        component, component_num = axis.axis_number_to_component(pt)
        offset = axes.offset(
            path | {axis.label: component.label}, indices | {axis.label: component_num}
        )
        offsets[pt] = offset

    for pt in ilocal:
        component, component_num = axis.axis_number_to_component(pt)
        ndofs[pt] = mysize(axes, axis, component)
        offset = axes.offset(
            path | {axis.label: component.label}, indices | {axis.label: component_num}
        )
        offsets[pt] = offset
    # for i, component in enumerate(axis.components):
    #     for j in range(component_counts[i]):
    #         offset = axes.offset(
    #             path | {axis.label: component.label}, indices | {axis.label: j}
    #         )
    #         ndof = mysize(axes, axis, component)
    #
    #         offsets[component_offsets[i] + j] = offset
    #         ndofs[component_offsets[i] + j] = ndof
    print_with_rank("offsets: ", offsets)
    print_with_rank("ndofs: ", ndofs)
    print_with_rank("ilocal: ", ilocal)
    print_with_rank("iremote: ", iremote)

    # now communicate this

    # TODO use a single buffer
    to_offsets = np.zeros_like(offsets)

    dtype, _ = get_mpi_dtype(np.dtype(IntType))
    bcast_args = dtype, offsets, to_offsets, MPI.REPLACE
    point_sf.bcastBegin(*bcast_args)
    point_sf.bcastEnd(*bcast_args)

    print_with_rank("to offsets: ", to_offsets)

    # construct a new SF with these offsets
    local_offsets = []
    remote_offsets = []
    for i, (rank, _) in zip(ilocal, iremote):
        # maybe inverse
        # j = axis._inverse_numbering[i]
        # j = axis.numbering[i]
        # print_with_rank(i, j)
        j = i
        for d in range(ndofs[i]):
            local_offsets.append(offsets[j] + d)
            remote_offsets.append((rank, to_offsets[j] + d))

    local_offsets = np.array(local_offsets, dtype=IntType)
    remote_offsets = np.array(remote_offsets, dtype=IntType)

    print_with_rank("local offsets: ", local_offsets)
    print_with_rank("remote offsets: ", remote_offsets)

    return (nroots, local_offsets, remote_offsets)

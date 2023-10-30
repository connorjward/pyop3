from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import pmap

from pyop3.dtypes import IntType, get_mpi_dtype
from pyop3.extras.debug import print_with_rank
from pyop3.utils import strict_int


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


def renumber_sf(sf, numbering):
    # TODO use a single buffer
    to_numbering = np.zeros_like(numbering)

    cdim = 1
    dtype, _ = get_mpi_dtype(np.dtype(IntType), cdim)
    bcast_args = dtype, numbering, to_numbering, MPI.REPLACE
    sf.bcastBegin(*bcast_args)
    sf.bcastEnd(*bcast_args)

    # construct a new SF with these offsets
    nroots, ilocal, iremote = sf.getGraph()

    local_offsets = []
    remote_offsets = []
    i = 0
    for new_pt, old_pt in enumerate(numbering):
        if old_pt in ilocal:
            local_offsets.append(new_pt)
            # first arg is rank
            remote_offsets.append((iremote[i, 0], to_numbering[old_pt]))
            i += 1

    local_offsets = np.array(local_offsets, dtype=IntType)
    remote_offsets = np.array(remote_offsets, dtype=IntType)

    new_sf = PETSc.SF().create(sf.comm)
    new_sf.setGraph(nroots, local_offsets, remote_offsets)
    return new_sf


def collect_sf_graphs(axes, axis=None, path=pmap(), indices=pmap()):
    # NOTE: This function does not check for nested SFs (which should error)
    from pyop3.axes.tree import _axis_component_size

    axis = axis or axes.root

    if axis.sf is not None:
        return (grow_dof_sf(axes, axis, path, indices),)
    else:
        graphs = []
        for component in axis.components:
            subaxis = axes.child(axis, component)
            if subaxis is not None:
                # FIXME will fail if non-integer
                for pt in range(component.count):
                    graphs.extend(
                        collect_sf_graphs(
                            axes,
                            subaxis,
                            path | {axis.label: component.label},
                            indices | {axis.label: pt},
                        )
                    )
        return tuple(graphs)


def grow_dof_sf(axes: FrozenAxisTree, axis, path, indices):
    # effectively build the section
    component_counts = tuple(c.count for c in axis.components)
    *component_offsets, npoints = [0] + list(np.cumsum(component_counts))
    # TODO this is overkill since we only need to compute the owned/halo data
    offsets = np.full(npoints, -1, IntType)
    ndofs = np.copy(offsets)
    for i, component in enumerate(axis.components):
        for j in range(component_counts[i]):
            offset = axes.offset(
                path | {axis.label: component.label}, indices | {axis.label: j}
            )
            ndof = mysize(axes, axis, component)

            offsets[component_offsets[i] + j] = offset
            ndofs[component_offsets[i] + j] = ndof

    # now communicate this
    point_sf = axis.sf

    # TODO use a single buffer
    to_offsets = np.zeros_like(offsets)

    # TODO send offsets and dofs together, makes cdim 2?
    cdim = 1
    dtype, _ = get_mpi_dtype(np.dtype(IntType), cdim)
    bcast_args = dtype, offsets, to_offsets, MPI.REPLACE
    point_sf.bcastBegin(*bcast_args)
    point_sf.bcastEnd(*bcast_args)

    # now send dofs
    to_ndofs = np.zeros_like(ndofs)

    # TODO send offsets and dofs together, makes cdim 2?
    cdim = 1
    dtype, _ = get_mpi_dtype(np.dtype(IntType), cdim)
    bcast_args = dtype, ndofs, to_ndofs, MPI.REPLACE
    point_sf.bcastBegin(*bcast_args)
    point_sf.bcastEnd(*bcast_args)

    # construct a new SF with these offsets
    nroots, ilocal, iremote = point_sf.getGraph()

    local_offsets = []
    remote_offsets = []
    i = 0
    for pt in range(npoints):
        if pt in ilocal:
            for d in range(ndofs[pt]):
                local_offsets.append(offsets[pt] + d)
                # first arg is rank
                remote_offsets.append((iremote[i, 0], to_offsets[pt] + d))
            i += 1

    local_offsets = np.array(local_offsets, dtype=IntType)
    remote_offsets = np.array(remote_offsets, dtype=IntType)

    return (nroots, local_offsets, remote_offsets)

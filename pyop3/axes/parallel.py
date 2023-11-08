from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import pmap

from pyop3.axes.tree import step_size
from pyop3.dtypes import IntType, get_mpi_dtype
from pyop3.extras.debug import print_with_rank
from pyop3.utils import just_one, strict_int


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
    component_offsets = [0] + list(np.cumsum(component_counts))
    npoints = component_offsets[-1]

    # should be a property really, get the root indices so we can compute the offset for them
    buffer = np.full(npoints, False, dtype=bool)
    # buffer[-len(ilocal) :] = True
    buffer[ilocal] = True
    dtype, _ = get_mpi_dtype(buffer.dtype)
    bcast_args = dtype, buffer, buffer, MPI.REPLACE
    point_sf.reduceBegin(*bcast_args)
    point_sf.reduceEnd(*bcast_args)
    buffer[ilocal] = False

    iroot = just_one(np.nonzero(buffer))

    print_with_rank("iroots: ", iroot)

    # effectively build the section
    # TODO this is overkill since we only need to broadcast the roots
    offsets = np.full(npoints, -1, IntType)
    ndofs = np.copy(offsets)
    for pt in iroot:
        # this isn't right
        # rpt = axis.numbering[pt]
        # this works because we are using the default numbering
        # component, component_num = axis.axis_number_to_component(pt)

        # inverse numbering maps from default -> renumbered
        renumbered_pt = axis._inverse_numbering[pt]
        selected_component = None
        component_num = None
        for i, (min_, max_) in enumerate(zip(component_offsets, component_offsets[1:])):
            if min_ <= renumbered_pt < max_:
                selected_component = axis.components[i]
                component_num = renumbered_pt - component_offsets[i]
                break
        assert selected_component is not None
        assert component_num is not None

        offset = axes.offset(
            path | {axis.label: selected_component.label},
            indices | {axis.label: component_num},
            insert_zeros=True,
        )
        offsets[pt] = offset

    print_with_rank("int. offsets: ", offsets)

    for pt in ilocal:
        renumbered_pt = axis._inverse_numbering[pt]
        component = None
        component_num = None
        for i, (min_, max_) in enumerate(zip(component_offsets, component_offsets[1:])):
            if min_ <= renumbered_pt < max_:
                component = axis.components[i]
                component_num = renumbered_pt - component_offsets[i]
                break
        assert component is not None
        assert component_num is not None
        ndofs[pt] = step_size(axes, axis, component)
        offset = axes.offset(
            path | {axis.label: component.label},
            indices | {axis.label: component_num},
            insert_zeros=True,
        )
        offsets[pt] = offset

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

from __future__ import annotations

import functools

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from pyrsistent import pmap

from pyop3.axtree.layout import _as_int, _axis_component_size, step_size
from pyop3.dtypes import IntType, as_numpy_dtype, get_mpi_dtype
from pyop3.utils import checked_zip, just_one, strict_int


def reduction_op(op, invec, inoutvec, datatype):
    dtype = as_numpy_dtype(datatype)
    invec = np.frombuffer(invec, dtype=dtype)
    inoutvec = np.frombuffer(inoutvec, dtype=dtype)
    inoutvec[:] = op(invec, inoutvec)


_contig_min_op = MPI.Op.Create(
    functools.partial(reduction_op, np.minimum), commute=True
)
_contig_max_op = MPI.Op.Create(
    functools.partial(reduction_op, np.maximum), commute=True
)


def partition_ghost_points(axis, sf):
    npoints = sf.size
    is_owned = np.full(npoints, True, dtype=bool)
    is_owned[sf.ileaf] = False

    numbering = np.empty(npoints, dtype=IntType)
    owned_ptr = 0
    ghost_ptr = npoints - sf.nleaves
    points = axis.numbering.data_ro if axis.numbering is not None else range(npoints)
    for pt in points:
        if is_owned[pt]:
            numbering[owned_ptr] = pt
            owned_ptr += 1
        else:
            numbering[ghost_ptr] = pt
            ghost_ptr += 1

    assert owned_ptr == npoints - sf.nleaves
    assert ghost_ptr == npoints
    return numbering


def collect_sf_graphs(axes, axis=None, path=pmap(), indices=pmap()):
    # NOTE: This function does not check for nested SFs (which should error)
    axis = axis or axes.root

    if axis.sf is not None:
        return (grow_dof_sf(axes, axis, path, indices),)
    else:
        graphs = []
        for component in axis.components:
            subaxis = axes.child(axis, component)
            if subaxis is not None:
                for pt in range(_as_int(component.count, indices, path)):
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
def grow_dof_sf(axes, axis, path, indices):
    point_sf = axis.sf
    # TODO, use convenience methods
    nroots, ilocal, iremote = point_sf._graph

    component_counts = tuple(c.count for c in axis.components)
    component_offsets = [0] + list(np.cumsum(component_counts))
    npoints = component_offsets[-1]

    # renumbering per component, can skip if no renumbering present
    if axis.numbering is not None:
        renumbering = [np.empty(c.count, dtype=int) for c in axis.components]
        counters = [0] * len(axis.components)
        for new_pt, old_pt in enumerate(axis.numbering.data_ro):
            for cidx, (min_, max_) in enumerate(
                zip(component_offsets, component_offsets[1:])
            ):
                if min_ <= old_pt < max_:
                    renumbering[cidx][old_pt - min_] = counters[cidx]
                    counters[cidx] += 1
                    break
        assert all(
            count == c.count for count, c in checked_zip(counters, axis.components)
        )
    else:
        renumbering = [np.arange(c.count, dtype=int) for c in axis.components]

    # effectively build the section
    new_nroots = 0
    root_offsets = np.full(npoints, -1, IntType)
    for pt in point_sf.iroot:
        # convert to a component-wise numbering
        selected_component = None
        component_num = None
        for cidx, (min_, max_) in enumerate(
            zip(component_offsets, component_offsets[1:])
        ):
            if min_ <= pt < max_:
                selected_component = axis.components[cidx]
                component_num = renumbering[cidx][pt - component_offsets[cidx]]
                break
        assert selected_component is not None
        assert component_num is not None

        offset = axes.offset(
            indices | {axis.label: component_num},
            path | {axis.label: selected_component.label},
        )
        root_offsets[pt] = offset
        new_nroots += step_size(
            axes,
            axis,
            selected_component,
            indices | {axis.label: component_num},
            # path | {axis.label: selected_component.label},
        )

    point_sf.broadcast(root_offsets, MPI.REPLACE)

    # for sanity reasons remove the original root values from the buffer
    root_offsets[point_sf.iroot] = -1

    local_leaf_offsets = np.empty(point_sf.nleaves, dtype=IntType)
    leaf_ndofs = local_leaf_offsets.copy()
    for myindex, pt in enumerate(ilocal):
        # convert to a component-wise numbering
        selected_component = None
        component_num = None
        for cidx, (min_, max_) in enumerate(
            zip(component_offsets, component_offsets[1:])
        ):
            if min_ <= pt < max_:
                selected_component = axis.components[cidx]
                component_num = renumbering[cidx][pt - component_offsets[cidx]]
                break
        assert selected_component is not None
        assert component_num is not None

        offset = axes.offset(
            indices | {axis.label: component_num},
            path | {axis.label: selected_component.label},
        )
        local_leaf_offsets[myindex] = offset
        leaf_ndofs[myindex] = step_size(axes, axis, selected_component)

    # construct a new SF with these offsets
    ndofs = sum(leaf_ndofs)
    local_leaf_dof_offsets = np.empty(ndofs, dtype=IntType)
    remote_leaf_dof_offsets = np.empty((ndofs, 2), dtype=IntType)
    counter = 0
    for leaf, pos in enumerate(point_sf.ilocal):
        for d in range(leaf_ndofs[leaf]):
            local_leaf_dof_offsets[counter] = local_leaf_offsets[leaf] + d

            rank = point_sf.iremote[leaf][0]
            remote_leaf_dof_offsets[counter] = [rank, root_offsets[pos] + d]
            counter += 1

    return (new_nroots, local_leaf_dof_offsets, remote_leaf_dof_offsets)

import functools
from typing import Hashable

import pytools
from petsc4py import PETSc

from pyop3.axis import Axis, AxisTree

DEFAULT_AXIS_PRIORITY = 100


class InvalidConstraintsException(Exception):
    pass


class ConstrainedAxis(pytools.ImmutableRecord):
    fields = {"axis", "priority", "within_labels"}
    # TODO We could use 'label' to set the priority
    # via commandline options

    def __init__(
        self,
        axis: Axis,
        *,
        priority: int = DEFAULT_AXIS_PRIORITY,
        within_labels: frozenset[Hashable] = frozenset(),
    ):
        self.axis = axis
        self.priority = priority
        self.within_labels = frozenset(within_labels)
        super().__init__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(axis=({', '.join(str(axis_cpt) for axis_cpt in self.axis)}), priority={self.priority}, within_labels={self.within_labels})"


class Space:
    def __init__(self, mesh, layout):
        # TODO mesh.axes is an iterable (could be extruded for example)
        if mesh.axes.depth > 1:
            raise NotImplementedError("need to unpack here somehow")
        meshaxis = ConstrainedAxis(mesh.axes.root, priority=10)
        axes = order_axes([meshaxis] + layout)

        self.mesh = mesh
        self.layout = layout
        self.axes = axes

    @property
    def comm(self):
        return self.mesh.comm

    # I don't like that this is an underscored property. I think internal_comm might be better
    @property
    def _comm(self):
        return self.mesh._comm

    # TODO I think that this could be replaced with callbacks or something.
    # DMShell supports passing a callback
    # https://petsc.org/release/manualpages/DM/DMShellSetCreateGlobalVector/
    # so we could avoid allocating something here.
    @functools.cached_property
    def layout_vec(self):
        """A PETSc Vec compatible with the dof layout of this DataSet."""
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes((self.axes.calc_size(self.axes.root), None), bsize=1)
        vec.setUp()
        return vec

    @functools.cached_property
    def dm(self):
        from firedrake import dmhooks
        from firedrake.mg.utils import get_level

        dm = PETSc.DMShell().create(comm=self._comm)
        dm.setGlobalVector(self.layout_vec)
        _, level = get_level(self.mesh)

        # We need to pass sf and section for preconditioners that are not
        # implemented in Python (e.g. PCPATCH). For Python-level preconditioning
        # we can extract all of this information from the function space.
        # Since pyop3 spaces are more dynamic than a classical PETSc Vec we can only
        # emit sections for "simple" structures where we have points and DoFs/point.
        # TODO for "mixed" problems we could use fields in this section as well.
        # it is still very fragile compared with pyop3
        # Extruded meshes are currently outlawed (axis tree depth must be 2) and to
        # get them to work the numbering would need to be flattened.
        # Ephemeral meshes would probably be rather helpful here.

        # FIXME! this is failing because we are treating zeros as ones when we compute
        # the layouts. Also we need to fix the section computation below I think. At least
        # we could use SectionSetPermutation instead of using our own layout calculations?
        # figure out the right interplay.
        raise NotImplementedError("currently broken")

        # this algorithm is basically equivalent to get_global_numbering
        if self.axes.depth == 2 and self.axes.root.label == "mesh":
            # section = self._shared_data.global_numbering
            section = PETSc.Section().create(comm=self._comm)
            tdim = self.mesh.plex.getDimension()
            entity_counts = [self.mesh.num_entities(d) for d in range(tdim + 1)]
            section.setChart(0, sum(entity_counts) - 1)
            for d in range(tdim + 1):
                layout = self.axes.layouts[(d,)]
                # in pyop3 points per tdim are counted from zero
                counter = 0
                for pt in range(*self.mesh.plex.getDepthStratum(d)):
                    section.setOffset(pt, layout.get_value((counter,)))
                    counter += 1
            section.setUp()
            sf = (self.mesh.plex.getPointSF(),)
        else:
            section = None
            sf = None

        dmhooks.attach_hooks(dm, level=level, section=section, sf=sf)
        # Remember the function space so we can get from DM back to FunctionSpace.
        dmhooks.set_function_space(dm, self)
        return dm


def order_axes(layout):
    axes = AxisTree()
    layout = list(layout)
    axis_to_constraint = {caxis.axis.label: caxis for caxis in layout}
    history = set()
    while layout:
        if tuple(layout) in history:
            raise ValueError("Seen this before, cyclic")
        history.add(tuple(layout))

        constrained_axis = layout.pop(0)
        axes, inserted = _insert_axis(
            axes, constrained_axis, axes.root, axis_to_constraint
        )
        if not inserted:
            layout.append(constrained_axis)
    return axes


def _insert_axis(
    axes: AxisTree,
    new_caxis: ConstrainedAxis,
    current_axis: Axis,
    axis_to_caxis: dict[Axis, ConstrainedAxis],
    path: dict[Hashable] | None = None,
):
    path = path or {}

    within_labels = set(path.items())

    # alias - remove
    axis_to_constraint = axis_to_caxis

    if not axes.root:
        if not new_caxis.within_labels:
            axes = axes.put_node(new_caxis.axis)
            return axes, True
        else:
            return axes, False

    # current_axis = current_axis or axes.root
    current_caxis = axis_to_constraint[current_axis.label]

    if new_caxis.priority < current_caxis.priority:
        raise NotImplementedError("TODO subtrees")
        if new_caxis.within_labels <= within_labels:
            # diagram or something?
            parent_axis = axes.parent(current_axis)
            subtree = axes.pop_subtree(current_axis)
            betterid = new_caxis.axis.copy(id=next(Axis._id_generator))
            if not parent_axis:
                axes.add_node(betterid)
            else:
                axes.add_node(betterid, path)

            # must already obey the constraints - so stick  back in for all sub components
            for comp in betterid.components:
                stree = subtree.copy()
                # stree.replace_node(stree.root.copy(id=next(MultiAxis._id_generator)))
                mypath = (axes._node_to_path[betterid.id] or {}) | {
                    betterid.label: comp.label
                }
                axes.add_subtree(stree, mypath, uniquify=True)
                axes._parent_and_label_to_child[(betterid, comp.label)] = stree.root.id
                # need to register the right parent label
            return True
        else:
            # The priority is less so the axes should definitely
            # not be inserted below here - do not recurse
            return False
    elif axes.is_leaf(current_axis):
        assert new_caxis.priority >= current_caxis.priority
        for cidx, cpt in enumerate(current_axis.components):
            if new_caxis.within_labels <= within_labels | {(current_axis.label, cidx)}:
                # bad uniquify
                betterid = new_caxis.axis.copy(id=next(Axis._id_generator))
                axes = axes.put_node(betterid, current_axis.id, cidx)
        return axes, True
    else:
        inserted = False
        for cidx, cpt in enumerate(current_axis.components):
            subaxis = axes.children(current_axis)[cidx]
            # if not subaxis then we dont insert here
            if subaxis:
                # axes can be unchanged
                axes, now_inserted = _insert_axis(
                    axes,
                    new_caxis,
                    subaxis,
                    axis_to_constraint,
                    path | {current_axis.label: cidx},
                )
                inserted = inserted or now_inserted
        return axes, inserted
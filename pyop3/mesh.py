import dataclasses

from pyop3.tensors import *
from pyop3.utils import Tree



class Mesh:
    map_cache = {}

    def __init__(self, strata_sizes):
        self.strata_sizes = strata_sizes
        tdim = len(strata_sizes)
        self.dim_tree = Tree(MixedDim(tdim),
            tuple(Tree(UniformDim(size)) for size in strata_sizes)
        )

    @property
    def tdim(self):
        return len(self.strata_sizes)

    @property
    def has_substructure(self) -> bool:
        """Does the mesh have some kind of substructure?"""

    # TODO decorate with some sort of cache
    def cone(self, point_set):
        """Return a map."""

    @property
    def cells(self):
        # being very verbose
        dtree = self.dim_tree
        indices = (IntIndex(dtree.value, 0), Slice(dtree.children[0].value))
        stencil = (indices,)
        stencils = frozenset({stencil})
        return StencilGroup(stencils, mesh=self)

    @property
    def ncells(self):
        return self.strata_sizes[0]



class UnstructuredMesh(Mesh):
    # just for now
    NCELLS_IN_CELL_CLOSURE = 1
    NEDGES_IN_CELL_CLOSURE = 3
    NVERTS_IN_CELL_CLOSURE = 3
    CLOSURE_SIZE = NCELLS_IN_CELL_CLOSURE + NEDGES_IN_CELL_CLOSURE + NVERTS_IN_CELL_CLOSURE

    def closure(self, stencils):
        key = (self.closure.__name__, stencils)
        try:
            return self.map_cache[key]
        except KeyError:
            # be very assertive
            stencil, = stencils
            indices, = stencil
            stratum, index = indices

            assert stratum.value == 0  # for now

            edge_map = pyop3.NonAffineMap(index, self.dim_tree.children[1].value, arity=self.NEDGES_IN_CELL_CLOSURE)
            vert_map = pyop3.NonAffineMap(index, self.dim_tree.children[2].value, arity=self.NVERTS_IN_CELL_CLOSURE)
            mapped_stencils = frozenset({  # stencils (i.e. partitions)
                (  # stencil (i.e. temporary)
                    (pyop3.IntIndex(stratum.dim, 0), index),  # indices (i.e. loopy instructions)
                    (pyop3.IntIndex(stratum.dim, 1), edge_map),
                    (pyop3.IntIndex(stratum.dim, 2), vert_map),
                )
            })
            return self.map_cache.setdefault(key, mapped_stencils)


class StructuredMesh(Mesh):
    ...


class ExtrudedMesh(Mesh):
    # just for now
    NCELLS_IN_CELL_CLOSURE = 1
    NEDGES_IN_CELL_CLOSURE = 4
    NVERTS_IN_CELL_CLOSURE = 4
    CLOSURE_SIZE = NCELLS_IN_CELL_CLOSURE + NEDGES_IN_CELL_CLOSURE + NVERTS_IN_CELL_CLOSURE

    NBASE_VERTS_IN_CELL_CLOSURE = 2

    def __init__(self, strata_sizes):
        self.strata_sizes = strata_sizes
        nbase_edges, nbase_verts = self.strata_sizes

        layer_dims =  MixedDim(
            2,
            (
                UniformDim(  # base edges
                    strata_sizes[0],
                    MixedDim(2)
                ),
                UniformDim(  # base verts
                    strata_sizes[1],
                    MixedDim(2)
                )
            )
        )
        self.layer_count = Tensor(layer_dims, name="layer_count")

    def closure(self, stencils):
        """
        For semi-structured meshes we return a tuple of slices for plex ops. The
        first index is for the base mesh entities touched by the stencil. This can have variable arity depending
        on the number of incident base entities (for example a cubed-sphere mesh can touch
        just faces, faces and edges or vertices). The inner map is simply an affine start and offset
        such that the correct memory address can be retrieved given an input point. Like the map, this can have
        non-unity arity (e.g. if there are 5 DoFs in the cell (edge of base mesh) column but only 2 in the facet column
        (vertex in base mesh)). The initial start point of the map can also depend on the outer dimension. This
        is needed for variable layers in extrusion.

        For this example, let's assume that we have a cell that looks like this (+'s are DoFs):

        +----+----+
        |         |
        +   +++   +
        |         |
        +----+----+

        If we take the closure of the cell then we touch 3 base entities - 2 vertices and 1 edge - so the base map
        has arity 3. Due to the differing DoF counts for each entity, the arities of the column maps is different
        for each: 3, 5 and 3 respectively with offsets (0, 1, 2), (0, 1, 2, 3, 4) and (0, 1, 2).

        The base map points to the first entity in each column from which the data is recovered by applying a fixed offset and stride.
        This resolves problems with variable layers in extruded meshes where we might have a mesh that looks something like:

        +---+
        | 4 |
        +---+---+
        | 3 | 2 |
        z-a-y---+
            | 1 |
            x---+

        We can get around any problems by having base_map(3) return (a, y, z) rather than (a, x, z). If we do that
        then we can forget about it as an issue later on during code generation.

        Something like:
        for cell
            for layer
                for i
                    dat[map[i], offset[i]+layer*layersize]
        """
        key = (self.closure.__name__, stencils)
        try:
            return self.map_cache[key]
        except KeyError:
            # be very assertive
            # this will only work for cells at the moment
            stencil, = stencils
            indices, = stencil
            base_stratum, base_index, extr_stratum, extr_index = indices

            assert base_stratum.value == 0  # for now
            assert extr_stratum.value == 0  # for now

            base_vert_map = pyop3.NonAffineMap(
                base_index, pyop3.UniformDim(self.NBASE_VERTS_IN_CELL_CLOSURE)
            )

            inner_edge_map = pyop3.AffineMap(
                extr_index,
                arity=2
            )
            outer_edge_map = pyop3.AffineMap(
                extr_index,
                arity=1,
            )
            outer_vert_map = pyop3.AffineMap(
                extr_index,
                arity=2,
            )

            mapped_stencils = frozenset({  # stencils (i.e. partitions)
                (  # stencil (i.e. temporary)
                    # indices (i.e. loopy instructions)
                    # cell data
                    (pyop3.IntIndex(0), base_index, pyop3.IntIndex(0), extr_index),
                    # in-cell edges
                    (pyop3.IntIndex(0), base_index, pyop3.IntIndex(1), inner_edge_map),
                    # out-cell edges
                    (pyop3.IntIndex(1), base_vert_map, pyop3.IntIndex(0), outer_edge_map),
                    # out-cell verts
                    (pyop3.IntIndex(1), base_vert_map, pyop3.IntIndex(1), outer_vert_map),
                )
            })
            return self.map_cache.setdefault(key, mapped_stencils)

    @property
    def cells(self):
        # being very verbose
        base_cells = pyop3.Slice()
        extr_cells = pyop3.Slice()
        indices = (pyop3.IntIndex(0), base_cells, pyop3.IntIndex(0), extr_cells)
        stencil = (indices,)
        stencils = frozenset({stencil})
        return pyop3.StencilGroup(stencils, mesh=self)


class CubeSphereMesh(Mesh):
    ...


"""
Looping over meshes with substructure
-------------------------------------

If we want to loop efficiently over meshes with substructure (e.g. extruded, cube-sphere)
then we need to access data in a strided, rather than indirection-based, manner. For
extruded meshes this is straightforward as we access the base mesh with an indirection
and then stride up the columns. Things get complicated though for cube-sphere meshes
because we have two dimensional strides to deal with as well as different orientations of
the panels.

My proposed solution looks something like as follows:

1.
    Partition the iteration set (e.g. mesh cells) into groupings requiring a single starting
    indirection (i.e. every point getting iterated over in the partition can be directly
    accessed with strides). This partitioning is accomplished by checking what points are
    touched by the plex stencil operation (e.g. cone) and then grouping the results by the base
    entities of the touched points. This is especially needed for cube-sphere as it allows
    us to iterate up the edge of the mesh but avoiding the vertices whose data lives elsewhere
    and is not iterable.

    For an extruded mesh this would look something like:

    {
        (edge1, vertex1, vertex2): {start_points: [...], strides: [...]},
        (edge2, vertex2, vertex3): {start_points: [...], strides: [...]},
        ...
    }

    where edge1, vertex1, etc are the base entities, start_points are the initial offsets
    and strides tells us how far to jump for each subsequent iteration point.

    N.B. This map represents some plex operation (e.g. cone) so it tells us how to access
    the data from some plex operation that applies to a particular iteration point.

    For a cube-sphere mesh we would have something like:

    {
        (cell1, cell2, edge2): {start_points: [...], strides: [...]},
        (cell2, cell3, edge3, vertex1): {start_points: [...], strides: [...]},
        ...
    }

    Note how for extruded meshes the base mesh is always 1 edge and 2 vertices whereas
    cube-sphere meshes can have a variable number of base entities.

    Also, the size of this table is very much larger for extruded meshes (since there is
    an entry for every base cell) than for cube-sphere since there are far fewer base entities.

2.
    Generate code s.t. the partitions are iterated over in turn.
"""


@dataclasses.dataclass
class IndexStrider:
    # TODO This should optionally be a tuple to allow for cube-sphered
    # and different orientations
    stride: int
    """Stride between adjacent points."""

    start: int = 0
    """The starting point in the array."""


def compute_strides(points, plex_op):
    """

    points: iterable of plex points (ints) to iterate over
    plex_op: the plex operation to perform (e.g. cone())

    """
    strides_per_base_point_group = {}
    for point in points:
        base_points = frozenset({p.base_point for p in plex_op(point)})

        # we only need to do this computation once per column/entity group
        if base_points not in strides_per_base_point_group:
            striders = tuple(IndexStrider(p.stride, p.start) for p in plex_op(point))
            strides_per_base_point_group[base_points] = striders


def closure(itree):
    return itree.mesh.closure(itree)


def star(itree):
    return itree.mesh.star(itree)

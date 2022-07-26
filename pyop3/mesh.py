import dataclasses

from pyop3.tensors import *



class Mesh:
    map_cache = {}

    @property
    def tdim(self):
        return len(self.strata_sizes)

    @property
    def has_substructure(self) -> bool:
        """Does the mesh have some kind of substructure?"""

    # TODO decorate with some sort of cache
    def cone(self, point_set):
        """Return a map."""



class UnstructuredMesh(Mesh):
    # just for now
    NCELLS_IN_CELL_CLOSURE = 1
    NEDGES_IN_CELL_CLOSURE = 3
    NVERTS_IN_CELL_CLOSURE = 3
    CLOSURE_SIZE = NCELLS_IN_CELL_CLOSURE + NEDGES_IN_CELL_CLOSURE + NVERTS_IN_CELL_CLOSURE
    CELL_CONE_SIZE = 3
    EDGE_CONE_SIZE = 2

    def __init__(self, strata_sizes):
        self.strata_sizes = strata_sizes
        self.dims = Tree(Dim(strata_sizes))
        self.dim_tree = self.dims  # deprecated

        ncells, nedges, nverts = self.dims.root.sizes

        cone_map0_dim = Tree.from_nest([Dim(ncells), [Dim(self.CELL_CONE_SIZE)]])
        cone_map0_tensor = Tensor(cone_map0_dim, mesh=self, name="cone0")
        cone_map0 = NonAffineMap(self.dims.root, 0, 1, cone_map0_tensor)

        cone_map1_dim = Tree.from_nest([Dim(nedges), [Dim(self.EDGE_CONE_SIZE)]])
        cone_map1_tensor = Tensor(cone_map1_dim, mesh=self, name="cone1")
        cone_map1 = NonAffineMap(self.dims.root, 1, 2, cone_map1_tensor)

        closure02_dim = Tree.from_nest([Dim(ncells), [Dim(self.NVERTS_IN_CELL_CLOSURE)]])
        closure02_tensor = Tensor(closure02_dim, mesh=self, name="closure02")
        closure02 = NonAffineMap(self.dims.root, 0, 2, closure02_tensor)

        self.maps = {
            self.cone.__name__: {
                0: cone_map0,
                1: cone_map1,
            },
            self.closure.__name__: {
                (0, 0): Slice(self.dims.root, 0),
                (0, 1): cone_map0,
                (0, 2): closure02,
                (1, 1): Slice(self.dims.root, 1),
                (1, 2): cone_map1,
            }
        }

    @property
    def cells(self):
        return StencilGroup([Stencil([
            (Slice(self.dims.root, 0),)
        ])])

    @property
    def ncells(self):
        return self.strata_sizes[0]

    def closure(self, stencils):
        new_stencils = []
        for stencil in stencils:
            new_stencil = []
            for indices in stencil:
                idx, = indices
                for stratum in range(idx.stratum, self.tdim):
                    new_stencil.append((self.maps[self.closure.__name__][(idx.stratum, stratum)],))
            new_stencils.append(Stencil(new_stencil))
        return StencilGroup(new_stencils)

    def cone(self, stencils):
        return StencilGroup([
            Stencil([
                tuple(
                    self.maps[self.cone.__name__][(idx.stratum, idx.stratum+1)]
                    for idx in idxs
                )
                for idxs in stencil
            ])
            for stencil in as_stencil_group(stencils, self.dims)
        ])


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

        layer_count00 = Tensor(Tree(UniformDim(strata_sizes[0])), name="layer_count00")
        layer_count01 = Tensor(Tree(UniformDim(strata_sizes[0])), name="layer_count01")
        layer_count10 = Tensor(Tree(UniformDim(strata_sizes[1])), name="layer_count10")
        layer_count11 = Tensor(Tree(UniformDim(strata_sizes[1])), name="layer_count11")

        self.dim_tree = Tree(
            MixedDim(2),
            (
                # base edges
                Tree(
                    UniformDim(strata_sizes[0]),
                    (
                        Tree(
                            MixedDim(2),
                            (
                                Tree(UniformDim(layer_count00)),
                                Tree(UniformDim(layer_count01))
                            )
                        )
                    )
                ),
                # base verts
                Tree(
                    UniformDim(strata_sizes[1]),
                    (
                        Tree(
                            MixedDim(2),
                            (
                                Tree(UniformDim(layer_count10)),
                                Tree(UniformDim(layer_count11)),
                            )
                        )
                    )
                ),
            )
        )


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
            base_index, extr_index = indices

            base_vert_map = pyop3.NonAffineMap(
                base_index, self.dim_tree.children[1].value, self.NBASE_VERTS_IN_CELL_CLOSURE, name="base_vert_map"
            )

            inner_edge_map = pyop3.AffineMap(
                extr_index,
                self.dim_tree.children[0].child.children[1].value,
                arity=2,
                name="inner_edge_map",
            )
            outer_edge_map = pyop3.AffineMap(
                extr_index,
                self.dim_tree.children[1].child.children[0].value,
                arity=1,
                name="outer_edge_map",
            )
            outer_vert_map = pyop3.AffineMap(
                extr_index,
                self.dim_tree.children[1].child.children[1].value,
                arity=2,
                name="outer_vert_map",
            )

            mapped_stencils = frozenset({  # stencils (i.e. partitions)
                (  # stencil (i.e. temporary)
                    # indices (i.e. loopy instructions)
                    # cell data
                    (base_index, extr_index),
                    # in-cell edges
                    (base_index, inner_edge_map),
                    # out-cell edges
                    (base_vert_map, outer_edge_map),
                    # out-cell verts
                    (base_vert_map, outer_vert_map),
                )
            })
            return self.map_cache.setdefault(key, mapped_stencils)

    @property
    def cells(self):
        dtree = self.dim_tree

        # being very verbose
        base_cells = pyop3.Range(dtree.children[0].value, dtree.children[0].value.size)
        extr_cells = pyop3.Range(dtree.children[0].child.children[1].value, dtree.children[0].child.children[1].value.size)
        indices = (base_cells, extr_cells)
        stencil = (indices,)
        return (stencil,)


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


@dataclasses.dataclass(frozen=True)
class PlexOp:
    index: Any


class Closure(PlexOp):
    pass


class Cone(PlexOp):
    pass


class Star(PlexOp):
    pass


def closure(index):
    return Closure(index)


def cone(index):
    return Cone(index)


def star(index):
    return Star(index)

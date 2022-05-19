import dataclasses

from pyop3.tensors import *



class Mesh:
    def __init__(self, strata_sizes):
        self.strata_sizes = strata_sizes

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
        return IndexTree(None, (IndexTree(Range("start", "end")), None, None), mesh=self)

    @property
    def ncells(self):
        return self.strata_sizes[0]




class UnstructuredMesh(Mesh):
    ...


class StructuredMesh(Mesh):
    ...


class ExtrudedMesh(Mesh):

    MYARITY = 8

    def __init__(self):
        domain = DenseDomain(0, self.MYARITY)
        self.offsets = Dat(domain, "offsets")
        self.layer_count = Dat(domain, "layer_count")

    def closure(self, index):
        """

        Something like:
        for cell
            for layer
                for i
                    dat[map[i], offset[i]+layer*layersize]
        """

        # map from the cell to the starting points for the extruded traversal
        base_domain = SparseDomain(index, MYARITY)

        # strided access to the layers
        layer_domain = DenseDomain(0, self.layer_count[index]*self.offsets[index], self.offsets[index])

        return base_domain, layer_domain


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

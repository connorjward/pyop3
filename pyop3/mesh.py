import dataclasses
import functools

from petsc4py import PETSc

from pyop3.utils import NameGenerator, as_tuple


__all__ = ["Mesh"]


class Mesh:
    map_cache = {}

    _name_generator = NameGenerator(prefix="mesh")

    def __init__(self, dm, *, name=None):
        parts = []
        for i in range(self.height + 1):
            start, stop = self.dm.getHeightStratum(i)
            parts.append(AxisPart(stop - start, id=(i,)))
        axis = MultiAxis(id=name)

        self.dm = dm
        self.axis = axis
        self.name = name or self._name_generator.next()

    @functools.cached_property
    def axis(self):
        ...

    @property
    def dim(self):
        return self.dm.getDimension()

    @property
    def ncells(self):
        start, stop = self.dm.getHeightStratum(0)
        return stop - start

    @property
    def nedges(self):
        start, stop = self.dm.getHeightStratum(1)
        return stop - start

    @property
    def nfacets(self):
        start, stop = self.dm.getDepthStratum(1)
        return stop - start

    @property
    def nvertices(self):
        start, stop = self.dm.getDepthStratum(0)
        return stop - start

    # TODO I need to create a better index type for this sort of thing
    @property
    def cells(self):
        return self[0]
        return [Slice(self.ncells, npart=0, mesh=self)]

    @property
    def edges(self):
        if self.dim < 2:
            raise RuntimeError
        return [Slice(self.nedges, npart=1, mesh=self)]

    @property
    def facets(self):
        if self.dim < 3:
            raise RuntimeError
        return [Slice(self.nfacets, npart=self.dim - 2, mesh=self)]

    @property
    def vertices(self):
        if self.dim < 1:
            raise RuntimeError
        return [Slice(self.nvertices, npart=self.dim - 1, mesh=self)]

    @property
    def depth(self):
        return self.dm.getDepth()

    @property
    def height(self):
        return self.depth

    @functools.lru_cache
    def cone(self, index):
        (index,) = as_tuple(index)
        from_stratum = index.npart
        to_stratum = from_stratum + 1

        start, stop = self.dm.getHeightStratum(from_stratum)
        offset, _ = self.dm.getHeightStratum(to_stratum)

        # We assume a constant cone size
        arity = self.dm.getConeSize(start)
        data = np.zeros(((stop - start), arity), dtype=np.int32) - 1

        for i, pt in enumerate(range(start, stop)):
            data[i] = self.dm.getCone(pt) - offset

        assert (data >= 0).all()

        axes = MultiAxis(AxisPart(stop - start, id="myid")).add_subaxis("myid", arity)
        tensor = MultiArray.new(axes, prefix="map", data=data, dtype=np.int32)
        return NonAffineMap(
            tensor[[index]], arity=arity, npart=from_stratum + 1, mesh=self
        )

    @classmethod
    def create_interval(cls, ncells, length_or_left, right=None):
        """
        Generate a uniform mesh of an interval.

        :arg ncells: The number of the cells over the interval.
        :arg length_or_left: The length of the interval (if ``right``
             is not provided) or else the left hand boundary point.
        :arg right: (optional) position of the right
             boundary point (in which case ``length_or_left`` should
             be the left boundary point).
        """
        if right is None:
            left = 0
            right = length_or_left
        else:
            left = length_or_left

        if ncells <= 0 or ncells % 1:
            raise ValueError("Number of cells must be a positive integer")
        length = right - left
        if length < 0:
            raise ValueError("Requested mesh has negative length")
        dx = length / ncells
        # This ensures the rightmost point is actually present.
        coords = np.arange(left, right + 0.01 * dx, dx, dtype=np.double).reshape(-1, 1)
        cells = np.dstack(
            (
                np.arange(0, len(coords) - 1, dtype=np.int32),
                np.arange(1, len(coords), dtype=np.int32),
            )
        ).reshape(-1, 2)
        dm = PETSc.DMPlex().createFromCellList(1, cells, coords)
        return cls(dm)

    @classmethod
    def create_square(cls, nx, ny, L, **kwargs):
        """Generate a square mesh

        :arg nx: The number of cells in the x direction
        :arg ny: The number of cells in the y direction
        :arg L: The extent in the x and y directions
        """
        return cls.create_rectangle(nx, ny, L, L, **kwargs)

    @classmethod
    def create_rectangle(cls, nx, ny, Lx, Ly, **kwargs):
        """Generate a rectangular mesh

        :arg nx: The number of cells in the x direction
        :arg ny: The number of cells in the y direction
        :arg Lx: The extent in the x direction
        :arg Ly: The extent in the y direction
        """

        for n in (nx, ny):
            if n <= 0 or n % 1:
                raise ValueError("Number of cells must be a positive integer")

        xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
        ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
        return cls.create_tensor_rectangle(xcoords, ycoords, **kwargs)

    @classmethod
    def create_tensor_rectangle(cls, xcoords, ycoords, quadrilateral=False):
        """Generate a rectangular mesh

        :arg xcoords: mesh points for the x direction
        :arg ycoords: mesh points for the y direction
        :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
        """
        xcoords = np.unique(xcoords)
        ycoords = np.unique(ycoords)
        nx = np.size(xcoords) - 1
        ny = np.size(ycoords) - 1

        for n in (nx, ny):
            if n <= 0:
                raise ValueError("Number of cells must be a postive integer")

        coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
        # cell vertices
        i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))

        cells = [
            i * (ny + 1) + j,
            i * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j,
        ]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)

        if not quadrilateral:
            idx = [0, 1, 3, 1, 2, 3]
            cells = cells[:, idx].reshape(-1, 3)

        dm = PETSc.DMPlex().createFromCellList(2, cells, coords)
        return cls(dm)

    @classmethod
    def create_box(cls, nx, ny, nz, Lx, Ly, Lz):
        """Generate a mesh of a 3D box.

        :arg nx: The number of cells in the x direction
        :arg ny: The number of cells in the y direction
        :arg nz: The number of cells in the z direction
        :arg Lx: The extent in the x direction
        :arg Ly: The extent in the y direction
        :arg Lz: The extent in the z direction
        """
        for n in (nx, ny, nz):
            if n <= 0 or n % 1:
                raise ValueError("Number of cells must be a positive integer")

        xcoords = np.linspace(0, Lx, nx + 1, dtype=np.double)
        ycoords = np.linspace(0, Ly, ny + 1, dtype=np.double)
        zcoords = np.linspace(0, Lz, nz + 1, dtype=np.double)
        # X moves fastest, then Y, then Z
        coords = (
            np.asarray(np.meshgrid(xcoords, ycoords, zcoords))
            .swapaxes(0, 3)
            .reshape(-1, 3)
        )
        i, j, k = np.meshgrid(
            np.arange(nx, dtype=np.int32),
            np.arange(ny, dtype=np.int32),
            np.arange(nz, dtype=np.int32),
        )

        v0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1) * (ny + 1)
        v5 = v1 + (nx + 1) * (ny + 1)
        v6 = v2 + (nx + 1) * (ny + 1)
        v7 = v3 + (nx + 1) * (ny + 1)

        cells = [
            v0,
            v1,
            v3,
            v7,
            v0,
            v1,
            v7,
            v5,
            v0,
            v5,
            v7,
            v4,
            v0,
            v3,
            v2,
            v7,
            v0,
            v6,
            v4,
            v7,
            v0,
            v2,
            v6,
            v7,
        ]
        cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)

        dm = PETSc.DMPlex().createFromCellList(dim, cells, coords)
        return cls(dm)


class ProductMesh:
    def __init__(self, *meshes):
        self.meshes = meshes

    def __getitem__(self, indices):
        """Return the correct set of indices..."""
        if len(indices) != len(self.meshes):
            raise ValueError

        slices = []
        for mesh, idx in zip(self.meshes, indices):
            slices.extend(mesh[idx])
        return slices

    @functools.cached_property
    def axis(self):
        return self._productify_mesh_axes(self.meshes)

    def find_axis_part(self, strata):
        return self.axis.find_part(strata)

    def _productify_mesh_axes(self, meshes, parent_ids=()):
        if not meshes:
            return None

        mesh, *rest = meshes

        new_parts = []
        for part in mesh.axis.parts:
            new_subaxis = self._productify_mesh_axes(rest, parent_ids + part.id)
            new_part = part.copy(subaxis=new_subaxis, id=parent_ids + part.id)
            new_parts.append(new_part)
        return mesh.axis.copy(parts=new_parts)


def closure(index):
    raise NotImplementedError


def cone(index):
    maps = []
    for idx in as_tuple(index):
        maps.append(idx.mesh.cone(idx))
    return tuple(maps)


def star(index):
    raise NotImplementedError


# Old notes below:

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

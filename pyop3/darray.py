import abc
import itertools


class DistributedArray(abc.ABC):
    """Base class for all :mod:`pyop3` parallel objects."""

    # @abc.abstractmethod
    # def sync(self):
    #     pass


# TODO make MultiArray inherit from this too


class PetscVec(DistributedArray, abc.ABC):
    def __new__(cls, *args, **kwargs):
        # dispatch to different vec types based on -vec_type
        raise NotImplementedError


class PetscVecStandard(PetscVec):
    ...


class PetscVecNest(PetscVec):
    ...


class PetscMat(DistributedArray, abc.ABC):
    # def __new__(cls, *args, **kwargs):
        # dispatch to different mat types based on -mat_type
        # raise NotImplementedError

    # TODO: What is the best way to instantiate these things?
    @classmethod
    def new(cls, axes, maps):
        raxis, caxis = axes
        rmap, cmap = maps
        # only certain types of axes can form particular matrix types (e.g. MATNEST
        # requires a mixed space "outside")
        # need to "flatten" axes and maps s.t. there is a constant inner shape

        # or similar
        can_be_mat_nest = all(part.size == 1 for part in itertools.chain(raxis.parts, caxis.parts))


        """
        * The axes reference only the nodes, not the cells
        * The maps map from cells (e.g.) to the nodes

        This makes matrices specific to the loop in which they are used as they
        depend on the indices. This is not dissimilar to how vector overlaps depend
        on the sizes of the stencils.

        This is not a pleasing abstraction. Can we do things lazily?

        I suppose that matrix allocation is dependent upon the discretisation (i.e.
        the overlap). Therefore it should be allocat-able prior to a loop. The parloop
        should check that the operations are being provided with valid data structures,
        not allocating the data structures to conform to the operation.
        """
        


class PetscMatDense(PetscMat):
    ...


class PetscMatAIJ(PetscMat):
    ...


class PetscMatBAIJ(PetscMat):
    ...


class PetscMatNest(PetscMat):
    def __init__(axes, ...):
        ...


class PetscMatPython(PetscMat):
    ...

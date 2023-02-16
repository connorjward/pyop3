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

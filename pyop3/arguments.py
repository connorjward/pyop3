import collections.abc
import itertools

import pyop3.domains
import pyop3.exprs


class IndexedTensor:
    def __init__(self, tensor, indices):
        if type(tensor) != Dat:
            raise NotImplementedError("Only dealing with dats atm")

        self.tensor = tensor
        self.indices = indices

    @property
    def name(self):
        return self.tensor.name


class Tensor:
    def __getitem__(self, indices):
        """You can index a dat with a domain to get an argument to a loop.

        **collective**
        """
        if not isinstance(indices, collections.abc.Sequence):
            indices = (indices,)

        if len(indices) > len(self.shape):
            raise ValueError

        # handle things like vector dats
        new_indices = []
        for index, dim in itertools.zip_longest(indices, self.shape):
            if index:
                # index must be a PointIndex (which is a multiindex plus maps)
                # because you can index either with dat[star(p)] or dat[star(p).index]
                if isinstance(index, pyop3.domains.Domain):
                    index = index.index
                new_indices.append(index)
            else:
                # if not indexed loop over whole subdomain
                new_indices.append(
                    pyop3.domains.Domain(
                        dim, new_indices[-1].mesh, new_indices[-1]
                    ).index
                )

        return IndexedTensor(self, tuple(new_indices))


class Global(Tensor):

    shape = ()

    def __init__(self, name):
        self.name = name


class Dat(Tensor):
    def __init__(self, shape, name):
        if not isinstance(shape, collections.abc.Sequence):
            shape = (shape,)

        self.shape = tuple(shape)
        self.name = name

    def __str__(self):
        return self.name

# TODO what is the difference between a vector dat and a mat?
# different domains

class Mat(Tensor):
    def __init__(self, shape, name):
        ...

import abc

import dtl

import pyop3.domains
import pyop3.exprs


class Global(dtl.TensorVariable):
    def __init__(self, name):
        self.name = name


class Dat(dtl.TensorVariable):

    def __str__(self):
        return self.name

    def __getitem__(self, index):
        """You can index a dat with a domain to get an argument to a loop.

        **collective**
        """
        if isinstance(index, pyop3.domains.RestrictedPointSet):
            restriction = index.restriction_tensor
            p = index.parent_index
            i = dtl.Index("itest", restriction.space.spaces[1].dim)
            j = dtl.Index("jtest", restriction.space.spaces[2].dim)
            # contract one index since we want something that is 2D back
            """
            This magic line says:
            
            (for p)  # outside
                for i in range(P)  # big
                    for j in range(arity)  # small
                        t0[j] += R[p, i, j] * dat[i]
            """
            return (restriction[p, i, j] * self[i]).forall(p, j)

        return super().__getitem__(index)


class Mat(dtl.TensorVariable):
    ...

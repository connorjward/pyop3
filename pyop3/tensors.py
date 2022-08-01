import functools
import numpy as np
import operator
import abc
import itertools
import collections
import dataclasses
from typing import Tuple, Union, Any, Optional, Sequence
import numbers
import pyrsistent

import pymbolic as pym

import pytools
import pyop3.exprs
import pyop3.utils
from pyop3.utils import as_tuple, checked_zip, NameGenerator



class Dim(pytools.ImmutableRecord):
    fields = {"sizes", "permutation", "labels", "subdims"}

    _label_generator = NameGenerator("dim")

    def __init__(self, sizes=(), *, permutation=None, labels=None, subdims=()):
        if permutation and not all(isinstance(s, numbers.Integral) for s in sizes):
            raise NotImplementedError("This turns out to be very complicated")
            

        if not isinstance(sizes, collections.abc.Sequence):
            sizes = (sizes,)
        if not labels:
            labels = tuple(self._label_generator.next() for _ in sizes)

        assert len(labels) == len(sizes)

        if subdims:
            assert len(sizes) == len(subdims)

        self.sizes = sizes
        self.permutation = permutation
        self.labels = labels
        self.subdims = subdims
        super().__init__()

    def __bool__(self):
        assert len(self.sizes) == len(self.labels)
        return bool(self.sizes)

    # Could I tie sizes and subdims together?

    @property
    def strata(self):
        return tuple(range(len(self.sizes)))

    @property
    def size(self):
        try:
            s, = self.sizes
            return s
        except ValueError:
            raise RuntimeError

    @property
    def label(self):
        try:
            l, = self.labels
            return l
        except ValueError:
            raise RuntimeError

    @property
    def subdim(self):
        try:
            sdim, = self.subdims
            return sdim
        except ValueError:
            raise RuntimeError

    @property
    def offsets(self):
        # size can be `None` if scalar
        sizes = [size or 1 for size in self.sizes]
        return tuple(sum(sizes[:i]) for i, _ in enumerate(sizes))


# TODO delete `id`
class Index(pytools.ImmutableRecord, abc.ABC):
    """Does it make sense to index a tensor with this object?"""
    fields = {"label", "is_loop_index", "id"}

    _id_generator = NameGenerator("idx")

    def __init__(self, label, is_loop_index=False, *, id=None):
        # self.size = size
        self.label = label
        self.is_loop_index = is_loop_index
        self.id = id or self._id_generator.next()
        super().__init__()

    @property
    def within(self):
        import warnings
        warnings.warn("dontuse", DeprecationWarning)
        return self.is_loop_index


class ScalarIndex(Index):
    """Not a fancy index (scalar-valued)"""


class FancyIndex(Index):
    """Name inspired from numpy. This allows you to slice something with a
    list of indices."""


class Slice(FancyIndex):
    fields = FancyIndex.fields | {"size", "start", "step", "offset"}

    def __init__(self, size, start=0, step=1, offset=0, **kwargs):
        self.size = size
        if isinstance(size, Tensor):
            assert size.indices is not None
        self.start = start
        self.step = step
        self.offset = offset
        super().__init__(**kwargs)

    @classmethod
    def from_dim(cls, dim, subdim_id, *, parent_indices=None, **kwargs):
        size = dim.sizes[subdim_id]
        label = dim.labels[subdim_id]
        offset = dim.offsets[subdim_id]
        return cls(size=size, label=label, offset=offset, **kwargs)

    # @functools.singledispatch
    @staticmethod
    def _as_pym_var(param):
        if pym.primitives.is_valid_operand(param):
            return param
        elif isinstance(param, Tensor):
            return pym.var(param.name)
        else:
            raise TypeError

    # @_as_pym_var.register
    # def _(param: "Tensor"):
    #     return pym.var(param.name)


class Map(FancyIndex, abc.ABC):
    fields = FancyIndex.fields | {"arity"}
    ...
    # fields = FancyIndex.fields | {"from_index", "arity", "name"}
    #
    # _name_generator = NameGenerator("map")
    #
    # def __init__(self, from_index, dim, arity, *, name=None, **kwargs):
    #     if not name:
    #         name = self._name_generator.next()
    #
    #     self.from_index = from_index
    #     # self.arity = arity
    #     self.name = name
    #     super().__init__(dim=dim, **kwargs)
    #
    # @property
    # def index(self):
    #     return LoopIndex(self)

    @property
    def size(self):
        return self.arity


class IndexFunction(Map):
    """The idea here is that we provide an expression, say, "2*x0 + x1 - 3"
    and then use pymbolic maps to replace the xN with the correct inames for the
    outer domains. We could also possibly use pN (or pym.var subclass called Parameter)
    to describe parameters."""
    fields = Map.fields | {"expr", "vardims"}
    def __init__(self, expr, arity, vardims, **kwargs):
        """
        vardims:
            iterable of 2-tuples of the form (var, label) where var is the
            pymbolic Variable in expr and label is the dim label associated with
            it (needed to select the right iname) - note, this is ordered
        """
        self.expr = expr
        self.arity = arity
        self.vardims = vardims

        # the dim label associated with the map is the final entry in vardims
        label = vardims[-1][1]
        super().__init__(label, **kwargs)

    @property
    def size(self):
        return self.arity


class NonAffineMap(Map):
    fields = Index.fields | {"tensor"}

    # TODO is this ever not valid?
    offset = 0

    def __init__(self, tensor, **kwargs):
        self.tensor = tensor
        if "label" in kwargs:
            assert kwargs["label"] == self.tensor.indices[-1].label
        else:
            kwargs["label"] = self.tensor.indices[-1].label
        super().__init__(**kwargs)

    @property
    def input_indices(self):
        return self.tensor.indices[:-1]

    @property
    def map(self):
        return self.tensor

    @property
    def arity(self):
        return self.tensor.indices[-1].size
        # dim = self.tensor.dim
        # while dim.subdims:
        #     dim = dim.subdim
        # return dim.size

    # @property
    # def start(self):
    #     return 0
    #
    # @property
    # def stop(self):
    #     return self.arity
    #
    # @property
    # def step(self):
    #     return 1



class Tensor(pym.primitives.Variable, pytools.ImmutableRecordWithoutPickling):

    fields = {"dim", "indicess", "dtype", "sections", "mesh", "name", "data", "max_value"}

    name_generator = pyop3.utils.MultiNameGenerator()
    prefix = "ten"

    def __init__(self, dim=None, indicess=None, dtype=None, sections=None, *, mesh = None, name: str = None, prefix: str=None, data=None, max_value=32):
        self.data = data
        self.params = {}
        self._param_namer = NameGenerator(f"{name}_p")
        assert dtype is not None

        self.dim = dim
        # if not self._is_valid_indices(indices, dim.root):
        if dim:
            assert all(self._is_valid_indices(idxs, dim) for idxs in indicess)
        else:
            assert indicess is None
            indicess = [[]]
        self.indicess = indicess # self._parse_indices(dim.root, indices)

        # this is a map from dim label to a data layout map (could be an index function
        # or non-affine)
        self.sections = sections
        # self.linear_indices
        # import pdb; pdb.set_trace()

        self.mesh = mesh
        self.dtype = dtype
        self.max_value = max_value
        super().__init__(name)

    @classmethod
    def new(cls, dim=None, indicess=None, *args, prefix=None, name=None, **kwargs):
        # import pdb; pdb.set_trace()
        name = name or cls.name_generator.next(prefix or cls.prefix)

        if dim:
            if not indicess:
                indicess = cls._fill_with_slices(dim)
                # import pdb; pdb.set_trace()
            else:
                indicess = [cls._parse_indices(dim, idxs) for idxs in indicess]
        else:
            assert indicess is None

        # iport pdb; pdb.set_trace()
        sections = cls.collect_sections(dim)

        return cls(dim, indicess, sections=sections, *args, name=name, **kwargs)

    @classmethod
    def collect_sections(cls, dim):
        # FIXME not happy with this, hopefully DimSection (with zero size) will resolve
        if not dim:
            return None

        # this is a map from dim label (should be unique for each tensor) to a tensor
        # or index function
        if dim.permutation:
            secs = cls._collect_sections_permuted(dim)
        else:
            secs = cls._collect_sections_unpermuted(dim)

        for subdim in dim.subdims:
            secs |= cls.collect_sections(subdim)
        # import pdb; pdb.set_trace()
        return secs

    @classmethod
    def _collect_sections_permuted(cls, dim, *, imap=None):
        assert dim.permutation
        if not all(isinstance(s, numbers.Integral) for s in dim.sizes):
            raise NotImplementedError

        if not imap:
            imap = {}

        sections = collections.defaultdict(dict)
        offset = 0
        npoints = sum(dim.sizes)
        for pt in range(npoints):
            pt = dim.permutation[pt]

            subdim, label, i = cls._get_subdim(dim, pt)
            sections[label][pt] = offset

            # increment the pointer by the size of the step for this subdim
            if dim.subdims:
                offset += cls._get_full_dim_size(subdim, imap | {label: pt})
            else:
                offset += 1

        return sections, offset

    @classmethod
    def _collect_sections_unpermuted(cls, dim):
        assert not dim.permutation

        sections = collections.defaultdict(list)
        # used to create a tensor at the end
        # parent_label_size_tracker = collections.defaultdict(list)
        offset = 0
        for subdim_id, label in enumerate(dim.labels):
            dsize = dim.sizes[subdim_id]
            if isinstance(dsize, Tensor):
                for idx_map in cls._generate_indices(dsize):
                    # parent_label_size_tracker[label].append((mylabel, mysize))
                    sections[label].append(offset)
                    if dim.subdims:
                        offset += cls._get_full_dim_size(dim.subdims[subdim_id], idx_map)
                    else:
                        offset += 1
            else:
                for i in range(dsize):
                    sections[label].append(offset)
                    if dim.subdims:
                        offset += cls._get_full_dim_size(dim.subdims[subdim_id], {label: i})
                    else:
                        offset += 1

        # convert to a nice index-type representation
        new_sections = {}
        for label, idxs in sections.items():
            subdim_id = dim.labels.index(label)

            idxs = np.array(idxs, dtype=np.int32)

            # this is hard
            steps = set(idxs[1:] - idxs[:-1])
            if len(steps) == 1:
                step, = steps
                x0 = pym.var("x0")
                expr = x0 * step  # + offset
                new_section = IndexFunction(expr, 1, [(x0, label)])
            else:
                new_section = Tensor.new(Dim(len(idxs), labels=(label,)), data=idxs, prefix="sec", dtype=np.int32)

            new_sections[label] = new_section
        return new_sections

    @classmethod
    def _generate_indices(cls, tensor, dim=None):
        # start from root
        if dim is None:
            dim = tensor.dim

        assert len(dim.sizes) == 1
        assert isinstance(dim.size, numbers.Integral)

        if dim.subdims:
            inner = cls._generate_indices(tensor, dim.subdim)
            result = []
            for i in range(dim.size):
                result.append({dim.label: i} | inner)
            return result
        else:
            return [{dim.label: i} for i in range(dim.size)]

    # @classmethod
    # def _collect_sections_permuted(cls, dim):
    #     ...
    #     sections = {}
    #
    #     # we can create some IndexFunction sections provided that
    #     # we are not permuted
    #     easy_sections = {}
    #     if not dim.permutation:
    #         offset = 0
    #         for i, label in enumerate(dim.labels):
    #             if dim.subdims:
    #                 subdim_size = cls._get_full_dim_size(dim.subdims[i])
    #             else:
    #                 subdim_size = 1
    #
    #             if not cls._requires_nonaffine(dim, label):
    #                 x0 = pym.var("x0")
    #                 expr = x0 * subdim_size + offset
    #                 sec = IndexFunction(expr, 1, [(x0, label)])
    #                 easy_sections[label] = sec
    #             else:
    #                 size = ...
    #
    #             offset += dim.sizes[i] * size
    #
    #
    #
    #     if not dim.subdims:
    #         for label in dim.labels:
    #             if cls._requires_nonaffine(dim, label):
    #                 vals, _ = cls._make_offset_map(dim)
    #                 for label, ten in vals.items():
    #                     sections[label] = ten
    #             else:
    #                 x0 = pym.var("x0")
    #                 sections[label] = IndexFunction(x0, 1, [(x0, label)])
    #     else:
    #         offset = 0
    #         for i, (label, subdim) in enumerate(zip(dim.labels, dim.subdims)):
    #             if cls._requires_nonaffine(dim, label):
    #                 vals, _ = cls._make_offset_map(dim)
    #                 for label, ten in vals.items():
    #                     sections[label] = ten
    #             else:
    #                 x0 = pym.var("x0")
    #                 sections[label] = IndexFunction(x0, 1, [(x0, label)])
    #                 size = cls._get_full_dim_size(subdim)
    #                 x0 = pym.var("x0")
    #                 expr = x0 * size + offset
    #                 sec = IndexFunction(expr, 1, [(x0, label)])
    #                 sections[label] = sec
    #
    #                 # FIXME how does this related to maps?
    #                 offset += dim.sizes[i] * size
    #     return sections

    @classmethod
    def _get_full_dim_size(cls, dim, idx_map=None):
        if not idx_map:
            idx_map = {}

        total_size = 0
        for subdim_id, (size, label) in enumerate(zip(dim.sizes, dim.labels)):
            if isinstance(size, Tensor):
                for i in range(cls._read_tensor(size, idx_map)):
                    if dim.subdims:
                         total_size += cls._get_full_dim_size(
                            dim.subdims[subdim_id], idx_map | {label: i})
                    else:
                        total_size += 1
            else:
                for i in range(size):
                    if dim.subdims:
                         total_size += cls._get_full_dim_size(
                            dim.subdims[subdim_id], idx_map | {label: i})
                    else:
                        total_size += 1

        return total_size

    @classmethod
    def _make_offset_map(cls, dim, *, imap=None):
        # import pdb; pdb.set_trace()
        if imap is None:
            imap = {}

        # build the thing over the entire dim (inc. multi parts) as it's easier to keep
        # track of things

        # how big does the overall map need to be?
        npoints = 0
        for size in dim.sizes:
            # read the entry from the tensor if needed
            if isinstance(size, Tensor):
                size = cls._read_tensor(size, imap)
            npoints += size

        # construct the offset map by increasing a pointer along its length and then
        # reshuffling/splitting at the end
        ptr = 0
        offsets = {}
        for point in range(npoints):

            # e.g. the 2nd point may actually be the 6th, so we need to include the right step
            if dim.permutation:
                point = dim.permutation[point]

            subdim, label, i = cls._get_subdim(dim, point)

            if label in offsets:
                offsets[label][point] = ptr
            else:
                offsets[label] = {point: ptr}

            # increment the pointer by the size of the step for this subdim
            if dim.subdims:
                ptr += cls._get_full_dim_size(subdim, imap | {label: point})
            else:
                ptr += 1

        final = {}
        for label, values in offsets.items():
            subdim_id = dim.labels.index(label)
            myvals = np.array([values[i] for i in sorted(offsets[label].keys())], dtype=np.int32)
            assert len(myvals) == dim.sizes[subdim_id]
            map_dim = Dim(dim.sizes[subdim_id], labels=(label,))
            final[label] = Tensor.new(map_dim, data=myvals, dtype=myvals.dtype, prefix="sec")

        return final, ptr

    @classmethod
    def _requires_nonaffine(cls, dim, label):
        subdim_id = dim.labels.index(label)
        check1 = isinstance(dim.sizes[subdim_id], pym.primitives.Expression) or dim.permutation

        if dim.subdims:
            check2 = False
            for label in dim.subdims[subdim_id].labels:
                if cls._requires_nonaffine(dim.subdims[subdim_id], label):
                    check2 = True
                    break
            return check1 or check2
        else:
            return check1

    @classmethod
    def _get_subdim(cls, dim, point):
        subdims = dim.subdims

        if not subdims:
            return None, None, None

        bounds = list(np.cumsum(dim.sizes))
        for i, (start, stop) in enumerate(zip([0]+bounds, bounds)):
            if start <= point < stop:
                stratum = i
                break
        return subdims[stratum], dim.labels[stratum], stratum

    @classmethod
    def _read_tensor(cls, tensor, idx_map):
        ptr = 0
        for label, idx in idx_map.items():
            section = tensor.sections[label]

            if isinstance(section, Tensor):
                ptr += section.data[idx]
            elif isinstance(section, IndexFunction):
                context = {str(var): idx_map[label] for var, label in section.vardims}
                ptr += pym.evaluate(section.expr, context)
            else:
                raise AssertionError

        return tensor.data[ptr]

    def __getitem__(self, indicess):
        """The plan of action here is as follows:

        - if a tensor is indexed by a set of stencils then that's great.
        - if it is indexed by a set of slices and integers then we convert
          that to a set of stencils.
        - if passed a combination of stencil groups and integers/slices then
          the integers/slices are first converted to stencil groups and then
          the groups are concatenated, multiplying where required.

        N.B. for matrices, we want to take a tensor product of the stencil groups
        rather than concatenate them. For example:

        mat[f(p), g(q)]

        where f(p) is (0, map[i]) and g(q) is (0, map[j]) would produce the wrong
        thing because we would get mat[0, map[i], 0, map[j]] where what we really
        want is mat[0, 0, map[i], map[j]]. Therefore we instead write:

        mat[f(p)*g(q)]

        to get the correct behaviour.
        """

        # TODO Add support for already indexed items
        # This is complicated because additional indices should theoretically index
        # pre-existing slices, rather than get appended/prepended as is currently
        # assumed.
        # if self.is_indexed:
        #     raise NotImplementedError("Needs more thought")

        # if self.name ==  "nnz":
        #     import pdb; pdb.set_trace()
        indicess = [self._parse_indices(self.dim, idxs) for idxs in indicess]
        return self.copy(indicess=indicess)

    @classmethod
    def _fill_with_slices(cls, dim, parent_indices=None):
        # import pdb; pdb.set_trace()
        if not parent_indices:
            parent_indices = []

        idxs = []
        for i, size in enumerate(dim.sizes):
            if size is None:
                idxs.append([])
                continue
            idx = Slice.from_dim(dim, i, parent_indices=parent_indices)
            if dim.subdims:
                idxs += [[idx, *subidxs]
                    for subidxs in cls._fill_with_slices(dim.subdims[i], parent_indices+[idx])]
            else:
                idxs += [[idx]]
        return idxs

    @classmethod
    def _is_valid_indices(cls, indices, dim):
        if not indices and dim.sizes and None in dim.sizes:
            return True

        if dim.sizes and not indices:
            return False

        # scalar case
        if not dim.sizes and not indices:
            return True

        idx, *subidxs = indices
        assert idx.label in dim.labels

        if isinstance(idx, NonAffineMap):
            mapvalid = cls._is_valid_indices(idx.tensor.indices, idx.tensor.dim)
            if not mapvalid:
                return False
            # import pdb; pdb.set_trace()
            subdim_id = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[subdim_id]
                if not cls._is_valid_indices(subidxs, subdim):
                    return False
            return True
        elif isinstance(idx, (Slice, IndexFunction)):
            subdim_id = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[subdim_id]
                if not cls._is_valid_indices(subidxs, subdim):
                    return False
            return True
        else:
            raise TypeError

    def __str__(self):
        return self.name

    @property
    def is_indexed(self):
        return all(self._check_indexed(self.dim, idxs) for idxs in self.indicess)

    def _check_indexed(self, dim, indices):
        for label, size in zip(dim.labels, dim.sizes):
            try:
                (index, subindices), = [(idx, subidxs) for idx, subidxs in indices if idx.label == label]

                subdim_id = dim.labels.index(index.label)

                if subdims := self.dim.get_children(dim):
                    subdim = subdims[subdim_id]
                    return self._check_indexed(subdim, subindices)
                else:
                    return index.size != size
            except:
                return True

    @classmethod
    def _parse_indices(cls, dim, indices, parent_indices=None):
        # import pdb; pdb.set_trace()
        if not parent_indices:
            parent_indices = []

        if not indices:
            if len(dim.sizes) > 1:
                raise ValueError
            else:
                indices = [Slice.from_dim(dim, 0, parent_indices=parent_indices)]

        # import pdb; pdb.set_trace()
        idx, *subidxs = indices

        if isinstance(idx, Map):
            subdim_id = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[subdim_id]
                return [idx] + cls._parse_indices(subdim, subidxs, parent_indices+[idx])
            else:
                return [idx]
        elif isinstance(idx, Slice):
            # reindex dim.size s.t. it references the correct parent indices
            if isinstance(idx.size, pym.primitives.Expression):
                if not isinstance(idx.size, Tensor):
                    raise NotImplementedError
                # myidxs = parent_indices[-idx.size.order:]
                # import pdb; pdb.set_trace()
                # idx = idx.copy(size=idx.size[[myidxs]])

            subdim_id = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[subdim_id]
                return [idx] + cls._parse_indices(subdim, subidxs, parent_indices+[idx])
            else:
                return [idx]
        else:
            raise TypeError

    @property
    def indices(self):
        try:
            idxs, = self.indicess
            return idxs
        except ValueError:
            raise RuntimeError

    # @property
    # def linear_indicess(self):
    #     # import pdb; pdb.set_trace()
    #     if not self.indices:
    #         return [[]]
    #     return [val for item in self.indices for val in self._linearize(item)]

    def _linearize(self, item):
        # import pdb; pdb.set_trace()
        value, children = item

        if children:
            return [[value] + result for child in children for result in self._linearize(child)]
        else:
            return [[value]]

    @property
    def indexed_shape(self):
        try:
            sh, = self.indexed_shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def indexed_shapes(self):
        if not self.dim:
            return ((),)
        return indexed_shapes(self)

    @property
    def indexed_size(self):
        return functools.reduce(operator.mul, self.indexed_shape, 1)

    @property
    def shape(self):
        try:
            sh, = self.shapes
            return sh
        except ValueError:
            raise RuntimeError

    @property
    def shapes(self):
        if not self.dim:
            return ((),)
        else:
            return self._compute_shapes(self.dim)

    @property
    def size(self):
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def order(self):
        return self._compute_order(self.dim)

    def _parametrise_if_needed(self, value):
        if isinstance(value, Tensor):
            if (param := pym.var(value.name)) in self.params:
                assert self.params[param] == value
            else:
                self.params[param] = value
            return param
        else:
            return value

    def _compute_order(self, dim):
        subdims = dim.subdims
        ords = {self._compute_order(subdim) for subdim in subdims}

        if len(ords) == 0:
            return 1
        elif len(ords) == 1:
            return 1 + ords.pop()
        if len(ords) > 1:
            raise Exception("tensor order cannot be established (subdims are different depths)")

    def _merge_stencils(self, stencils1, stencils2):
        return _merge_stencils(stencils1, stencils2, self.dim)

    def _compute_shapes(self, dim):
        # import pdb; pdb.set_trace()
        if not dim:
            return ((),)

        if subdims := dim.subdims:
            # FIXME (see below)
            return tuple(
                (dim.size, *sh) for subdim in subdims for sh in self._compute_shapes(subdim)
            )
        else:
            return tuple((size or 1,) for size in dim.sizes)


def indexed_shapes(tensor):
    if not tensor.dim:
        assert not tensor.indicess
        return ((),)
    return tuple(_compute_indexed_shape(idxs) for idxs in tensor.indicess)


def _compute_indexed_shape(indices):
    if not indices:
        return ()

    index, *subindices = indices

    return index_shape(index) + _compute_indexed_shape(subindices)


def _compute_indexed_shape2(flat_indices):
    shape = ()
    for index in flat_indices:
        shape += index_shape(index)

    return shape


@functools.singledispatch
def index_shape(index):
    raise TypeError

@index_shape.register(Slice)
@index_shape.register(IndexFunction)
def _(index):
    # import pdb; pdb.set_trace()
    if index.is_loop_index:
        return ()
    return (index.size,)

@index_shape.register(NonAffineMap)
def _(index):
    if index.is_loop_index:
        return ()
    else:
        return index.tensor.indexed_shape


def _merge_stencils(stencils1, stencils2, dims):
    stencils1 = as_stencil_group(stencils1, dims)
    stencils2 = as_stencil_group(stencils2, dims)

    return StencilGroup(
        Stencil(
            idxs1+idxs2
            for idxs1, idxs2 in itertools.product(stc1, stc2)
        )
        for stc1, stc2 in itertools.product(stencils1, stencils2)
    )

def as_stencil_group(stencils, dims):
    if isinstance(stencils, StencilGroup):
        return stencils

    is_sequence = lambda seq: isinstance(seq, collections.abc.Sequence)
    # case 1: dat[x]
    if not is_sequence(stencils):
        return StencilGroup([
            Stencil([
                _construct_indices([stencils], dims, dims.root)
            ])
        ])
    # case 2: dat[x, y]
    elif not is_sequence(stencils[0]):
        return StencilGroup([
            Stencil([
                _construct_indices(stencils, dims, dims.root)
            ])
        ])
    # case 3: dat[(a, b), (c, d)]
    elif not is_sequence(stencils[0][0]):
        return StencilGroup([
            Stencil([
                _construct_indices(idxs, dims, dims.root)
                for idxs in stencils
            ])
        ])
    # case 4: dat[((a, b), (c, d)), ((e, f), (g, h))]
    elif not is_sequence(stencils[0][0][0]):
        return StencilGroup([
            Stencil([
                _construct_indices(idxs, dims, dims.root)
                for idxs in stencil
            ])
            for stencil in stencils
        ])
    # default
    else:
        raise ValueError


def _construct_indices(input_indices, dims, current_dim, parent_indices=None):
    if not parent_indices:
        parent_indices = []
    # import pdb; pdb.set_trace()
    if not current_dim:
        return ()

    if not input_indices:
        if len(dims.get_children(current_dim)) > 1:
            raise RuntimeError("Ambiguous subdim_id")
        input_indices = [Slice.from_dim(current_dim, 0, parent_indices=parent_indices)]

    index, *subindices = input_indices

    subdim_id = current_dim.labels.index(index.label)

    if subdims := dims.get_children(current_dim):
        subdim = subdims[subdim_id]
    else:
        subdim = None

    return (index,) + _construct_indices(subindices, dims, subdim, parent_indices + [index.copy(is_loop_index=True)])



def index(indicess):
    """wrap all slices and maps in loop index objs."""
    return tuple(tuple(_index(idx) for idx in idxs) for idxs in indicess)


def _index(idx):
    if isinstance(idx, NonAffineMap):
        return idx.copy(is_loop_index=True, tensor=idx.tensor.copy(indicess=index(idx.tensor.indicess)))
    else:
        return idx.copy(is_loop_index=True)


def _break_mixed_slices(stencils, dtree):
    return tuple(
        tuple(idxs
            for indices in stencil
            for idxs in _break_mixed_slices_per_indices(indices, dtree)
        )
        for stencil in stencils
    )


def _break_mixed_slices_per_indices(indices, dtree):
    """Every slice over a mixed dim should branch the indices."""
    if not indices:
        yield ()
    else:
        index, *subindices = indices
        for i, idx in _partition_slice(index, dtree):
            subtree = dtree.children[i]
            for subidxs in _break_mixed_slices_per_indices(subindices, subtree):
                yield (idx, *subidxs)


"""
so I like it if we could go dat[:mesh.ncells, 2] to access the right part of the mixed dim. How to do multiple stencils?

dat[(), ()] for a single stencil
or dat[((), ()), ((), ())] for stencils

also dat[:] should return multiple indicess (for each part of the mixed dim) (but same stencil/temp)

how to handle dat[2:mesh.ncells] with raggedness? Global offset required.

N.B. dim tree no longer used for codegen - can probably get removed. Though a tree still needed since dims should
be independent of their children.

What about LoopIndex?

N.B. it is really difficult to support partial indexing (inc. integers) because, for ragged tensors, the initial offset
is non-trivial and needs to be computed by summing all preceding entries.
"""


def _partition_slice(slice_, dtree):
    if isinstance(slice_, slice):
        ptr = 0
        for i, child in enumerate(dtree.children):
            dsize = child.value.size
            dstart = ptr
            dstop = ptr + dsize

            # check for overlap
            if ((slice_.stop is None or dstart < slice_.stop)
                    and (slice_.start is None or dstop > slice_.start)):
                start = max(dstart, slice_.start) if slice_.start is not None else dstart
                stop = min(dstop, slice_.stop) if slice_.stop is not None else dstop
                yield i, slice(start, stop)
            ptr += dsize
    else:
        yield 0, slice_


class AffineMap(Map):
    pass




class Section:
    def __init__(self, dofs):
        # dofs is a list of dofs per stratum in the mesh (for now)
        self.dofs = dofs


def Global(*, name: str = None):
    return Tensor(name=name, prefix="glob")


def Dat(mesh, dofs: Section, *, prefix="dat", **kwargs) -> Tensor:
    dims = mesh.dim_tree.copy()
    for i, _ in enumerate(dims.root.sizes):
        dims = dims.add_child(dims.root, Dim(dofs.dofs[i]))
    return Tensor(dims, mesh=mesh, prefix=prefix, **kwargs)


def VectorDat(mesh, dofs, count, **kwargs):
    dim = MixedDim(
        mesh.tdim,
        tuple(
            UniformDim(
                mesh.strata_sizes[stratum],
                UniformDim(dofs.dofs[stratum], UniformDim(count))
            )
            for stratum in range(mesh.tdim)
        )
    )
    return Tensor(dim, **kwargs)


def ExtrudedDat(mesh, dofs, **kwargs):
    # dim = MixedDim(
    #     2,
    #     (
    #         UniformDim(  # base edges
    #             mesh.strata_sizes[0],
    #             MixedDim(
    #                 2,
    #                 (
    #                     UniformDim(mesh.layer_count),  # extr cells
    #                     UniformDim(mesh.layer_count),  # extr 'inner' edges
    #                 )
    #             )
    #         ),
    #         UniformDim(  # base verts
    #             mesh.strata_sizes[1],
    #             MixedDim(
    #                 2,
    #                 (
    #                     UniformDim(mesh.layer_count),  # extr 'outer' edges
    #                     UniformDim(mesh.layer_count),  # extr verts
    #                 )
    #             )
    #         )
    #     )
    # )
    # dim = mesh.dim.copy(children=(
    #     mesh.dim.children[0].copy(children=(
    #         
    #     ))
    # ))
    # TODO Actually attach a section to it.
    return Tensor(mesh.dim_tree, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    raise NotImplementedError

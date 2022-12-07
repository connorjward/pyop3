import copy
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
from pyop3.utils import as_tuple, checked_zip, NameGenerator, unique


def as_multiaxis(axis):
    if isinstance(axis, MultiAxis):
        return axis
    elif isinstance(axis, AbstractAxisPart):
        return MultiAxis(axis)
    else:
        raise TypeError


def compute_offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)[:-1]], dtype=np.int32)


class MultiAxis(pytools.ImmutableRecord):
    fields = {"parts", "permutation", "id"}

    def __init__(self, parts=(), *, permutation=None, id=None):
        parts = tuple(self._parse_part(pt) for pt in as_tuple(parts))

        if permutation and not all(isinstance(pt.size, numbers.Integral) for pt in parts):
            raise NotImplementedError

        self.parts = parts
        self.permutation = permutation
        self.id = id
        super().__init__()

    @property
    def part(self):
        try:
            pt, = self.parts
            return pt
        except ValueError:
            raise RuntimeError

    def find_part(self, part_id):
        if part_id not in self._all_part_ids:
            raise ValueError

        for pt in self.parts:
            if pt.id == part_id:
                return pt

        for pt in self.parts:
            if pt.subaxis and part_id in pt.subaxis._all_part_ids:
                return pt.subaxis.find_part(part_id)

    def find_part_from_indices(self, indices):
        """Traverse axis to find things

        indices is a list of integers"""
        index, *rest = indices

        if not rest:
            return self.parts[index]
        else:
            return self.parts[index].subaxis.find_part_from_indices(rest)


    # TODO I think I would prefer to subclass tuple here s.t. indexing with
    # None works iff len(self.parts) == 1
    def get_part(self, npart):
        if npart is None:
            if len(self.parts) != 1:
                raise RuntimeError
            return self.parts[0]
        else:
            return self.parts[npart]

    @property
    def nparts(self):
        return len(self.parts)

    @property
    def size(self):
        try:
            s, = self.sizes
            return s
        except ValueError:
            raise RuntimeError

    def add_part(self, axis_id, *args):
        if axis_id not in self._all_axis_ids:
            raise ValueError

        part = self._parse_part(*args)
        return self._add_part(axis_id, part)

    def add_subaxis(self, part_id, *args):
        if part_id not in self._all_part_ids:
            raise ValueError

        subaxis = self._parse_multiaxis(*args)
        return self._add_subaxis(part_id, subaxis)

    @functools.cached_property
    def _all_axis_ids(self):
        all_ids = [self.id]
        for part in self.parts:
            if part.subaxis:
                all_ids.extend(part.subaxis._all_axis_ids)

        if len(unique(all_ids)) != len(all_ids):
            raise RuntimeError("Axis IDs must be unique")
        return frozenset(all_ids)

    @functools.cached_property
    def _all_part_ids(self):
        all_ids = []
        for part in self.parts:
            if part.id is not None:
                all_ids.append(part.id)
            if part.subaxis:
                all_ids.extend(part.subaxis._all_part_ids)

        if len(unique(all_ids)) != len(all_ids):
            raise RuntimeError("ids must be unique")
        return frozenset(all_ids)

    def _add_part(self, axis_id, part):
        if axis_id == self.id:
            return self.copy(parts=self.parts+(part,))
        elif axis_id not in self._all_axis_ids:
            return self
        else:
            new_parts = []
            for pt in self.parts:
                if pt.subaxis:
                    new_subaxis = pt.subaxis._add_part(axis_id, part)
                    new_parts.append(pt.copy(subaxis=new_subaxis))
                else:
                    new_parts.append(pt)
            return self.copy(parts=new_parts)

    def _add_subaxis(self, part_id, subaxis):
        # TODO clean this up
        if part_id in self._all_part_ids:
            new_parts = []
            for part in self.parts:
                if part.id == part_id:
                    if part.subaxis:
                        raise RuntimeError("Already has a subaxis")
                    new_part = part.copy(subaxis=subaxis)
                else:
                    if part.subaxis:
                        new_part = part.copy(subaxis=part.subaxis._add_subaxis(part_id, subaxis))
                    else:
                        new_part = part
                new_parts.append(new_part)
            return self.copy(parts=new_parts)
        else:
            return self

    @staticmethod
    def _parse_part(*args):
        if len(args) == 1 and isinstance(args[0], AbstractAxisPart):
            return args[0]
        else:
            return AxisPart(*args)

    @staticmethod
    def _parse_multiaxis(*args):
        if len(args) == 1 and isinstance(args[0], MultiAxis):
            return args[0]
        else:
            return MultiAxis(*args)


class AbstractAxisPart(pytools.ImmutableRecord, abc.ABC):
    fields = set()


class AxisPart(AbstractAxisPart):
    fields = AbstractAxisPart.fields | {"size", "subaxis", "layout", "id"}

    def __init__(self, size, subaxis=None, *, layout=None, id=None):
        if subaxis:
            subaxis = as_multiaxis(subaxis)

        if not isinstance(size, (numbers.Integral, pym.primitives.Expression)):
            raise TypeError

        # hack
        # but WHY
        # I think that the layout thing here needs to be a callable that takes types, idxs and inames
        # and returns a collection of instructions that generate an offset.
        # if layout is None:
        #     x0 = pym.var("x0")
        #     layout = IndexFunction(x0, 1, [x0], npart=None)

        self.size = size
        self.subaxis = subaxis
        self.layout = layout
        self.id = id
        super().__init__()


class ScalarAxisPart(AbstractAxisPart):

    @property
    def size(self):
        return 1


# not used
class ExpressionTemplate:
    """A thing that evaluates to some collection of loopy instructions when
    provided with the right inames.

    Useful for (e.g.) map0_getSize() since function calls are not allowed for GPUs.
    """

    def __init__(self, fn):
        self._fn = fn
        """Callable taking indices that evaluates to a pymbolic expression"""

    def generate(self, _):
        pass


class Index:
    def __init__(self, *args):
        raise NotImplementedError("deprecated")


class IndexSet(pytools.ImmutableRecord):
    """A set of entries to iterate over."""
    fields = {"size", "subset_indices"}

    def __init__(self, size, subset_indices=None):
        self.size = size
        self.subset_indices = subset_indices
        """indices is not None if we are dealing with a subset (e.g. mesh.interior_facets)"""


class TypedIndex(pytools.ImmutableRecord):
    fields = {"part", "iset"}

    def __init__(self, part: int, iset: IndexSet):
        self.part = part
        self.iset = iset


class MultiIndex(pytools.ImmutableRecord):
    fields = {"typed_indices"}

    def __init__(self, typed_indices):
        if any(not isinstance(idx, TypedIndex) for idx in typed_indices):
            raise TypeError
        self.typed_indices = tuple(typed_indices)

    def __iter__(self):
        return iter(self.typed_indices)

    @property
    def depth(self):
        return len(self.indices)


class MultiIndexCollection(pytools.ImmutableRecord, abc.ABC):
    fields = {"multi_indices"}

    def __init__(self, multi_indices):
        if not all(isinstance(idx, MultiIndex) for idx in multi_indices):
            raise ValueError

        self.multi_indices = tuple(multi_indices)

    def __iter__(self):
        return iter(self.multi_indices)


# class Index(pytools.ImmutableRecord, abc.ABC):
#     fields = {"nparts", "sizes"}
#
#     def __init__(self, parts, sizes, depth: int=1):
#         self.parts = parts
#         """List of integers selecting the parts produced by this index."""
#         self.sizes = sizes
#         """Function returning an integer given a part number describing the size of the loop."""
#
#         self.depth = depth
#         """The multi-index size"""
#         super().__init__()


class Map(MultiIndexCollection):
    fields = MultiIndexCollection.fields | {"from_multi_indices"}

    def __init__(self, multi_indices, from_multi_indices):
        super().__init__(multi_indices=multi_indices)
        self.from_multi_indices = from_multi_indices


class Slice(Map):
    fields = Map.fields | {"start", "step"}

    def __init__(self, indices, from_indices, start=None, step=None):
        # FIXME need to think about how slices with starts and steps work
        # with multi-part axes
        if start or step:
            raise NotImplementedError

        super().__init__(indices, from_indices)
        self.start = start
        self.step = step


class IndirectMap(Map):
    fields = Map.fields | {"data"}

    def __init__(self, indices, from_indices, data):
        super().__init__(indices, from_indices)
        self.data = data


# TODO need to specify the output types I reckon - parents can vary but base outputs
# are absolutely needed.
# class Map(Index):
#     fields = Index.fields | {"from_index", "to"}
#
#     def __init__(self, from_, depth: int, parts, sizes, to):
#         if depth != from_.depth:
#             raise ValueError("Can only map between multi-indices of the same size")
#
#         super().__init__(parts=parts,sizes=sizes, depth=depth)
#         """The number of indices 'consumed' by this map"""
#
#         self.from_index = from_
#         """The input multi-index mapped from"""
#
#         self.to = to
#         """A function mapping between multi-indices"""


class LayoutFunction(pytools.ImmutableRecord, abc.ABC):
    fields = set()


class AffineLayoutFunction(LayoutFunction):
    fields = LayoutFunction.fields | {"step", "start"}

    def __init__(self, step, start=0):
        self.step = step
        self.start = start


class IndexFunction(Map):
    """The idea here is that we provide an expression, say, "2*x0 + x1 - 3"
    and then use pymbolic maps to replace the xN with the correct inames for the
    outer domains. We could also possibly use pN (or pym.var subclass called Parameter)
    to describe parameters."""
    fields = Map.fields | {"expr", "vars"}
    def __init__(self, expr, arity, vars, **kwargs):
        """
        vardims:
            iterable of 2-tuples of the form (var, label) where var is the
            pymbolic Variable in expr and label is the dim label associated with
            it (needed to select the right iname) - note, this is ordered
        """
        self.expr = expr
        self.vars = as_tuple(vars)
        super().__init__(arity=arity, **kwargs)

    @property
    def size(self):
        return self.arity


class NonAffineMap(Map):
    fields = Map.fields | {"tensor"}

    # TODO is this ever not valid?
    offset = 0

    def __init__(self, tensor, **kwargs):
        self.tensor = tensor

        # TODO this is AWFUL
        arity_ = self.tensor.indices[-1].size
        if "arity" in kwargs:
            assert arity_ == kwargs["arity"] 
            super().__init__(**kwargs)
        else:
            super().__init__(arity=arity_, **kwargs)

    @property
    def input_indices(self):
        return self.tensor.indices[:-1]

    @property
    def map(self):
        return self.tensor


class MultiArray(pym.primitives.Variable, pytools.ImmutableRecordWithoutPickling):

    fields = {"dim", "indices", "dtype", "mesh", "name", "data", "max_value"}

    name_generator = pyop3.utils.MultiNameGenerator()
    prefix = "ten"

    def __init__(self, dim, indices=None, dtype=None, *, mesh = None, name: str = None, prefix: str=None, data=None, max_value=32):
        dim = as_multiaxis(dim)

        self.data = data
        self.params = {}
        self._param_namer = NameGenerator(f"{name}_p")
        assert dtype is not None

        self.dim = dim
        # if not self._is_valid_indices(indices, dim.root):
        # assert all(self._is_valid_indices(idxs, dim) for idxs in indicess)
        self.indices = indices or [] # self._parse_indices(dim.root, indices)

        self.mesh = mesh
        self.dtype = dtype
        self.max_value = max_value
        super().__init__(name)

    # TODO delete this and just use constructor
    @classmethod
    def new(cls, dim, indices=None, *args, prefix=None, name=None, **kwargs):
        name = name or cls.name_generator.next(prefix or cls.prefix)

        dim = as_multiaxis(dim)

        # if not indicess:
        #     indicess = cls._fill_with_slices(dim)
        # else:
        #     if not isinstance(indicess[0], collections.abc.Sequence):
        #         indicess = (indicess,)
        #     indicess = [cls._parse_indices(dim, idxs) for idxs in indicess]

        # dim = cls.compute_layouts(dim)

        return cls(dim, indices, *args, name=name, **kwargs)

    @classmethod
    def compute_part_size(cls, part):
        size = 0
        if isinstance(part.size, numbers.Integral):
            return part.size
        return size

    @classmethod
    def compute_layouts(cls, axis):
        if axis.permutation:
            layouts = cls.make_offset_map(axis)
        else:
            layouts = [None] * len(axis.parts)

        new_parts = []
        offset = 0  # for mixed
        for part, mylayout in zip(axis.parts, layouts):
            if isinstance(part, ScalarAxisPart):
                # FIXME may not work with mixed
                new_part = part
                offset += 1
            else:
                subaxis = cls.compute_layouts(part.subaxis) if part.subaxis else None

                if axis.permutation:
                    layout = mylayout, 0  # offset here is always 0 as accounted for in map
                    # import pdb; pdb.set_trace()
                else:
                    if isinstance(part.size, pym.primitives.Expression):
                        offsets = compute_offsets(part.size.data)
                        layout = part.size.copy(name=part.size.name+"c", data=offsets), offset
                    else:
                        layout = part.size, offset
                new_part = part.copy(layout=layout, subaxis=subaxis)
                # import pdb; pdb.set_trace()
                offset += cls._compute_full_part_size(part)

            new_parts.append(new_part)
        return axis.copy(parts=new_parts)

    @classmethod
    def _get_part_size(cls, part, parent_indices):
        size = part.size
        if isinstance(size, numbers.Integral):
            return size
        elif isinstance(size, MultiArray):
            return cls._read_tensor(size, parent_indices)
        else:
            raise TypeError

    @classmethod
    def make_offset_map(cls, axis):
        offsets = collections.defaultdict(dict)
        offset = 0
        npoints = sum(p.size for p in axis.parts)
        for pt in range(npoints):
            pt = axis.permutation[pt]

            npart, part = cls._get_subdim(axis, pt)
            offsets[npart][pt] = offset

            # increment the pointer by the size of the step for this subdim
            # FIXME This does not work for ragged as the wrong result is returned here...
            if part.subaxis:
                offset += cls._compute_full_axis_size(part.subaxis, [pt])
            else:
                offset += 1

        layouts = []
        for npart in sorted(offsets):
            idxs = np.array([offsets[npart][i] for i in sorted(offsets[npart])], dtype=np.int32)
            new_section = MultiArray.new(MultiAxis(len(idxs)), data=idxs, prefix="sec", dtype=np.int32)
            layouts.append(new_section)

        return layouts

    @classmethod
    def _generate_looping_indices(cls, part):
        if isinstance(part.size, numbers.Integral):
            return [([], range(part.size))]
        else:
            result = []
            for parent_indices in part.size.mygenerateindices():
                result.append([parent_indices, range(cls._read_tensor(part.size, parent_indices))])
            return result

    @classmethod
    def _generate_indices(cls, part, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        if isinstance(part.size, MultiArray):
            # there must already be an outer dim or this makes no sense
            assert parent_indices
            idxs = [i for i in range(cls._read_tensor(part.size, parent_indices))]
        else:
            idxs = [i for i in range(part.size)]

        if part.subaxis:
            idxs = [
                [i, *subidxs]
                for i in idxs
                for subidxs in cls._generate_indices(part.subaxis, parent_indices=parent_indices+[i])
            ]
        else:
            idxs = [[i] for i in idxs]

        return idxs

    def mygenerateindices(self):
        return self._mygenindices(self.dim)

    @classmethod
    def _mygenindices(cls, axis, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        idxs = []
        for i in range(cls._get_size(axis, parent_indices)):
            if axis.part.subaxis:
                for subidxs in cls._mygenindices(axis.part.subaxis, parent_indices+[i]): 
                    idxs.append([i] + subidxs)
            else:
                idxs.append([i])
        return idxs

    @classmethod
    def _compute_full_part_size(cls, part, parent_indices=None, current_size=1):
        if not parent_indices:
            parent_indices = []

        if isinstance(part, ScalarAxisPart):
            return 1

        # if we encounter an array then discard everything before and make this the new size
        # e.g. if we have 2 * 2 * [1, 2, 3, 4] then the actual size is 1+2+3+4 = 10
        if isinstance(part.size, MultiArray):
            d = cls._slice_marray(part.size, parent_indices)
            current_size = sum(d)
        else:
            current_size *= part.size

        if part.subaxis:
            return sum(cls._compute_full_part_size(pt, parent_indices, current_size) for pt in part.subaxis.parts)
        else:
            return current_size

    @classmethod
    def _slice_marray(cls, marray, parent_indices):
        def compute_subaxis_size(subaxis, idxs):
            if subaxis:
                return cls._compute_full_axis_size(subaxis, idxs)
            else:
                return 1

        if not parent_indices:
            return marray.data
        elif len(parent_indices) == 1:
            if marray.dim.part.subaxis:
                ptr = 0
                parent_idx, = parent_indices
                for i in range(parent_idx):
                    ptr += compute_subaxis_size(marray.dim.part.subaxis, parent_indices+[i])
                return marray.data[ptr:ptr+compute_subaxis_size(marray.dim.part.subaxis, parent_indices+[parent_idx])]
            else:
                idx, = parent_indices
                return marray.data[idx],
        else:
            raise NotImplementedError

    @classmethod
    def _compute_full_axis_size(cls, axis, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        return sum(cls._compute_full_part_size(pt, parent_indices) for pt in axis.parts)

    @classmethod
    def _get_subdim(cls, dim, point):
        bounds = list(np.cumsum([p.size for p in dim.parts]))
        for i, (start, stop) in enumerate(zip([0]+bounds, bounds)):
            if start <= point < stop:
                npart = i
                break
        return npart, dim.parts[npart]

    @classmethod
    def _read_tensor(cls, tensor, idx_map):
        # TODO we don't do any truncation here... access in reverse?

        idx_map_copy = copy.deepcopy(idx_map)

        # for each dim in the tree (must not be mixed), find the section offset and accumulate
        current_dim = tensor.dim
        ptr = 0
        while True:
            npart = 0
            section = current_dim.parts[npart].layout

            idx = idx_map_copy.pop(0)

            if isinstance(section, MultiArray):
                ptr += section.data[idx]
            elif isinstance(section, IndexFunction):
                # basically a very complicated way of turning 4*x into 4*n (where n is a number)
                # so here we need to perform the right substitution.
                context = {str(var): list(reversed(idx_map))[i] for i, var in enumerate(section.vars)}
                ptr += pym.evaluate(section.expr, context)
            else:
                raise AssertionError

            if current_dim.parts[npart].subaxis:
                current_dim = current_dim.parts[npart].subaxis
            else:
                break

        return tensor.data[ptr]

    @property
    def axes(self):
        return self.dim

    def __getitem__(self, indices):
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

        # if not isinstance(indicess[0], collections.abc.Sequence):
        #     indicess = (indicess,)
        # indicess = [self._parse_indices(self.dim, idxs) for idxs in indicess]
        return self.copy(indices=indices)

    def select_axes(self, indices):
        selected = []
        current_axis = self.axes
        for idx in indices:
            selected.append(current_axis)
            current_axis = current_axis.get_part(idx.npart).subaxis
        return tuple(selected)

    @classmethod
    def _fill_with_slices(cls, axis, parent_indices=None):
        if not parent_indices:
            parent_indices = []

        idxs = []
        for i, part in enumerate(axis.parts):
            if isinstance(part, ScalarAxisPart):
                idxs.append([])
                continue
            idx = Slice(part.size, npart=i)
            if part.subaxis:
                idxs += [[idx, *subidxs]
                    for subidxs in cls._fill_with_slices(part.subaxis, parent_indices+[idx])]
            else:
                idxs += [[idx]]
        return idxs

    @classmethod
    def _is_valid_indices(cls, indices, dim):
        # deal with all of this later - need a good scalar solution before this will make sense I think.
        return True

        # not sure what I'm trying to do here
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
            npart = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[npart]
                if not cls._is_valid_indices(subidxs, subdim):
                    return False
            return True
        elif isinstance(idx, (Slice, IndexFunction)):
            npart = dim.labels.index(idx.label)
            if subdims := dim.subdims:
                subdim = subdims[npart]
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

                npart = dim.labels.index(index.label)

                if subdims := self.dim.get_children(dim):
                    subdim = subdims[npart]
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

        # cannot assume a slice if we are mixed - is this right?
        # could duplicate parent indices I suppose
        if not indices:
            if len(dim.parts) > 1:
                raise ValueError
            else:
                indices = [Slice(dim.part.size)]

        # import pdb; pdb.set_trace()
        idx, *subidxs = indices

        if isinstance(idx, Map):
            npart = idx.npart
            part = dim.get_part(npart)
            if part.subaxis:
                return [idx] + cls._parse_indices(part.subaxis, subidxs, parent_indices+[idx])
            else:
                return [idx]
        elif isinstance(idx, Slice):
            if isinstance(idx.size, pym.primitives.Expression):
                if not isinstance(idx.size, MultiArray):
                    raise NotImplementedError

            part = dim.get_part(idx.npart)
            if part.subaxis:
                return [idx] + cls._parse_indices(part.subaxis, subidxs, parent_indices+[idx])
            else:
                return [idx]
        else:
            raise TypeError

    # @property
    # def indices(self):
    #     try:
    #         idxs, = self.indicess
    #         return idxs
    #     except ValueError:
    #         raise RuntimeError

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
        return self._compute_shapes(self.dim)

    @property
    def size(self):
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def order(self):
        return self._compute_order(self.dim)

    def _parametrise_if_needed(self, value):
        if isinstance(value, MultiArray):
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

    def _compute_shapes(self, axis):
        shapes = []
        for part in axis.parts:
            if isinstance(part, ScalarAxisPart):
                shapes.append(())
            elif part.subaxis:
                for shape in self._compute_shapes(part.subaxis):
                    shapes.append((part.size, *shape))
            else:
                shapes.append((part.size,))
        return tuple(shapes)


def indexed_shapes(tensor):
    return tuple(_compute_indexed_shape(idxs) for idxs in tensor.indicess)


def _compute_indexed_shape(indices):
    if not indices:
        return ()

    index, *subindices = indices

    return index_shape(index) + _compute_indexed_shape(subindices)


def _compute_indexed_shape2(flat_indices):
    import warnings
    warnings.warn("need to remove", DeprecationWarning)
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
            raise RuntimeError("Ambiguous npart")
        input_indices = [Slice.from_dim(current_dim, 0)]

    index, *subindices = input_indices

    npart = current_dim.labels.index(index.label)

    if subdims := dims.get_children(current_dim):
        subdim = subdims[npart]
    else:
        subdim = None

    return (index,) + _construct_indices(subindices, dims, subdim, parent_indices + [index.copy(is_loop_index=True)])



def index(indices):
    """wrap all slices and maps in loop index objs."""
    # cannot be multiple sets of indices if we are shoving this into a loop
    if isinstance(indices[0], collections.abc.Sequence):
        (indices,) = indices
    return tuple(_index(idx) for idx in indices)


def _index(idx):
    if isinstance(idx, NonAffineMap):
        return idx.copy(is_loop_index=True, tensor=idx.tensor.copy(indicess=(index(idx.tensor.indices),)))
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


def Global(*, name: str = None):
    raise NotImplementedError
    # return MultiArray(name=name, prefix="glob")


def Dat(mesh, dofs, **kwargs):
    """
    dofs:
        A dict mapping part IDs (usually topological dims) to a new
        subaxis.
    """
    axes = mesh.axis
    for id, subaxis in dofs.items():
        axes = axes.add_subaxis(id, subaxis)
    return MultiArray.new(axes, prefix="dat", **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    raise NotImplementedError

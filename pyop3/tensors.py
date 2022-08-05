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


class MultiAxis(pytools.ImmutableRecord):
    fields = {"parts", "permutation", "id"}

    def __init__(self, parts, *, permutation=None, id=None):
        parts = tuple(self._parse_part(pt) for pt in as_tuple(parts))

        if permutation and not all(isinstance(pt.size, numbers.Integral) for pt in parts):
            raise NotImplementedError("This turns out to be very complicated")

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

    def get_part(self, npart):
        if npart is None:
            if len(self.parts) != 1:
                raise RuntimeError
            return self.parts[0]
        else:
            return self.parts[npart]

    @property
    def subdims(self):
        return tuple(pt.subdim for pt in self.parts)

    @property
    def size(self):
        try:
            s, = self.sizes
            return s
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
    def subaxis(self):
        return self.subdim

    def add_subaxis(self, part_id, *args):
        if part_id not in self._all_part_ids:
            raise ValueError

        subaxis = self._parse_multiaxis(*args)
        return self._add_subaxis(part_id, subaxis)

    @functools.cached_property
    def _all_part_ids(self):
        all_ids = []
        for part in self.parts:
            if part.id is not None:
                all_ids.append(part.id)
            if part.subdim:
                all_ids.extend(part.subdim._all_part_ids)

        if len(unique(all_ids)) != len(all_ids):
            raise RuntimeError("ids must be unique")
        return frozenset(all_ids)

    def _add_subaxis(self, part_id, subaxis):
        if part_id in self._all_part_ids:
            new_parts = []
            for part in self.parts:
                if part.id == part_id:
                    if part.subaxis:
                        raise RuntimeError("Already has a subaxis")
                    new_part = part.copy(subdim=subaxis)
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
    fields = AbstractAxisPart.fields | {"size", "subdim", "layout", "id"}

    def __init__(self, size, subdim=None, layout=None, *, id=None):
        if subdim:
            subdim = as_multiaxis(subdim)

        if not isinstance(size, (numbers.Integral, pym.primitives.Expression)):
            raise TypeError

        self.size = size
        self.subdim = subdim
        self.layout = layout
        self.id = id
        super().__init__()

    @property
    def section(self):
        import warnings
        warnings.warn("use layout instead", DeprecationWarning)
        return self.layout

    @property
    def subaxis(self):
        return self.subdim


class ScalarAxisPart(AbstractAxisPart):

    @property
    def size(self):
        return 1


class Index(pytools.ImmutableRecord, abc.ABC):
    fields = {"npart", "is_loop_index"}

    def __init__(self, npart=None, is_loop_index=False):
        self.npart = npart
        self.is_loop_index = is_loop_index
        super().__init__()


class Slice(Index):
    fields = Index.fields | {"size", "start", "step"}

    def __init__(self, size, start=0, step=1, **kwargs):
        self.size = size
        self.start = start
        self.step = step
        super().__init__(**kwargs)


class Map(Index, abc.ABC):
    fields = Index.fields | {"arity"}

    def __init__(self, arity, **kwargs):
        self.arity = arity
        super().__init__(**kwargs)

    @property
    def size(self):
        return self.arity


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

    fields = {"dim", "indicess", "dtype", "mesh", "name", "data", "max_value"}

    name_generator = pyop3.utils.MultiNameGenerator()
    prefix = "ten"

    def __init__(self, dim, indicess=None, dtype=None, *, mesh = None, name: str = None, prefix: str=None, data=None, max_value=32):
        dim = as_multiaxis(dim)

        self.data = data
        self.params = {}
        self._param_namer = NameGenerator(f"{name}_p")
        assert dtype is not None

        self.dim = dim
        # if not self._is_valid_indices(indices, dim.root):
        assert all(self._is_valid_indices(idxs, dim) for idxs in indicess)
        self.indicess = indicess or [[]] # self._parse_indices(dim.root, indices)

        self.mesh = mesh
        self.dtype = dtype
        self.max_value = max_value
        super().__init__(name)

    @classmethod
    def new(cls, dim, indicess=None, *args, prefix=None, name=None, **kwargs):
        # import pdb; pdb.set_trace()
        name = name or cls.name_generator.next(prefix or cls.prefix)

        dim = as_multiaxis(dim)

        if not indicess:
            indicess = cls._fill_with_slices(dim)
            # import pdb; pdb.set_trace()
        else:
            if not isinstance(indicess[0], collections.abc.Sequence):
                indicess = (indicess,)
            indicess = [cls._parse_indices(dim, idxs) for idxs in indicess]

        # iport pdb; pdb.set_trace()
        dim = cls.collect_sections(dim)

        return cls(dim, indicess, *args, name=name, **kwargs)

    @classmethod
    def collect_sections(cls, dim):
        # this is a map from dim label to a tensor or index function. This is not
        # unique for each tensor so we need to construct a stack of them.
        if dim.permutation:
            layouts = cls._collect_sections_permuted(dim)
        else:
            layouts = cls._collect_sections_unpermuted(dim)

        new_parts = []
        for part, layout in zip(dim.parts, layouts):
            if isinstance(part, ScalarAxisPart):
                assert layout is None
                new_part = part
            else:
                if part.subdim:
                    new_part = part.copy(layout=layout, subdim=cls.collect_sections(part.subdim))
                else:
                    new_part = part.copy(layout=layout)
            new_parts.append(new_part)
        return dim.copy(parts=tuple(new_parts))

    @classmethod
    def _collect_sections_permuted(cls, dim, *, idx_map=None):
        assert dim.permutation
        if not all(isinstance(part.size, numbers.Integral) for part in dim.parts):
            raise NotImplementedError

        if not idx_map:
            idx_map = []

        sections = collections.defaultdict(dict)
        offset = 0
        npoints = sum(p.size for p in dim.parts)
        for pt in range(npoints):
            pt = dim.permutation[pt]

            npart, part = cls._get_subdim(dim, pt)
            sections[npart][pt] = offset

            # increment the pointer by the size of the step for this subdim
            if part.subdim:
                new_idx_map = copy.deepcopy(idx_map)
                new_idx_map.append(pt)
                offset += cls._get_full_dim_size(part.subdim, idx_map=new_idx_map)
            else:
                offset += 1

        new_sections = [None] * len(dim.parts)
        for npart in sections:
            assert isinstance(npart, int)

            part = dim.parts[npart]

            idxs = np.array([sections[npart][i] for i in sorted(sections[npart])], dtype=np.int32)
            new_section = MultiArray.new(MultiAxis(len(idxs)), data=idxs, prefix="sec", dtype=np.int32)
            new_sections[npart] = new_section

        return new_sections

    @classmethod
    def _collect_sections_unpermuted_array(cls, dim, include_size=False):
        assert not dim.permutation

        sections = []
        all_sizes = []
        offset = 0
        for npart, part in enumerate(dim.parts):

            idxs = []
            sizes = []

            if isinstance(part, ScalarAxisPart):
                idxs = [offset]
                sizes = [1]
                offset += 1
            elif isinstance(dsize := part.size, MultiArray):
                # FIXME this is missing an extra index somehow - requires thought!!
                for idx_map in cls._generate_indices(dsize):
                    idxs.append(offset)
                    if part.subdim:
                        size = cls._get_full_dim_size(part.subdim, idx_map=idx_map)
                    else:
                        size = 1
                    sizes.append(size)
                    offset += size
            else:
                for i in range(dsize):
                    idxs.append(offset)
                    if part.subdim:
                        idx_map = [i]
                        size = cls._get_full_dim_size(part.subdim, idx_map=idx_map)
                    else:
                        size = 1
                    sizes.append(size)
                    offset += size

            sections.append(np.array(idxs, dtype=np.int32))
            all_sizes.append(np.array(sizes, dtype=np.int32))

        if include_size:
            return sections, all_sizes
        else:
            return sections

    @classmethod
    def _collect_sections_unpermuted(cls, dim):
        sections = cls._collect_sections_unpermuted_array(dim)
        # convert to a nice index-type representation
        new_sections = []
        for npart, idxs in enumerate(sections):
            part = dim.parts[npart]

            if isinstance(part, ScalarAxisPart):
                new_section = None
            else:
                # see if we can represent this as an affine transformation or not
                steps = set(idxs[1:] - idxs[:-1])
                if not steps:
                    x0 = pym.var("x0")
                    expr = x0
                    new_section = IndexFunction(expr, 1, [x0], npart=npart)
                elif len(steps) == 1:
                    start = idxs[0]
                    step, = steps
                    x0 = pym.var("x0")
                    expr = x0 * step  + start
                    new_section = IndexFunction(expr, 1, [x0], npart=npart)
                else:
                    new_section = MultiArray.new(MultiAxis(len(idxs)), data=idxs, prefix="sec", dtype=np.int32)

            new_sections.append(new_section)
        return new_sections

    @classmethod
    def _generate_indices(cls, tensor, dim=None):
        # start from root
        if dim is None:
            dim = tensor.dim

        assert len(dim.parts) == 1
        assert isinstance(dim.part.size, numbers.Integral)

        if dim.part.subdim:
            raise NotImplementedError("I don't think I actually touch this code atm")
            inner = cls._generate_indices(tensor, dim.part.subdim)
            result = []
            for i in range(dim.part.size):
                result.append({dim.part.label: i} | inner)
            return result
        else:
            return [[i] for i in range(dim.part.size)]

    @classmethod
    def _get_full_dim_size(cls, dim, *, idx_map=None):
        if not idx_map:
            idx_map = collections.defaultdict(list)

        total_size = 0
        for npart, part in enumerate(dim.parts):
            if isinstance(part.size, MultiArray):
                for i in range(cls._read_tensor(part.size, idx_map=idx_map)):
                    if part.subdim:
                        raise NotImplementedError
                        new_idx_map = copy.deepcopy(idx_map)
                        new_idx_map[part.label].append(i)

                        total_size += cls._get_full_dim_size(
                            part.subdim, idx_map=new_idx_map)
                    else:
                        total_size += 1
            else:
                for i in range(part.size):
                    if part.subdim:
                        raise NotImplementedError
                        new_idx_map = copy.deepcopy(idx_map)
                        new_idx_map[part.label].append(i)
                        total_size += cls._get_full_dim_size(part.subdim, idx_map=new_idx_map)
                    else:
                        total_size += 1

        return total_size

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
                # so here we need to perform the right substitution. I think that dim labels
                # are right to use here as we could theoretically get duplicates and we want
                # to go in reverse
                # idx_map_copy = idx_map.copy()
                # context = {}
                # for var, label in reversed(section.vardims):
                #     ilabel, _, iidx = idx_map_copy.pop()
                #     while ilabel != label:
                #         ilabel, _, iidx = idx_map_copy.pop()
                #     context[str(var)] = iidx
                # FIXME assumes no duplicates
                context = {str(var): idx_map[0] for var in section.vars}
                ptr += pym.evaluate(section.expr, context)
            else:
                raise AssertionError

            if current_dim.parts[npart].subdim:
                current_dim = current_dim.parts[npart].subdim
            else:
                break

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

        if not isinstance(indicess[0], collections.abc.Sequence):
            indicess = (indicess,)
        indicess = [self._parse_indices(self.dim, idxs) for idxs in indicess]
        return self.copy(indicess=indicess)

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
            if part.subdim:
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
            if part.subdim:
                return [idx] + cls._parse_indices(part.subdim, subidxs, parent_indices+[idx])
            else:
                return [idx]
        elif isinstance(idx, Slice):
            if isinstance(idx.size, pym.primitives.Expression):
                if not isinstance(idx.size, MultiArray):
                    raise NotImplementedError

            part = dim.get_part(idx.npart)
            if part.subdim:
                return [idx] + cls._parse_indices(part.subdim, subidxs, parent_indices+[idx])
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

    def _compute_shapes(self, dim):
        shapes = []
        for part in dim.parts:
            if isinstance(part, ScalarAxisPart):
                shapes.append(())
            elif part.subdim:
                for shape in self._compute_shapes(part.subdim):
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


class AffineMap(Map):
    pass




class Section:
    def __init__(self, dofs):
        # dofs is a list of dofs per stratum in the mesh (for now)
        self.dofs = dofs


def Global(*, name: str = None):
    return MultiArray(name=name, prefix="glob")


def Dat(mesh, dofs: Section, *, prefix="dat", **kwargs) -> MultiArray:
    dims = mesh.dim_tree.copy()
    for i, _ in enumerate(dims.root.sizes):
        dims = dims.add_child(dims.root, MultiAxis(dofs.dofs[i]))
    return MultiArray(dims, mesh=mesh, prefix=prefix, **kwargs)


def VectorDat(mesh, dofs, count, **kwargs):
    dim = MixedMultiAxis(
        mesh.tdim,
        tuple(
            UniformMultiAxis(
                mesh.strata_sizes[stratum],
                UniformMultiAxis(dofs.dofs[stratum], UniformMultiAxis(count))
            )
            for stratum in range(mesh.tdim)
        )
    )
    return MultiArray(dim, **kwargs)


def ExtrudedDat(mesh, dofs, **kwargs):
    # dim = MixedMultiAxis(
    #     2,
    #     (
    #         UniformMultiAxis(  # base edges
    #             mesh.strata_sizes[0],
    #             MixedMultiAxis(
    #                 2,
    #                 (
    #                     UniformMultiAxis(mesh.layer_count),  # extr cells
    #                     UniformMultiAxis(mesh.layer_count),  # extr 'inner' edges
    #                 )
    #             )
    #         ),
    #         UniformMultiAxis(  # base verts
    #             mesh.strata_sizes[1],
    #             MixedMultiAxis(
    #                 2,
    #                 (
    #                     UniformMultiAxis(mesh.layer_count),  # extr 'outer' edges
    #                     UniformMultiAxis(mesh.layer_count),  # extr verts
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
    return MultiArray(mesh.dim_tree, **kwargs)


def Mat(shape: Tuple[int, ...], *, name: str = None):
    raise NotImplementedError

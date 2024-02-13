import abc
import collections
import functools
import itertools
import operator
import warnings
from typing import Any, Collection, Hashable, Optional

import numpy as np
import pytools
from pyrsistent import pmap


class UniqueNameGenerator(pytools.UniqueNameGenerator):
    """Class for generating unique names."""

    def __call__(self, prefix: str) -> str:
        # To skip using prefix as a unique name we declare it as already used
        self.add_name(prefix, conflicting_ok=True)
        return super().__call__(prefix)


_unique_name_generator = UniqueNameGenerator()
"""Generator for creating globally unique names."""


def unique_name(prefix: str) -> str:
    return _unique_name_generator(prefix)


class auto:
    pass


# type aliases
Id = Hashable
Label = Hashable


class Identified(abc.ABC):
    def __init__(self, id):
        self.id = id if id is not None else self.unique_id()

    @classmethod
    def unique_id(cls) -> str:
        return unique_name(f"_id_{cls.__name__}")


class Labelled(abc.ABC):
    def __init__(self, label):
        self.label = label if label is not None else self.unique_label()

    @classmethod
    def unique_label(cls) -> str:
        return unique_name(f"_label_{cls.__name__}")


# TODO is Identified really useful?
class UniqueRecord(pytools.ImmutableRecord, Identified):
    fields = {"id"}

    def __init__(self, id=None):
        pytools.ImmutableRecord.__init__(self)
        Identified.__init__(self, id)


def as_tuple(item):
    if isinstance(item, collections.abc.Sequence):
        return tuple(item)
    else:
        return (item,)


def split_at(iterable, index):
    return iterable[:index], iterable[index:]


class PrettyTuple(tuple):
    """Implement a tuple with nice syntax for recursive functions. Like set notation."""

    def __or__(self, other):
        return type(self)(self + (other,))


def checked_zip(*iterables):
    if not pytools.is_single_valued(set(len(it) for it in iterables)):
        raise ValueError
    return zip(*iterables)


def rzip(*iterables):
    if any(not isinstance(it, collections.abc.Sized) for it in iterables):
        raise ValueError("Can only rzip with objects that have a known length")

    max_length = max(len(it) for it in iterables)
    return zip(*(pad(it, max_length, False) for it in iterables))


def pad(iterable, length, after=True, padding_value=None):
    missing = [padding_value] * (length - len(iterable))
    if after:
        return itertools.chain(iterable, missing)
    else:
        return itertools.chain(missing, iterable)


single_valued = pytools.single_valued
is_single_valued = pytools.is_single_valued


def merge_dicts(dicts, persistent=True):
    merged = {}
    for dict_ in dicts:
        merged.update(dict_)
    return pmap(merged) if persistent else merged


def unique(iterable):
    unique_items = []
    for item in iterable:
        if item not in unique_items:
            unique_items.append(item)
    return tuple(unique_items)


def has_unique_entries(iterable):
    # duplicate the iterator in case it can only be iterated over once (e.g. a generator)
    it1, it2 = itertools.tee(iterable, 2)
    return len(unique(it1)) == len(list(it2))


def is_sequence(item):
    return isinstance(item, collections.abc.Sequence)


def flatten(iterable):
    """Recursively flatten a nested iterable."""
    if isinstance(iterable, np.ndarray):
        return iterable.flatten()
    if not isinstance(iterable, (list, tuple)):
        return (iterable,)
    return tuple(item_ for item in iterable for item_ in flatten(item))


def some_but_not_all(iterable):
    # duplicate the iterable in case using any/all consumes it
    it1, it2 = itertools.tee(iterable)
    return any(it1) and not all(it2)


def strictly_all(iterable):
    """Returns ``all(iterable)`` but raises an exception if values are inconsistent."""
    if not isinstance(iterable, collections.abc.Iterable):
        raise TypeError("Expecting an iterable")

    # duplicate the iterable in case using any/all consumes it
    it1, it2 = itertools.tee(iterable)
    if (result := any(it1)) and not all(it2):
        raise ValueError("Iterable contains inconsistent values")
    return result


def just_one(iterable):
    # bit of a hack
    iterable = list(iterable)

    if len(iterable) == 0:
        raise ValueError("Empty iterable found")
    if len(iterable) > 1:
        raise ValueError("Too many values")
    return iterable[0]


class MultiStack:
    """Keyed stack."""

    def __init__(self, data=None):
        raise NotImplementedError("shouldnt be needed")
        self._data = data or collections.defaultdict(PrettyTuple)

    def __str__(self):
        return str(dict(self._data))

    def __repr__(self):
        return f"{self.__class__}({self._data!r})"

    def __getitem__(self, key):
        return self._data[key]

    def __or__(self, other):
        new_data = self._data.copy()
        if isinstance(other, collections.abc.Mapping):
            for key, value in other.items():
                new_data[key] += value
            return type(self)(new_data)
        else:
            return NotImplemented


def popwhen(predicate, iterable):
    """Pop the first instance from iterable where predicate is ``True``."""
    if not isinstance(iterable, list):
        raise TypeError("Expecting iterable to be a list")

    for i, item in enumerate(iterable):
        if predicate(item):
            return iterable.pop(i)
    raise KeyError("Predicate does not hold for any items in iterable")


def steps(sizes, drop_last=False):
    sizes = tuple(sizes)
    steps_ = (0,) + tuple(np.cumsum(sizes, dtype=int))
    return steps_[:-1] if drop_last else steps_


def pairwise(iterable):
    return zip(iterable, iterable[1:])


# stolen from stackoverflow
# https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
def invert(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p)  # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def strict_cast(obj, cast):
    new_obj = cast(obj)
    if new_obj != obj:
        raise TypeError(f"Invalid cast from {obj} to {new_obj}")
    return new_obj


def strict_int(num):
    return strict_cast(num, int)


def apply_at(func, iterable, index):
    if index < 0 or index >= len(iterable):
        raise IndexError

    result = []
    for i, item in enumerate(iterable):
        if i == index:
            result.append(func(item))
        else:
            result.append(item)
    return tuple(result)


def map_when(func, when_func, iterable):
    for item in iterable:
        if when_func(item):
            yield func(item)
        else:
            yield item


def readonly(array):
    """Return a readonly view of a numpy array."""
    view = array.view()
    view.setflags(write=False)
    return view


def deprecated(prefer=None, internal=False):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            msg = f"{fn.__qualname__} is deprecated and will be removed"
            if prefer:
                msg += f", please use {prefer} instead"
            warning_type = DeprecationWarning if internal else FutureWarning
            warnings.warn(msg, warning_type)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


class FrozenRecordException(TypeError):
    pass


def _disabled_record_copy(self, **kwargs):
    raise FrozenRecordException("Cannot call copy on a frozen record class")


def frozen_record(cls):
    """Class decorator that disables record copying.

    This is required to handle the case where we have `pytools.Record` subclasses
    that have "correlated" attributes. Consider a case where we have class
    ``MyClass`` with attributes ``a`` and ``b``, where ``a`` and ``b`` are in some
    sense related. It is therefore invalid to call ``myobj.copy(a=new_a)`` or
    ``myobj.copy(b=new_b)`` as that will break the connection between ``a``
    and ``b``.

    The primary use case for this decorator is for `AxisTree` (non-frozen) and
    `SetUpAxisTree` (frozen). We want to inherit the full set of methods from
    `LabelledTree` into `AxisTree`, but when we call `AxisTree.set_up` we no longer
    want to allow "mutator" methods that add additional axes since the tree now
    has correlated attributes such as the layout functions and star forest and
    adding new axes would break them.

    Notes
    -----
    This behaviour has been implemented as a class decorator as opposed to
    a mixin class because, for a mixin class, the disabling behaviour would
    be dependent on the ordering of the classes in the inheritance hierarchy.

    """
    if not issubclass(cls, pytools.Record):
        raise TypeError("frozen_record is only valid for subclasses of pytools.Record")
    cls.copy = _disabled_record_copy
    return cls

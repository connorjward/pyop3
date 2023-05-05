import collections
import itertools
from typing import Any, Collection

import pytools


# a tree
class Node(pytools.ImmutableRecord):
    fields = {"value", "children"}

    def __init__(self, value, children=()):
        self.value = value
        self.children = tuple(children)
        super().__init__()


class MultiNameGenerator:
    def __init__(self):
        self._namers = {}

    def next(self, prefix="", suffix=""):
        key = prefix, suffix
        try:
            namer = self._namers[key]
        except KeyError:
            namer = self._namers.setdefault(key, NameGenerator(prefix, suffix))
        return namer.next()

    def reset(self):
        self._namers = {}


class NameGenerator:
    def __init__(self, prefix="", suffix=""):
        if not (prefix or suffix):
            raise ValueError

        self._prefix = prefix
        self._suffix = suffix
        self._counter = itertools.count()

    def next(self):
        return f"{self._prefix}{next(self._counter)}{self._suffix}"


def as_tuple(item):
    if isinstance(item, collections.abc.Sequence):
        return tuple(item)
    else:
        return (item,)


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
    if not isinstance(iterable, collections.abc.Iterable):
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


def strict_cast(obj, cast):
    new_obj = cast(obj)
    if new_obj != obj:
        raise TypeError(f"Invalid cast from {obj} to {new_obj}")
    return new_obj


def strict_int(num):
    return strict_cast(num, int)

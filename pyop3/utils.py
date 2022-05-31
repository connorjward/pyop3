import collections
import itertools
import pytools


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


class CustomTuple(tuple):
    """Implement a tuple with nice syntax for recursive functions. Like set notation."""

    def __or__(self, other):
        return self + (other,)

def checked_zip(*iterables):
    if not pytools.is_single_valued(set(len(it) for it in iterables)):
        raise ValueError
    return zip(*iterables)


class Tree(pytools.ImmutableRecord):
    def __init__(self, value, children=()):
        children = as_tuple(children)
        super().__init__(value=value, children=children)

    @property
    def child(self):
        return pytools.one(self.children)

    @property
    def is_leaf(self):
        return not self.children


def unique(iterable):
    unique_items = []
    for item in iterable:
        if item not in unique_items:
            unique_items.append(item)
    return tuple(unique_items)

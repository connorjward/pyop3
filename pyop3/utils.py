import collections
import itertools
import pyrsistent
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


class PrettyTuple(tuple):
    """Implement a tuple with nice syntax for recursive functions. Like set notation."""

    def __or__(self, other):
        return self + (other,)

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


def unique(iterable):
    unique_items = []
    for item in iterable:
        if item not in unique_items:
            unique_items.append(item)
    return tuple(unique_items)


def is_sequence(item):
    return isinstance(item, collections.abc.Sequence)

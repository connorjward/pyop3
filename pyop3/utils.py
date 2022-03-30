import collections.abc
import itertools


class NameGenerator:
    def __init__(self, prefix="", suffix=""):
        if not (prefix or suffix):
            raise ValueError("Must specify either a prefix or suffix")
        self.prefix = prefix
        self.suffix = suffix
        self._counter = itertools.count()

    def __iter__(self):
        return self

    def __next__(self):
        return f"{self.prefix}{next(self._counter)}{self.suffix}"


def as_tuple(item):
    return tuple(item) if isinstance(item, collections.abc.Iterable) else (item,)

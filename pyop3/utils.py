import collections
import itertools


class UniqueNameGenerator:
    def __init__(self):
        self.name_generators = {}

    def generate(self, prefix="", suffix=""):
        try:
            namer = self.name_generators[(prefix, suffix)]
        except KeyError:
            namer = self.name_generators.setdefault(
                (prefix, suffix), _NameGenerator(prefix, suffix)
            )
        return next(namer)


class _NameGenerator:
    def __init__(self, prefix="", suffix=""):
        if not (prefix or suffix):
            raise ValueError

        self._prefix = prefix
        self._suffix = suffix
        self._counter = itertools.count()

    def __iter__(self):
        return self

    def __next__(self):
        return f"{self._prefix}{next(self._counter)}{self._suffix}"


def as_tuple(item):
    if isinstance(item, collections.abc.Sequence):
        return item
    else:
        return (item,)

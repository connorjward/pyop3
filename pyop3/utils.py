import collections
import itertools
import pytools


class UniqueNameGenerator:
    def __init__(self):
        self.name_generators = {}

    def next(self, prefix="", suffix=""):
        try:
            namer = self.name_generators[(prefix, suffix)]
        except KeyError:
            namer = self.name_generators.setdefault(
                (prefix, suffix), NameGenerator(prefix, suffix)
            )
        return namer.next()


class NameGenerator:
    def __init__(self, prefix="", suffix="", existing_names=frozenset()):
        if not (prefix or suffix):
            raise ValueError

        self._prefix = prefix
        self._suffix = suffix
        self._existing_names = existing_names
        self._counter = itertools.count()

    def next(self):
        while (name := f"{self._prefix}{next(self._counter)}{self._suffix}") in self._existing_names:
            pass
        return name


def as_tuple(item):
    if isinstance(item, collections.abc.Sequence):
        return item
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

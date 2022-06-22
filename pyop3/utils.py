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


class CustomTuple(tuple):
    """Implement a tuple with nice syntax for recursive functions. Like set notation."""

    def __or__(self, other):
        return self + (other,)

def checked_zip(*iterables):
    if not pytools.is_single_valued(set(len(it) for it in iterables)):
        raise ValueError
    return zip(*iterables)


class Tree(pytools.ImmutableRecord):
    fields = {"root", "children"}

    def __init__(self, root, children=None):
        self.root = root
        self.children = pyrsistent.pmap(children or {root: ()})
        super().__init__()

    def add_child(self, parent, child):
        new_children = dict(self.children)
        new_children[parent] += (child,)
        new_children[child] = ()
        return self.copy(children=new_children)

    def get_child(self, item):
        if children := self.get_children(item):
            return pytools.one(children)
        else:
            return None

    def get_children(self, item):
        return self.children[item]

    def is_leaf(self, item):
        return not self.get_children(item)

    @classmethod
    def from_nest(cls, nest):
        root, _ = nest
        children = cls._collect_children(nest)
        return cls(root, children)

    @classmethod
    def _collect_children(cls, nest):
        try:
            from_edge, subnests = nest
        except TypeError:
            from_edge, subnests = nest, ()

        to_edges = tuple(
            subnest if not is_sequence(subnest) else subnest[0]
            for subnest in subnests
        )

        subchildren = dict(
            ch for subnest in subnests for ch in cls._collect_children(subnest).items()
        )
        return {from_edge: to_edges} | subchildren


def unique(iterable):
    unique_items = []
    for item in iterable:
        if item not in unique_items:
            unique_items.append(item)
    return tuple(unique_items)


def is_sequence(item):
    return isinstance(item, collections.abc.Sequence)

import abc
import enum
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


class Restriction(enum.Enum):

    CLOSURE = enum.auto()
    STAR = enum.auto()

    def __str__(self):
        return self.name


class DomainIndex:
    """Class representing an index in a domain."""

    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

    def __str__(self):
        return self.name


class Domain(abc.ABC):

    _iname_generator = NameGenerator(prefix="i")

    def __init__(self):
        self._index = DomainIndex(next(self._iname_generator), self)

    @property
    def index(self):
        return self._index


class FreeDomain(Domain):
    """An unrestricted domain (must be the outermost one)."""

    def __init__(self, extent):
        self.extent = extent
        super().__init__()

    def __str__(self):
        return f"range({self.extent})"


class RestrictedDomain(Domain):
    """E.g. closure(i)."""

    def __init__(self, parent_index: DomainIndex, restriction: Restriction):
        self.restriction = restriction
        self.parent_index = parent_index
        super().__init__()

    def __str__(self):
        return f"{self.restriction}({self.parent_index})"


def closure(index):
    return RestrictedDomain(index, Restriction.CLOSURE)


def star(index):
    return RestrictedDomain(index, Restriction.STAR)

import collections.abc


def as_tuple(item):
    return tuple(item) if isinstance(item, collections.abc.Iterable) else (item,)

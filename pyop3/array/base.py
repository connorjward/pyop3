import abc

from pyop3.axtree import ContextAware
from pyop3.axtree.tree import Terminal
from pyop3.lang import FunctionArgument, Assignment
from pyop3.utils import UniqueNameGenerator


class Array(ContextAware, FunctionArgument, Terminal, abc.ABC):
    _prefix = "array"
    _name_generator = UniqueNameGenerator()

    def __init__(self, name=None, *, prefix=None) -> None:
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")
        self.name = name or self._name_generator(prefix or self._prefix)

    def assign(self, other, /, *, eager=False):
        expr = Assignment(self, other, "write")
        return expr() if eager else expr

    # TODO: Add this to different types
    # @abc.abstractmethod
    # def reshape(self, *axes):
    #     pass

    @abc.abstractmethod
    def with_context(self):
        pass

    @property
    @abc.abstractmethod
    def context_free(self):
        pass

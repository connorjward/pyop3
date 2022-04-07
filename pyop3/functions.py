import dtl
import dtlpp.monads


class TemporaryFunctionArgument(dtl.Node):

    operands = ()

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class Function(dtl.Node):

    operands = ()

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, *args):
        args = tuple(_replace_temporary(a) for a in args)
        return dtlpp.monads.FunctionCall(self, args)


def _replace_temporary(argument):
    if isinstance(argument, str):
        return TemporaryFunctionArgument(argument)
    else:
        return argument

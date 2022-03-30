class Function:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, *args):
        from pyop3.exprs import FunctionCall

        return FunctionCall(self, args)

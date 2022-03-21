class Dat:
    def __init__(self, name):
        self.name = name
    
    def __getitem__(self, domain):
        """You can index a dat with a domain to get an argument to a loop."""
        return DatArgument(self, domain)


class DatArgument:

    def __init__(self, dat, domain):
        self.dat = dat
        self.domain = domain

    def __str__(self):
        return self.dat.name

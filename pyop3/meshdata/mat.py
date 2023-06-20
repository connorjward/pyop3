from pyop3.meshdata.base import MeshDataCarrier
class Mat(MeshDataCarrier):
    def __init__(self, spaces):
        raise NotImplementedError

    @property
    def array(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def comm(self):
        raise NotImplementedError

    @property
    def internal_comm(self):
        raise NotImplementedError

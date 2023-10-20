import abc
import contextlib


class OffloadingDevice(abc.ABC):
    pass


class CPUDevice(OffloadingDevice):
    pass


class GPUDevice(OffloadingDevice, abc.ABC):
    def __init__(self, num_threads=32):
        self.num_threads = num_threads


class CUDADevice(GPUDevice):
    pass


class OpenCLDevice(GPUDevice):
    pass


host_device = CPUDevice()
offloading_device = host_device


@contextlib.contextmanager
def offloading(device: OffloadingDevice):
    global offloading_device

    orig_offloading_device = offloading_device
    if not isinstance(orig_offloading_device, CPUDevice):
        raise NotImplementedError("Not sure what to do when offloading from not a CPU")

    offloading_device = device
    yield
    offloading_device = orig_offloading_device

import collections
import weakref

import numpy as np

from pyop3.dtypes import ScalarType
from pyop3.device import host_device, offloading_device, OffloadingDevice

try:
    import pycuda
except ImportError:
    pycuda = None

try:
    import pyopencl
except ImportError:
    pyopencl = None


class NoValidDevicesError(RuntimeError):
    pass


class InvalidDeviceError(RuntimeError):
    pass


class MirroredArray:
    """An array that is transparently available on different devices."""

    def __init__(self, data_or_shape, dtype=None):
        # option 1: passed shape
        if isinstance(data_or_shape, tuple):
            shape = data_or_shape
            if not dtype:
                dtype = ScalarType
            data_per_device = weakref.WeakKeyDictionary()
            device_validity = collections.defaultdict(bool)
        # option 2: passed a numpy array
        elif isinstance(data_or_shape, np.ndarray):
            data = data_or_shape
            shape = data.shape
            if not dtype:
                dtype = data.dtype
            data_per_device = weakref.WeakKeyDictionary({host_device: np.asarray(data, dtype)})
            device_validity = collections.defaultdict(bool, {host_device: True})
        # option 3: passed a CUDA array
        elif pycuda and isinstance(data_or_shape, pycuda.gpuarray.GPUArray):
            # TODO I suppose the device could be passed as a kwarg
            if not isinstance(offloading_device, OffloadingDevice.CUDADevice):
                raise InvalidDeviceError(
                    "Cannot pass a CUDA array to the MirroredArray constructor "
                    "outside a CUDA offloading context"
                )
            data = data_or_shape
            shape = data.shape
            if not dtype:
                dtype = data.dtype
            data_per_device = weakref.WeakKeyDictionary({offloading_device: data.astype(dtype)})
            device_validity = collections.defaultdict(bool, {offloading_device: True})
        # option 4: passed an OpenCL array
        elif pyopencl and isinstance(data_or_shape, pyopencl.array.Array):
            if not isinstance(offloading_device, OffloadingDevice.OpenCLDevice):
                raise InvalidDeviceError(
                    "Cannot pass an OpenCL array to the MirroredArray constructor "
                    "outside an OpenCL offloading context"
                )
            data = data_or_shape
            shape = data.shape
            if not dtype:
                dtype = data.dtype
            data_per_device = weakref.WeakKeyDictionary({offloading_device: data.astype(dtype)})
            device_validity = collections.defaultdict(bool, {offloading_device: True})
        else:
            raise ValueError("Unexpected arguments encountered")

        self.shape = shape
        self.dtype = dtype
        self._data_per_device = data_per_device
        self._device_validity = device_validity

        # counter used to keep track of modifications
        self.state = 0

    @property
    def data(self):
        return self.data_rw

    @property
    def data_rw(self):
        self.state += 1
        self._ensure_valid_on_device(offloading_device)
        self._invalidate_other_devices(offloading_device)
        return self._as_rw_array(self._device_array(offloading_device))

    @property
    def data_ro(self):
        self._ensure_valid_on_device(offloading_device)
        return self._as_ro_array(self._device_array(offloading_device))

    @property
    def data_wo(self):
        self.state += 1
        self._invalidate_other_devices(offloading_device)
        self._device_validity[offloading_device] = True
        return self._as_wo_array(self._device_array(offloading_device))

    @property
    def ptr_rw(self):
        return self._as_ptr(self.data_rw)

    @property
    def ptr_ro(self):
        return self._as_ptr(self.data_ro)

    @property
    def ptr_wo(self):
        return self._as_ptr(self.data_wo)

    @property
    def size(self):
        return np.prod(self.shape, dtype=int)

    def _ensure_valid_on_device(device):
        if not self._device_validity[device]:
            if self._device_validity[host_device]:
                self._host_to_device_copy(device)
            else:
                self._device_to_host_copy(self._first_valid_device)
                self._device_validity[host_device] = True
                self._host_to_device_copy(device)
            self._device_validity[device] = True

    def _invalidate_other_devices(device):
        for dev in self._device_validity.keys():
            if dev is not device:
                self._device_validity[dev] = False

    @functools.singledispatchmethod
    def _host_to_device_copy(self, device):
        raise TypeError(f"No handler registered for {type(device).__name__}")

    @host_to_device_copy.register(OffloadingDevice.CPUDevice)
    def _(self, device):
        if device is host_device:
            return
        else:
            raise NotImplementedError("Cannot offload to other CPUs")

    @host_to_device_copy.register(OffloadingDevice.CUDADevice)
    def _(self, device):
        self._data_per_device[device].set(self._data_per_device[host_device])

    @host_to_device_copy.register
    def _(self, device: OffloadingDevice.OpenCLDevice):
        self._data_per_device[device].set(self._data_per_device[host_device])

    @functools.singledispatchmethod
    def device_to_host_copy(self, device):
        raise TypeError(f"No handler registered for {type(device).__name__}")

    @device_to_host_copy.register
    def _(self, device: OffloadingDevice.CPUDevice):
        if device is host_device:
            return
        else:
            raise NotImplementedError("Cannot offload to other CPUs")

    @device_to_host_copy.register
    def _(self, device: OffloadingDevice.CUDADevice):
        self._data_per_device[device].get(self._data_per_device[host_device])

    @device_to_host_copy.register
    def _(self, device: OffloadingDevice.OpenCLDevice):
        self._data_per_device[device].get(self._data_per_device[host_device])

    @property
    def _first_valid_device(self):
        for device, valid in self._device_validity.items():
            if valid:
                return device
        raise NoValidDevicesError("No valid devices found")

    def _device_array(self, device):
        try:
            return self._data_per_device[device]
        except KeyError:
            if isinstance(device, OffloadingDevice.CPUDevice):
                data = self._alloc_cpu()
            elif isinstance(device, OffloadingDevice.CUDADevice):
                data = self._alloc_cuda()
            elif isinstance(device, OffloadingDevice.OpenCLDevice):
                data = self._alloc_opencl(device.queue)
            else:
                raise AssertionError
            return self._data_per_device.setdefault(offloading_device, data)

    def _alloc_cpu(self):
        return np.zeros(self.shape, self.dtype)

    def _alloc_cuda(self):
        return pycuda.gpuarray.zeros(
            shape=self.shape, dtype=self.dtype
        )

    def _alloc_opencl(self, queue):
        raise return pyopencl.array.zeros(
            queue, shape=self.shape, dtype=self.dtype
        )

    @staticmethod
    def _as_rw_array(array):
        # pycuda and pyopencl are optional dependencies so can't be singledispatch-ed
        if isinstance(array, np.ndarray):
            rw_array = array.view()
            rw_array.setflags(write=True)
        elif pycuda and isinstance(array, pycuda.gpuarray.GPUArray) or pyopencl and isinstance(array, pyopencl.array.Array):
            rw_array = array
        else:
            raise TypeError(f"No handler provided for {type(array).__name__}")
        return rw_array

    @staticmethod
    def _as_ro_array(array):
        # pycuda and pyopencl are optional dependencies so can't be singledispatch-ed
        if isinstance(array, np.ndarray):
            ro_array = array.view()
            ro_array.setflags(write=False)
        elif pycuda and isinstance(array, pycuda.gpuarray.GPUArray) or pyopencl and isinstance(array, pyopencl.array.Array):
            # don't have specific readonly arrays
            ro_array = array
        else:
            raise TypeError(f"No handler provided for {type(array).__name__}")
        return ro_array

    @staticmethod
    def _as_wo_array(array):
        return self._as_rw_array(array)

    @staticmethod
    def _as_ptr(array):
        if isinstance(array, np.ndarray):
            return array.ctypes.data
        elif pycuda and isinstance(array, pycuda.gpuarray.GPUArray):
            return array.gpudata
        elif pyopencl and isinstance(array, pyopencl.array.Array):
            return array.data
        else:
            raise TypeError(f"No handler provided for {type(array).__name__}")

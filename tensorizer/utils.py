import contextlib
import contextvars
import threading
from typing import (
    Callable,
    ContextManager,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

import torch

try:
    import resource
except ImportError:
    resource = None

import psutil

try:
    import pynvml

    try:
        pynvml.nvmlInit()
    except pynvml.nvml.NVMLError_LibraryNotFound:
        pynvml = None
except ImportError:
    pynvml = None

__all__ = [
    "convert_bytes",
    "get_device",
    "GlobalGPUMemoryUsage",
    "TorchGPUMemoryUsage",
    "CPUMemoryUsage",
    "MemoryUsage",
    "get_mem_usage",
    "get_gpu_name",
    "no_init_or_tensor",
]


# Silly function to convert to human bytes
def convert_bytes(num, decimal=True) -> str:
    """
    Convert bytes to MB, GB, etc.

    Args:
        num: Quantity of bytes to format.
        decimal: Whether to use decimal or binary units
            (e.g. KB = 1000 bytes vs. KiB = 1024 bytes).

    Returns:
        A string in the format ``<n> <units>`` (e.g. ``123.4 MB``).
    """
    if decimal:
        step_unit = 1000.0
        units = ("bytes", "KB", "MB", "GB", "TB", "PB")
    else:
        step_unit = 1024.0
        units = ("bytes", "KiB", "MiB", "GiB", "TiB", "PiB")

    for unit in units[:-1]:
        if num < step_unit:
            break
        num /= step_unit
    else:
        unit = units[-1]
    return "%3.1f %s" % (num, unit)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlobalGPUMemoryUsage(NamedTuple):
    """Total memory usage statistics across all processes for a single GPU."""

    total: int
    free: int
    used: int

    @classmethod
    def now(cls, device=None) -> Optional["GlobalGPUMemoryUsage"]:
        """
        Capture a snapshot of the current total memory usage on a single GPU.

        Args:
            device: The GPU to gather memory statistics for.
                If None, the current GPU is used,
                as determined by ``torch.cuda.current_device()``.

        Returns:
            A tuple of (`total`, `free`, `used`) VRAM in bytes, if possible.

            None if neither PyTorch >=1.10 nor pynvml is available.
        """
        if not torch.cuda.is_available():
            return None

        # torch.cuda.mem_get_info() was introduced in PyTorch 1.10
        mem_get_info = getattr(torch.cuda, "mem_get_info", None)
        if mem_get_info is not None:
            free, total = mem_get_info(device)
            return cls(total, free, total - free)
        elif pynvml is not None:
            # Normalize the device to an int
            if isinstance(device, (int, str, bytes)):
                device = torch.device(device)
            if isinstance(device, torch.device):
                if device.type == "cpu":
                    return None
                else:
                    device = device.index
            if device is None:
                device = torch.cuda.current_device()
            nvml_device = pynvml.nvmlDeviceGetHandleByIndex(device)
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
            return cls(gpu_info.total, gpu_info.free, gpu_info.used)
        else:
            return None

    def __str__(self):
        return "GPU: (U: {:,}MiB F: {:,}MiB T: {:,}MiB)".format(
            self.used >> 20, self.free >> 20, self.total >> 20
        )


class TorchGPUMemoryUsage(NamedTuple):
    """Memory usage statistics for PyTorch on a single GPU."""

    reserved: int
    reserved_max: int
    used: int
    used_max: int

    @classmethod
    def now(cls, device=None) -> Optional["TorchGPUMemoryUsage"]:
        """
        Capture a snapshot of the current total memory usage on a single GPU.

        Args:
            device: The GPU to gather memory statistics for.
                If None, the current GPU is used,
                as determined by ``torch.cuda.current_device()``.

        Returns:
            A tuple of (`reserved`, `reserved_max`, `used`, `used_max`)
            memory statistics for PyTorch in bytes, if possible.

            None if CUDA isn't available.
        """
        if torch.cuda.is_available():
            stats = torch.cuda.memory.memory_stats(device)
            return cls(
                stats.get("reserved_bytes.all.current", 0),
                stats.get("reserved_bytes.all.peak", 0),
                stats.get("allocated_bytes.all.current", 0),
                stats.get("allocated_bytes.all.peak", 0),
            )
        else:
            return None

    def __str__(self):
        return "TORCH: (R: {:,}MiB/{:,}MiB, A: {:,}MiB/{:,}MiB)".format(
            self.reserved >> 20,
            self.reserved_max >> 20,
            self.used >> 20,
            self.used_max >> 20,
        )


class CPUMemoryUsage(NamedTuple):
    """Memory usage statistics for CPU RAM."""

    maxrss: int
    free: int

    @classmethod
    def now(cls) -> "CPUMemoryUsage":
        """
        Capture a snapshot of the current CPU RAM usage.

        Returns:
            A tuple of (`maxrss`, `free`) RAM statistics in bytes,
            where maxrss is the max resident set size of the current process.

            On Unix, the system call ``getrusage(2)`` is used
            to measure the maxrss, so the granularity is 1024 bytes,
            but the unit is still bytes.
        """
        if resource is not None:
            maxrss = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            ) << 10
        else:
            process = psutil.Process()
            maxrss = process.memory_info().rss + sum(
                p.memory_info().rss for p in process.children(True)
            )
        vmem = psutil.virtual_memory()
        return cls(maxrss, vmem.free)

    def __str__(self):
        return "CPU: (maxrss: {:,}MiB F: {:,}MiB)".format(
            self.maxrss >> 20, self.free >> 20
        )


class MemoryUsage(NamedTuple):
    """
    Combined statistics for CPU, total GPU, and PyTorch memory usage.

    Gathers `CPUMemoryUsage`, `GlobalGPUMemoryUsage`, and `TorchGPUMemoryUsage`
    together in one tuple.
    """

    cpu: CPUMemoryUsage
    gpu: Optional[GlobalGPUMemoryUsage]
    torch: Optional[TorchGPUMemoryUsage]

    @classmethod
    def now(cls, device=None):
        """
        Capture a snapshot of CPU, total GPU, and PyTorch memory usage.
        If GPU memory usage statistics are not available,
        the `gpu` and `torch` fields of the resulting tuple are None.

        Args:
            device: The GPU to gather both total and PyTorch-specific
                memory statistics for. If None, the current GPU is used,
                as determined by ``torch.cuda.current_device()``.

        Returns:
            A tuple of (`cpu`, `gpu`, `torch`) memory statistics.
            If GPU memory usage statistics are not available,
            the `gpu` and `torch` fields are None.

            See the respective classes, `CPUMemoryUsage`,
            `GlobalGPUMemoryUsage`, and `TorchGPUMemoryUsage`,
            for more information on each component.
        """
        gpu_info = torch_info = None
        try:
            gpu_info = GlobalGPUMemoryUsage.now(device)
            torch_info = TorchGPUMemoryUsage.now(device)
        except AssertionError:
            pass
        return cls(CPUMemoryUsage.now(), gpu_info, torch_info)

    def __str__(self):
        return " ".join(str(item) for item in self if item)


def get_mem_usage() -> str:
    """
    Captures and formats memory usage statistics for the CPU, GPU, and PyTorch.

    Equivalent to ``str(MemoryUsage.now())``.

    Returns:
        A formatted string summarizing memory usage
        across the CPU, GPU, and PyTorch.
    """
    return str(MemoryUsage.now())


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return "N/A"


Model = TypeVar("Model")


def no_init_or_tensor(
    loading_code: Optional[Callable[..., Model]] = None
) -> Union[Model, ContextManager]:
    """
    Suppress the initialization of weights while loading a model.

    Can either directly be passed a callable containing model-loading code,
    which will be evaluated with weight initialization suppressed,
    or used as a context manager around arbitrary model-loading code.

    Args:
        loading_code: Either a callable to evaluate
            with model weight initialization suppressed,
            or None (the default) to use as a context manager.

    Returns:
        The return value of `loading_code`, if `loading_code` is callable.

        Otherwise, if `loading_code` is None, returns a context manager
        to be used in a `with`-statement.

    Examples:
        As a context manager::

            from transformers import AutoConfig, AutoModelForCausalLM
            config = AutoConfig("EleutherAI/gpt-j-6B")
            with no_init_or_tensor():
                model = AutoModelForCausalLM.from_config(config)

        Or, directly passing a callable::

            from transformers import AutoConfig, AutoModelForCausalLM
            config = AutoConfig("EleutherAI/gpt-j-6B")
            model = no_init_or_tensor(lambda: AutoModelForCausalLM.from_config(config))
    """
    if loading_code is None:
        return _NoInitOrTensorImpl.context_manager()
    elif callable(loading_code):
        with _NoInitOrTensorImpl.context_manager():
            return loading_code()
    else:
        raise TypeError(
            "no_init_or_tensor() expected a callable to evaluate,"
            " or None if being used as a context manager;"
            f' got an object of type "{type(loading_code).__name__}" instead.'
        )


class _NoInitOrTensorImpl:
    # Implementation of the thread-safe, async-safe, re-entrant context manager
    # version of no_init_or_tensor().
    # This class essentially acts as a namespace.
    # It is not instantiable, because modifications to torch functions
    # inherently affect the global scope, and thus there is no worthwhile data
    # to store in the class instance scope.
    _MODULES = (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)
    _MODULE_ORIGINALS = tuple((m, m.reset_parameters) for m in _MODULES)
    _ORIGINAL_EMPTY = torch.empty

    is_active = contextvars.ContextVar(
        "_NoInitOrTensorImpl.is_active", default=False
    )
    _count_active: int = 0
    _count_active_lock = threading.Lock()

    @classmethod
    @contextlib.contextmanager
    def context_manager(cls):
        if cls.is_active.get():
            yield
            return

        with cls._count_active_lock:
            cls._count_active += 1
            if cls._count_active == 1:
                for mod in cls._MODULES:
                    mod.reset_parameters = cls._disable(mod.reset_parameters)
                # When torch.empty is called, make it map to meta device by replacing
                # the device in kwargs.
                torch.empty = cls._meta_empty
        reset_token = cls.is_active.set(True)

        try:
            yield
        finally:
            cls.is_active.reset(reset_token)
            with cls._count_active_lock:
                cls._count_active -= 1
                if cls._count_active == 0:
                    torch.empty = cls._ORIGINAL_EMPTY
                    for mod, original in cls._MODULE_ORIGINALS:
                        mod.reset_parameters = original

    @staticmethod
    def _disable(func):
        def wrapper(*args, **kwargs):
            # Behaves as normal except in an active context
            if not _NoInitOrTensorImpl.is_active.get():
                return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def _meta_empty(*args, **kwargs):
        # Behaves as torch.empty except in an active context
        if _NoInitOrTensorImpl.is_active.get():
            kwargs["device"] = "meta"
        return _NoInitOrTensorImpl._ORIGINAL_EMPTY(*args, **kwargs)

    __init__ = None

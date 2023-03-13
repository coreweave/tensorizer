import torch
try:
    import resource
except ImportError:
    resource = None

import pynvml
import psutil

try:
    pynvml.nvmlInit()
    nvml_device = pynvml.nvmlDeviceGetHandleByIndex(0)
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None


# Silly function to convert to human bytes
def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    step_unit = 1000.0

    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mem_usage() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1e6)
        gpu_free = int(gpu_info.free / 1e6)
        gpu_used = int(gpu_info.used / 1e6)
        gpu_str = (
            f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb "
            f"T: {gpu_total:,}mb) "
        )
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1e6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1e6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1e6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1e6)
        torch_str = (
            f"TORCH: (R: {torch_reserved_gpu:,}mb/"
            f"{torch_reserved_max:,}mb, "
            f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
        )
    except AssertionError:
        pass
    cpu_maxrss = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e3
    )
    cpu_shared = int(resource.getrusage(resource.RUSAGE_SELF).ru_ixrss / 1e3)
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1e6)
    return (
        f"CPU: (maxrss: {cpu_maxrss:,}mb idrss: {cpu_shared} F: {cpu_free:,}mb) "
        f"{gpu_str}"
        f"{torch_str}"
    )

def get_vram_ram_usage_str() -> str:
    return f"{get_mem_usage()} {get_vram_usage_str()}"

def get_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"


def no_init_or_tensor(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy
    original_empty = torch.empty

    # when torch.empty is called, make it map to meta device by replacing the
    # device in kwargs.
    torch.empty = lambda *args, **kwargs: original_empty(
        *args, **{**kwargs, "device": "meta"}
    )

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]
    torch.empty = original_empty

    return result

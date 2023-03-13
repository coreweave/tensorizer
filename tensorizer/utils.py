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


def _get_gpu_mem_usage():
    """
    Returns a tuple of (total_vram, free_vram, used_vram) in bytes if possible.
    Returns None if neither PyTorch >=1.10 nor pynvml is available.
    """
    mem_get_info = getattr(torch.cuda, "mem_get_info", None)
    if mem_get_info is not None:
        free, total = mem_get_info()
        return total, free, total - free
    elif pynvml is not None:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        return gpu_info.total, gpu_info.free, gpu_info.used
    else:
        return None


def get_mem_usage() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        gpu_info = _get_gpu_mem_usage()
        if gpu_info is not None:
            gpu_total, gpu_free, gpu_used = gpu_info
            gpu_total >>= 20
            gpu_free >>= 20
            gpu_used >>= 20
            gpu_str = (
                f"GPU: (U: {gpu_used:,}MiB F: {gpu_free:,}MiB T: {gpu_total:,}MiB) "
            )
        torch_reserved_gpu = torch.cuda.memory.memory_reserved() >> 20
        torch_reserved_max = torch.cuda.memory.max_memory_reserved() >> 20
        torch_used_gpu = torch.cuda.memory_allocated() >> 20
        torch_max_used_gpu = torch.cuda.max_memory_allocated() >> 20
        torch_str = (
            f"TORCH: (R: {torch_reserved_gpu:,}MiB/"
            f"{torch_reserved_max:,}MiB, "
            f"A: {torch_used_gpu:,}MiB/{torch_max_used_gpu:,}MiB)"
        )
    except AssertionError:
        pass
    if resource is not None:
        cpu_maxrss = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        ) >> 10
    else:
        process = psutil.Process()
        cpu_maxrss = (
            process.memory_info().rss
            + sum(p.memory_info().rss for p in process.children(True))
        ) >> 20
    cpu_vmem = psutil.virtual_memory()
    cpu_free = cpu_vmem.free >> 20
    return (
        f"CPU: (maxrss: {cpu_maxrss:,}MiB F: {cpu_free:,}MiB) "
        f"{gpu_str}{torch_str}"
    )


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

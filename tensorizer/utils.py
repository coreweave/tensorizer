import resource
import torch

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

def get_ram_usage_str() -> str:
    if resource is not None:
        maxrss_b4 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1000 * 1000
        )
        maxrss_b4_gb = f"{maxrss_b4:0.2f}gb CPU RAM used"
    else:
        maxrss_b4_gb = "unknown CPU RAM used"
    return maxrss_b4_gb


def get_vram_usage_str() -> str:
    if torch.cuda.is_available():
        gb_gpu = int(
            torch.cuda.get_device_properties(0).total_memory
            / (1000 * 1000 * 1000)
        )
        return f"{str(gb_gpu)}gb"
    return "N/A"


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
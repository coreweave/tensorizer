import time
import os
from typing import Optional, Dict, List

import torch
from collections import defaultdict
from pathlib import Path
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

MODEL_ID = os.environ.get("MODEL_ID", "EleutherAI/gpt-neo-125M")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "./models"))
USE_FP16 = os.environ.get("USE_FP16", False)
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", 1))

if USE_FP16:
    MODEL_PATH /= "fp16"
    # extra_args = {"revision": "float16", "torch_dtype": torch.float16}
    dtype = torch.float16
else:
    # extra_args = {}
    dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
).to(dtype)

AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(MODEL_PATH / "hf")


def shared_pointers(tensors) -> List:
    """Find tensors that share the same data."""

    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    shared = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            shared.append(names)
    return shared


def convert_shared_tensors(
    pt_filename: Optional[str] = None,
    state_dict=None
) -> Dict:
    """
    Clone data shared between tensors.

    If no state_dict is given, then it will be read from the model file saved
    at pt_filename.

    Shared models are only supported by passing in the state dict.
    """

    if state_dict is None:
        loaded = torch.load(pt_filename, map_location="cpu")
        state_dict = loaded["state_dict"]

    shared = shared_pointers(state_dict)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            state_dict[name] = state_dict[name].clone()

    # For tensors to be contiguous
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    return state_dict


def save_hf() -> float:
    dest = MODEL_PATH / "hf"
    dest.mkdir(parents=True, exist_ok=True)

    start = time.time()
    model.save_pretrained(MODEL_PATH / "hf")
    end = time.time()

    print(f"Huggingface saved the model in {end - start:0.2f}s")
    return end - start


def save_tzr() -> float:
    start = time.time()
    serializer = TensorSerializer(MODEL_PATH / "model.tensors")
    serializer.write_module(model)
    serializer.close()
    end = time.time()

    print(f"Serialized {serializer.total_tensor_bytes} btyes in {end - start:0.2f}s")
    return end - start


def save_st() -> float:
    sf_filename = MODEL_PATH / "hf" / "model.safetensors"

    start = time.time()

    state_dict = convert_shared_tensors(state_dict=model.state_dict())
    save_file(state_dict, sf_filename, metadata={"format": "pt"})

    end = time.time()

    print(f"Saved the safetensors file in {end - start:0.2f}s")
    return end - start


# Huggingface save
hf_times = [save_hf() for _ in range(NUM_TRIALS)]
print("Average huggingface save:", sum(hf_times) / len(hf_times))
print("~" * 25)

# Tensorizer save
tzr_times = [save_tzr() for _ in range(NUM_TRIALS)]
print("Average tensorizer serialization:", sum(tzr_times) / len(tzr_times))
print("~" * 25)

# Safetensors save
st_times = [save_st() for _ in range(NUM_TRIALS)]
print("Average safetensors serialization:", sum(st_times) / len(st_times))

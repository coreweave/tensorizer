import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from tensorizer import TensorDeserializer
from tensorizer.utils import convert_bytes, get_mem_usage

# disable missing keys and unexpected key warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Improve safetensors performance
os.environ["SAFETENSORS_FAST_GPU"] = "1"

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def convert_to_bool(val: str) -> bool:
    return val.strip().lower() not in ("", "0", "no", "f", "false")


MODEL_PATH = Path(os.environ.get("MODEL_PATH", f"./models"))
TZR_PATH = MODEL_PATH / "model.tensors"
HF_PATH = MODEL_PATH / "hf"
RES_PATH = Path(os.environ.get("RES_PATH", "./results"))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", 1))
SKIP_HF = convert_to_bool(os.environ.get("SKIP_HF", ""))
SKIP_TZR = convert_to_bool(os.environ.get("SKIP_TZR", ""))
SKIP_ST = convert_to_bool(os.environ.get("SKIP_ST", ""))
SKIP_INFERENCE = convert_to_bool(os.environ.get("SKIP_INFERENCE", ""))
CURL_PATH = shutil.which("curl")


RES_PATH.mkdir(parents=True, exist_ok=True)
config = AutoConfig.from_pretrained(HF_PATH)

DEVICE = torch.device("cuda:0")


def run_inference(model):
    if SKIP_INFERENCE:
        return
    # Tokenize and generate
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH / "hf")
    input_ids = tokenizer.encode(
        "We the people, of the United States of America", return_tensors="pt"
    ).to(DEVICE)

    torch.manual_seed(100)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50, do_sample=True)

    print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")


def hf_load() -> float:
    print("~" * 25)

    before_mem = get_mem_usage()

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_PATH,
        # revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        config=config,
        use_safetensors=False,
        device_map="auto",
    )
    duration = time.time() - start

    after_mem = get_mem_usage()
    print(f"Loaded huggingface model in {duration:0.2f}s")
    print(f"Memory usage before: {before_mem}")
    print(f"Memory usage after: {after_mem}")
    run_inference(model)

    return duration


def tzr_load() -> float:
    print("~" * 25)
    before_mem = get_mem_usage()

    start = time.time()
    # This ensures that the model is not initialized.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    deserializer = TensorDeserializer(TZR_PATH, plaid_mode=True, device=DEVICE)
    deserializer.load_into_module(model)
    end = time.time()

    # Brag about how fast we are.
    total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
    duration = end - start
    per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
    after_mem = get_mem_usage()
    deserializer.close()
    print(f"Deserialized {total_bytes_str} in {duration:0.2f}s, {per_second}/s")
    print(f"Memory usage before: {before_mem}")
    print(f"Memory usage after: {after_mem}")
    run_inference(model)

    return duration


def st_load() -> float:
    print("~" * 25)

    before_mem = get_mem_usage()

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_PATH,
        low_cpu_mem_usage=True,
        config=config,
        use_safetensors=True,
        device_map="auto",
    )
    end = time.time()

    after_mem = get_mem_usage()
    duration = end - start

    print(f"Deserialized safetensors in {duration:0.2f}s")
    print(f"Memory usage before: {before_mem}")
    print(f"Memory usage after: {after_mem}")

    run_inference(model)

    return duration


if not SKIP_TZR:
    print("\nRunning Tensorizer...")
    tzr_times = [tzr_load() for _ in range(NUM_TRIALS)]
    print(
        "Average tensorizer deserialization:", sum(tzr_times) / len(tzr_times)
    )
    with open(RES_PATH / f"tzr_times_{time.time()}.npy", "wb") as f:
        np.save(f, np.array(tzr_times))

if not SKIP_HF:
    print("\nRunning Huggingface...")
    hf_times = [hf_load() for _ in range(NUM_TRIALS)]
    print("Average huggingface load:", sum(hf_times) / len(hf_times))
    with open(RES_PATH / f"hf_times_{time.time()}.npy", "wb") as f:
        np.save(f, np.array(hf_times))

if not SKIP_ST:
    print("\nRunning Safetensors...")
    st_times = [st_load() for _ in range(NUM_TRIALS)]
    print("Average safetensors load:", sum(st_times) / len(st_times))
    with open(RES_PATH / f"st_times_{time.time()}.npy", "wb") as f:
        np.save(f, np.array(st_times))

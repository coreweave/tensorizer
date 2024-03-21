import argparse
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import tensorizer.serialization
from tensorizer import DecryptionParams, TensorDeserializer
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor

parser = argparse.ArgumentParser("deserialize")
parser.add_argument("--source", default=None, help="local path or URL")
parser.add_argument("--model-ref", default="EleutherAI/gpt-j-6B")
parser.add_argument("--no-plaid", action="store_true")
parser.add_argument("--lazy-load", action="store_true")
parser.add_argument("--verify-hash", action="store_true")
parser.add_argument("--encryption", action="store_true")
parser.add_argument("--viztracer", action="store_true")
parser.add_argument("--num-readers", type=int, default=1)

args = parser.parse_args()

model_ref = args.model_ref
# To run this at home, swap this with the line below for a smaller example:
# model_ref = "EleutherAI/gpt-neo-125M"
model_name = model_ref.split("/")[-1]

if args.source is None:
    args.source = f"s3://{s3_bucket}/{model_name}.tensors"

tracer = None
if args.viztracer:
    import viztracer

    tracer = viztracer.VizTracer(pid_suffix=True)

decryption_params = None
if args.encryption:
    decryption_params = DecryptionParams.from_string(
        os.getenv("SUPER_SECRET_STRONG_PASSWORD")
    )

config = AutoConfig.from_pretrained(model_ref)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# This ensures that the pretrained model weights are not initialized,
# and non-persistent buffers (generated at runtime) are on the correct device.
with torch.device(device), no_init_or_tensor():
    model = AutoModelForCausalLM.from_config(config)

input(f"PID {os.getpid()}")
print(f"Deserializing to {device}:")
before_mem = get_mem_usage()


# Lazy load the tensors from S3 into the model.
if tracer is not None:
    tracer.start()
start = time.perf_counter()
deserializer = TensorDeserializer(
    args.source,
    device=device,
    plaid_mode=not args.no_plaid,
    lazy_load=args.lazy_load,
    encryption=decryption_params,
    num_readers=args.num_readers,
    verify_hash=args.verify_hash,
)
deserializer.load_into_module(model)
end = time.perf_counter()
if tracer is not None:
    tracer.stop()
    tracer.save()
after_mem = get_mem_usage()

# Brag about how fast we are.
total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
duration = end - start
per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
deserializer.close()
print(f"Deserialized {total_bytes_str} in {duration:0.2f}s, {per_second}/s")
print(f"Memory usage before: {before_mem}")
print(f"Memory usage after: {after_mem}")

# Tokenize and generate
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_ref)
eos = tokenizer.eos_token_id
input_ids = tokenizer.encode(
    "¡Hola! Encantado de conocerte. hoy voy a", return_tensors="pt"
).to(device)

with torch.no_grad():
    output = model.generate(
        input_ids, max_new_tokens=50, do_sample=True, pad_token_id=eos
    )

print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

if tensorizer.serialization._enable_perf_stats:
    perf_stats = tensorizer.serialization._get_perf_stats()
    to_device_bytes = perf_stats["tensor_to_device_bytes"]
    to_device_secs = perf_stats["tensor_to_device_secs"]
    to_device_speed = to_device_bytes / to_device_secs if to_device_secs else 0
    readinto_bytes = perf_stats["file_readinto_bytes"]
    readinto_secs = perf_stats["file_readinto_secs"]
    readinto_speed = readinto_bytes / readinto_secs if readinto_secs else 0

    print(
        f"to CUDA stats: {to_device_bytes} bytes in"
        f" {to_device_secs:.3f}s, {convert_bytes(to_device_speed, False)}/s"
    )
    print(
        f"readinto stats: {readinto_bytes} bytes in"
        f" {readinto_secs:.3f}s, {convert_bytes(readinto_speed, False)}/s"
    )

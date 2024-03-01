import mmap
import os
import time

import torch
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM
from tensorizer._syscalls import try_fallocate

import viztracer

model_ref = "EleutherAI/gpt-j-6B"
# For less intensive requirements, swap above with the line below:
# model_ref = "EleutherAI/gpt-neo-125M"
model_name = model_ref.split("/")[-1]

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    model_ref,
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print("Done loading. To Cuda")
model = model.to("cuda")

tracer = None
if os.environ.get("VIZTRACER"):
    tracer = viztracer.VizTracer(pid_suffix=True)
    tracer.start()
start = time.perf_counter()
print("Serializing")
fd = os.open("/scratch/out", os.O_RDWR | os.O_CREAT)
try_fallocate(fd, 0, 12219504910)
m = mmap.mmap(fd, 12219504910, flags=mmap.MAP_SHARED)
# m = '/scratch/out'
serializer = TensorSerializer(m)
serializer.write_module(model)
serializer.close()
end = time.perf_counter()
if tracer is not None:
    tracer.stop()
    tracer.save()

print(f"Serializing took {(end-start):.3f}s")

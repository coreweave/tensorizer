import os
import time
import torch
import viztracer
from tensorizer import TensorSerializer, EncryptionParams
from transformers import AutoModelForCausalLM

model_ref = "EleutherAI/gpt-j-6B"
# For less intensive requirements, swap above with the line below:
# model_ref = "EleutherAI/gpt-neo-125M"
model_name = model_ref.split("/")[-1]
# Change this to your S3 bucket.
s3_bucket = "bucket"
s3_uri = f"s3://{s3_bucket}/{model_name}.tensors"
s3_uri = '/scratch/gpt-j-6B.tensors.tmp'

source: str = os.getenv("SUPER_SECRET_STRONG_PASSWORD", "")
encryption_params = EncryptionParams.from_string(source) if source else None

model = AutoModelForCausalLM.from_pretrained(
    model_ref,
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

tracer = viztracer.VizTracer(pid_suffix=True, tracer_entries=int(2e7))
tracer.start()
start = time.perf_counter()

serializer = TensorSerializer(s3_uri, encryption=encryption_params)
serializer.write_module(model)
serializer.close()

end = time.perf_counter()
tracer.stop()
tracer.save()

print(f"Serialized model to {s3_uri} in {end - start:.2f} seconds.")
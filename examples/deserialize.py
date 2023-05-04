import time
import torch
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor, convert_bytes, get_mem_usage

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_ref = "EleutherAI/gpt-j-6B"
# To run this at home, swap this with the line below for a smaller example:
# model_ref = "EleutherAI/gpt-neo-125M"
model_name = model_ref.split("/")[-1]
# Change this to your S3 bucket.
s3_bucket = "bucket"
s3_uri = f"s3://{s3_bucket}/{model_name}.tensors"

config = AutoConfig.from_pretrained(model_ref)

# This ensures that the model is not initialized.
with no_init_or_tensor():
    model = AutoModelForCausalLM.from_config(config)

before_mem = get_mem_usage()

# Lazy load the tensors from S3 into the model.
start = time.time()
deserializer = TensorDeserializer(s3_uri, plaid_mode=True)
deserializer.load_into_module(model)
end = time.time()

# Brag about how fast we are.
total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
duration = end - start
per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
after_mem = get_mem_usage()
deserializer.close()
print(f"Deserialized {total_bytes_str} in {end - start:0.2f}s, {per_second}/s")
print(f"Memory usage before: {before_mem}")
print(f"Memory usage after: {after_mem}")

# Tokenize and generate
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_ref)
eos = tokenizer.eos_token_id
input_ids = tokenizer.encode(
    "¡Hola! Encantado de conocerte. hoy voy a", return_tensors="pt"
).to("cuda")

with torch.no_grad():
    output = model.generate(
        input_ids, max_new_tokens=50, do_sample=True, pad_token_id=eos
    )

print(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

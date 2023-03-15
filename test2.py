import torch
import os
import time
from tensorizer.serialization import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from collections import OrderedDict

# disable missing keys and unexpected key warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoModelForCausalLM, AutoTokenizer

model_ref = "EleutherAI/gpt-j-6B"
model_name = model_ref.split("/")[-1]
s3_uri = f"s3://bucket/{model_name}.tensors"

# This ensures that the model is not initialized.
model = no_init_or_tensor(
    lambda: AutoModelForCausalLM.from_pretrained(
        model_ref, state_dict=OrderedDict()
    )
)

# Lazy load the tensors from S3 into the model.
start = time.time()
deserializer = TensorDeserializer("test-gpt-j-6B.tensors",
                                  plaid_mode=True)
deserializer.load_into_module(model)
end = time.time()

# Brag about how fast we are.
print(f"Deserialized model in {end - start:0.2f} seconds")

# Tokenize and generate
tokenizer = AutoTokenizer.from_pretrained(model_ref)
input_ids = tokenizer.encode(
    "¡Hola! Encantado de conocerte. hoy voy a", return_tensors="pt"
).to("cuda")

with torch.no_grad():
    output = model.generate(
        input_ids, max_new_tokens=50, do_sample=True
    )

print(
    f"Test Output: {tokenizer.decode(output[0], skip_special_tokens=True)}"
)
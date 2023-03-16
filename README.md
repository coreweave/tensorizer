# tensorizer
Module, Model, and Tensor Serialization/Deserialization

## TLDR
Extremely fast model loads from HTTP/HTTPS and S3 endpoints. GPT-J
(`20GB`) loads at wire-speed (`~5GB/s`) on a 40GbE network, and is
only bottlenecked by the Linux kernel TCP stack.

## Rationale
CoreWeave and our customers use KNative to deploy models as serverless
functions. How long a model takes to load is a major factor in the latency
of KNative scale-up. `tensorizer` is a tool to serialize models and their
associated tensors into a single file that can be loaded quickly and
efficiently off an HTTP/HTTPS or S3 endpoint.

By not embedding the model in the container image, we can reduce the
container image size and the time it takes to load the model. This is
especially important for models that are large in size, such as
[EleutherAI/gpt-neox-20B](https://huggingface.co/EleutherAI/gpt-neox-20B)
that weighs in at `~40GB`.

This decoupling of the model from the container image also allows us to
update the model without having to rebuild the container image. This allows
us to quickly iterate on the model and deploy new versions without having
to wait for the container image to build or for the container image cache
to be populated.

`tensorizer` has S3 support, so we can store the serialized model in S3
object storage, and perform streaming loads from S3. This allows us to
stream the model directly from S3 into the container without having to
download the model to the container's local filesystem. This also
pertains to HTTP/HTTPS endpoints, as S3 is just an HTTP/HTTPS endpoint.

`tensorizer` also has support for loading models from a local filesystem,
so you can use it to serialize models locally and load them locally. This
is extremely fast, as the same principles that make it fast for HTTP/HTTPS
and S3 endpoints also apply to local filesystems.

## Basic Usage
Serialization is done with the `TensorSerializer` class. It takes a
`path_uri` argument that can be a local filesystem path, an HTTP/HTTPS
endpoint, or an S3 endpoint.

`write_module` is the main method of the `TensorSerializer` class. It
takes a `torch.nn.Module` and serializes the tensors to the `path_uri`
endpoint.

The below example serializes the `EleutherAI/gpt-j-6B` model to an S3
endpoint. It assumes that you have already configured your S3
credentials in `~/.s3cfg`.

```python
from transformers import AutoModelForCausalLM
from tensorizer.serialization import TensorSerializer

model_name = "EleutherAI/gpt-j-6B"
output_dir = model_name.split("/")[-1]
s3_uri = f"s3://bucket/{output_dir}.tensors"

model = AutoModelForCausalLM.from_pretrained(model_name)

serializer = TensorSerializer(s3_uri)
serializer.write_module(model)
```

Conversely, deserialization is done with the `TensorDeserializer` class.
It takes a `path_uri` argument that can be a local filesystem path, an
HTTP/HTTPS endpoint, or an S3 endpoint.

`load_into_module` is the main method of the `TensorDeserializer` class.
It takes a `torch.nn.Module` and loads the tensors from the `path_uri`
endpoint into the `torch.nn.Module`.

The below example loads the `EleutherAI/gpt-j-6B` model from an S3
endpoint.

```python
import torch
import os
import time
from tensorizer.serialization import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from collections import OrderedDict

# disable missing keys and unexpected key warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_ref = "EleutherAI/gpt-j-6B"
model_name = model_ref.split("/")[-1]
s3_uri = f"s3://bucket/{model_name}.tensors"

config = AutoConfig.from_pretrained(model_ref)

# This ensures that the model is not initialized.
model = no_init_or_tensor(
    lambda: AutoModelForCausalLM.from_pretrained(
        None, config=config, state_dict=OrderedDict()
    )
)

# Lazy load the tensors from S3 into the model.
start = time.time()
deserializer = TensorDeserializer(s3_uri, plaid_mode=True)
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
```

It should produce output similar to the following:
```
Deserialized model in 6.25 seconds
Test Output: ¡Hola! Encantado de conocerte. hoy voy a comentar por primera
vez una teoría de trineo, que quizá te parezca
algo desconocido, ya que en este mundo han
llegado a dominar tantos
```

More practical examples for the usage of `tensorizer` can be found in
[examples/hf_serialization.py](examples/hf_serialization.py),
where `df_main()` serializes models from
[HuggingFace Diffusers](https://github.com/huggingface/diffusers)
and `hf_main()` serializes
[HuggingFace Transformers](https://github.com/huggingface/transformers) models.

## Additional Features
`tensorizer` has a few additional features that make it more useful than
just a serialization/deserialization tool.

### Plaid Mode
`tensorizer` has a `plaid_mode` argument that can be passed to the
`TensorDeserializer` class. When `plaid_mode` is `True`, `tensorizer`
will load the tensors extremely fast. This is done by loading the tensors
into a `torch.nn.Module` that is not initialized, by overriding the
`__init__` method of the `torch.nn.Module` to do nothing.

The tensors are them loaded into a buffer, and the buffer is zero-copied
into the uninitialized `torch.nn.Module`. This is unsafe, and should only
be used in inference cases where the model is not being trained.

### `state_dict` Support
The `TensorDeserializer` object can be used as-is as a `state_dict` for
`torch.nn.Module.load_state_dict`. This is useful for loading the tensors
into a `torch.nn.Module` that is already initialized, or for inspection.

Keep in mind that `load_state_dict` is not a fast operation, and will
likely be much slower than `load_into_module`.

The `state_dict` can also be used to initialize a Huggingface Transfomers
AutoModel. But Hugginface Transformers performs three or more copies of
the data, so memory use will explode.
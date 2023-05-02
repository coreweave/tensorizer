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

## Installation

### From PyPI
`tensorizer` can be installed from PyPI with `pip`:
```bash
python -m pip install tensorizer
```

### From Source
You can also install `tensorizer` from source using `pip`.

To clone the repository and install `tensorizer` in
[editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs),
run:
```bash
git clone https://github.com/coreweave/tensorizer
cd tensorizer
python -m pip install -e .
```

Or, run the following for `pip` to install `tensorizer`
[directly from GitHub](https://pip.pypa.io/en/stable/topics/vcs-support/#git):
```bash
python -m pip install git+https://github.com/coreweave/tensorizer
```

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

**NOTE:** Loading and serializing `gpt-j-6B` will take a lot of CPU RAM,
up to `~20GB`. Additionally, when loading `gpt-j-6B` into a GPU, you
will need about `~16GB` of VRAM. If you don't have that much RAM or VRAM,
you can use the smaller `gpt-neo-125M` model instead.

**NOTE2:** The below examples require the `transformers` and `accelerate`
libraries. You can install them with `pip`:
```bash
python -m pip install transformers accelerate
```

[serialize.py](examples/serialize.py)
```python
import torch
from tensorizer import TensorSerializer
from transformers import AutoModelForCausalLM

model_ref = "EleutherAI/gpt-j-6B"
# For less intensive requirements, swap above with the line below:
# model_ref = "EleutherAI/gpt-neo-125M"
model_name = model_ref.split("/")[-1]
# Change this to your S3 bucket.
s3_bucket = "bucket"
s3_uri = f"s3://{s3_bucket}/{model_name}.tensors"

model = AutoModelForCausalLM.from_pretrained(
    model_ref,
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

serializer = TensorSerializer(s3_uri)
serializer.write_module(model)
serializer.close()
```

Conversely, deserialization is done with the `TensorDeserializer` class.
It takes a `path_uri` argument that can be a local filesystem path, an
HTTP/HTTPS endpoint, or an S3 endpoint.

`load_into_module` is the main method of the `TensorDeserializer` class.
It takes a `torch.nn.Module` and loads the tensors from the `path_uri`
endpoint into the `torch.nn.Module`.

The below example loads the `EleutherAI/gpt-j-6B` model from an S3
endpoint.

[deserialize.py](examples/deserialize.py)
```python
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
```

It should produce output similar to the following, with GPT-J-6B:
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

## Available Pre-Tensorized Models on the CoreWeave Cloud

The following models are available on the CoreWeave Cloud for free, and can be
used with the `TensorDeserializer` class. The S3 support defaults to the
`accel-object.ord1.coreweave.com` endpoint, and the bucket to use as `tensorized`.

We name the keys in the S3 bucket after the HuggingFace model identifier, and
append the `/fp16` suffix for the half-precision version.

For example, the S3 URI for the `EleutherAI/gpt-j-6B` model is:
`s3://tensorized/EleutherAI/gpt-j-6B/fp16/model.tensors`

The below table shows the available models and their S3 URIs.

### Large Language Models

| Model                                                                                   | Precision | S3 URI                                                              |
|-----------------------------------------------------------------------------------------|-----------|---------------------------------------------------------------------|
| [EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M)               | `fp32`    | `s3://tensorized/EleutherAI/gpt-neo-125M/model.tensors`             |
| [EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M)               | `fp16`    | `s3://tensorized/EleutherAI/gpt-neo-125M/fp16/model.tensors`        |
| [EleutherAI/gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)               | `fp32`    | `s3://tensorized/EleutherAI/gpt-neo-1.3B/model.tensors`             |
| [EleutherAI/gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)               | `fp16`    | `s3://tensorized/EleutherAI/gpt-neo-1.3B/fp16/model.tensors`        |
| [EleutherAI/gpt-neo-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B)               | `fp32`    | `s3://tensorized/EleutherAI/gpt-neo-2.7B/model.tensors`             |
| [EleutherAI/gpt-neo-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B)               | `fp16`    | `s3://tensorized/EleutherAI/gpt-neo-2.7B/fp16/model.tensors`        |
| [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B)                       | `fp32`    | `s3://tensorized/EleutherAI/gpt-j-6B/model.tensors`                 |
| [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B)                       | `fp16`    | `s3://tensorized/EleutherAI/gpt-j-6B/fp16/model.tensors`            |
| [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)               | `fp32`    | `s3://tensorized/EleutherAI/gpt-neox-20b/model.tensors`             |
| [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)               | `fp16`    | `s3://tensorized/EleutherAI/gpt-neox-20b/fp16/model.tensors`        |
| [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m)                   | `fp32`    | `s3://tensorized/EleutherAI/pythia-70m/model.tensors`               |
| [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m)                   | `fp16`    | `s3://tensorized/EleutherAI/pythia-70m/fp16/model.tensors`          |
| [EleutherAI/pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)                 | `fp32`    | `s3://tensorized/EleutherAI/pythia-1.4b/model.tensors`              |
| [EleutherAI/pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)                 | `fp16`    | `s3://tensorized/EleutherAI/pythia-1.4b/fp16/model.tensors`         |
| [EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)                 | `fp32`    | `s3://tensorized/EleutherAI/pythia-2.8b/model.tensors`              |
| [EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)                 | `fp16`    | `s3://tensorized/EleutherAI/pythia-2.8b/fp16/model.tensors`         |
| [EleutherAI/pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b)                 | `fp32`    | `s3://tensorized/EleutherAI/pythia-6.9b/model.tensors`              |
| [EleutherAI/pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b)                 | `fp16`    | `s3://tensorized/EleutherAI/pythia-6.9b/fp16/model.tensors`         |
| [EleutherAI/pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)                   | `fp32`    | `s3://tensorized/EleutherAI/pythia-12b/model.tensors`               |
| [EleutherAI/pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)                   | `fp16`    | `s3://tensorized/EleutherAI/pythia-12b/fp16/model.tensors`          |
| [EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)   | `fp32`    | `s3://tensorized/EleutherAI/pythia-70m-deduped/model.tensors`       |
| [EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)   | `fp16`    | `s3://tensorized/EleutherAI/pythia-70m-deduped/fp16/model.tensors`  |
| [EleutherAI/pythia-1.4b-deduped](https://huggingface.co/EleutherAI/pythia-1.4b-deduped) | `fp32`    | `s3://tensorized/EleutherAI/pythia-1.4b-deduped/model.tensors`      |
| [EleutherAI/pythia-1.4b-deduped](https://huggingface.co/EleutherAI/pythia-1.4b-deduped) | `fp16`    | `s3://tensorized/EleutherAI/pythia-1.4b-deduped/fp16/model.tensors` |
| [EleutherAI/pythia-2.8b-deduped](https://huggingface.co/EleutherAI/pythia-2.8b-deduped) | `fp32`    | `s3://tensorized/EleutherAI/pythia-2.8b-deduped/model.tensors`      |
| [EleutherAI/pythia-2.8b-deduped](https://huggingface.co/EleutherAI/pythia-2.8b-deduped) | `fp16`    | `s3://tensorized/EleutherAI/pythia-2.8b-deduped/fp16/model.tensors` |
| [EleutherAI/pythia-6.9b-deduped](https://huggingface.co/EleutherAI/pythia-6.9b-deduped) | `fp32`    | `s3://tensorized/EleutherAI/pythia-6.9b-deduped/model.tensors`      |
| [EleutherAI/pythia-6.9b-deduped](https://huggingface.co/EleutherAI/pythia-6.9b-deduped) | `fp16`    | `s3://tensorized/EleutherAI/pythia-6.9b-deduped/fp16/model.tensors` |
| [EleutherAI/pythia-12b-deduped](https://huggingface.co/EleutherAI/pythia-12b-deduped)   | `fp32`    | `s3://tensorized/EleutherAI/pythia-12b-deduped/model.tensors`       |
| [EleutherAI/pythia-12b-deduped](https://huggingface.co/EleutherAI/pythia-12b-deduped)   | `fp16`    | `s3://tensorized/EleutherAI/pythia-12b-deduped/fp16/model.tensors`  |
| [KoboldAI/fairseq-dense-125M](https://huggingface.co/KoboldAI/fairseq-dense-125M)       | `fp32`    | `s3://tensorized/KoboldAI/fairseq-dense-125M/model.tensors`         |
| [KoboldAI/fairseq-dense-125M](https://huggingface.co/KoboldAI/fairseq-dense-125M)       | `fp16`    | `s3://tensorized/KoboldAI/fairseq-dense-125M/fp16/model.tensors`    |
| [KoboldAI/fairseq-dense-355M](https://huggingface.co/KoboldAI/fairseq-dense-355M)       | `fp32`    | `s3://tensorized/KoboldAI/fairseq-dense-355M/model.tensors`         |
| [KoboldAI/fairseq-dense-355M](https://huggingface.co/KoboldAI/fairseq-dense-355M)       | `fp16`    | `s3://tensorized/KoboldAI/fairseq-dense-355M/fp16/model.tensors`    |
| [KoboldAI/fairseq-dense-2.7B](https://huggingface.co/KoboldAI/fairseq-dense-2.7B)       | `fp32`    | `s3://tensorized/KoboldAI/fairseq-dense-2.7B/model.tensors`         |
| [KoboldAI/fairseq-dense-2.7B](https://huggingface.co/KoboldAI/fairseq-dense-2.7B)       | `fp16`    | `s3://tensorized/KoboldAI/fairseq-dense-2.7B/fp16/model.tensors`    |
| [KoboldAI/fairseq-dense-6.7B](https://huggingface.co/KoboldAI/fairseq-dense-6.7B)       | `fp32`    | `s3://tensorized/KoboldAI/fairseq-dense-6.7B/model.tensors`         |
| [KoboldAI/fairseq-dense-6.7B](https://huggingface.co/KoboldAI/fairseq-dense-6.7B)       | `fp16`    | `s3://tensorized/KoboldAI/fairseq-dense-6.7B/fp16/model.tensors`    |
| [KoboldAI/fairseq-dense-13B](https://huggingface.co/KoboldAI/fairseq-dense-13B)         | `fp32`    | `s3://tensorized/KoboldAI/fairseq-dense-13B/model.tensors`          |
| [KoboldAI/fairseq-dense-13B](https://huggingface.co/KoboldAI/fairseq-dense-13B)         | `fp16`    | `s3://tensorized/KoboldAI/fairseq-dense-13B/fp16/model.tensors`     |
| [Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono)     | `fp32`    | `s3://tensorized/Salesforce/codegen-350M-mono/model.tensors`        |
| [Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono)     | `fp16`    | `s3://tensorized/Salesforce/codegen-350M-mono/fp16/model.tensors`   |
| [Salesforce/codegen-350M-multi](https://huggingface.co/Salesforce/codegen-350M-multi)   | `fp32`    | `s3://tensorized/Salesforce/codegen-350M-multi/model.tensors`       |
| [Salesforce/codegen-350M-multi](https://huggingface.co/Salesforce/codegen-350M-multi)   | `fp16`    | `s3://tensorized/Salesforce/codegen-350M-multi/fp16/model.tensors`  |
| [Salesforce/codegen-2B-multi](https://huggingface.co/Salesforce/codegen-2B-multi)       | `fp32`    | `s3://tensorized/Salesforce/codegen-2B-multi/model.tensors`         |
| [Salesforce/codegen-2B-multi](https://huggingface.co/Salesforce/codegen-2B-multi)       | `fp16`    | `s3://tensorized/Salesforce/codegen-2B-multi/fp16/model.tensors`    |
| [Salesforce/codegen-6B-mono](https://huggingface.co/Salesforce/codegen-6B-mono)         | `fp32`    | `s3://tensorized/Salesforce/codegen-6B-mono/model.tensors`          |
| [Salesforce/codegen-6B-mono](https://huggingface.co/Salesforce/codegen-6B-mono)         | `fp16`    | `s3://tensorized/Salesforce/codegen-6B-mono/fp16/model.tensors`     |
| [Salesforce/codegen-6B-multi](https://huggingface.co/Salesforce/codegen-6B-multi)       | `fp32`    | `s3://tensorized/Salesforce/codegen-6B-multi/model.tensors`         |
| [Salesforce/codegen-6B-multi](https://huggingface.co/Salesforce/codegen-6B-multi)       | `fp16`    | `s3://tensorized/Salesforce/codegen-6B-multi/fp16/model.tensors`    |
| [Salesforce/codegen-16B-mono](https://huggingface.co/Salesforce/codegen-16B-mono)       | `fp32`    | `s3://tensorized/Salesforce/codegen-16B-mono/model.tensors`         |
| [Salesforce/codegen-16B-mono](https://huggingface.co/Salesforce/codegen-16B-mono)       | `fp16`    | `s3://tensorized/Salesforce/codegen-16B-mono/fp16/model.tensors`    |
| [Salesforce/codegen-16B-multi](https://huggingface.co/Salesforce/codegen-16B-multi)     | `fp32`    | `s3://tensorized/Salesforce/codegen-16B-multi/model.tensors`        |
| [Salesforce/codegen-16B-multi](https://huggingface.co/Salesforce/codegen-16B-multi)     | `fp16`    | `s3://tensorized/Salesforce/codegen-16B-multi/fp16/model.tensors`   |

## S3 Usage Notes
`tensorizer` uses the `boto3` library to interact with S3. The easiest way
to use `tensorizer` with S3 is to configure your S3 credentials in
`~/.s3cfg`.

If you don't want to use `~/.s3cfg`, or wish to use a `.s3cfg` config file
saved at a nonstandard location (e.g. under `/var/run`), you can also specify
your S3 credentials using the `tensorizer.stream_io.open_stream()` function,
and then pass that into the `TensorSerializer` or `TensorDeserializer`
constructor.

The `stream_io.open_stream()` function takes a `path_uri` argument, which can
be an `s3://` URI, and accepts the following keyword arguments:
* `s3_access_key_id`: S3 access key ID
* `s3_secret_access_key`: S3 secret access key
* `s3_endpoint`: S3 endpoint

*Or,*

* `s3_config_path`: Alternative filesystem path to a `.s3cfg` config file

For example:
```python
TensorSerializer(
    open_stream(s3_uri,
                "wb",
                s3_access_key_id=ACCESS_KEY,
                s3_secret_access_key=SECRET_KEY,
                s3_endpoint="object.ord1.coreweave.com"))
```

and...

```python
TensorDeserializer(
    open_stream(s3_uri,
                "rb",
                s3_access_key_id=ACCESS_KEY,
                s3_secret_access_key=SECRET_KEY,
                s3_endpoint="object.ord1.coreweave.com"))
```

**NOTE:** For faster object downloads in the CoreWeave Cloud, you can use
the `accel-object.ord1.coreweave.com` endpoint. This endpoint is optimized
for object downloads, and will be faster than the `object.ord1.coreweave.com`
endpoint once the object is cached.

**NOTE2:** The cache above does not get invalidated when the object is updated
in S3. If you update an object in S3, you will need to wait for the cache to
expire before you can download the updated object. This takes 24 hours since
the last download.

For this reason, it is recommended to use a unique S3 key for each version
of a model if you use the `accel-object.ord1.coreweave.com` endpoint.

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

The `state_dict` can also be used to initialize a HuggingFace Transformers
AutoModel. But HuggingFace Transformers performs three or more copies of
the data, so memory use will explode.

## Running Tests
`tensorizer` uses `unittest` for testing.
The tests have their own set of dependencies, which can be installed with
`pip install -r tests/requirements.txt`.

Some tests require a GPU, and will be skipped if no GPU is available.
To run the tests, run the following in the root of the repository:

```bash
python -m pip install -e .
python -m pip install -r tests/requirements.txt
python -m unittest discover tests/ --verbose
```

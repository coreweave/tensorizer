# tensorizer
Module, Model, and Tensor Serialization/Deserialization

## TLDR
Extremely fast model loads from HTTP/HTTPS and S3 endpoints. GPT-J
(`20gb`) loads at wire-speed (`~5GB/s`) on a 40gige network, and is
only bottlenecked by the Linux kernel TCP stack.

## Rationale
CoreWeave and our customers use KNative to deploy models as serverless
functions. How long a model takes to load is a major factor in the latency
of KNative scale-up. `tensorizer` is a tool to serialize models and their
associated tensors into a single file that can be loaded quickly and
efficiently off a HTTP/HTTPS or S3 endpoint.

By not embedding the model in the container image, we can reduce the
container image size and the time it takes to load the model. This is
especially important for models that are large in size, such as
[EleutherAI/gpt-neox-20B](https://huggingface.co/EleutherAI/gpt-neox-20B)
that weight in at `~40GB`.

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

`load_module` is the main method of the `TensorDeserializer` class. It
takes a `torch.nn.Module` and loads the tensors from the `path_uri`
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


## Quick Examples

### Transformers

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tensorizer import load_model, serialize_model

model_name = "EleutherAI/gpt-neo-125M"
output_dir = model_name.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# The first model is the CLIP Text Encoder.
serialize_model(
    model=model,
    config=model.config,
    model_directory=output_dir,
)

# OPTIONAL: You can also save the tokenizer to the output directory.
tokenizer.save_pretrained(output_dir)

# To load the models, all we have to do is call load_model().
model = load_model(
    path_uri=output_dir,
    modelclass=AutoModelForCausalLM,
    configclass=AutoConfig
)

# Validate that the model is working. The model is already on the GPU.
# If you want to use the model on the CPU, you can call model.cpu().
print(tokenizer.decode(model.generate(
    tokenizer.encode("I walked my dog", return_tensors="pt").to("cuda"),
    max_new_tokens=20,
    pad_token_id=tokenizer.eos_token_id,
)[0]))
```

### Stable Diffusion

```py
import os
from diffusers import StableDiffusionPipeline
from tensorizer import load_model, serialize_model

hf_api_token = os.environ.get("HF_API_TOKEN")
model_name = "runwayml/stable-diffusion-v1-5"
model_id = model_name.split("/")[-1]

pipeline = StableDiffusionPipeline.from_pretrained(
    model_name, use_auth_token=hf_api_token
)

# StableDiffusionPipeline is just a collection of models,
# so we serialize each model in the pipeline individually.

# The first model is the CLIP Text Encoder.
serialize_model(
    model=pipeline.text_encoder,
    config=pipeline.text_encoder.config,
    model_directory=model_id,
    model_prefix="encoder",
)

# The second model is the VAE.
serialize_model(
    model=pipeline.vae,
    config=None,
    model_directory=model_id,
    model_prefix="vae",
)

# The third model is the UNet.
serialize_model(
    model=pipeline.unet,
    config=None,
    model_directory=model_id,
    model_prefix="unet",
)

# This is optional, but you can also save the CLIP tokenizer and the Stable
# Diffusion scheduler to the model directory.
pipeline.tokenizer.save_pretrained(model_id)

# We can also save the scheduler.
pipeline.scheduler.save_config(model_id)

# To load the models into a blank Stable Diffusion pipeline, we have to import
# the individual models first so we can initialize the pipeline with them.

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

vae = load_model(model_id, AutoencoderKL, None, "vae")
unet = load_model(model_id, UNet2DConditionModel, None, "unet")
encoder = load_model(model_id, CLIPTextModel, CLIPTextConfig, "encoder")

pipeline = StableDiffusionPipeline(
    text_encoder=encoder,
    vae=vae,
    unet=unet,
    scheduler=DDIMScheduler.from_config(model_id),
    tokenizer=CLIPTokenizer.from_pretrained(model_id),
    safety_checker=None,
    feature_extractor=None,
)

# Now we can use the pipeline to generate images.
pipeline.to("cuda")
pipeline("a photo of an astronaut riding a horse on mars").images[0].save(
    "image.png"
)
```

More practical examples for usage of the Tensorizer can be found inside of
[tensorizer.py](tensorizer.py), where `df_main()` serializes models from
[HuggingFace Diffusers](https://github.com/huggingface/diffusers) and `hf_main()`
serializes [HuggingFace Transformers](https://github.com/huggingface/transformers)
models.

# tensorizer
Module, Model, and Tensor Serialization/Deserialization

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

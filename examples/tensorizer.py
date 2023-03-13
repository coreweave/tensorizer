import argparse
import json
import os
from collections import OrderedDict
from typing import Optional, Union

import tensorizer.utils as utils
from tensorizer.serialization import TensorSerializer, TensorDeserializer
import tensorizer.stream_io as stream_io
import logging
import time
import torch
import tempfile

os.environ[
    "TRANSFORMERS_VERBOSITY"
] = "error"  # disable missing keys and unexpected key warnings

from transformers import (
    AutoConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextConfig,
)
from transformers.modeling_utils import PreTrainedModel

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    LMSDiscreteScheduler,
)
from diffusers.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
)
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

def serialize_model(
    model: torch.nn.Module,
    config: Optional[Union[ConfigMixin, AutoConfig, dict]],
    model_directory: str,
    model_prefix: str = "model",
):
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays and serialize them to GooseTensor format. The stripped
    model is also serialized to pytorch format.

    Args:
        model: The model to serialize.
        config: The model's configuration. This is optional and only
            required for HuggingFace Transformers models. Diffusers
            models do not require this.
        model_directory: The directory to save the serialized model to.
        model_prefix: The prefix to use for the serialized model files. This
            is purely optional and it allows for multiple models to be
            serialized to the same directory. A good example are Stable
            Diffusion models. Default is "model".
    """

    os.makedirs(model_directory, exist_ok=True)
    dir_prefix = f"{model_directory}/{model_prefix}"

    if config is None:
        config = model
    if config is not None:
        if hasattr(config, "to_json_file"):
            config.to_json_file(f"{dir_prefix}-config.json")
        if isinstance(config, dict):
            open(f"{dir_prefix}-config.json", "w").write(
                json.dumps(config, indent=2)
            )

    ts = TensorSerializer(open(f"{dir_prefix}.tensors", "wb"))
    ts.write_module(model)
    ts.close()



def load_model(
    path_uri: str,
    modelclass: Union[PreTrainedModel, ModelMixin, ConfigMixin] = None,
    configclass: Optional[Union[ConfigMixin, AutoConfig]] = None,
    model_prefix: str = "model",
    device: torch.device = utils.get_device(),
    dtype: str = None,
) -> torch.nn.Module:
    """
    Given a path prefix, load the model with a custom extension

    Args:
        path_uri: path to the model. Can be a local path or a URI
        modelclass: The model class to load the tensors into.
        configclass: The config class to load the model config into. This must be
            set if you are loading a model from HuggingFace Transformers.
        model_prefix: The prefix to use to distinguish between multiple serialized
            models. The default is "model".
        dtype: The dtype to load the tensors into. If None, the dtype is inferred from
            the model.
    """

    if model_prefix is None:
        model_prefix = "model"

    begin_load = time.time()
    ram_usage = utils.get_mem_usage()

    config_uri = f"{path_uri}/{model_prefix}-config.json"
    tensors_uri = f"{path_uri}/{model_prefix}.tensors"
    tensor_stream = stream_io.open_stream(tensors_uri)

    logger.info(f"Loading {tensors_uri}, {ram_usage}")

    tensor_deserializer = TensorDeserializer(tensor_stream)

    if configclass is not None:
        try:
            with tempfile.TemporaryDirectory() as dir:
                open(os.path.join(dir, "config.json"), "w").write(
                    stream_io.open_stream(config_uri).read().decode("utf-8"))
                config = configclass.from_pretrained(dir)
                config.gradient_checkpointing = True
        except ValueError:
            config = configclass.from_pretrained(config_uri)
        model = utils.no_init_or_tensor(
            lambda: modelclass.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )
    else:
        try:
            config = json.loads(
                stream_io.open_stream(config_uri).read().decode("utf-8")
            )
        except ValueError:
            with open(config_uri, "r") as f:
                config = json.load(f)
        model = utils.no_init_or_tensor(lambda: modelclass(**config))

    tensor_deserializer.load_tensors(model, device=device, dtype=dtype)

    tensor_load_s = time.time() - begin_load
    rate_str = utils.convert_bytes(
        tensor_deserializer.total_bytes_read / tensor_load_s
    )
    tensors_sz = utils.convert_bytes(tensor_deserializer.total_bytes_read)
    logger.info(
        f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
        + f"{tensors_sz} @ {rate_str}/s, {utils.get_mem_usage()}"
    )

    return model



def df_main(args: argparse.Namespace) -> None:
    output_prefix = args.output_prefix
    print("MODEL PATH:", args.input_directory)
    print("OUTPUT PREFIX:", output_prefix)

    hf_api_token = os.environ.get("HF_API_TOKEN")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.input_directory, use_auth_token=hf_api_token
    )

    logger.info("Serializing model")
    logger.info("GPU: " + utils.get_gpu_name())
    logger.info("GPU RAM: " + utils.get_vram_usage_str())
    logger.info("PYTHON USED RAM: " + utils.get_mem_usage())

    serialize_model(
        pipeline.text_encoder.eval(),
        pipeline.text_encoder.config,
        output_prefix,
        "encoder",
    )
    serialize_model(pipeline.vae.eval(), None, output_prefix, "vae")
    serialize_model(pipeline.unet.eval(), None, output_prefix, "unet")

    pipeline.tokenizer.save_pretrained(output_prefix)

    if args.validate:
        device = utils.get_device()

        logger.info("Validating serialization")
        vae = load_model(output_prefix, AutoencoderKL, None, "vae", device)
        unet = load_model(output_prefix, UNet2DConditionModel, None, "unet")
        encoder = load_model(
            output_prefix, CLIPTextModel, CLIPTextConfig, "encoder", device
        )

        pipeline = StableDiffusionPipeline(
            text_encoder=encoder,
            vae=vae,
            unet=unet,
            tokenizer=CLIPTokenizer.from_pretrained(
                args.input_directory, subfolder="tokenizer"
            ),
            scheduler=LMSDiscreteScheduler(
                beta_end=0.012,
                beta_schedule="scaled_linear",
                beta_start=0.00085,
                num_train_timesteps=1000,
                trained_betas=None,
            ),
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        prompt = "a photo of an astronaut riding a horse on mars"
        with torch.autocast(
            "cuda" if torch.cuda.is_available() else "cpu"
        ):  # for some reason device_type needs to be a string instead of an actual device
            pipeline(prompt).images[0]


def hf_main(args):
    output_prefix = args.output_prefix
    print("MODEL PATH:", args.input_directory)
    print("OUTPUT PREFIX:", output_prefix)

    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    model_config = AutoConfig.from_pretrained(args.input_directory)
    model = AutoModelForCausalLM.from_pretrained(
        args.input_directory, config=model_config, torch_dtype=torch.float16
    )

    logger.info("Serializing model")
    logger.info("GPU: " + utils.get_gpu_name())
    logger.info("GPU RAM: " + utils.get_vram_usage_str())
    logger.info("PYTHON USED RAM: " + utils.get_mem_usage())

    serialize_model(model, model_config, output_prefix)

    tokenizer = AutoTokenizer.from_pretrained(
        args.input_directory
    ).save_pretrained(output_prefix)

    if args.validate:
        device = utils.get_device()
        logger.info("Validating serialization")
        model = load_model(
            output_prefix,
            AutoModelForCausalLM,
            AutoConfig,
            None,
            device,
            "float16",
        ).eval()
        # test generation
        tokenizer = AutoTokenizer.from_pretrained(args.input_directory)
        input_ids = tokenizer.encode(
            "¡Hola! Encantado de conocerte. hoy voy a", return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=50, do_sample=True
            )
        logger.info(
            f"Test Output: {tokenizer.decode(output[0], skip_special_tokens=True)}"
        )


def main():
    # usage: tensorizer [input-directory] [output-prefix] [model-type]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_directory",
        type=str,
        help="Path to model directory or HF model ID",
    )
    parser.add_argument(
        "output_prefix", type=str, help="Path to output directory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["transformers", "diffusers"],
        required=True,
        help="Framework used for the model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate serialization by running a test inference",
    )
    args = parser.parse_args()

    if args.model_type == "transformers":
        hf_main(args)
    elif args.model_type == "diffusers":
        df_main(args)
    else:
        raise ValueError(
            f"Unknown model type {args.model_type} (transformers or diffusers)"
        )


if __name__ == "__main__":
    main()

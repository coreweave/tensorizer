import argparse
import gc
import json
import logging
import os
import tempfile
import time
import zipfile
from functools import partial
from pathlib import Path
from typing import Optional, Type, Union

import torch
from diffusers import (
    AutoencoderKL,
    ConfigMixin,
    ModelMixin,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPTextConfig,
    CLIPTextModel,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from tensorizer import TensorDeserializer, TensorSerializer, stream_io, utils

s3_access_key_id = os.environ.get("S3_ACCESS_KEY_ID") or None
s3_secret_access_key = os.environ.get("S3_SECRET_ACCESS_KEY") or None
s3_endpoint = os.environ.get("S3_ENDPOINT_URL") or None

_read_stream, _write_stream = (
    partial(
        stream_io.open_stream,
        mode=mode,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        s3_endpoint=s3_endpoint,
    )
    for mode in ("rb", "wb+")
)


def setup_logger():
    _logger = logging.getLogger(__name__)
    _logger.setLevel(level=logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    return _logger


logger = setup_logger()


def set_logger_verbosity(verbosity: int = 2):
    """
    Configure the verbosity of the global logger.
    Args:
        verbosity: Verbosity level. Clamped to [0, 3].
    """
    levels = (logging.CRITICAL, logging.WARNING, logging.INFO, logging.DEBUG)
    verbosity: int = min(len(levels) - 1, max(0, verbosity))
    logger.setLevel(levels[verbosity])


class ValidationError(RuntimeError):
    pass


class ModuleValidationError(ValidationError):
    pass


class TokenizerValidationError(ValidationError):
    pass


class SchedulerValidationError(ValidationError):
    pass


def assert_module_equal(
    before: torch.nn.Module, after: torch.nn.Module, label: Optional[str] = None
) -> None:
    """
    Check that the state dicts of two modules are equal.
    Args:
        before: The original module to compare against.
        after: The secondary module to compare.
        label: Optional label to include in error messages.

    Raises:
        ModuleValidationError: If the modules do not match.
    """
    label: str = f"{label}: " if label else ""
    before_sd = before.state_dict()
    after_sd = after.state_dict()
    if before_sd.keys() != after_sd.keys():
        raise ModuleValidationError(
            f"{label}Validating deserialized model failed: different keys"
        )
    for name, before_tensor in before_sd.items():
        after_tensor = after_sd[name]
        if not torch.equal(before_tensor, after_tensor):
            raise ModuleValidationError(
                f"{label}Validating deserialized model failed:"
                f" weights for tensor {name} don't match"
            )


def assert_tokenizer_equal(
    before: PreTrainedTokenizer, after: PreTrainedTokenizer
) -> None:
    msg = "Validating deserialized tokenizer failed: different {}".format
    if type(before) is not type(after):
        raise TokenizerValidationError(msg("types"))
    missing = object()
    for prop in (
        "vocab_size",
        "model_max_length",
        "is_fast",
        "special_tokens",
        "added_tokens_decoder",
    ):
        if getattr(before, prop, missing) != getattr(after, prop, missing):
            raise TokenizerValidationError(msg(prop))


def file_is_non_empty(file: str):
    """
    Check if file exists and is not empty. If the file is found locally,
    it is checked for emptiness with `os.stat`. If the file is found
    on S3, it is checked for emptiness by reading the first byte of the file.

    Args:
        file: The path to check for existence and emptiness. This can
           either be a local path or an S3 URI.
    """
    try:
        return os.stat(file).st_size > 0
    except FileNotFoundError:
        uri = file.lower()
        if not any(map(uri.startswith, ("s3://", "http://", "https://"))):
            return False
        try:
            with _read_stream(file) as f:
                return bool(f.read(1))
        except OSError:
            return False


def serialize_model(
    model: torch.nn.Module,
    config: Union[ConfigMixin, AutoConfig, dict],
    model_directory: str,
    model_prefix: str = "model",
    force: bool = False,
):
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays and serialize them to Tensorizer format. The stripped
    model is also serialized to pytorch format.

    Args:
        model: The model to serialize.
        config: The model's configuration.
        model_directory: The directory to save the serialized model to.
        model_prefix: The prefix to use for the serialized model files. This
            is purely optional, and it allows for multiple models to be
            serialized to the same directory. A good example are Stable
            Diffusion models. Default is "model".
        force: Force upload serialized tensors to `output_prefix`
            even if they already exist
    """

    dir_prefix: str = f"{model_directory}/{model_prefix}"
    config_path: str = f"{dir_prefix}-config.json"
    model_path: str = f"{dir_prefix}.tensors"
    paths = (config_path, model_path)
    write: bool = force or any(not file_is_non_empty(path) for path in paths)
    if write:
        logger.info(f"Writing config to {config_path}")
        with _write_stream(config_path) as f:
            config_dict = (
                config.to_dict() if hasattr(config, "to_dict") else config
            )
            f.write(json.dumps(config_dict, indent=2).encode("utf-8"))
        logger.info(f"Writing tensors to {model_path}")
        with _write_stream(model_path) as f:
            serializer = TensorSerializer(f)
            serializer.write_module(model, include_non_persistent_buffers=False)
            serializer.close()
    else:
        logger.warning(
            "Skipping serialization because files already"
            f" exist at {' & '.join(paths)}."
            " Use the --force option to force an overwrite instead."
        )


def load_model(
    path_uri: str,
    model_class: Union[
        Type[PreTrainedModel], Type[ModelMixin], Type[ConfigMixin]
    ],
    config_class: Optional[
        Union[Type[PretrainedConfig], Type[ConfigMixin], Type[AutoConfig]]
    ] = None,
    model_prefix: Optional[str] = "model",
    device: torch.device = utils.get_device(),
    dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    """
    Given a path prefix, load the model with a custom extension

    Args:
        path_uri: path to the model. Can be a local path or a URI
        model_class: The model class to load the tensors into.
        config_class: The config class to load the model config into. This must be
            set if you are loading a model from HuggingFace Transformers.
        model_prefix: The prefix to use to distinguish between multiple serialized
            models. The default is "model".
        device: The device onto which to load the model.
        dtype: The dtype to load the tensors into. If None, the dtype is inferred from
            the model.
    """
    if model_prefix is None:
        model_prefix = "model"

    begin_load = time.perf_counter()
    ram_usage = utils.get_mem_usage()

    path_uri: str = path_uri.rstrip("/")
    config_uri: str = f"{path_uri}/{model_prefix}-config.json"
    tensors_uri: str = f"{path_uri}/{model_prefix}.tensors"

    logger.info(f"Loading {tensors_uri}, {ram_usage}")

    if config_class is None:
        config_loader = model_class.load_config
    else:
        config_loader = config_class.from_pretrained
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = os.path.join(temp_dir, "config.json")
            with open(temp_config_path, "wb") as temp_config:
                logger.info(f"Loading {config_uri}")
                with _read_stream(config_uri) as config_file:
                    temp_config.write(config_file.read())
            config = config_loader(temp_dir)
            if isinstance(config, PretrainedConfig):
                config.gradient_checkpointing = True
    except ValueError:
        config = config_loader(config_uri)
    with utils.no_init_or_tensor():
        # AutoModels instantiate from a config via their from_config()
        # method, while other classes can usually be instantiated directly.
        model_loader = getattr(model_class, "from_config", model_class)
        model = model_loader(config)

    with _read_stream(tensors_uri) as tensor_stream, TensorDeserializer(
        tensor_stream, device=device, dtype=dtype
    ) as tensor_deserializer:
        tensor_deserializer.load_into_module(model)
        tensor_load_s = time.perf_counter() - begin_load
        bytes_read: int = tensor_deserializer.total_bytes_read

    rate_str = utils.convert_bytes(bytes_read / tensor_load_s)
    tensors_sz = utils.convert_bytes(bytes_read)
    logger.info(
        f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
        f"{tensors_sz} @ {rate_str}/s, {utils.get_mem_usage()}"
    )

    return model


def serialize_pretrained(
    component, path_uri: str, prefix: str, force: bool = False
):
    save_path: str = f"{path_uri.rstrip('/')}/{prefix}.zip"
    if force or not file_is_non_empty(save_path):
        logger.info(f"Writing {save_path}")
        with _write_stream(save_path) as stream, zipfile.ZipFile(
            stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=5
        ) as file, tempfile.TemporaryDirectory() as directory:
            component.save_pretrained(directory)
            for path in Path(directory).iterdir():
                file.write(filename=path, arcname=path.name)
    else:
        logger.warning(
            "Skipping saving non-tensor component because files already"
            f" exist at {save_path}."
            " Use the --force option to force an overwrite instead."
        )


def load_pretrained(
    component_class: Union[
        Type[PreTrainedTokenizer], Type[AutoTokenizer], Type[ConfigMixin]
    ],
    path_uri: str,
    prefix: str,
):
    load_path: str = f"{path_uri.rstrip('/')}/{prefix}.zip"
    logger.info(f"Loading {load_path}")
    with _read_stream(load_path) as stream, zipfile.ZipFile(
        stream, mode="r"
    ) as file, tempfile.TemporaryDirectory() as directory:
        file.extractall(path=directory)
        return component_class.from_pretrained(directory, local_files_only=True)


def df_main(args: argparse.Namespace) -> None:
    output_prefix = args.output_prefix
    output_prefix = output_prefix.rstrip("/")

    logger.info(f"MODEL PATH: {args.input_directory}")
    logger.info(f"OUTPUT PREFIX: {output_prefix}")

    hf_api_token = os.environ.get("HF_API_TOKEN")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.input_directory, use_auth_token=hf_api_token
    )

    logger.info("Serializing model")
    logger.info("GPU: " + utils.get_gpu_name())
    logger.info("PYTHON USED RAM: " + utils.get_mem_usage())

    for prefix, component in (
        ("encoder", pipeline.text_encoder),
        ("vae", pipeline.vae),
        ("unet", pipeline.unet),
    ):
        component.eval()
        config = component.config
        if isinstance(config, dict):
            config = {k: v for k, v in config.items() if k != "_name_or_path"}
        serialize_model(
            component, config, output_prefix, prefix, force=args.force
        )

    serialize_pretrained(
        pipeline.tokenizer, output_prefix, "tokenizer", force=args.force
    )
    serialize_pretrained(
        pipeline.scheduler, output_prefix, "scheduler", force=args.force
    )
    tokenizer_type = type(pipeline.tokenizer)
    scheduler_type = type(pipeline.scheduler)

    if args.validate:
        device = utils.get_device()
        pipeline = pipeline.to(device)
        gc.collect()

        logger.info("Validating serialization")
        vae = load_model(
            output_prefix,
            AutoencoderKL,
            None,
            "vae",
            device,
        )
        assert_module_equal(pipeline.vae, vae, "vae")

        unet = load_model(
            output_prefix,
            UNet2DConditionModel,
            None,
            "unet",
            device,
        )
        assert_module_equal(pipeline.unet, unet, "unet")

        encoder = load_model(
            output_prefix,
            CLIPTextModel,
            CLIPTextConfig,
            "encoder",
            device,
        )
        assert_module_equal(pipeline.text_encoder, encoder, "encoder")

        tokenizer = load_pretrained(tokenizer_type, output_prefix, "tokenizer")
        assert_tokenizer_equal(pipeline.tokenizer, tokenizer)
        scheduler = load_pretrained(scheduler_type, output_prefix, "scheduler")
        if not isinstance(scheduler, scheduler_type):
            raise SchedulerValidationError(
                "Validating deserialized scheduler failed"
            )

        deserialized_pipeline = StableDiffusionPipeline(
            text_encoder=encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)

        prompt = "a photo of an astronaut riding a horse on mars"
        with torch.autocast(device.type):
            deserialized_pipeline(prompt).images[0].save("test.png")


def hf_main(args):
    output_prefix = args.output_prefix
    output_prefix = output_prefix.rstrip("/")

    logger.info(f"MODEL PATH: {args.input_directory}")
    logger.info(f"OUTPUT PREFIX: {output_prefix}")

    dtype = torch.float16
    model_config = AutoConfig.from_pretrained(args.input_directory)
    model = AutoModelForCausalLM.from_pretrained(
        args.input_directory, config=model_config, torch_dtype=dtype
    )

    logger.info("Serializing model")
    logger.info("GPU: " + utils.get_gpu_name())
    logger.info("PYTHON USED RAM: " + utils.get_mem_usage())

    serialize_model(model, model_config, output_prefix, force=args.force)

    if args.validate:
        device = utils.get_device()
        model = model.to(device)
        gc.collect()
        logger.info("Validating serialization")
        tensorizer_model = load_model(
            output_prefix,
            AutoModelForCausalLM,
            AutoConfig,
            None,
            device,
            dtype,
        ).eval()
        # Comparing model parameters
        logger.debug("Testing for sameness of model parameters")
        assert_module_equal(model, tensorizer_model)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "An example script that uses Tensorizer to serialize"
            " a HuggingFace model to an output directory or object storage."
        )
    )
    parser.add_argument(
        "input_directory",
        type=str,
        help="Path to model directory or HF model ID",
    )
    parser.add_argument(
        "output_prefix", type=str, help="Path to output directory"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["transformers", "diffusers"],
        required=True,
        help="Framework used for the model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate serialization by checking if deserialized weights match",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing serialized tensors in the output directory",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        help="Show less output",
        default=0,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Show more output",
        default=2,
    )
    args = parser.parse_args(argv)

    set_logger_verbosity(args.verbose - args.quiet)
    if args.force:
        logger.debug(f"Forcing serialization to {args.output_prefix}")
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

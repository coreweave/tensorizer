import os
import tempfile
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tensorizer import (
    DecryptionParams,
    EncryptionParams,
    TensorDeserializer,
    TensorSerializer,
)
from tensorizer.utils import no_init_or_tensor

model_ref = "EleutherAI/gpt-neo-2.7B"


def original_model(ref) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(ref)


def empty_model(ref) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(ref)
    with no_init_or_tensor():
        return AutoModelForCausalLM.from_config(config)


# Set a strong string or bytes passphrase here
source: str = os.getenv("SUPER_SECRET_STRONG_PASSWORD", "") or input(
    "Source string to create an encryption key: "
)

fd, path = tempfile.mkstemp(prefix="encrypted-tensors")

try:
    # Encrypt a model during serialization
    encryption_params = EncryptionParams.from_string(source)

    model = original_model(model_ref)
    serialization_start = time.monotonic()

    serializer = TensorSerializer(path, encryption=encryption_params)
    serializer.write_module(model)
    serializer.close()

    serialization_end = time.monotonic()
    del model

    # Then decrypt it again during deserialization
    decryption_params = DecryptionParams.from_string(source)

    model = empty_model(model_ref)
    deserialization_start = time.monotonic()

    deserializer = TensorDeserializer(
        path, encryption=decryption_params, plaid_mode=True
    )
    deserializer.load_into_module(model)
    deserializer.close()

    deserialization_end = time.monotonic()
    del model
finally:
    os.close(fd)
    os.unlink(path)


def print_speed(prefix, start, end, size):
    mebibyte = 1 << 20
    gibibyte = 1 << 30
    duration = end - start
    rate = size / duration
    print(
        f"{prefix} {size / gibibyte:.2f} GiB model in {duration:.2f} seconds,"
        f" {rate / mebibyte:.2f} MiB/s"
    )


print_speed(
    "Serialized and encrypted",
    serialization_start,
    serialization_end,
    serializer.total_tensor_bytes,
)

print_speed(
    "Deserialized encrypted",
    deserialization_start,
    deserialization_end,
    deserializer.total_tensor_bytes,
)

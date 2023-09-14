import argparse
import gc
import os
import time

import torch

from tensorizer.serialization import TensorDeserializer
from tensorizer.stream_io import CURLStreamFile

# Read in model name from command line, or env var, or default to gpt-j-6B
model_name_default = os.getenv("MODEL_NAME") or "EleutherAI/gpt-j-6B/fp16"
parser = argparse.ArgumentParser(
    description="Test CURLStreamFile download speeds"
)
parser.add_argument(
    "model",
    nargs="?",
    type=str,
    default=model_name_default,
    help=(
        "Model from the s3://tensorized bucket"
        f" with which to test deserialization (default: {model_name_default})"
    ),
)
args = parser.parse_args()

model_name: str = args.model

http_uri = (
    "http://tensorized.accel-object.ord1.coreweave.com"
    f"/{model_name}/model.tensors"
)

kibibyte = 1 << 10
mebibyte = 1 << 20


def io_test(
    source=http_uri, read_size=256 * kibibyte, buffer_size=256 * mebibyte
):
    # Read the stream `read_size` at a time.
    buffer = bytearray(read_size)
    total_sz = 0
    start = time.monotonic()
    io = CURLStreamFile(source, buffer_size=buffer_size)
    while True:
        try:
            sz = io.readinto(buffer)
            total_sz += sz
        except OSError:
            break

        if sz == 0:
            break
    end = time.monotonic()

    resp_headers = getattr(io, "response_headers", {})
    cached_by = resp_headers.get("x-cache-trace", None)
    cached = resp_headers.get("x-cache-status", False)

    # Print the total size of the stream, and the speed at which it was read.
    print(
        f"Read {total_sz / mebibyte:0.2f} MiB at "
        f"{total_sz / mebibyte / (end - start):0.2f} MiB/s, "
        f"{read_size / kibibyte} KiB read size, "
        f"{buffer_size / kibibyte} KiB stream buffer size, "
        f"cached: {cached} by {cached_by}"
    )


def deserialize_test(
    source=http_uri,
    plaid_mode=False,
    verify_hash=False,
    lazy_load=False,
    buffer_size=256 * kibibyte,
):
    start = time.monotonic()
    test_dict = TensorDeserializer(
        CURLStreamFile(source, buffer_size=buffer_size),
        verify_hash=verify_hash,
        plaid_mode=plaid_mode,
        lazy_load=lazy_load,
    )

    if lazy_load or plaid_mode:
        for name in test_dict:
            test_dict[name]

    end = time.monotonic()

    resp_headers = getattr(test_dict._file, "response_headers", {})
    cached_by = resp_headers.get("x-cache-trace", None)
    cached = resp_headers.get("x-cache-status", False)
    total_sz = test_dict.total_bytes_read

    print(
        f"Deserialized {total_sz / mebibyte:0.2f} MiB at "
        f"{total_sz / mebibyte / (end - start):0.2f} MiB/s, "
        f"{buffer_size / kibibyte} KiB stream buffer size, "
        f"plaid: {plaid_mode}, "
        f"verify_hash: {verify_hash}, "
        f"lazy_load: {lazy_load or plaid_mode}, "
        f"cached: {cached} by {cached_by}"
    )

    test_dict.close()
    del test_dict
    torch.cuda.synchronize()
    gc.collect()


# Test the speed of reading from a stream,
# with different buffer sizes ranging from 128 KiB to 256 MiB.
for buffer_size_power in range(17, 28):
    buffer_size = 1 << buffer_size_power
    for sample in range(10):
        io_test(read_size=32 * kibibyte, buffer_size=buffer_size)
        deserialize_test(source=http_uri, buffer_size=buffer_size)
        deserialize_test(
            source=http_uri, plaid_mode=True, buffer_size=buffer_size
        )

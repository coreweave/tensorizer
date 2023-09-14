import gc
import torch
import time
from tensorizer.serialization import TensorDeserializer
from tensorizer.stream_io import CURLStreamFile
import os
import sys

# Read in model name from command line, or env var, or default
# to gpt-j-6b
model_name = os.getenv("MODEL_NAME", "EleutherAI/gpt-j-6b/fp16")
if len(sys.argv) > 1:
    model_name = sys.argv[1]

output_dir = model_name.split("/")[-1]
s3_uri = f"s3://tensorized/{model_name}.tensors"
http_uri = f"http://tensorized.accel-object.ord1.coreweave.com/{model_name}/model.tensors"


def io_test(
    source=http_uri, read_size=256 * 1024, buffer_size=256 * 1024 * 1024
):
    # Read the stream `read_size` at a time.
    buffer = memoryview(bytearray(read_size))
    total_sz = 0
    start = time.time()
    io = CURLStreamFile(source, buffer_size=buffer_size)
    while True:
        try:
            sz = io.readinto(buffer)
            total_sz += sz
        except OSError:
            break

        if sz == 0:
            break
    end = time.time()

    resp_headers = {}
    if hasattr(io, "response_headers"):
        resp_headers = io.response_headers
    cached_by = resp_headers.get("x-cache-trace", None)
    cached = resp_headers.get("x-cache-status", False)

    # Print the total size of the stream, and the speed at which it was read.
    print(
        f"Read {total_sz / 1024 / 1024:0.2f}mb at "
        f"{total_sz / 1024 / 1024 / (end - start):0.2f} mb/s, "
        f"{read_size / 1024}kb read size, "
        f"{buffer_size / 1024}kb stream buffer size, "
        f"cached: {cached} by {cached_by}"
    )


def deserialize_test(
    source=http_uri,
    plaid_mode=False,
    verify_hash=False,
    lazy_load=False,
    buffer_size=2**18,  # 256kb
):
    start = time.time()
    test_dict = TensorDeserializer(
        CURLStreamFile(source, buffer_size=buffer_size),
        verify_hash=verify_hash,
        plaid_mode=plaid_mode,
        lazy_load=lazy_load,
    )

    if lazy_load or plaid_mode:
        for name in test_dict:
            test_dict[name]

    end = time.time()

    resp_headers = {}
    if hasattr(test_dict._file, "response_headers"):
        resp_headers = test_dict._file.response_headers
    cached_by = resp_headers.get("x-cache-trace", None)
    cached = resp_headers.get("x-cache-status", False)
    total_sz = test_dict.total_bytes_read

    print(
        f"Deserialized {total_sz / 1024 / 1024:0.2f}mb at "
        f"{total_sz / 1024 / 1024 / (end - start):0.2f} mb/s, "
        f"{buffer_size / 1024}kb stream buffer size, "
        f"plaid: {plaid_mode}, "
        f"verify_hash: {verify_hash}, "
        f"lazy_load: {lazy_load or plaid_mode}, "
        f"cached: {cached} by {cached_by}"
    )

    test_dict.close()
    del test_dict
    torch.cuda.synchronize()
    gc.collect()


# Test the speed of reading from a stream, with different buffer sizes ranging from
# 128kb to 256mb.
for buffer_size in range(17, 28):
    for sample in range(10):
        io_test(read_size=2**15, buffer_size=2**buffer_size)
        deserialize_test(source=http_uri, buffer_size=2**buffer_size)
        deserialize_test(
            source=http_uri, plaid_mode=True, buffer_size=2**buffer_size
        )

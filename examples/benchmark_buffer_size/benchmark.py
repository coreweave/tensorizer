import argparse
import gc
import logging
import os
import socket
import time

import redis
import torch

from tensorizer.serialization import TensorDeserializer
from tensorizer.stream_io import CURLStreamFile, RedisStreamFile

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Read in model name from command line, or env var, or default to gpt-neo-2.7B
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

print(f"Testing {http_uri}")

kibibyte = 1 << 10
mebibyte = 1 << 20
gibibyte = 1 << 30

# Get nodename from environment, or default to os.uname().nodename
nodename = os.getenv("K8S_NODE_NAME") or os.uname().nodename

# Collect GPU data
try:
    cudadev = torch.cuda.current_device()
    gpu_gb = int(torch.cuda.get_device_properties(0).total_memory / gibibyte)
    gpu_name = torch.cuda.get_device_name(cudadev)
except AssertionError:
    gpu_gb = 0
    gpu_name = "CPU"


redis_client = redis.Redis(host="localhost", port=6379, db=0)


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
    logging.info(
        f"{nodename} -- "
        f"gpu: {gpu_name} ({gpu_gb} GiB), streamed "
        f"{total_sz / mebibyte:0.2f} MiB at "
        f"{total_sz / mebibyte / (end - start):0.2f} MiB/s, "
        f"{buffer_size / kibibyte} KiB stream buffer size, "
        f"{read_size / kibibyte} KiB read size, "
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

    logging.info(
        f"{nodename} -- "
        f"gpu: {gpu_name} ({gpu_gb} GiB), loaded  "
        f" {total_sz / mebibyte:0.2f} MiB at"
        f" {total_sz / mebibyte / (end - start):0.2f} MiB/s,"
        f" {buffer_size / kibibyte} KiB stream buffer size, plaid:"
        f" {plaid_mode}, verify_hash: {verify_hash}, lazy_load:"
        f" {lazy_load or plaid_mode}, cached: {cached} by {cached_by}"
    )

    test_dict.close()
    del test_dict
    torch.cuda.synchronize()
    gc.collect()


def test_read_performance():
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


def load_redis():
    start = time.monotonic()
    test_dict = TensorDeserializer(http_uri, lazy_load=True)
    test_dict.to_redis(redis_client, model_name)
    end = time.monotonic()
    rate = test_dict.total_bytes_read / (end - start)
    print(
        f"Loaded {test_dict.total_bytes_read / gibibyte:0.2f} GiB at"
        f" {rate / mebibyte:0.2f} MiB/s in {end - start:0.2f}s"
    )


def bench_redis():
    # Establish raw TCP connection to redis server.
    RECV_BUF_SIZE = 256 * mebibyte
    redis_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bufsize = redis_tcp.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    # redis_tcp.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 0)
    # redis_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)
    after_bufsize = redis_tcp.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)

    print("Buffer size [Before]:%d" % bufsize)
    print("Buffer size [After]:%d" % after_bufsize)

    redis_tcp.connect(("localhost", 6379))

    buf = bytearray(512 * mebibyte)

    # Loop over redis keys, and read them into memory.
    for key in redis_client.scan_iter(f"{model_name}:*:module"):
        start = time.monotonic()
        redis_tcp.send(f"GET {key.decode('utf-8')}\r\n".encode("utf-8"))
        # Loop over tcp bytes until we get a \r\n
        sz_resp = b""
        while True:
            b = redis_tcp.recv(1)
            sz_resp += b
            if sz_resp[-2:] == b"\r\n":
                break
        sz_str = sz_resp.decode("utf-8").strip()[1:]
        if sz_str == "-1":
            print("Key not found")
            break
        sz = int(sz_str)
        left = sz
        mv = memoryview(buf)
        read_ct = 0
        while left > 0:
            num_bytes = redis_tcp.recv_into(mv, left, socket.MSG_WAITALL)
            mv = mv[num_bytes:]
            left -= num_bytes
            read_ct += 1
        end = time.monotonic()
        rate = sz / (end - start)
        # read trailing \r\n
        redis_tcp.recv(2)

        print(
            f"{key.decode('utf-8')}: Read {sz / mebibyte:0.2f} MiB at"
            f" {rate / mebibyte:0.2f} MiB/s in {(end - start) * 1000:0.2f}ms"
            f" in {read_ct} reads"
        )


# load_redis()

test_dict = TensorDeserializer(
    RedisStreamFile(f"redis://localhost:6379/{model_name}"),
    lazy_load=True,
)

all_begin = time.monotonic()
for name in test_dict:
    begin = time.monotonic()
    test_dict[name]
    end = time.monotonic()
    print(f"{name}: {end - begin:0.2f}s")
all_end = time.monotonic()

print(
    f"Total bytes read: {test_dict.total_bytes_read / gibibyte:0.2f} GiB in"
    f" {all_end - all_begin:0.2f}s"
)


exit(0)

import argparse
import gc
import logging
import os
import socket
import time

import redis
import torch

from tensorizer.serialization import TensorDeserializer
from tensorizer.stream_io import (
    CURLStreamFile,
    RedisStreamFile,
    default_s3_read_endpoint,
    open_stream,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Read in model name from command line, or env var, or default to gpt-neo-2.7B
model_name_default = os.getenv("MODEL_NAME") or "EleutherAI/gpt-neo-2.7B/fp16"
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
parser.add_argument(
    "--redis",
    type=str,
    default="redis://localhost:6379",
    help="Redis URI to use for testing (default: redis://localhost:6379)",
)
parser.add_argument(
    "--load_redis",
    type=bool,
    default=True,
    help="Whether to load the model into redis (default: True)",
)
parser.add_argument(
    "--start",
    type=int,
    default=18,
    help="Starting buffer size power (default: 18)",
)
parser.add_argument(
    "--end",
    type=int,
    default=28,
    help="Ending buffer size power (default: 28)",
)
args = parser.parse_args()

model_name: str = args.model

http_uri = (
    "http://tensorized.accel-object.ord1.coreweave.com"
    f"/{model_name}/model.tensors"
)

s3_uri = f"s3://tensorized/{model_name}/model.tensors"

# Get nodename from environment, or default to os.uname().nodename
nodename = os.getenv("K8S_NODE_NAME") or os.uname().nodename

logging.info(f"{nodename} -- Testing {http_uri}")

kibibyte = 1 << 10
mebibyte = 1 << 20
gibibyte = 1 << 30


# Collect GPU data
try:
    cudadev = torch.cuda.current_device()
    gpu_gb = int(torch.cuda.get_device_properties(0).total_memory / gibibyte)
    gpu_name = torch.cuda.get_device_name(cudadev)
    has_gpu = True
except AssertionError:
    gpu_gb = 0
    gpu_name = "CPU"
    has_gpu = False

# Parse redis URI
redis_uri = args.redis
redis_host = redis_uri.split("://")[1].split(":")[0]
redis_port = int(redis_uri.split("://")[1].split(":")[1])

redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)


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
        f"{nodename} -- curl:  "
        f"gpu: {gpu_name} ({gpu_gb} GiB), raw read "
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
    scheme = source.split("://")[0]
    scheme_pad = " " * (5 - len(scheme))
    source = open_stream(
        source, s3_endpoint=default_s3_read_endpoint, buffer_size=buffer_size
    )
    start = time.monotonic()
    test_dict = TensorDeserializer(
        source,
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
        f"{nodename} -- {scheme}:{scheme_pad} "
        f"gpu: {gpu_name} ({gpu_gb} GiB), loaded  "
        f" {total_sz / mebibyte:0.2f} MiB at"
        f" {total_sz / mebibyte / (end - start):0.2f} MiB/s,"
        f" {buffer_size / kibibyte} KiB stream buffer size, plaid:"
        f" {plaid_mode}, verify_hash: {verify_hash}, lazy_load:"
        f" {lazy_load or plaid_mode}, cached: {cached} by {cached_by}"
    )

    test_dict.close()
    del test_dict
    if hasattr(torch, "cuda"):
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
    logging.info(
        f"{nodename} -- redis: Loaded"
        f" {test_dict.total_bytes_read / gibibyte:0.2f} GiB at"
        f" {rate / mebibyte:0.2f} MiB/s in {end - start:0.2f}s"
    )


def bench_redis(
    plaid_mode=False,
    verify_hash=False,
    lazy_load=False,
    buffer_size=256 * kibibyte,
):
    test_dict = TensorDeserializer(
        RedisStreamFile(
            f"redis://{redis_host}:{redis_port}/{model_name}",
            buffer_size=buffer_size,
        ),
        lazy_load=lazy_load,
        plaid_mode=plaid_mode,
        verify_hash=verify_hash,
    )

    start = time.monotonic()
    for name in test_dict:
        test_dict[name]
    end = time.monotonic()
    total_sz = test_dict.total_bytes_read

    logging.info(
        f"{nodename} -- redis: "
        f"gpu: {gpu_name} ({gpu_gb} GiB), loaded  "
        f" {total_sz / mebibyte:0.2f} MiB at"
        f" {total_sz / mebibyte / (end - start):0.2f} MiB/s,"
        f" {buffer_size / kibibyte} KiB stream buffer size, plaid:"
        f" {plaid_mode}, verify_hash: {verify_hash}, lazy_load:"
        f" {lazy_load or plaid_mode}"
    )

    del test_dict

    if hasattr(torch, "cuda"):
        torch.cuda.synchronize()
    gc.collect()


def io_test_redis(buffer_size=256 * kibibyte):
    # Establish raw TCP connection to redis server.
    redis_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    redis_tcp.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 0)
    redis_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    redis_tcp.connect((redis_host, redis_port))

    buf = bytearray(512 * mebibyte)

    total_sz = 0

    start = time.monotonic()

    # Loop over redis keys, and read them into memory.
    for key in redis_client.scan_iter(f"{model_name}:*"):
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
            total_sz += num_bytes
            read_ct += 1
        redis_tcp.recv(2)

    end = time.monotonic()
    redis_tcp.close()
    logging.info(
        f"{nodename} -- redis: "
        f"gpu: {gpu_name} ({gpu_gb} GiB), raw read "
        f"{total_sz / mebibyte:0.2f} MiB at "
        f"{total_sz / mebibyte / (end - start):0.2f} MiB/s, "
        f"{buffer_size / kibibyte} KiB stream buffer size"
    )


if args.load_redis:
    load_redis()
for buffer_size_power in range(args.start, args.end):
    buffer_size = 1 << buffer_size_power
    for sample in range(5):
        io_test(buffer_size=buffer_size)
        io_test_redis(buffer_size=buffer_size)
        if has_gpu:
            bench_redis(buffer_size=buffer_size, plaid_mode=True)
        bench_redis(buffer_size=buffer_size, lazy_load=True)
        if has_gpu:
            deserialize_test(buffer_size=buffer_size, plaid_mode=True)
        deserialize_test(buffer_size=buffer_size, lazy_load=True)
        deserialize_test(source=s3_uri, buffer_size=buffer_size)
        deserialize_test(source=s3_uri, buffer_size=buffer_size, lazy_load=True)

exit(0)

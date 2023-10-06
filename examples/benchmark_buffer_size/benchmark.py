import argparse
import gc
import json
import logging
import os
import platform
import socket
import time
from typing import Optional

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

kibibyte = 1 << 10
mebibyte = 1 << 20
gibibyte = 1 << 30

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
    "--no-load-redis",
    action="store_true",
    default=False,
    help="Don't load the model into redis",
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
parser.add_argument(
    "--json",
    action="store_true",
    default=False,
    help="Output JSON lines instead of logging",
)
args = parser.parse_args()

model_name: str = args.model

http_uri = (
    "http://tensorized.accel-object.ord1.coreweave.com"
    f"/{model_name}/model.tensors"
)
https_uri = http_uri.replace("http://", "https://")
s3_uri = f"s3://tensorized/{model_name}/model.tensors"

# Get nodename from environment, or default to os.uname().nodename
nodename = os.getenv("K8S_NODE_NAME") or os.uname().nodename

# Get our region, pod name, link speed from environment
region = os.getenv("K8S_POD_REGION")
pod_name = os.getenv("K8S_POD_NAME")
link_speed = os.getenv("K8S_LINK_SPEED")

if not args.json:
    logging.info(f"{nodename} -- Testing {http_uri}")


# Collect CPU data
cpu_arch = platform.processor()
# Read CPU name from /proc/cpuinfo
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu_name = line.split(":")[1].strip()
                break
except FileNotFoundError:
    cpu_name = platform.machine()

# Collect GPU data
try:
    cudadev = torch.cuda.current_device()
    gpu_gb = torch.cuda.get_device_properties(0).total_memory // gibibyte
    gpu_name = torch.cuda.get_device_name(cudadev)
    has_gpu = True
except AssertionError:
    gpu_gb = 0
    gpu_name = cpu_name
    has_gpu = False


# Parse redis URI
redis_uri = args.redis
redis_host = redis_uri.split("://")[1].split(":")[0]
redis_port = int(redis_uri.split("://")[1].split(":")[1])

redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)


def log(
    total_sz: int,
    start: float,
    end: float,
    raw_read: bool,
    source: str,
    force_http: Optional[bool] = None,
    lazy_load: Optional[bool] = None,
    plaid_mode: Optional[bool] = None,
    verify_hash: Optional[bool] = None,
    response_headers: Optional[dict] = None,
    read_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
):
    scheme = source.split("://")[0]
    if scheme == "s3" and not force_http:
        scheme = "s3s"
        source = source.replace("s3://", "s3s://")
    scheme_pad = " " * (5 - len(scheme))

    if response_headers is None:
        response_headers = {}
    cached_by = response_headers.get("x-cache-trace", None)
    cached = response_headers.get("x-cache-status", False)

    if cached_by is not None:
        remote_peer = cached_by
    elif "localhost" in source:
        remote_peer = "localhost"
    else:
        remote_peer = source.split("//")[1].split("/")[0].split(":")[0]

    if args.json:
        log_json(
            scheme=scheme,
            start=start,
            end=end,
            duration=end - start,
            total_bytes_read=total_sz,
            rate=total_sz / (end - start),
            source=source,
            raw_read=raw_read,
            force_http=force_http,
            lazy_load=lazy_load,
            plaid_mode=plaid_mode,
            verify_hash=verify_hash,
            cached=cached,
            cached_by=cached_by,
            remote_peer=remote_peer,
            response_headers=dict(response_headers),
            read_size=read_size,
            buffer_size=buffer_size,
        )
        return

    verb_str = "raw read" if raw_read else "deserialized"
    postamble = ""
    if buffer_size is not None:
        postamble += f", {buffer_size / kibibyte} KiB buffer size"
    if read_size is not None:
        postamble += f", {read_size / kibibyte} KiB read size"
    if plaid_mode is not None:
        postamble += f", plaid: {plaid_mode}"
    if lazy_load is not None or plaid_mode is not None:
        postamble += f", lazy_load: {lazy_load or plaid_mode}"
    if verify_hash is not None:
        postamble += f", verify_hash: {verify_hash}"
    if cached_by is not None:
        postamble += f", cached: {cached} by {cached_by}"

    logging.info(
        f"{nodename} -- {scheme}:{scheme_pad} "
        f"gpu: {gpu_name} ({gpu_gb} GiB), {verb_str} "
        f"{total_sz / mebibyte:0.2f} MiB at "
        f"{total_sz / mebibyte / (end - start):0.2f} MiB/s"
        f"{postamble}"
    )


def log_json(
    **kwargs,
):
    jsonl = {
        "ts": time.time(),
        "nodename": nodename,
        "region": region,
        "pod_name": pod_name,
        "link_speed": link_speed,
        "cpu_arch": cpu_arch,
        "cpu_name": cpu_name,
        "gpu_name": gpu_name,
        "gpu_gb": gpu_gb,
        **kwargs,
    }
    print(json.dumps(jsonl))


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

    log(
        total_sz,
        start,
        end,
        True,
        source,
        buffer_size=buffer_size,
        read_size=read_size,
    )


def deserialize_test(
    source=http_uri,
    plaid_mode=False,
    verify_hash=False,
    lazy_load=False,
    force_http=False,
    buffer_size=256 * kibibyte,
):
    stream = open_stream(
        source,
        s3_endpoint=default_s3_read_endpoint,
        buffer_size=buffer_size,
        force_http=force_http,
    )
    start = time.monotonic()
    test_dict = TensorDeserializer(
        stream,
        verify_hash=verify_hash,
        plaid_mode=plaid_mode,
        lazy_load=lazy_load,
    )

    if lazy_load or plaid_mode:
        for name in test_dict:
            test_dict[name]

    end = time.monotonic()

    log(
        test_dict.total_bytes_read,
        start,
        end,
        False,
        source,
        buffer_size=buffer_size,
        force_http=force_http,
        lazy_load=lazy_load,
        plaid_mode=plaid_mode,
        verify_hash=verify_hash,
        response_headers=getattr(stream, "response_headers", {}),
    )

    test_dict.close()
    del test_dict
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()


def test_read_performance():
    # Test the speed of reading from a stream,
    # with different buffer sizes ranging from 128 KiB to 256 MiB.
    for buffer_size_power in range(args.start, args.end):
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
    if not args.json:
        logging.info(
            f"{nodename} -- redis: Loaded"
            f" {test_dict.total_bytes_read / gibibyte:0.2f} GiB at"
            f" {rate / mebibyte:0.2f} MiB/s in {end - start:0.2f}s"
        )


def bench_redis(
    uri=f"redis://{redis_host}:{redis_port}/{model_name}",
    plaid_mode=False,
    verify_hash=False,
    lazy_load=False,
    buffer_size=256 * kibibyte,
):
    start = time.monotonic()
    test_dict = TensorDeserializer(
        RedisStreamFile(
            uri,
            buffer_size=buffer_size,
        ),
        lazy_load=lazy_load,
        plaid_mode=plaid_mode,
        verify_hash=verify_hash,
    )

    if lazy_load or plaid_mode:
        for name in test_dict:
            test_dict[name]
    end = time.monotonic()

    log(
        test_dict.total_bytes_read,
        start,
        end,
        False,
        uri,
        lazy_load=lazy_load,
        plaid_mode=plaid_mode,
        verify_hash=verify_hash,
        buffer_size=buffer_size,
    )

    del test_dict

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()


def io_test_redis(buffer_size=256 * kibibyte):
    # Establish raw TCP connection to redis server.
    start = time.monotonic()
    redis_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    redis_tcp.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 0)
    redis_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    redis_tcp.connect((redis_host, redis_port))

    buf = bytearray(512 * mebibyte)

    total_sz = 0

    # Loop over redis keys, and read them into memory.
    for key in redis_client.scan_iter(f"{model_name}:*"):
        redis_tcp.send(f"GET {key.decode('utf-8')}\r\n".encode("utf-8"))
        # Loop over tcp bytes until we get a \r\n
        sz_resp = bytearray()
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
    log(total_sz, start, end, True, redis_uri, buffer_size=buffer_size)


if not args.no_load_redis:
    load_redis()
for buffer_size_power in range(args.start, args.end):
    buffer_size = 1 << buffer_size_power
    for sample in range(5):
        io_test(buffer_size=buffer_size)
        io_test_redis(buffer_size=buffer_size)
        bench_redis(buffer_size=buffer_size, lazy_load=True)
        if has_gpu:
            bench_redis(buffer_size=buffer_size, plaid_mode=True)
        deserialize_test(buffer_size=buffer_size, lazy_load=True)
        if has_gpu:
            deserialize_test(buffer_size=buffer_size, plaid_mode=True)
        deserialize_test(
            source=https_uri, buffer_size=buffer_size, lazy_load=True
        )
        if has_gpu:
            deserialize_test(
                source=https_uri, buffer_size=buffer_size, plaid_mode=True
            )
        deserialize_test(source=s3_uri, buffer_size=buffer_size, lazy_load=True)
        if has_gpu:
            deserialize_test(
                source=s3_uri, buffer_size=buffer_size, plaid_mode=True
            )
        deserialize_test(
            source=s3_uri,
            buffer_size=buffer_size,
            lazy_load=True,
            force_http=True,
        )
        if has_gpu:
            deserialize_test(
                source=s3_uri,
                buffer_size=buffer_size,
                plaid_mode=True,
                force_http=True,
            )

import argparse
import gc
import json
import logging
import os
import platform
import socket
import time
from typing import Optional

import psutil
import redis
import torch

from tensorizer.serialization import TensorDeserializer, TensorSerializer
from tensorizer.stream_io import (
    CURLStreamFile,
    RedisStreamFile,
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

access_key = os.getenv("S3_ACCESS_KEY")
secret_key = os.getenv("S3_SECRET_KEY")

# Read in model name from command line, or env var, or default to gpt-neo-2.7B
model_name_default = os.getenv("MODEL_NAME") or "EleutherAI/pythia-12b/fp16"
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
    "--start-buffer-size",
    type=int,
    default=18,
    help="Starting buffer size power (default: 18)",
)
parser.add_argument(
    "--end-buffer-size",
    type=int,
    default=28,
    help="Ending buffer size power (default: 28)",
)
parser.add_argument(
    "--start-plaid-buffers",
    type=int,
    default=1,
    help="Starting plaid buffers power (default: 1)",
)
parser.add_argument(
    "--end-plaid-buffers",
    type=int,
    default=4,
    help="Ending plaid buffers power (default: 4)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=5,
    help="Number of iterations to run per permutation (default: 5)",
)
parser.add_argument(
    "--test-https",
    action="store_true",
    default=False,
    help="Test HTTPS download speeds",
)
parser.add_argument(
    "--test-s3",
    action="store_true",
    default=False,
    help="Test S3 download speeds",
)
parser.add_argument(
    "--local-only",
    action="store_true",
    default=False,
    help="Only test local speeds",
)
parser.add_argument(
    "--json",
    action="store_true",
    default=False,
    help="Output JSON lines instead of logging",
)
parser.add_argument(
    "--file-prefix", default="", help="Prefix for file names, can include path"
)
parser.add_argument(
    "--convert-json", default="", help="Convert JSON to human readable"
)
parser.add_argument(
    "--s3-endpoint",
    type=str,
    help="The S3 storage URL to load the models from (default: accel-object.ord1.coreweave.com)",
    default="accel-object.ord1.coreweave.com"
)
parser.add_argument(
    "--bucket",
    type=str,
    help="The bucket where the models are located (default: tensorized)",
    default="tensorized"
)
args = parser.parse_args()

model_name: str = args.model

http_uri = f"http://{args.s3_endpoint}/{args.bucket}/{model_name}/model.tensors"

https_uri = http_uri.replace("http://", "https://")
s3_uri = f"s3://{args.bucket}/{model_name}/model.tensors"
sanitized_model_file = model_name.replace("/", "_")
file_uri = f"{args.file_prefix}{sanitized_model_file}.tensors"
local_uri = f"http://localhost:3000/{sanitized_model_file}.tensors"

s3_endpoint = f"http://{args.s3_endpoint}"
if args.test_https:
    s3_endpoint = s3_endpoint.replace("http://", "https://")

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
        for cpu_line in f:
            if cpu_line.startswith("model name"):
                cpu_name = cpu_line.split(":")[1].strip()
                break
except FileNotFoundError:
    cpu_name = platform.machine()

# Collect GPU data
try:
    cudadev = torch.cuda.current_device()
    gpu_gb = torch.cuda.get_device_properties(0).total_memory // gibibyte
    gpu_name = torch.cuda.get_device_name(cudadev)
    map_device = torch.device("cuda")
    has_gpu = True
except (AssertionError, RuntimeError):
    gpu_gb = psutil.virtual_memory().total // gibibyte
    gpu_name = cpu_name
    has_gpu = False
    map_device = torch.device("cpu")


# Parse redis URI
redis_uri = args.redis
redis_host = redis_uri.split("://")[1].split(":")[0]
redis_port = int(redis_uri.split("://")[1].split(":")[1])

redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)

# Build range of buffer sizes to test
buffer_sizes = [
    1 << i for i in range(args.start_buffer_size, args.end_buffer_size)
]
plaid_sizes = [
    1 << i
    for i in range(args.start_plaid_buffers - 1, args.end_plaid_buffers + 1)
]


def log(
    total_sz: int,
    duration: float,
    raw_read: bool,
    source: str,
    nodename: str = nodename,
    cpu_name: str = cpu_name,
    cpu_arch: str = cpu_arch,
    gpu_name: str = gpu_name,
    gpu_gb: int = gpu_gb,
    force_http: Optional[bool] = None,
    lazy_load: Optional[bool] = None,
    plaid_mode: Optional[bool] = None,
    plaid_mode_buffers: Optional[int] = None,
    verify_hash: Optional[bool] = None,
    response_headers: Optional[dict] = None,
    read_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
):
    scheme = source.split("://")[0]
    if scheme not in ["http", "https", "s3", "s3s", "file", "redis"]:
        if source.endswith(".pt"):
            scheme = "torch"
        else:
            scheme = "file"
    if scheme == "http" and "localhost" in source:
        scheme = "local"
    if scheme == "s3" and not force_http:
        scheme = "s3s"
        source = source.replace("s3://", "s3s://")
    scheme_pad = " " * (5 - len(scheme))

    if response_headers is None:
        response_headers = {}
    response_headers = {k.lower(): v for k, v in response_headers.items()}
    cached_by = response_headers.get("x-cache-location", None)
    cached = response_headers.get("x-cache-status", False)

    if cached_by is not None:
        remote_peer = cached_by
    elif "localhost" in source:
        remote_peer = "localhost"
    else:
        if "//" in source:
            remote_peer = source.split("//")[1].split("/")[0].split(":")[0]
        else:
            remote_peer = "filesystem"

    if args.json:
        if not plaid_mode:
            plaid_mode_buffers = None
        log_json(
            nodename=nodename,
            cpu_name=cpu_name,
            cpu_arch=cpu_arch,
            gpu_name=gpu_name,
            gpu_gb=gpu_gb,
            scheme=scheme,
            duration=duration,
            total_bytes_read=total_sz,
            rate=total_sz / duration,
            source=source,
            raw_read=raw_read,
            force_http=force_http,
            lazy_load=lazy_load,
            plaid_mode=plaid_mode,
            plaid_buffers=plaid_mode_buffers,
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
        postamble += f", plaid_mode: {plaid_mode}"
    if plaid_mode_buffers is not None and plaid_mode:
        postamble += f", plaid_buffers: {plaid_mode_buffers}"
    if lazy_load is not None:
        postamble += f", lazy_load: {lazy_load}"
    if verify_hash is not None:
        postamble += f", verify_hash: {verify_hash}"
    if cached_by is not None:
        postamble += f", cached: {cached} by {cached_by}"

    logging.info(
        f"{nodename} -- {scheme}:{scheme_pad} "
        f"gpu: {gpu_name} ({gpu_gb} GiB), {verb_str} "
        f"{total_sz / mebibyte:0.2f} MiB at "
        f"{total_sz / mebibyte / duration:0.2f} MiB/s"
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


def json_to_human(path: str):
    with open(path) as f:
        for line in f:
            jsonl = json.loads(line)
            log(
                jsonl["total_bytes_read"],
                jsonl["duration"],
                jsonl["raw_read"],
                jsonl["source"],
                nodename=jsonl["nodename"],
                cpu_name=jsonl["cpu_name"],
                cpu_arch=jsonl["cpu_arch"],
                gpu_name=jsonl["gpu_name"],
                gpu_gb=jsonl["gpu_gb"],
                force_http=jsonl.get("force_http", None),
                lazy_load=jsonl.get("lazy_load", None),
                plaid_mode=jsonl.get("plaid_mode", None),
                plaid_mode_buffers=jsonl.get("plaid_buffers", None),
                verify_hash=jsonl.get("verify_hash", None),
                response_headers=jsonl.get("response_headers", {}),
                read_size=jsonl.get("read_size", None),
                buffer_size=jsonl.get("buffer_size", None),
            )


def io_test(
    source=http_uri, read_size=256 * kibibyte, buffer_size=256 * mebibyte
):
    # Read the stream `read_size` at a time.
    buffer = bytearray(read_size)
    total_sz = 0
    start = time.monotonic()
    if ":" not in source:
        # presume local file path
        io = open(source, mode="rb", buffering=buffer_size)
    else:
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
        end - start,
        True,
        source,
        buffer_size=buffer_size,
        read_size=read_size,
    )


def deserialize_test(
    source=http_uri,
    s3_endpoint=s3_endpoint,
    access_key=access_key,
    secret_key=secret_key,
    plaid_mode=False,
    verify_hash=False,
    lazy_load=False,
    force_http=False,
    plaid_mode_buffers=4,
    buffer_size=256 * kibibyte,
):
    if not plaid_mode:
        plaid_mode_buffers = None
    stream = open_stream(
        source,
        s3_endpoint=s3_endpoint,
        s3_access_key_id=access_key,
        s3_secret_access_key=secret_key,
        buffer_size=buffer_size,
        force_http=force_http,
    )
    start = time.monotonic()
    test_dict = TensorDeserializer(
        stream,
        verify_hash=verify_hash,
        plaid_mode=plaid_mode,
        lazy_load=lazy_load,
        plaid_mode_buffers=plaid_mode_buffers,
    )

    if lazy_load:
        for name in test_dict:
            test_dict[name]

    end = time.monotonic()

    log(
        test_dict.total_bytes_read,
        end - start,
        False,
        source,
        buffer_size=buffer_size,
        force_http=force_http,
        lazy_load=lazy_load,
        plaid_mode=plaid_mode,
        plaid_mode_buffers=plaid_mode_buffers,
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


def prep_local():
    logging.info(f"{http_uri} http uri")
    # Store on Redis
    start = time.monotonic()
    test_dict = TensorDeserializer(http_uri)
    if not args.no_load_redis:
        test_dict.to_redis(redis_client, model_name)
    end = time.monotonic()
    bytes_read = test_dict.total_bytes_read
    rate = bytes_read / (end - start)
    if not args.json and not args.no_load_redis:
        logging.info(
            f"{nodename} -- redis: Loaded"
            f" {bytes_read / gibibyte:0.2f} GiB at"
            f" {rate / mebibyte:0.2f} MiB/s in {end - start:0.2f}s"
        )
    state_dict = dict(test_dict)
    # Serialize to local file
    start = time.monotonic()
    ts = TensorSerializer(file_uri)
    ts.write_state_dict(state_dict)
    ts.close()
    end = time.monotonic()
    rate = bytes_read / (end - start)
    if not args.json:
        logging.info(
            f"{nodename} -- file: Serialized"
            f" {bytes_read / gibibyte:0.2f} GiB at"
            f" {rate / mebibyte:0.2f} MiB/s in {end - start:0.2f}s"
        )
    # Save using torch
    start = time.monotonic()
    torch.save(state_dict, f"{args.file_prefix}{sanitized_model_file}.pt")
    end = time.monotonic()
    rate = bytes_read / (end - start)
    if not args.json:
        logging.info(
            f"{nodename} -- torch: Serialized"
            f" {bytes_read / gibibyte:0.2f} GiB at"
            f" {rate / mebibyte:0.2f} MiB/s in {end - start:0.2f}s"
        )


def bench_torch():
    # Load from torch
    start = time.monotonic()
    state_dict = torch.load(
        f"{sanitized_model_file}.pt", map_location=map_device
    )
    end = time.monotonic()
    bytes_read = os.path.getsize(f"{sanitized_model_file}.pt")
    log(
        bytes_read,
        end - start,
        False,
        f"{sanitized_model_file}.pt",
    )
    del state_dict

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()


def bench_redis(
    uri=f"redis://{redis_host}:{redis_port}/{model_name}",
    plaid_mode=False,
    verify_hash=False,
    lazy_load=False,
    buffer_size=256 * kibibyte,
    plaid_mode_buffers=4,
):
    if not plaid_mode:
        plaid_mode_buffers = None
    start = time.monotonic()
    test_dict = TensorDeserializer(
        RedisStreamFile(
            uri,
            buffer_size=buffer_size,
        ),
        lazy_load=lazy_load,
        plaid_mode=plaid_mode,
        plaid_mode_buffers=plaid_mode_buffers,
        verify_hash=verify_hash,
    )

    if lazy_load:
        for name in test_dict:
            test_dict[name]
    end = time.monotonic()

    log(
        test_dict.total_bytes_read,
        end - start,
        False,
        uri,
        lazy_load=lazy_load,
        plaid_mode=plaid_mode,
        plaid_mode_buffers=plaid_mode_buffers,
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
    log(total_sz, end - start, True, redis_uri, buffer_size=buffer_size)


if args.convert_json:
    json_to_human(args.convert_json)
    exit(0)

prep_local()

for buffer_size in buffer_sizes:
    for sample in range(args.iterations):
        bench_torch()
        if not args.local_only:
            io_test(buffer_size=buffer_size)
        io_test(source=local_uri, buffer_size=buffer_size)
        io_test_redis(buffer_size=buffer_size)
        deserialize_test(file_uri, buffer_size=buffer_size, lazy_load=False)
        deserialize_test(file_uri, buffer_size=buffer_size, lazy_load=True)
        if has_gpu:
            for plaid_buffers in plaid_sizes:
                deserialize_test(
                    file_uri,
                    buffer_size=buffer_size,
                    plaid_mode=True,
                    plaid_mode_buffers=plaid_buffers,
                )
        bench_redis(buffer_size=buffer_size, lazy_load=False)
        bench_redis(buffer_size=buffer_size, lazy_load=True)
        if has_gpu:
            for plaid_buffers in plaid_sizes:
                bench_redis(
                    buffer_size=buffer_size,
                    plaid_mode=True,
                    plaid_mode_buffers=plaid_buffers,
                )
        deserialize_test(
            source=local_uri, buffer_size=buffer_size, lazy_load=False
        )
        deserialize_test(
            source=local_uri, buffer_size=buffer_size, lazy_load=True
        )
        if has_gpu:
            for plaid_buffers in plaid_sizes:
                deserialize_test(
                    source=local_uri,
                    buffer_size=buffer_size,
                    plaid_mode=True,
                    plaid_mode_buffers=plaid_buffers,
                )
        if not args.local_only:
            deserialize_test(buffer_size=buffer_size, lazy_load=False)
            deserialize_test(buffer_size=buffer_size, lazy_load=True)
            if has_gpu:
                for plaid_buffers in plaid_sizes:
                    deserialize_test(
                        buffer_size=buffer_size,
                        plaid_mode=True,
                        plaid_mode_buffers=plaid_buffers,
                    )
        if args.test_https and not args.local_only:
            deserialize_test(
                source=https_uri, buffer_size=buffer_size, lazy_load=False
            )
            deserialize_test(
                source=https_uri, buffer_size=buffer_size, lazy_load=True
            )
            if has_gpu:
                for plaid_buffers in plaid_sizes:
                    deserialize_test(
                        source=https_uri,
                        buffer_size=buffer_size,
                        plaid_mode=True,
                        plaid_mode_buffers=plaid_buffers,
                    )
        if args.test_s3 and not args.local_only:
            deserialize_test(
                source=s3_uri,
                buffer_size=buffer_size,
                lazy_load=False,
                force_http=True,
            )
            deserialize_test(
                source=s3_uri,
                buffer_size=buffer_size,
                lazy_load=True,
                force_http=True,
            )
            if has_gpu:
                for plaid_buffers in plaid_sizes:
                    deserialize_test(
                        source=s3_uri,
                        buffer_size=buffer_size,
                        plaid_mode=True,
                        plaid_mode_buffers=plaid_buffers,
                        force_http=True,
                    )
            if args.test_https:
                deserialize_test(
                    source=s3_uri,
                    buffer_size=buffer_size,
                    lazy_load=False,
                )
                deserialize_test(
                    source=s3_uri,
                    buffer_size=buffer_size,
                    lazy_load=True,
                )
                if has_gpu:
                    for plaid_buffers in plaid_sizes:
                        deserialize_test(
                            source=s3_uri,
                            buffer_size=buffer_size,
                            plaid_mode=True,
                            plaid_mode_buffers=plaid_buffers,
                        )

import argparse
import contextlib
import json
import mmap
import pathlib
import time
from hashlib import sha256
from typing import Optional, Union

from ._cgroup_cpu_count import effective_cpu_count
from ._encryption import (
    AsymmetricParams,
    ChunkedEncryption,
    SequentialEncryption,
    SymmetricParams,
    as_ucstr,
    crypto_box_detached,
    crypto_box_open_detached,
    crypto_secretbox_detached,
    crypto_secretbox_open_detached,
)

cpu_count: int = effective_cpu_count()

parser = argparse.ArgumentParser(
    description="internal command for testing encryption performance"
)
parser.add_argument(
    "--generate",
    metavar="NUMBYTES",
    type=int,
    nargs="?",
    default=None,
    help="generate a message to test with this size (default: 256 MiB)",
)
parser.add_argument(
    "-f",
    "--file",
    type=pathlib.Path,
    default=None,
    help=(
        "encrypt or decrypt a file inplace"
        " (requires --crypt-info as well as one of --encrypt or --decrypt)"
    ),
)
parser.set_defaults(encrypt=None)
parser.add_argument(
    "--encrypt", action="store_true", help="encrypt a file (requires --file)"
)
parser.add_argument(
    "--decrypt",
    action="store_false",
    dest="encrypt",
    help="decrypt a file (requires --file)",
)
parser.add_argument(
    "-i",
    "--crypt-info",
    type=pathlib.Path,
    default=None,
    help="path to store and retrieve key, nonces, and MACs (requires --file)",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=2 << 20,
    help="chunk size for parallel encryption",
)
parser.add_argument(
    "-d",
    "--delayed",
    action="store_true",
    help="enable delayed verification",
)
parser.add_argument(
    "-t",
    "--threads",
    type=int,
    default=cpu_count,
    help=(
        "maximum number of threads for parallel encryption"
        f" (default: {cpu_count})"
    ),
)
args = parser.parse_args()

if args.file is not None:
    if args.generate is not None:
        parser.error("Cannot specify both --file and --generate")
    if args.encrypt is None:
        parser.error("Must specify either --encrypt or --decrypt with --file")
    if args.crypt_info is None:
        parser.error("Must specify --crypt-info with --file")
    if not args.encrypt:
        if not args.file.is_file():
            parser.error("--file path is not a file and cannot be decrypted")
        if not args.crypt_info.is_file():
            parser.error("--crypt-info path is not a file")
else:
    if (args.crypt_info, args.encrypt) != (None, None):
        parser.error(
            "--crypt-info, --encrypt, and --decrypt are only valid"
            " when used with --file"
        )
    if args.generate is None:
        args.generate = 256 << 20


class Timer(contextlib.AbstractContextManager):
    def __init__(self):
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        self.start = time.monotonic()
        return super().__enter__()

    def __exit__(self, __exc_type, __exc_value, __traceback):
        self.end = time.monotonic()
        self.elapsed = self.end - self.start

    def rate(self, count) -> Union[float, str]:
        return "\u221e" if self.elapsed == 0 else count / self.elapsed


def num_chunks(total_size: int, chunk_size: int) -> int:
    return (total_size // chunk_size) + (total_size % chunk_size != 0)


def preview(text, length: int, as_hex: bool = False):
    with memoryview(text) as mv:
        truncated = mv[:length].hex() if as_hex else str(bytes(mv[:length]))
        return truncated + ("..." if len(mv) > length else "")


def sequential_test(
    buffer: Union[bytearray, memoryview], asymmetric: bool = False
):
    if asymmetric:
        params = AsymmetricParams.random()
        key_views = {"pk": as_ucstr(params.pk), "sk": as_ucstr(params.sk)}
        encrypt = crypto_box_detached
        decrypt = crypto_box_open_detached
    else:
        params = SymmetricParams.random()
        key_views = {"k": as_ucstr(params.k)}
        encrypt = crypto_secretbox_detached
        decrypt = crypto_secretbox_open_detached
    buffer_view = as_ucstr(buffer)
    mac_view = as_ucstr(params.mac)
    nonce_view = as_ucstr(params.nonce)
    m_len = len(buffer)

    timer = Timer()
    mebibyte = 1 << 20
    with timer:
        encrypt(
            c=buffer_view,
            mac=mac_view,
            m=buffer_view,
            mlen=m_len,
            n=nonce_view,
            **key_views,
        )

    print(
        f"Encrypted {m_len} bytes in {timer.elapsed:.6f} seconds,"
        f" {timer.rate(m_len / mebibyte)} MiB/s",
        f"Contents: {preview(buffer, 32, True)}",
        f"MAC: {params.mac.hex()}",
        sep="\n",
    )

    with timer:
        decrypt(
            m=buffer_view,
            c=buffer_view,
            mac=mac_view,
            clen=m_len,
            n=nonce_view,
            **key_views,
        )

    print(
        f"Decrypted {m_len} bytes in {timer.elapsed:.6f} seconds,"
        f" {timer.rate(m_len / mebibyte)} MiB/s",
        preview(buffer, 64),
        sep="\n",
    )


def symmetric_sequential_test(
    buffer: Union[bytearray, memoryview], key: Optional[bytes] = None
):
    if key is None:
        key = SymmetricParams.random().k
    timer = Timer()
    mebibyte = 1 << 20

    crypto = SequentialEncryption(key, buffer)
    with timer:
        crypto.encrypt()

    print(
        f"Encrypted {len(buffer)} bytes"
        f" in {timer.elapsed:.6f} seconds,"
        f" {timer.rate(len(buffer) / mebibyte)} MiB/s",
        f"Contents: {preview(buffer, 32, True)}",
        f"MACs: {preview(crypto.mac, 32, True)}",
        sep="\n",
    )

    with timer:
        crypto.decrypt()

    print(
        f"Decrypted {len(buffer)} bytes"
        f" in {timer.elapsed:.6f} seconds,"
        f" {timer.rate(len(buffer) / mebibyte)} MiB/s",
        preview(buffer, 64),
        sep="\n",
    )


def parallel_test(
    buffer: Union[bytearray, memoryview],
    chunk_size: int,
    automatic_verification: bool,
    num_threads: int,
    key: Optional[bytes] = None,
):
    if key is None:
        key = SymmetricParams.random().k

    timer = Timer()
    mebibyte = 1 << 20

    with ChunkedEncryption(
        key,
        buffer,
        chunk_size,
        num_threads=num_threads,
        automatic_verification=automatic_verification,
    ) as crypto:
        with timer:
            crypto.encrypt_all(wait=True, timeout=None)

        macs = crypto.concatenated_macs()

        print(
            f"Encrypted {len(buffer)} bytes"
            f" in {timer.elapsed:.6f} seconds,"
            f" {timer.rate(len(buffer) / mebibyte)} MiB/s",
            f"Contents: {preview(buffer, 32, True)}",
            f"MACs: {preview(macs, 32, True)}",
            sep="\n",
        )

        with timer:
            crypto.decrypt_all(wait=True, timeout=None)

        print(
            f"Decrypted {len(buffer)} bytes"
            f" in {timer.elapsed:.6f} seconds,"
            f" {timer.rate(len(buffer) / mebibyte)} MiB/s",
            preview(buffer, 64),
            sep="\n",
        )


def parallel_transform_file(
    encrypt: bool,
    path: pathlib.Path,
    crypt_info_path: pathlib.Path,
    chunk_size: int,
    automatic_verification: bool,
    num_threads: int,
):
    with contextlib.ExitStack() as context:
        file = context.enter_context(path.open("rb+"))
        buffer = context.enter_context(mmap.mmap(file.fileno(), 0))
        context.callback(buffer.flush)
        num_bytes = len(buffer)
        context = context.pop_all()
    if encrypt:
        params = SymmetricParams.random()
        key = params.k
        nonces = None
        macs = None
    else:
        with crypt_info_path.open("rb") as crypt_info_file:
            json_params = json.load(crypt_info_file)
        key = bytes.fromhex(json_params["key"])
        nonces = tuple(map(bytes.fromhex, json_params["nonces"]))
        macs = tuple(map(bytes.fromhex, json_params["macs"]))

    timer = Timer()
    mebibyte = 1 << 20

    with context, ChunkedEncryption(
        key,
        buffer,
        chunk_size,
        nonces=nonces,
        macs=macs,
        num_threads=num_threads,
        automatic_verification=automatic_verification,
    ) as crypto:
        try:
            if encrypt:
                with timer:
                    crypto.encrypt_all(wait=True, timeout=None)
            else:
                with timer:
                    crypto.decrypt_all(wait=True, timeout=None)
            macs = crypto.macs
            del crypto
        finally:
            import gc

            gc.collect()

    if encrypt:
        with crypt_info_path.open("w") as crypt_info_file:
            json_params = {
                "key": key.hex(),
                "nonces": [n.hex() for n in nonces],
                "macs": [m.hex() for m in macs],
            }
            json.dump(json_params, crypt_info_file)

    print(
        f"{'Encrypted' if encrypt else 'Decrypted'}"
        f" {num_bytes} bytes in {timer.elapsed:.6f} seconds"
        f" {timer.rate(num_bytes / mebibyte)} MiB/s"
    )


def run_tests(
    message: bytearray,
    chunk_size: int,
    threads: int,
    automatic_verification: bool,
):
    original_hash = sha256(message).digest()

    print("Asymmetric")
    sequential_test(message, asymmetric=True)

    assert sha256(message).digest() == original_hash

    print("\nSymmetric")
    sequential_test(message, asymmetric=False)

    assert sha256(message).digest() == original_hash

    key: bytes = SymmetricParams.random().k

    print("\nSymmetric (OO)")
    symmetric_sequential_test(message, key)

    assert sha256(message).digest() == original_hash

    print("\nParallel")
    parallel_test(
        message,
        chunk_size=chunk_size,
        automatic_verification=automatic_verification,
        num_threads=threads,
        key=key,
    )

    assert sha256(message).digest() == original_hash


if args.generate is not None:
    message = bytearray(b"Hello, World!")
    message_size = args.generate
    message *= num_chunks(message_size, len(message))
    message[message_size:] = b""
    assert len(message) == message_size
    run_tests(message, args.chunk_size, args.threads, not args.delayed)
else:
    parallel_transform_file(
        args.encrypt,
        args.file,
        args.crypt_info,
        chunk_size=args.chunk_size,
        automatic_verification=not args.delayed,
        num_threads=args.threads,
    )

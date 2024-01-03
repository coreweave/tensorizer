import argparse
import dataclasses
import fcntl
import io
import itertools
import json
import os
import re
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Sequence


def positive_int(i: str) -> int:
    i = int(i)
    if i <= 0:
        raise ValueError("must be positive")
    return i


def non_negative_int(i: str) -> int:
    i = int(i)
    if i < 0:
        raise ValueError("must be non-negative")
    return i


def byte_size(size: str) -> int:
    match = re.match(r"(\d+)\s*(?:([KMGTPEZY])i?)?B?$", size, re.IGNORECASE)
    if not match:
        raise ValueError("not a valid size")
    val = positive_int(match.group(1))
    multiplier = match.group(2)
    if multiplier:
        multipliers = dict(zip("KMGTPEZY", range(10, 81, 10)))
        val *= 1 << multipliers[multiplier.upper()]
    return val


def atomic_increment(
    file, metadata_func: Optional[Callable[..., bytes]] = None
) -> int:
    with lock((fd := file.fileno()), 0, 0):
        file.seek(0)
        val_bytes: bytes = file.read().split(b",", maxsplit=1)[0]
        if val_bytes:
            val = int(val_bytes)
        else:
            val = 0
        file.truncate(0)
        file.seek(0)
        if metadata_func:
            file.write(b"%d,%s" % (val + 1, metadata_func()))
        else:
            file.write(b"%d" % (val + 1))
        file.flush()
        os.fdatasync(fd)
    return val


def find_index(path: str) -> int:
    with open(path, "ab+") as file:
        return atomic_increment(file)


def synchronized_sleep(max_delay: float) -> None:
    now = time.time()
    duration = -now % max_delay
    time.sleep(duration)


def barrier(
    path: str, count: int, poll_period: float, final_delay: float = 0
) -> None:
    def time_metadata() -> bytes:
        return struct.pack("<d", time.time())

    def parse_time(t: bytes) -> float:
        return struct.unpack("<d", t)[0]

    sleep_until = 0
    try:
        with open(path, "ab+") as file:
            rank = atomic_increment(file, time_metadata)
            assert rank <= count
            if rank == count - 1:
                os.unlink(path)
            fd = file.fileno()
            for sec in range(300):
                with lock(fd, 0, 0, shared=True):
                    file.seek(0)
                    current, timestamp = file.read().split(b",", 1)
                    if int(current) == count:
                        sleep_until = parse_time(timestamp) + final_delay
                        break
                synchronized_sleep(poll_period)
            else:
                raise TimeoutError("Barrier timed out")
    except BaseException:
        if os.path.exists(path):
            os.unlink(path)
        raise
    now = time.time()
    if sleep_until > now:
        time.sleep(sleep_until - now)


def index(arg: str) -> int:
    parts = arg.split(":", maxsplit=2)
    if len(parts) == 2:
        command = parts[0].lower()
        if command == "env":
            return non_negative_int(os.environ[parts[1]])
        elif command == "find":
            return find_index(parts[1])
    else:
        return non_negative_int(arg)


@dataclasses.dataclass
class ChunkSpec:
    __slots__ = "path", "index", "index_in_file", "size"
    path: str
    index: int
    index_in_file: int
    size: int

    @property
    def file_offset(self):
        return self.size * self.index_in_file


@dataclasses.dataclass
class SubChunkSpec(ChunkSpec):
    __slots__ = "sub_chunk_size", "sub_chunk_index"
    sub_chunk_size: int
    sub_chunk_index: int

    @property
    def sub_chunk_file_offset(self):
        return self.file_offset + self.sub_chunk_size * self.sub_chunk_index


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "benchmark concurrent read/write speed on a shared filesystem"
        )
    )
    parser.add_argument(
        "--chunk", type=index, required=True, help="chunk index"
    )
    parser.add_argument("--chunk-size", type=byte_size, required=True)
    parser.add_argument("--write-size", type=byte_size, default=2 << 20)
    parser.add_argument("--read-size", type=byte_size, default=2 << 40)
    parser.add_argument("--read-split", type=positive_int, default=1)
    parser.add_argument("--max-chunks", type=positive_int)
    parser.add_argument("--separate", action="store_true")
    parser.add_argument("--super-separate", action="store_true")
    parser.add_argument("--fallocate", action="store_true")
    parser.add_argument("--barrier", type=str)
    parser.add_argument("--read-barrier", type=str)
    parser.add_argument("--finish-barrier", type=str)
    parser.add_argument("--barrier-poll-period", type=float, default=4)
    parser.add_argument("--barrier-final-delay", type=float, default=4)
    parser.add_argument("--pre-read-delay", type=float, default=0)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--no-lock", dest="lock", action="store_false")
    parser.add_argument("--report-timestamps", action="store_true")
    parser.add_argument("--data-src", type=str)
    args = parser.parse_args(argv)
    chunk_size: int = args.chunk_size
    if args.max_chunks:
        args.chunk = args.chunk % args.max_chunks
    if args.write_size > chunk_size:
        args.write_size = chunk_size
    elif chunk_size % args.write_size != 0:
        parser.error("--write-size must divide --chunk-size")
    if chunk_size % args.read_split != 0:
        parser.error("--read-split must divide --chunk-size")
    args.split_chunk_size = chunk_size // args.read_split
    if args.read_size > args.split_chunk_size:
        args.read_size = args.split_chunk_size
    elif args.split_chunk_size % args.read_size != 0:
        parser.error("--read-size must divide --chunk-size / --read-split")
    args.original_chunk = args.chunk
    args.separate = args.separate or args.super_separate
    base_file = Path(args.file)
    if args.separate:
        chunk_suffix = f".{args.chunk}"
        args.file = str(base_file.with_suffix(chunk_suffix + base_file.suffix))
        args.chunk = 0
        args.lock = False
        if args.super_separate:
            num_writes = chunk_size // args.write_size
            args.files = [
                str(
                    base_file.with_suffix(
                        f"{chunk_suffix}.{i}{base_file.suffix}"
                    )
                )
                for i in range(num_writes)
            ]
        else:
            args.files = [args.file]
    else:
        args.files = [args.file]
    if args.barrier and args.chunk == 0:
        if args.super_separate:
            total_size: int = args.write_size
        elif args.separate:
            total_size: int = chunk_size
        else:
            total_size = chunk_size * args.max_chunks
        # Delaying on this is fine, since it's synced by a barrier anyway
        for path in args.files:
            if not os.path.isfile(path) or os.stat(path).st_size != total_size:
                with open(path, "wb") as file:
                    if args.fallocate:
                        os.posix_fallocate(file.fileno(), 0, total_size)
                    file.truncate(total_size)
    elif not args.barrier:
        for path in args.files:
            if not os.path.isfile(path):
                parser.error(f"file not found: {path!r}")
    if args.barrier and not args.max_chunks:
        parser.error("--barrier requires --max-chunks")

    if args.read_barrier and not args.max_chunks:
        parser.error("--read-barrier requires --max-chunks")
    elif args.read_barrier:
        opposite_chunk: int = (
            args.original_chunk + args.max_chunks // 2
        ) % args.max_chunks
        sub_chunk_size: int = args.split_chunk_size
        if args.super_separate:
            parser.error("cannot test reads with --super-separate")
        read_chunks = [
            i % args.max_chunks
            for i in range(opposite_chunk, opposite_chunk + args.read_split)
        ]
        if args.separate:
            read_files = [
                str(base_file.with_suffix(f".{c}{base_file.suffix}"))
                for c in read_chunks
            ]
        else:
            read_files = [args.file] * len(read_chunks)
        args.read_specs = [
            SubChunkSpec(
                path=f,
                index=c,
                index_in_file=c * (not args.separate),
                size=chunk_size,
                sub_chunk_size=sub_chunk_size,
                sub_chunk_index=i % args.read_split,
            )
            for i, (f, c) in enumerate(zip(read_files, read_chunks))
        ]
    else:
        args.read_specs = []

    if args.barrier_poll_period < 0.1 or args.barrier_poll_period > 3600:
        parser.error("--barrier-poll-period must be between [0.1, 3600]")
    if args.barrier_final_delay < 0 or args.barrier_final_delay > 3600:
        parser.error("--barrier-final-delay must be between [0, 3600]")
    if args.pre_read_delay < 0:
        parser.error("--pre-read-delay must be non-negative")
    return args


def data_chunk(char: int, size: int) -> bytes:
    return bytes([char & 0xFF]) * size


def data_view(path: str, chunk_index: int, chunk_size: int) -> bytes:
    """
    Return `chunk_size` bytes read from the file at `path`
    from indices ``chunk_index * chunk_size``
    to ``(chunk_index + 1) * chunk_size`` as if the file data looped infinitely.
    """
    buffer = io.BytesIO()
    original_size: int = os.stat(path).st_size
    if original_size == 0:
        raise ValueError("Cannot make a data view from an empty file")
    buffer.seek(chunk_size - 1)
    buffer.write(b"\x00")
    buffer.seek(0)
    absolute_start: int = chunk_index * chunk_size
    view_start: int = absolute_start % original_size
    view_end: int = view_start + chunk_size
    loops, remainder = divmod(view_end, original_size)
    with open(path, "rb") as file:
        file: io.BufferedReader
        file.seek(view_start)
        file.readinto(buffer.getbuffer())
        offset: int = original_size - view_start
        if loops >= 1:
            file.seek(0)
            file.readinto(buffer.getbuffer()[offset:])
            loops -= 1
            # Below only does anything if the previous read was the entire file
            file_start, file_end = offset, offset + original_size
            offset = file_end
            while loops > 1:
                next_offset = offset + original_size
                buffer.getbuffer()[offset:next_offset] = buffer.getbuffer()[
                    file_start:file_end
                ]
                offset = next_offset
                loops -= 1
            if loops == 1:
                assert offset + remainder == chunk_size
                prefix_end_offset = file_start + remainder
                buffer.getbuffer()[offset:chunk_size] = buffer.getbuffer()[
                    file_start:prefix_end_offset
                ]
    return buffer.getvalue()


@contextmanager
def lock(
    fd: int,
    offset: int,
    size: int,
    *,
    shared: bool = False,
    enable: bool = True,
):
    if not enable:
        yield
        return
    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    fcntl.lockf(fd, mode, size, offset, io.SEEK_SET)
    try:
        yield
    finally:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN, size, offset, io.SEEK_SET)
        except OSError:
            pass


def main(argv=None) -> None:
    args = parse_args(argv)
    chunk: int = args.chunk
    original_chunk: int = args.original_chunk
    max_chunks: int = args.max_chunks
    chunk_size: int = args.chunk_size
    file_path: str = args.file
    repeats: int = args.repeats
    do_lock: bool = args.lock
    barrier_path: Optional[str] = args.barrier
    read_barrier_path: Optional[str] = args.read_barrier
    data_src: Optional[str] = args.data_src
    report_timestamps: bool = args.report_timestamps
    barrier_poll_period: float = args.barrier_poll_period
    barrier_final_delay: float = args.barrier_final_delay
    write_size: Optional[int] = args.write_size
    super_separate: bool = args.super_separate
    file_paths: Sequence[str] = args.files
    read_specs: Sequence[SubChunkSpec] = args.read_specs
    read_size: int = args.read_size
    pre_read_delay: float = args.pre_read_delay
    finish_barrier_path: Optional[str] = args.finish_barrier

    start_offset: int = chunk * chunk_size
    end_offset: int = (chunk + 1) * chunk_size

    data_chunk_size: int = min(write_size, chunk_size)
    assert chunk_size % data_chunk_size == 0
    if data_src:
        data: bytes = data_view(data_src, original_chunk, chunk_size)
        blocks = (
            memoryview(data)[i : i + data_chunk_size]
            for i in range(0, chunk_size, data_chunk_size)
        )
        blocks = itertools.cycle(blocks)
    else:
        data: bytes = data_chunk(original_chunk, data_chunk_size)
        blocks = itertools.cycle(data)

    # noinspection PyUnusedLocal
    block = None
    try:
        if repeats < 1:
            repeats = 1
        for i in range(repeats):
            if i != 0:
                time.sleep(2)
            bytes_written: int = 0

            if barrier_path:
                barrier(
                    barrier_path,
                    max_chunks,
                    barrier_poll_period,
                    barrier_final_delay,
                )

            write_start_timestamp: int = time.time_ns()
            write_start_time: int = time.monotonic_ns()
            if not super_separate or len(file_paths) == 1:
                with open(file_path, "rb+") as file, lock(
                    (fd := file.fileno()),
                    start_offset,
                    chunk_size,
                    enable=do_lock,
                ):
                    write_opened_time: int = time.monotonic_ns()
                    for offset, block in zip(
                        range(start_offset, end_offset, data_chunk_size), blocks
                    ):
                        bytes_written += os.pwrite(fd, block, offset)
                write_end_time: int = time.monotonic_ns()
                write_end_timestamp: int = time.time_ns()
                write_open_duration: int = write_opened_time - write_start_time
                write_io_duration: int = write_end_time - write_opened_time
            else:
                assert super_separate
                write_open_duration: int = 0
                write_io_duration: int = 0
                for path, block in zip(file_paths, blocks):
                    open_start_time = time.monotonic_ns()
                    with open(path, "rb+") as file:
                        open_end_time: int = time.monotonic_ns()
                        write_open_duration += open_end_time - open_start_time
                        bytes_written += os.pwrite(file.fileno(), block, 0)
                        write_end_time: int = time.monotonic_ns()
                        write_io_duration += write_end_time - open_end_time
                write_end_time: int = time.monotonic_ns()
                write_end_timestamp: int = time.time_ns()
            write_total_duration: int = write_end_time - write_start_time
            assert (
                bytes_written == chunk_size
            ), f"{bytes_written} != {chunk_size}"
    finally:
        del blocks, block

    empty = {}
    write_timestamps = (
        {
            "write_start_timestamp": write_start_timestamp,
            "write_end_timestamp": write_end_timestamp,
        }
        if report_timestamps
        else empty
    )
    write_times = {
        **write_timestamps,
        "write_open_duration_ms": write_open_duration / 1e6,
        "write_io_duration_ms": write_io_duration / 1e6,
        "write_total_duration_ms": write_total_duration / 1e6,
    }

    if read_barrier_path:
        assert read_specs
        if pre_read_delay > 0:
            time.sleep(pre_read_delay)
        barrier(
            read_barrier_path,
            max_chunks,
            barrier_poll_period,
            barrier_final_delay,
        )
        read_start_timestamp: int = time.time_ns()
        read_start_time: int = time.monotonic_ns()
        bytes_read: int = 0
        read_open_duration: int = 0
        read_io_duration: int = 0
        for spec in read_specs:
            read_offset: int = spec.sub_chunk_file_offset
            read_chunk_size: int = spec.sub_chunk_size
            read_path: str = spec.path
            read_open_start_time: int = time.monotonic_ns()
            with open(read_path, "rb") as file, lock(
                (fd := file.fileno()),
                read_offset,
                read_chunk_size,
                shared=True,
                enable=do_lock,
            ):
                read_open_end_time: int = time.monotonic_ns()
                read_open_duration += read_open_end_time - read_open_start_time
                for read_block_offset in range(
                    read_offset, read_offset + read_chunk_size, read_size
                ):
                    bytes_read += len(
                        os.pread(fd, read_size, read_block_offset)
                    )
            read_io_end_time: int = time.monotonic_ns()
            read_io_duration += read_io_end_time - read_open_end_time
        read_end_time: int = time.monotonic_ns()
        read_end_timestamp: int = time.time_ns()
        read_total_duration = read_end_time - read_start_time
        assert bytes_read == chunk_size, f"{bytes_read} != {chunk_size}"
        read_timestamps = (
            {
                "read_start_timestamp": read_start_timestamp,
                "read_end_timestamp": read_end_timestamp,
            }
            if report_timestamps
            else empty
        )
        read_times = {
            **read_timestamps,
            "read_open_duration_ms": read_open_duration / 1e6,
            "read_io_duration_ms": read_io_duration / 1e6,
            "read_total_duration_ms": read_total_duration / 1e6,
        }
    else:
        read_times = empty
    node_name = os.getenv("SHARED_FS_BENCHMARK_NAME", "")
    node = {"node": node_name} if node_name else empty
    log_entry = {
        "chunk": original_chunk,
        **node,
        **write_times,
        **read_times,
    }
    print(json.dumps(log_entry, indent=None), flush=True)
    if finish_barrier_path:
        barrier(finish_barrier_path, max_chunks, min(2.0, barrier_poll_period))


if __name__ == "__main__":
    main()

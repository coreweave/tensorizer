import ctypes
import mmap
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from tensorizer._syscalls import prefault


def prefault_mmap(
    buffer: mmap.mmap, pool: Optional[ThreadPoolExecutor], num_threads: int
) -> None:
    size: int = len(buffer)
    if size <= 0:
        return
    elif num_threads <= 0:
        raise ValueError("num_threads must be positive")

    # The minimum chunk size is around 16 KiB, since the overhead of threading
    # outweighs the benefits of parallelism for too-small chunks.
    # If there is no thread pool, there is only one chunk.
    min_chunk_size: int = size if pool is None else 16384

    # Decide on a chunk size between `min_chunk_size` and `size`,
    # rounded up to the nearest multiple of the page size,
    # but not larger than `size`.
    chunk_size: int = size // num_threads
    chunk_size = min_chunk_size if chunk_size < min_chunk_size else chunk_size
    chunk_size -= chunk_size % -mmap.PAGESIZE
    chunk_size = size if size < chunk_size else chunk_size

    offsets = range(0, size, chunk_size)
    # num_chunks is bounded between [1, num_threads]
    num_chunks: int = len(offsets)

    buf = (ctypes.c_ubyte * size).from_buffer(buffer)
    try:
        if num_chunks == 1:
            prefault(buf, size)
        else:
            barrier = threading.Barrier(num_chunks, timeout=2e-2)

            def _synchronized_prefault(offset: int) -> None:
                # Try to wait for threads to spin up before starting to fault
                # pages, as those operations interfere with each other.
                # If the requested number of threads can't be reached
                # by the end of the timeout, proceed anyway.
                remaining: int = size - offset
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass
                prefault(
                    ctypes.byref(buf, offset),
                    remaining if remaining < chunk_size else chunk_size,
                )

            for _ in pool.map(_synchronized_prefault, offsets):
                pass
    finally:
        del buf

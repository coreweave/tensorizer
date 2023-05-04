import contextlib
import functools
import logging
import subprocess
import sys
import threading

# =============================================================================
# From `pipe(7)` manpage:
#
# Pipe capacity
# A pipe has a limited capacity. If the pipe is full, then a write(2) will
# block or fail, depending on whether the O_NONBLOCK flag is set (see below).
# Different implementations have different limits for the pipe capacity.
#
# Applications should not rely on a particular capacity: an application should
# be designed so that a reading process consumes data as soon as it is
# available, so that a writing process does not remain blocked.
#
# In Linux versions before 2.6.11, the capacity of a pipe was the same as the
# system page size (e.g., 4096 bytes on i386). Since Linux 2.6.11, the pipe
# capacity is 16 pages (i.e., 65,536 bytes in a system with a page size of
# 4096 bytes). Since Linux 2.6.35, the default pipe capacity is 16 pages, but
# the capacity can be queried and set using the fcntl(2) F_GETPIPE_SZ and
# F_SETPIPE_SZ operations. See fcntl(2) for more information.
#
# =============================================================================
# From `fcntl(2)` manpage:
#
# Changing the capacity of a pipe
#
# F_SETPIPE_SZ (int; since Linux 2.6.35)
# Change the capacity of the pipe referred to by fd to be at least arg bytes.
# An unprivileged process can adjust the pipe capacity to any value between the
# system page size and the limit defined in /proc/sys/fs/pipe−max−size
# (see proc(5)). Attempts to set the pipe capacity below the page size are
# silently rounded up to the page size. Attempts by an unprivileged process to
# set the pipe capacity above the limit in /proc/sys/fs/pipe−max−size yield the
# error EPERM; a privileged process (CAP_SYS_RESOURCE) can override the limit.
#
# When allocating the buffer for the pipe, the kernel may use a capacity larger
# than arg, if that is convenient for the implementation. (In the current
# implementation, the allocation is the next higher power-of-two page-size
# multiple of the requested size.) The actual capacity (in bytes) that is set
# is returned as the function result.
#
# Attempting to set the pipe capacity smaller than the amount of buffer space
# currently used to store data produces the error EBUSY.
#
# Note that because of the way the pages of the pipe buffer are employed when
# data is written to the pipe, the number of bytes that can be written may be
# less than the nominal size, depending on the size of the writes.
#
# F_GETPIPE_SZ (void; since Linux 2.6.35)
# Return (as the function result) the capacity of the pipe referred to by fd.
#
# =============================================================================
# Constant for `F_SETPIPE_SZ`, as Python's `fcntl` module doesn't have this
# defined until Python 3.10.
F_SETPIPE_SZ = 1031

_logger = logging.getLogger(__name__)

__all__ = ["get_max_pipe_size", "widen_pipe", "widen_new_pipes"]

# No-op default implementations
widen_new_pipes = contextlib.ExitStack


def widen_pipe(_fileno):
    pass


@functools.lru_cache(maxsize=None)
def get_max_pipe_size():
    pipe_buf_sz = 1024 * 1024
    if sys.platform != "win32":
        # Read our max-fd-size, fall back to 1mb if invalid.
        try:
            with open("/proc/sys/fs/pipe-max-size", "r") as pipe_file:
                pipe_buf_sz = int(pipe_file.read())
        except IOError as e:
            _logger.warning(
                f"Could not read /proc/sys/fs/pipe-max-size: {e.strerror}"
            )
    else:
        # Windows has no maximum pipe size,
        # so 256 MiB is chosen completely arbitrarily.
        pipe_buf_sz = 256 * 1024 * 1024
    _logger.debug(f"pipe-max-size: {pipe_buf_sz}")
    return pipe_buf_sz


if sys.platform != "win32":
    # Linux uses fcntl to resize an existing pipe.
    import fcntl

    def widen_pipe(fileno):
        pipe_buf_sz = get_max_pipe_size()
        try:
            fcntl.fcntl(fileno, F_SETPIPE_SZ, pipe_buf_sz)
        except PermissionError as e:
            _logger.warning(
                f"Couldn't fcntl F_SETPIPE_SZ to {pipe_buf_sz}: {e.strerror}"
            )

else:
    # Windows cannot change the size of a pipe after creation,
    # but it can set one's size during creation, so a context manager
    # is used to temporarily modify the creation of all pipes.
    _winapi = getattr(subprocess, "_winapi", None)
    if _winapi is not None and hasattr(_winapi, "CreatePipe"):

        class _LocalPipeSize(threading.local):
            pipe_size = 0

        _local = _LocalPipeSize()
        _original_create_pipe = _winapi.CreatePipe
        _pipe_routine_swap_mutex = threading.Lock()
        _pipe_widening_threads = 0

        def _create_wide_pipe(pipe_attrs, size):
            # The subprocess module creates new anonymous pipes on Windows as:
            # _winapi.CreatePipe(None, 0)
            # Where the first argument is ignored,
            # and the second is the pipe size (0 = default, usually 1 page).
            # To change this without reimplementing all of subprocess.Popen,
            # _winapi.CreatePipe itself is wrapped to override a size of 0
            # with a default of our choosing.
            #
            # This function is thread-safe in the sense that other threads
            # creating pipes while this is active will end up with
            # unchanged results due to the override value being thread-local.
            return _original_create_pipe(
                pipe_attrs, _local.pipe_size if size == 0 else size
            )

        @contextlib.contextmanager
        def widen_new_pipes():
            global _pipe_widening_threads
            # Thread-safe but not re-entrant.
            # Thread safety in this function only matters if multiple threads
            # try to separately invoke this context manager at the same time,
            # which would only happen if multiple CURLStreamFiles were being
            # opened in the same process at the same time. It is less important
            # than _create_wide_pipe being thread-safe.
            _local.pipe_size = get_max_pipe_size()
            with _pipe_routine_swap_mutex:
                _winapi.CreatePipe = _create_wide_pipe
                _pipe_widening_threads += 1
            try:
                yield
            finally:
                with _pipe_routine_swap_mutex:
                    _pipe_widening_threads -= 1
                    if _pipe_widening_threads == 0:
                        _winapi.CreatePipe = _original_create_pipe
                del _local.pipe_size

    else:
        _logger.warning("Couldn't increase default pipe size.")

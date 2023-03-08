import fcntl
import logging
import subprocess
import typing
import os
from io import SEEK_SET, SEEK_END
from typing import Union
import requests

logger = logging.getLogger(__name__)

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
# Constant for `F_SETPIPE_SZ`, as python3's `fcntl` module doesn't have this
# defined -- despite the documentation saying that they're there.
#
# TODO: Make this work or fail gracefully on non-Linux systems. Not sure if
#       this is really relevant, as I don't even know if CUDA is available on
#       non-Linux systems in a production sense.
F_SETPIPE_SZ = 1031


def find_curl() -> str:
    """
    Find the path to the `curl` binary on the system using PATH.
    """
    try:
        path_env = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + os.environ["PATH"]
        for path in path_env.split(":"):
            curl_path = os.path.join(path, "curl")
            if os.path.isfile(curl_path):
                return curl_path
    except KeyError:
        pass
    raise (IOError("Could not find curl binary -- will fall back to requests"))


try:
    curl_path = find_curl()
except IOError as e:
    logger.warning(e.strerror)
    curl_path = None


class CURLStreamFile(object):
    """
    CURLStreamFile implements a file-like object around an HTTP download, the
    intention being to not buffer more than we have to.
    """

    def __init__(self, uri: str) -> None:
        # NOTE: `256mb` buffer on the python IO object.
        self._curl = subprocess.Popen(
            [
                curl_path,
                "--header",
                "Accept-Encoding: identity",
                "-s",
                uri,
            ],
            stdout=subprocess.PIPE,
            bufsize=256 * 1024 * 1024,
        )
        # Read our max-fd-size, fall back to 1mb if invalid.
        pipe_buf_sz = 1024 * 1024
        try:
            pipe_file = open("/proc/sys/fs/pipe-max-size", "r")
            pipe_buf_sz = int(pipe_file.read())
            logger.debug(f"pipe-max-size: {pipe_buf_sz}")
        except IOError as e:
            logger.warning(
                f"Could not read /proc/sys/fs/pipe-max-size: {e.strerror}"
            )
        try:
            fcntl.fcntl(self._curl.stdout.fileno(), F_SETPIPE_SZ, pipe_buf_sz)
        except PermissionError as e:
            logger.warning(
                f"Couldn't fcntl F_SETPIPE_SZ to {pipe_buf_sz}: {e.strerror}"
            )
        self._curr = 0
        self.closed = False

    def _read_until(
            self, goal_position: int, ba: Union[bytearray, None] = None
    ) -> Union[bytes, int]:
        if ba is None:
            rq_sz = goal_position - self._curr
            ret_buff = self._curl.stdout.read(rq_sz)
            ret_buff_sz = len(ret_buff)
        else:
            rq_sz = len(ba)
            ret_buff_sz = self._curl.stdout.readinto(ba)
            ret_buff = ba
        if ret_buff_sz != rq_sz:
            self.closed = True
            err = self._curl.stderr.read()
            self._curl.terminate()
            if self._curl.returncode != 0:
                raise (IOError(f"curl error: {self._curl.returncode}, {err}"))
            else:
                raise (IOError(f"Requested {rq_sz} != {ret_buff_sz}"))
        self._curr += ret_buff_sz
        if ba is None:
            return ret_buff
        else:
            return ret_buff_sz

    def tell(self) -> int:
        return self._curr

    def readinto(self, ba: bytearray) -> int:
        goal_position = self._curr + len(ba)
        return self._read_until(goal_position, ba)

    def read(self, size=None) -> bytes:
        if self.closed:
            raise (IOError("CURLStreamFile closed."))
        if size is None:
            return self._curl.stdout.read()
        goal_position = self._curr + size
        return self._read_until(goal_position)

    @staticmethod
    def writable() -> bool:
        return False

    @staticmethod
    def fileno() -> int:
        return -1

    def close(self):
        self.closed = True
        self._curl.terminate()

    def readline(self):
        raise Exception("Unimplemented")

    """
    This seek() implementation is effectively a no-op, and will throw an
    exception for anything other than a seek to the current position.
    """

    def seek(self, position, whence=SEEK_SET):
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise (Exception("Unsupported `whence`"))
        else:
            raise (Exception("Seeking is unsupported"))


class RequestsStreamFile(object):
    """
    RequestsStreamFile implements a file-like object around an HTTP download, the
    intention being to not buffer more than we have to. Not as fast or efficient
    as CURLStreamFile, but it works on Windows.
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._curr = 0
        self._r = requests.get(uri, stream=True)
        self.closed = False

    def _read_until(
            self, goal_position: int, ba: Union[bytearray, None] = None
    ) -> Union[bytes, int]:
        if ba is None:
            rq_sz = goal_position - self._curr
            ret_buff = self._r.raw.read(rq_sz)
            ret_buff_sz = len(ret_buff)
        else:
            rq_sz = len(ba)
            ret_buff_sz = self._r.raw.readinto(ba)
            ret_buff = ba
        if ret_buff_sz != rq_sz:
            self.closed = True
            raise (IOError(f"Requested {rq_sz} != {ret_buff_sz}"))
        self._curr += ret_buff_sz
        if ba is None:
            return ret_buff
        else:
            return ret_buff_sz

    def tell(self) -> int:
        return self._curr

    def readinto(self, ba: bytearray) -> int:
        goal_position = self._curr + len(ba)
        return self._read_until(goal_position, ba)

    def read(self, size=None) -> bytes:
        if self.closed:
            raise (IOError("RequestsStreamFile closed."))
        if size is None:
            return self._r.raw.read()
        goal_position = self._curr + size
        return self._read_until(goal_position)

    @staticmethod
    def writable() -> bool:
        return False

    @staticmethod
    def fileno() -> int:
        return -1

    def close(self):
        self.closed = True
        self._r.close()
        del self._r

    def readline(self):
        raise Exception("Unimplemented")

    """
    This seek() implementation is effectively a no-op, and will throw an
    exception for anything other than a seek to the current position.
    """

    def seek(self, position, whence=SEEK_SET):
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise (Exception("Unsupported `whence`"))
        else:
            raise (Exception("Seeking is unsupported"))


def open_stream(
        path_uri: str,
        mode: str = "rb"
) -> Union[RequestsStreamFile, CURLStreamFile, typing.BinaryIO]:
    if path_uri.startswith("https://") or path_uri.startswith("http://"):
        if fcntl is not None and curl_path is not None:
            # We have fcntl and curl exists, so we can use the fast CURL-based loader.
            logger.debug(f"Using CURL for tensor streaming of {path_uri}")
            return CURLStreamFile(path_uri)
        else:
            # Fallback to slow requests-based loader.
            logger.debug(f"Using requests for tensor streaming {path_uri}")
            return RequestsStreamFile(path_uri)
    else:
        handle: typing.BinaryIO = open(path_uri, mode)
        handle.seek(0)
        return handle

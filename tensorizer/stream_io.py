import functools
import http.client
import io
import logging
import mmap
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import typing
import weakref
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import boto3
import botocore
import redis

import tensorizer._version as _version
import tensorizer._wide_pipes as _wide_pipes

__all__ = ["open_stream", "CURLStreamFile", "RedisStreamFile", "CAInfo"]

logger = logging.getLogger(__name__)

curl_path = shutil.which("curl")

default_s3_read_endpoint = "accel-object.ord1.coreweave.com"
default_s3_write_endpoint = "object.ord1.coreweave.com"

_BASE_USER_AGENT = f"Tensorizer/{_version.__version__}"

# Curl's user agent is curl/<version>, but it's not worth
# spawning another subprocess just to check that
_CURL_USER_AGENT = f"{_BASE_USER_AGENT} (curl)"

# Botocore appends " Botocore/<version>" to the end of the user agent on its own
_BOTO_USER_AGENT = f"{_BASE_USER_AGENT} (Boto3/{boto3.__version__})"

if sys.platform != "win32":
    _s3_default_config_paths = (os.path.expanduser("~/.s3cfg"),)
else:
    # s3cmd generates its config at a different path on Windows by default,
    # but it may have been manually placed at ~\.s3cfg instead, so check both.
    _s3_default_config_paths = tuple(
        map(os.path.expanduser, (r"~\.s3cfg", r"~\AppData\Roaming\s3cmd.ini"))
    )


class _ParsedCredentials(typing.NamedTuple):
    config_file: Optional[str]
    s3_endpoint: Optional[str]
    s3_access_key: Optional[str]
    s3_secret_key: Optional[str]


@functools.lru_cache(maxsize=None)
def _get_s3cfg_values(
    config_paths: Optional[
        Union[
            Tuple[Union[str, bytes, os.PathLike], ...], str, bytes, os.PathLike
        ]
    ] = None,
) -> _ParsedCredentials:
    """
    Gets S3 credentials from the .s3cfg file.

    Args:
        config_paths: The sequence of potential file paths to check
            for s3cmd config settings. If not provided or an empty tuple,
            platform-specific default search locations are used.
            When specifying a sequence, this argument must be a tuple,
            because this function is cached, and that requires
            all arguments to be hashable.

    Returns:
        A 4-tuple, config_file, s3_endpoint, s3_access_key, s3_secret_key,
        where each element may be None if not found,
        and config_file is the config file path used.
        If config_file is None, no valid config file
        was found, and nothing was parsed.

    Note:
        If the config_paths argument is not provided or is an empty tuple,
        platform-specific default search locations are used.
        This function is cached, and hence config_paths must be a
        (hashable) tuple when specifying a sequence.
    """
    if not config_paths:
        config_paths = _s3_default_config_paths
    elif isinstance(config_paths, (str, bytes, os.PathLike)):
        config_paths = (config_paths,)

    import configparser

    config = configparser.ConfigParser()

    # Stop on the first path that can be successfully read
    for config_path in config_paths:
        if config.read((config_path,)):
            break
    else:
        return _ParsedCredentials(None, None, None, None)

    if "default" not in config:
        raise ValueError(f"No default section in {config_path}")

    return _ParsedCredentials(
        config_file=os.fsdecode(config_path),
        s3_endpoint=config["default"].get("host_base"),
        s3_access_key=config["default"].get("access_key"),
        s3_secret_key=config["default"].get("secret_key"),
    )


class CAInfo:
    """
    Configuration class for cURL's SSL certificate verification behaviour.

    See Also:
        https://curl.se/docs/sslcerts.html
    """

    __slots__ = ("_curl_flags",)

    def __init__(
        self,
        *,
        cacert: Optional[str] = None,
        capath: Optional[str] = None,
        allow_untrusted: bool = False,
    ):
        """
        Creates a configuration defining custom paths to use for SSL certificate
        verification, or disables SSL certificate verification.

        Exactly one of `cacert`, `capath`, or `allow_untrusted` must be
        specified.
        Use None instead of an instance of this class for cURL's default
        certificate verification behaviour.

        Args:
            cacert: Path to a PEM file to use for certificates.
                Corresponds to ``--cacert`` or ``CURLOPT_CAINFO`` in cURL.
            capath: Path to a directory to scan for certificates.
                Corresponds to ``--capath`` or ``CURLOPT_CAPATH`` in cURL.
            allow_untrusted: Skip verification of SSL certificates entirely.
                Corresponds to ``-k``/``--insecure``
                or setting ``CURLOPT_SSL_VERIFYPEER`` to ``FALSE`` in cURL.
        """
        if (cacert, capath).count(None) + (not allow_untrusted) != 2:
            raise ValueError(
                "Exactly one of cacert, capath, or allow_untrusted=True"
                " must be specified. For default certificate verification"
                " behaviour, use certificate_handling=None instead of"
                " an explicit CAInfo instance"
            )
        if cacert is not None:
            if not os.path.isfile(cacert):
                raise ValueError(
                    "Invalid cacert file path, see"
                    " https://curl.se/docs/manpage.html#--cacert"
                )
            self._curl_flags = ("--cacert", str(cacert))
        elif capath is not None:
            if not os.path.isdir(capath):
                raise ValueError(
                    "Invalid capath directory path, see"
                    " https://curl.se/docs/manpage.html#--capath"
                )
            self._curl_flags = ("--capath", str(capath))
        elif allow_untrusted:
            self._curl_flags = ("--insecure",)
        else:
            raise RuntimeError()

    @property
    def curl_flags(self):
        return self._curl_flags

    def __hash__(self):
        return hash(self._curl_flags)


class CURLStreamFile(io.BufferedIOBase):
    """
    CURLStreamFile implements a file-like object around an HTTP download, the
    intention being to not buffer more than we have to. It is intended for
    tar-like files, where we start at the beginning and read until the end of
    the file.

    It does implement `seek` and `tell`, but only for the purpose of
    implementing `read`, and only for the purpose of reading the entire file.
    It does support seeking to an arbitrary position, but is very inefficient
    in doing so as it requires re-opening the connection to the server.

    Attributes:
        popen_latencies: A list of the time it took to start the cURL process.
        http_response_latencies: A list of the time it took to get the first
            HTTP response from the server.
        response_headers: A dictionary of the HTTP response headers.
        bytes_read: The number of bytes read from the stream.
        bytes_skipped: The number of bytes skipped from the stream.
        read_operations: The number of read operations performed on the stream.
    """

    def __init__(
        self,
        uri: str,
        begin: Optional[int] = None,
        end: Optional[int] = None,
        headers: Dict[str, Any] = None,
        *,
        buffer_size: Optional[int] = None,
        certificate_handling: Optional[CAInfo] = None,
    ) -> None:
        # Note that range requests consider both `begin` and `end` inclusive
        # But the `end` argument here is exclusive
        self._uri = uri
        if begin is None:
            begin = 0
        elif begin < 0:
            raise ValueError("Invalid begin, must be non-negative")
        self._begin = begin
        self._end = end
        self._headers = headers
        max_buffer_size: Optional[int] = None
        if buffer_size is None:
            buffer_size = 16 << 20  # 16 MiB
        if end is not None:
            if end <= begin:
                # It isn't possible to request an empty range if begin == end
                # Servers usually throw HTTP code 416, Range Not Satisfiable
                raise ValueError(
                    "End marker must come strictly after begin marker"
                    f" (requested range: {begin} to {end})"
                )
            # Don't use a buffer significantly larger than a requested range
            max_buffer_size: int = self._end - begin + 1
            # Round up to a multiple of the system page size
            max_buffer_size -= max_buffer_size % -mmap.PAGESIZE
            buffer_size = min(buffer_size, max_buffer_size)
        self._buffer_size = buffer_size
        self._certificate_handling = certificate_handling

        self._error_context = []
        self._curl = None

        if curl_path is None:
            raise RuntimeError(
                "cURL is a required dependency for streaming downloads"
                " and could not be found."
            )

        self._optional_curl_flags = []
        if certificate_handling is not None:
            if isinstance(certificate_handling, CAInfo):
                self._optional_curl_flags.extend(
                    certificate_handling.curl_flags
                )
            else:
                raise TypeError(
                    "certificate_handling must either be None"
                    " (for cURL's default SSL certificate verification),"
                    " or a CAInfo object,"
                    f" not {certificate_handling.__class__.__name__}"
                )

        cmd = [
            curl_path,
            *self._optional_curl_flags,
            "--header",
            "Accept-Encoding: identity",
            "--include",
            "-A",
            _CURL_USER_AGENT,
            "-s",
            "-f",
            uri,
        ]

        uses_range_request: bool = begin != 0 or end is not None
        if uses_range_request:
            # Range requests are inclusive, the `end` parameter is exclusive
            range_end: Union[int, str] = end - 1 if end is not None else ""
            cmd.extend(["--range", f"{begin}-{range_end}"])

        if headers is not None:
            for k, v in headers.items():
                cmd.extend(["--header", f"{k}: {v}"])

        with _wide_pipes.widen_new_pipes(max_buffer_size):  # Widen on Windows
            popen_start = time.monotonic()
            self._curl = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                bufsize=buffer_size,
            )
        popen_end = time.monotonic()

        # Widen the pipe on Linux
        _wide_pipes.widen_pipe(self._curl.stdout.fileno(), max_buffer_size)
        resp = self._curl.stdout.readline()  # Block on the http response header
        resp_begin = time.monotonic()

        # We reinitialize this object when seeking,
        # so we don't want to overwrite these tracking variables
        # if they already exist.
        self._init_vars()

        # Track the latency of the popen and http response
        self.popen_latencies.append(popen_end - popen_start)
        self.http_response_latencies.append(resp_begin - popen_end)

        if not resp.startswith((b"HTTP/1.1 2", b"HTTP/2 2")):
            # If the error was caused by an invalid response, display that
            explanation = f": {resp.decode('utf-8')}" if resp.strip() else ""
            if not explanation:
                # Otherwise, check if cURL gave a meaningful exit code
                return_code: Optional[int] = self._curl.poll()
                if isinstance(return_code, int) and return_code != 0:
                    explanation = (
                        f": cURL exit code {return_code:d};"
                        f" see https://curl.se/docs/manpage.html#{return_code}"
                    )

            self.close()
            raise IOError("Failed to open stream" + explanation)

        # Read the rest of the header response and parse it.
        # noinspection PyTypeChecker
        self.response_headers = http.client.parse_headers(self._curl.stdout)

        self._curr = begin
        self._closed = False

    def _init_vars(self):
        self.popen_latencies: List[float] = getattr(self, "popen_latencies", [])
        self.http_response_latencies: List[float] = getattr(
            self, "http_response_latencies", []
        )
        self.bytes_read: int = getattr(self, "bytes_read", 0)
        self.bytes_skipped: int = getattr(self, "bytes_skipped", 0)
        self.read_operations: int = getattr(self, "read_operations", 0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def register_error_context(self, msg: str) -> None:
        """
        Registers a message to serve as context for possible errors
        encountered later.
        This method serves to keep track of any dubious conditions
        under which the CURLStreamFile was opened for more descriptive
        error messages if those conditions eventually lead to an error,
        while still attempting to connect regardless, in accordance with
        the "easier to ask for forgiveness than permission" principle.

        Args:
            msg: The message to be registered.
        """
        self._error_context.append(msg)

    def _reproduce_and_capture_error(
        self, expect_code: Optional[int]
    ) -> Optional[str]:
        """
        Re-attempts the connection with stderr attached to a pipe
        to capture an HTTP error code.
        stderr is not a pipe on the original self._curl Popen object
        because it would get the same `bufsize` as stdout, and waste RAM,
        so this optimizes for the non-error path
        at the slight expense of the error path

        Args:
            expect_code: The error code to expect.
                If this doesn't match the new error, the original error
                is considered not to have been reproduced.

        Returns:
            The cURL error message if the error could be reproduced,
            otherwise None.
        """
        args = [
            curl_path,
            *self._optional_curl_flags,
            "-I",  # Only read headers
            "-XGET",  # Use a GET request (in case HEAD is not supported)
            "-A",  # Set a custom user agent
            _CURL_USER_AGENT,
            "-f",  # Don't return HTML/XML error webpages
            "-s",  # Silence most output
            "-S",  # Keep error messages
            self._uri,
        ]
        try:
            result = subprocess.run(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )

        except subprocess.TimeoutExpired:
            return None
        if result.returncode == 0 or (
            expect_code and result.returncode != expect_code
        ):
            return None
        error_text = result.stderr.strip()
        return error_text if error_text else None

    def _create_read_error_from_context(self) -> IOError:
        return_code = self._curl.poll()
        self._curl.terminate()
        if return_code:
            error = self._reproduce_and_capture_error(expect_code=return_code)
            if error is None:
                error = (
                    f"curl error: ({return_code}), see"
                    f" https://curl.se/docs/manpage.html#{return_code}"
                )
        else:
            error = ""

        if self._error_context:
            error += "\n" + "\n".join(self._error_context)

        error = error.strip()
        if not error:
            # No other context is available, so give a generic error description
            error = "Failed to read from stream"

        return IOError(error)

    def _read_until(
        self, goal_position: int, ba: Union[bytearray, None] = None
    ) -> Union[bytes, int]:
        self.read_operations += 1
        try:
            eof = self._end
            if ba is None:
                rq_sz = goal_position - self._curr
                if eof is not None and self._curr + rq_sz > eof:
                    rq_sz = eof - self._curr
                    if rq_sz <= 0:
                        return bytes()
                ret_buff = self._curl.stdout.read(rq_sz)
                ret_buff_sz = len(ret_buff)
            else:
                rq_sz = len(ba)
                if eof is not None and self._curr + rq_sz > eof:
                    rq_sz = eof - self._curr
                    if rq_sz <= 0:
                        return 0
                    with memoryview(ba)[:rq_sz] as mv:
                        ret_buff_sz = self._curl.stdout.readinto(mv)
                    ret_buff = ba
                else:
                    ret_buff_sz = self._curl.stdout.readinto(ba)
                    ret_buff = ba
            self.bytes_read += ret_buff_sz
            if ret_buff_sz != rq_sz:
                self._closed = True
                self._curl.terminate()
                raise IOError(f"Requested {rq_sz} != {ret_buff_sz}")
            self._curr += ret_buff_sz
            if ba is None:
                return ret_buff
            else:
                return ret_buff_sz
        except (IOError, OSError) as e:
            # Attach a maximally descriptive error message for cURL errors
            raise self._create_read_error_from_context() from e

    def tell(self) -> int:
        return self._curr

    def readinto(self, ba: bytearray) -> int:
        goal_position = self._curr + len(ba)
        return self._read_until(goal_position, ba)

    def read(self, size=None) -> bytes:
        if self._closed:
            raise IOError("CURLStreamFile closed.")
        if size is None:
            ret = self._curl.stdout.read()
            self._curr += len(ret)
            return ret
        goal_position = self._curr + size
        return self._read_until(goal_position)

    def writable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def fileno(self) -> int:
        return -1

    def close(self):
        self._closed = True
        if getattr(self, "_curl", None) is not None:
            if self._curl.poll() is None:
                self._curl.stdout.close()
                self._curl.terminate()
                self._curl.wait()
            else:
                # stdout is normally closed by the Popen.communicate() method,
                # which we skip in favour of Popen.stdout.read()
                self._curl.stdout.close()
        self._curl = None

    @property
    def closed(self):
        return self._closed

    def readline(self, size=-1) -> bytes:
        raise io.UnsupportedOperation("readline")

    """
    This seek() implementation should be avoided if you're seeking backwards,
    as it's not very efficient due to the need to restart the curl process.
    """

    def seek(self, position, whence=SEEK_SET):
        if whence == SEEK_CUR:
            position += self._curr
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise ValueError("Unsupported `whence`")
        elif position > self._curr:
            # We're seeking forward, so we just read until we get there.
            self._read_until(position)
            self.bytes_skipped += position - self._curr
        else:
            # To seek backwards, we need to close out our existing process and
            # start a new one.
            self.close()

            # And we reinitialize ourself.
            self.__init__(
                uri=self._uri,
                begin=position,
                end=self._end,
                headers=self._headers,
                buffer_size=self._buffer_size,
                certificate_handling=self._certificate_handling,
            )

    def _fork(
        self, begin: Optional[int] = None, end: Optional[int] = None
    ) -> "CURLStreamFile":
        """
        Reopen a file at a new byte range. Meant only for internal use
        by `TensorDeserializer` when creating multiple readers.

        Args:
            begin: Starting byte of the new byte range, inclusive,
                or None to start at the beginning of the file.
            end: Ending byte of the new byte range, exclusive,
                or None to include up to the end of the file.

        Returns:
            A new `CURLStreamFile` identical to this one,
            but with a different byte range.
        """
        clone = self.__class__(
            uri=self._uri,
            begin=begin,
            end=end,
            headers=self._headers,
            buffer_size=self._buffer_size,
            certificate_handling=self._certificate_handling,
        )
        clone._error_context.extend(self._error_context)
        return clone


def _parse_redis_uri(uri):
    uri_components = urlparse(uri)

    if uri_components.scheme.lower() != "redis":
        raise ValueError(f"Invalid Redis URI: {uri}")

    host = uri_components.hostname
    port = uri_components.port
    if port is None:
        port = 6379
    prefix = uri_components.path.lstrip("/")

    return host, port, prefix


# Detect if we're running on OSX, and if so, set max buffer size to 1 MiB.
if sys.platform == "darwin":
    _MAX_TCP_BUFFER_SIZE = 1 << 20  # 1 MiB if OSX
else:
    _MAX_TCP_BUFFER_SIZE = 16 << 20  # 16 MiB


class RedisStreamFile(io.BufferedIOBase):
    """
    RedisStreamFile implements a file-like object around a Redis key namespace. Each
    'file' is broken up into multiple keys, each of which is a slice of the file. Each
    key is named with the prefix, followed by a colon, a user-assigned name, another
    colon, and then the byte index of the key.

    For example, if the prefix is 'foo', and the user-assigned name is 'bar', then the
    keys would be 'foo:bar:0', 'foo:bar:16'. The first key would be the first 16 bytes
    of the file, and the remainder would be in the next key.

    On initialization, it performs a prefix scan of the keys for the prefix, and then
    sorts the keys by their byte index. This allows it to perform a seek operation
    efficiently, as it can find the key that contains the byte index, and then read
    from that key.

    RedisStreamFile has some optimizations, such as reading the entirety of a key into
    a buffer, and then serving subsequent reads from that buffer. It also has a buffer
    for the remainder of a key, so that it can serve partial reads of a key.

    Arguments:
        uri: The URI of the Redis key namespace. This should be prefixed by the
            'redis://' scheme, and include the key prefix. For example,
            'redis://localhost:6379/foo'.
        buffer_size: The size of the TCP buffer to use for the Redis connection.

    Attributes:
        setup_latency: The time it took to setup the Redis connections and enumerate.
        response_latencies: A list of the time it took to get the Redis responses.
        bytes_read: The number of bytes read from the stream.
        bytes_skipped: The number of bytes skipped from the stream.
        read_operations: The number of read operations performed on the stream.
    """

    def __init__(
        self,
        uri: str,
        *,
        begin: Optional[int] = None,
        end: Optional[int] = None,
        buffer_size: Optional[int] = None,
    ) -> None:
        if buffer_size is None:
            buffer_size = _MAX_TCP_BUFFER_SIZE
        init_begin = time.monotonic()
        host, port, prefix = _parse_redis_uri(uri)
        self._redis = redis.Redis(host=host, port=port, db=0)
        self._redis_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._redis_tcp.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size
        )
        self._redis_tcp.connect((host, port))
        self._redis_tcp_mutex = threading.Lock()

        # Do a key scan for the prefix, and collect all the indexes from the keys.
        self._indexes = []
        self._sizes = []
        keys = self._redis.scan_iter(f"{prefix}:*")
        # Sort the keys by their index.
        self._keys = sorted(
            keys, key=lambda x: int(x.decode("utf-8").split(":")[-1])
        )

        # Get indexes.
        largest = 0
        for key in self._keys:
            size = self._redis.strlen(key)
            if size > largest:
                largest = size
            self._sizes.append(size)
            self._indexes.append(int(key.decode("utf-8").split(":")[-1]))

        self._curr = 0
        self._curr_key = 0
        self._curr_buffer = bytearray(largest)
        self._curr_buffer_idx = -1
        self._curr_buffer_view = memoryview(self._curr_buffer)[0:0]

        init_end = time.monotonic()

        self.setup_latency = init_end - init_begin
        self.response_latencies = []
        self.bytes_read = 0
        self.bytes_skipped = 0
        self.read_operations = 0

        if begin:
            self.seek(begin)

    def _find_key_index(self, position):
        for i, index in enumerate(self._indexes):
            if position < index:
                return i - 1
        return len(self._keys) - 1

    def _read_from(
        self,
        position: int,
        size: int,
        ba: Union[bytearray, memoryview],
        no_buffer: bool = False,
    ) -> int:
        self.read_operations += 1
        curr_key = self._find_key_index(position)
        curr_key_pos = self._indexes[curr_key]
        curr_key_sz = self._sizes[curr_key]
        begin_idx = position - curr_key_pos
        position_end = position + size
        read_sz = min(size, curr_key_sz)
        curr_buffer_end = self._curr_buffer_idx + len(self._curr_buffer_view)

        # Check if we have a partial read of the current key in the buffer.
        # TODO: allow for reads that do not align with the key boundaries.
        if (
            not no_buffer
            and len(self._curr_buffer_view) > 0
            and self._curr_buffer_idx
            <= position
            < position_end
            < curr_buffer_end
        ):
            num_bytes = min(read_sz, len(self._curr_buffer_view))
            ba[:num_bytes] = self._curr_buffer_view[:num_bytes]
            self._curr_buffer_idx += num_bytes
            self._curr_buffer_view = self._curr_buffer_view[num_bytes:]
            return num_bytes

        with self._redis_tcp_mutex:
            # We have an aligned read, so we can use GET.
            if position == curr_key_pos:
                command = f"GET {self._keys[curr_key].decode('utf-8')}"
            else:
                command = (
                    f"GETRANGE {self._keys[curr_key].decode('utf-8')}"
                    f" {begin_idx} {begin_idx + read_sz - 1}"
                )

            begin_timer = time.monotonic()
            self._redis_tcp.send(command.encode("utf-8") + b"\r\n")
            value_sz = self._read_sz()
            tcp_read_sz = min(value_sz, read_sz)
            if tcp_read_sz == 0:
                return 0
            num_bytes = self._redis_tcp.recv_into(
                ba, tcp_read_sz, socket.MSG_WAITALL
            )

            end_timer = time.monotonic()
            self.response_latencies.append(end_timer - begin_timer)

            # We performed a partial read of the entire key, so we need to
            # store the remainder in the buffer.
            if value_sz > read_sz:
                tcp_read_sz = value_sz - num_bytes
                if tcp_read_sz > len(self._curr_buffer):
                    self._curr_buffer = bytearray(tcp_read_sz)
                self._curr_buffer_view = memoryview(self._curr_buffer)[
                    :tcp_read_sz
                ]
                self._redis_tcp.recv_into(
                    self._curr_buffer_view, tcp_read_sz, socket.MSG_WAITALL
                )
                self._curr_buffer_idx = position + num_bytes

            # Read the trailing \r\n and discard.
            if self._redis_tcp.recv(2) != b"\r\n":
                raise RuntimeError("Missing key footer")

        return num_bytes

    def _read_sz(self) -> int:
        # Loop until we get a \r\n
        sz_resp = bytearray()
        while True:
            b = self._redis_tcp.recv(1)
            if b == b"":
                raise IOError("Failed to read size")
            sz_resp += b
            if sz_resp[-2:] == b"\r\n":
                break
        sz_str = sz_resp.decode("utf-8").strip()[1:]
        if sz_str == "-1":
            raise IOError("Key not found")
        return int(sz_str)

    def _read_until(
        self, goal_position: int, ba: Optional[bytearray] = None
    ) -> Union[bytes, int]:
        if ba is None:
            rq_sz = goal_position - self._curr
            mv = memoryview(bytearray(rq_sz))
        else:
            rq_sz = len(ba)
            mv = memoryview(ba)
        orig_mv = mv
        left = rq_sz
        while left > 0:
            num_bytes = self._read_from(self._curr, left, mv)
            if num_bytes == 0:
                break
            left -= num_bytes
            self._curr += num_bytes
            if left == 0:
                break
            mv = mv[num_bytes:]
        self.bytes_read += rq_sz - left
        if ba is None:
            return orig_mv.obj
        else:
            return rq_sz - left

    def tell(self) -> int:
        return self._curr

    def readinto(self, ba: bytearray) -> int:
        goal_position = self._curr + len(ba)
        return self._read_until(goal_position, ba)

    def read(self, size=None) -> bytes:
        if self.closed:
            raise IOError("RedisStreamFile closed.")
        if size is None:
            return self._read_until(self._indexes[-1] + self._sizes[-1])
        goal_position = self._curr + size
        return self._read_until(goal_position)

    def writable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def fileno(self) -> int:
        return -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def seek(self, position, whence=SEEK_SET):
        if whence == SEEK_CUR:
            position += self._curr
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise ValueError("Unsupported `whence`")
        self._curr = position

    def close(self):
        if self._redis is not None:
            self._redis.close()
            self._redis = None
        if self._redis_tcp is not None:
            self._redis_tcp.close()
            self._redis_tcp = None
        super(RedisStreamFile, self).close()  # will set self.closed to True

    def readline(self, size=-1, /):
        raise io.UnsupportedOperation("readline")


def _ensure_https_endpoint(endpoint: str):
    scheme, *location = endpoint.split("://", maxsplit=1)
    scheme = scheme.lower() if location else None
    if scheme is None:
        return "https://" + endpoint
    elif scheme == "https":
        return endpoint
    else:
        raise ValueError("Non-HTTPS endpoint URLs are not allowed.")


def _new_s3_client(
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_endpoint: str,
    s3_region_name: Optional[str] = None,
    s3_signature_version: Optional[str] = None,
):
    if s3_secret_access_key is None:
        raise TypeError("No secret key provided")
    if s3_access_key_id is None:
        raise TypeError("No access key provided")
    if s3_endpoint is None:
        raise TypeError("No S3 endpoint provided")

    config_args = dict(user_agent=_BOTO_USER_AGENT)
    auth_args = {}

    if s3_region_name is not None:
        config_args["region_name"] = s3_region_name

    if s3_access_key_id == s3_secret_access_key == "":
        config_args["signature_version"] = botocore.UNSIGNED
    else:
        auth_args = dict(
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_secret_access_key,
        )
        if s3_signature_version is not None:
            config_args["signature_version"] = s3_signature_version

    config = boto3.session.Config(**config_args)

    return boto3.session.Session.client(
        boto3.session.Session(),
        endpoint_url=_ensure_https_endpoint(s3_endpoint),
        service_name="s3",
        config=config,
        **auth_args,
    )


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    uri_components = urlparse(uri)

    if uri_components.scheme.lower() != "s3":
        raise ValueError(f"Invalid S3 URI: {uri}")

    bucket = uri_components.netloc
    key = uri_components.path.lstrip("/")
    return bucket, key


def s3_upload(
    path: str,
    target_uri: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_endpoint: str = default_s3_write_endpoint,
    s3_region_name: Optional[str] = None,
    s3_signature_version: Optional[str] = None,
):
    bucket, key = _parse_s3_uri(target_uri)
    client = _new_s3_client(
        s3_access_key_id,
        s3_secret_access_key,
        s3_endpoint,
        s3_region_name=s3_region_name,
        s3_signature_version=s3_signature_version,
    )
    client.upload_file(path, bucket, key)


def _s3_download_url(
    path_uri: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_endpoint: str = default_s3_read_endpoint,
    s3_region_name: Optional[str] = None,
    s3_signature_version: Optional[str] = None,
) -> str:
    bucket, key = _parse_s3_uri(path_uri)
    # v2 signature is important to easily align the presigned URL expiry
    # times. This allows multiple clients to generate the exact same
    # presigned URL, and get hits on a HTTP caching proxy.
    #
    # why v2 signature?
    # V2 signature has ONE part: expiry timestamp, in epoch seconds.
    # V4 signature has TWO parts: x-amz-date & duration,
    # boto3 does not permit easy modification of x-amz-date.
    # See upstream bug https://github.com/boto/botocore/issues/2230
    #
    if not s3_signature_version:
        s3_signature_version = "s3"
    client = _new_s3_client(
        s3_access_key_id,
        s3_secret_access_key,
        s3_endpoint,
        s3_region_name=s3_region_name,
        s3_signature_version=s3_signature_version,
    )

    # Explanation with SIG_GRANULARITY=1h
    # compute an expiry that is aligned to the hour, at least 1 hour
    # away from present time
    # time=00:00:00 -> expiry=02:00:00
    # time=00:01:00 -> expiry=02:00:00
    # time=00:59:59 -> expiry=02:00:00
    # time=01:00:00 -> expiry=03:00:00
    #
    # if your files are so large that they take LONGER than 1 hour to
    # download, a larger SIG_GRANULARITY would be required.
    #
    # Caveats: If the time.time() call is at 3599.9̅ (9 repeating if your
    # unicode is broken), and the Boto call happens at 3600.x, then the expiry
    # will be at the NEXT boundary. Boto3 support specifying an absolute expiry
    # time (Boto2 did support it).

    SIG_GRANULARITY = 86400
    t = int(time.time())
    expiry = t - (t % SIG_GRANULARITY) + (SIG_GRANULARITY * 2)
    seconds_to_expiry = expiry - t

    url = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=seconds_to_expiry,
    )
    return url


def s3_download(
    path_uri: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_endpoint: str = default_s3_read_endpoint,
    s3_region_name: Optional[str] = None,
    s3_signature_version: Optional[str] = None,
    buffer_size: Optional[int] = None,
    force_http: bool = False,
    begin: Optional[int] = None,
    end: Optional[int] = None,
    certificate_handling: Optional[CAInfo] = None,
) -> CURLStreamFile:
    url = _s3_download_url(
        path_uri=path_uri,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        s3_endpoint=s3_endpoint,
        s3_region_name=s3_region_name,
        s3_signature_version=s3_signature_version,
    )
    if force_http and url.lower().startswith("https://"):
        url = "http://" + url[8:]
    return CURLStreamFile(
        url,
        buffer_size=buffer_size,
        certificate_handling=certificate_handling,
        begin=begin,
        end=end,
    )


def _infer_credentials(
    s3_access_key_id: Optional[str],
    s3_secret_access_key: Optional[str],
    s3_config_path: Optional[Union[str, bytes, os.PathLike]] = None,
) -> _ParsedCredentials:
    """
    Fill in a potentially incomplete S3 credential pair
    by parsing the s3cmd config file if necessary.
    An empty string ("") is considered a specified credential,
    while None is an unspecified credential.
    Use "" for public buckets.

    Args:
        s3_access_key_id: `s3_access_key_id` if explicitly specified,
            otherwise None.
            If None, the s3cmd config file is parsed for the key.
        s3_secret_access_key: `s3_secret_access_key` if explicitly specified,
            otherwise None.
            If None, the s3cmd config file is parsed for the key.
        s3_config_path: An explicit path to the s3cmd config file to search,
            if necessary.
            If None, platform-specific default paths are used.

    Returns:
        A `_ParsedCredentials` object with both the
        `s3_access_key` and `s3_secret_key` fields guaranteed to not be None.

    Raises:
        ValueError: If the credential pair is incomplete and the
            missing parts could not be found in any s3cmd config file.
        FileNotFoundError: If `s3_config_path` was explicitly provided,
            but the file specified does not exist.
    """
    if None not in (s3_access_key_id, s3_secret_access_key):
        # All required credentials were specified; don't parse anything
        return _ParsedCredentials(
            config_file=None,
            s3_endpoint=None,
            s3_access_key=s3_access_key_id,
            s3_secret_key=s3_secret_access_key,
        )

    # Try to find default credentials if at least one is not specified
    if s3_config_path is not None and not os.path.exists(s3_config_path):
        raise FileNotFoundError(
            "Explicitly specified s3_config_path does not exist:"
            f" {s3_config_path}"
        )
    try:
        parsed: _ParsedCredentials = _get_s3cfg_values(s3_config_path)
    except ValueError as parse_error:
        raise ValueError(
            "Attempted to access an S3 bucket,"
            " but credentials were not provided,"
            " and the fallback .s3cfg file could not be parsed."
        ) from parse_error

    if parsed.config_file is None:
        raise ValueError(
            "Attempted to access an S3 bucket,"
            " but credentials were not provided,"
            " and no default .s3cfg file could be found."
        )

    # Don't override a specified credential
    if s3_access_key_id is None:
        s3_access_key_id = parsed.s3_access_key
    if s3_secret_access_key is None:
        s3_secret_access_key = parsed.s3_secret_key

    # Verify that both keys were ultimately found
    for required_credential, credential_name in (
        (s3_access_key_id, "s3_access_key_id"),
        (s3_secret_access_key, "s3_secret_access_key"),
    ):
        if not required_credential:
            raise ValueError(
                "Attempted to access an S3 bucket,"
                f" but {credential_name} was not provided,"
                " and could not be found in the default"
                f" config file at {parsed.config_file}."
            )

    return _ParsedCredentials(
        config_file=parsed.config_file,
        s3_endpoint=parsed.s3_endpoint,
        s3_access_key=s3_access_key_id,
        s3_secret_key=s3_secret_access_key,
    )


def _temp_file_closer(file: io.IOBase, file_name: str, *upload_args):
    """
    Close, upload by name, and then delete the file.
    Meant to be placed as a hook before both .close() and .__exit__()
    on a particular instance of a temporary file-like wrapper object,
    as a callback to a weakref.finalize() registration on the wrapper.

    The reason this implementation is necessary is really complicated.

    ---

    boto3's upload_fileobj could be used before closing the
    file, instead of closing it and then uploading it by
    name, but upload_fileobj is less performant than
    upload_file as of boto3's s3 library s3transfer
    version 0.6.0.

    For details, see the implementation & comments:
    https://github.com/boto/s3transfer/blob/0.6.0/s3transfer/upload.py#L351

    TL;DR: s3transfer does multithreaded transfers
    that require multiple file handles to work properly,
    but Python cannot duplicate file handles such that
    they can be accessed in a thread-safe way,
    so they have to buffer it all in memory.
    """

    try:
        file.close()
        s3_upload(file_name, *upload_args)
    finally:
        try:
            os.unlink(file_name)
        except OSError:
            pass


def open_stream(
    path_uri: Union[str, os.PathLike],
    mode: str = "rb",
    s3_access_key_id: Optional[str] = None,
    s3_secret_access_key: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    s3_config_path: Optional[Union[str, bytes, os.PathLike]] = None,
    buffer_size: Optional[int] = None,
    force_http: bool = False,
    *,
    begin: Optional[int] = None,
    end: Optional[int] = None,
    s3_region_name: Optional[str] = None,
    s3_signature_version: Optional[str] = None,
    certificate_handling: Optional[CAInfo] = None,
) -> Union[CURLStreamFile, RedisStreamFile, typing.BinaryIO]:
    """
    Open a file path, http(s):// URL, or s3:// URI.

    Note:
        The file-like streams returned by this function can be passed directly
        to `tensorizer.TensorDeserializer` when ``mode="rb"``,
        and `tensorizer.TensorSerializer` when ``mode="wb"``.

    Args:
        path_uri: File path, http(s):// URL, or s3:// URI to open.
        mode: Mode with which to open the stream.
            Supported values are:

            * "rb" for http(s)://,
            * "rb", "wb[+]", and "ab[+]" for s3://,
            * All standard binary modes for file paths.

        s3_access_key_id: S3 access key, corresponding to
            "aws_access_key_id" in boto3. If not specified and
            an s3:// URI is being opened, and `~/.s3cfg` exists,
            `~/.s3cfg`'s "access_key" will be parsed as this credential.
            To specify blank credentials, for a public bucket,
            pass the empty string ("") rather than None.
        s3_secret_access_key: S3 secret key, corresponding to
            "aws_secret_access_key" in boto3. If not specified and
            an s3:// URI is being opened, and `~/.s3cfg` exists,
            `~/.s3cfg`'s "secret_key" will be parsed as this credential.
            To specify blank credentials, for a public bucket,
            pass the empty string ("") rather than None.
        s3_endpoint: S3 endpoint.
            If not specified and a host_base was found
            alongside previously parsed credentials, that will be used.
            Otherwise, ``object.ord1.coreweave.com`` is the default.
        s3_config_path: An explicit path to the `~/.s3cfg` config file
            to be parsed if full credentials are not provided.
            If None, platform-specific default paths are used.
        buffer_size: The size of the TCP or pipe buffer to use.
        force_http: If True, force the use of HTTP instead of HTTPS for
            S3 downloads. This will double the throughput, but at the cost
            of security.
        begin: if specified, the file or request will begin at this byte offset.
            Uses seek() for files and Range for HTTP and S3. This has no effect
            on writes.
        end: if specified, HTTP and S3 requests will use this for the end of the
            Range header. This has no effect on files or on writes.
            This offset is exclusive like Python slicing: 0-2 is two bytes
            (in contrast to HTTP range headers, which treat 0-2 as three bytes).
        s3_region_name: S3 region name, corresponding to
            "region_name" in boto3 config.
            The object storage region used in instantiating the client.
        s3_signature_version: S3 signature version, corresponding
            to "signature_version" in boto3 config.
            The signature version used when signing requests.
        certificate_handling: Customize handling of SSL CA certificates for
            HTTPS and S3 downloads.
            Pass None to use default certificate verification, or an instance of
            `tensorizer.stream_io.CAInfo` to use a different CA bundle
            or to disable certificate verification entirely.
            This option is useful when working with self-signed certificates.

    Returns:
        An opened file-like object representing the target resource.

    Examples:
        Opening an S3 stream for writing with manually specified credentials::

            with open_stream(
                "s3://some-private-bucket/my-model.tensors",
                mode="wb",
                s3_access_key_id=...,
                s3_secret_access_key=...,
                s3_endpoint=...,
            ) as stream:
                ...

        Opening an S3 stream for reading with credentials
        specified in `~/.s3cfg`::

            # Credentials in ~/.s3cfg are parsed automatically.
            stream = open_stream(
                "s3://some-private-bucket/my-model.tensors",
            )

        Opening an S3 stream for reading with credentials
        specified in `/etc/secrets/.s3cfg`
        (e.g., as may be mounted in a Kubernetes pod)::

            stream = open_stream(
                "s3://some-private-bucket/my-model.tensors",
                s3_config_path="/etc/secrets/.s3cfg",
            )

        Opening an http(s):// URI for reading::

            with open_stream(
                "https://raw.githubusercontent.com/EleutherAI/gpt-neo/master/README.md"
            ) as stream:
                print(stream.read(128))

        Opening an https:// URI for reading,
        while skipping SSL certificate verification::

            with open_stream(
                "https://127.0.0.1/my-model.tensors",
                certificate_handling=CAInfo(allow_untrusted=True),
            ) as stream:
                print(stream.read(128))
    """
    if isinstance(path_uri, os.PathLike):
        local_only: bool = True
        path_uri = os.fspath(path_uri)
    else:
        local_only: bool = False

    scheme, *location = path_uri.split("://", maxsplit=1)
    scheme = scheme.lower() if location else None

    normalized_mode = "".join(sorted(mode))

    if not local_only and scheme in ("http", "https"):
        if normalized_mode != "br":
            raise ValueError(
                'Only the mode "rb" is valid when opening http(s):// streams.'
            )
        return CURLStreamFile(
            path_uri,
            buffer_size=buffer_size,
            begin=begin,
            end=end,
            certificate_handling=certificate_handling,
        )

    elif not local_only and scheme == "redis":
        if normalized_mode != "br":
            raise ValueError(
                'Only the mode "rb" is valid when opening redis:// streams.'
            )
        return RedisStreamFile(
            path_uri, buffer_size=buffer_size, begin=begin, end=end
        )

    elif not local_only and scheme == "s3":
        if normalized_mode not in ("br", "bw", "ab", "+bw", "+ab"):
            raise ValueError(
                'Only the modes "rb", "wb[+]", and "ab[+]" are valid'
                " when opening s3:// streams."
            )
        is_s3_upload = "w" in mode or "a" in mode
        error_context = None
        try:
            s3 = _infer_credentials(
                s3_access_key_id, s3_secret_access_key, s3_config_path
            )
            s3_access_key_id = s3.s3_access_key
            s3_secret_access_key = s3.s3_secret_key

            # Not required to have been found,
            # and doesn't overwrite an explicitly specified endpoint.
            s3_endpoint = s3_endpoint or s3.s3_endpoint
        except (ValueError, FileNotFoundError) as e:
            # Uploads always require credentials here, but downloads may not
            if is_s3_upload:
                raise
            else:
                # Credentials may be absent because a public read
                # bucket is being used, so try blank credentials,
                # but provide a descriptive warning for future errors
                # that may occur due to this exception being suppressed.
                # Don't save the whole exception object since it holds
                # a stack trace, which can interfere with garbage collection.
                error_context = (
                    "Warning: empty credentials were used for S3."
                    f"\nReason: {e}"
                    "\nIf the connection failed due to missing permissions"
                    " (e.g. HTTP error 403), try providing credentials"
                    " directly with the tensorizer.stream_io.open_stream()"
                    " function."
                )
                s3_access_key_id = s3_access_key_id or ""
                s3_secret_access_key = s3_access_key_id or ""

        # Regardless of whether the config needed to be parsed,
        # the endpoint gets a default value based on the operation.

        if is_s3_upload:
            s3_endpoint = s3_endpoint or default_s3_write_endpoint

            # delete must be False or the file will be deleted by the OS
            # as soon as it closes, before it can be uploaded on platforms
            # with primitive temporary file support (e.g. Windows)
            temp_file = tempfile.NamedTemporaryFile(mode="wb+", delete=False)

            # Attach a callback to upload the temporary file when it closes.
            # weakref finalizers are idempotent, so this upload callback
            # is guaranteed to run at most once.
            guaranteed_closer = weakref.finalize(
                temp_file,
                _temp_file_closer,
                temp_file.file,
                temp_file.name,
                path_uri,
                s3_access_key_id,
                s3_secret_access_key,
                s3_endpoint,
                s3_region_name,
                s3_signature_version,
            )

            # Always run the close + upload procedure
            # before any code from Python's NamedTemporaryFile wrapper.
            # It isn't safe to call a bound method from a weakref finalizer,
            # but calling a weakref finalizer alongside a bound method
            # creates no problems, other than that the code outside the
            # finalizer is not guaranteed to be run at any point.
            # In this case, the weakref finalizer performs all necessary
            # cleanup itself, but the original NamedTemporaryFile methods
            # are invoked as well, just in case.
            wrapped_close = temp_file.close

            def close_wrapper():
                guaranteed_closer()
                return wrapped_close()

            # Python 3.12+ doesn't call NamedTemporaryFile.close() during
            # .__exit__(), so it must be wrapped separately.
            # Since guaranteed_closer is idempotent, it's fine to call it in
            # both methods, even if both are called back-to-back.
            wrapped_exit = temp_file.__exit__

            def exit_wrapper(exc, value, tb):
                guaranteed_closer()
                return wrapped_exit(exc, value, tb)

            temp_file.close = close_wrapper
            temp_file.__exit__ = exit_wrapper

            return temp_file
        else:
            s3_endpoint = s3_endpoint or default_s3_read_endpoint
            curl_stream_file = s3_download(
                path_uri,
                s3_access_key_id,
                s3_secret_access_key,
                s3_endpoint,
                s3_region_name=s3_region_name,
                s3_signature_version=s3_signature_version,
                buffer_size=buffer_size,
                force_http=force_http,
                begin=begin,
                end=end,
                certificate_handling=certificate_handling,
            )
            if error_context:
                curl_stream_file.register_error_context(error_context)
            return curl_stream_file

    else:
        if "b" not in normalized_mode:
            raise ValueError(
                'Only binary modes ("rb", "wb", "wb+", etc.)'
                " are valid when opening local file streams."
            )
        if any(map(normalized_mode.__contains__, "wax")):
            dirname = os.path.dirname(path_uri)
            if dirname:
                os.makedirs(os.path.dirname(path_uri), exist_ok=True)
        if buffer_size is None:
            buffer_size = io.DEFAULT_BUFFER_SIZE
        handle: typing.BinaryIO = open(path_uri, mode, buffering=buffer_size)
        handle.seek(begin or 0)
        return handle

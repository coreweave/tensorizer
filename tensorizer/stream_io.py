import fcntl
import functools
import logging
import os
import subprocess
import sys
import tempfile
import typing
from urllib.parse import urlparse

import boto3
from io import SEEK_SET, SEEK_END
from typing import Union, Optional, Dict, Any
import requests
import shutil

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

# Read our max-fd-size, fall back to 1mb if invalid.
PIPE_BUF_SZ = 1024 * 1024
try:
    pipe_file = open("/proc/sys/fs/pipe-max-size", "r")
    pipe_buf_sz = int(pipe_file.read())
    logger.debug(f"pipe-max-size: {pipe_buf_sz}")
except IOError as e:
    logger.warning(
        f"Could not read /proc/sys/fs/pipe-max-size: {e.strerror}"
    )

curl_path = shutil.which("curl")

default_s3_endpoint = "object.ord1.coreweave.com"
if sys.platform != "win32":
    _s3_config_paths = (os.path.expanduser("~/.s3cfg"),)
else:
    # s3cmd generates its config at a different path on Windows by default,
    # but it may have been manually placed at ~\.s3cfg instead, so check both.
    _s3_config_paths = tuple(map(os.path.expanduser,
                                 (r"~\.s3cfg",
                                  r"~\AppData\Roaming\s3cmd.ini")))


class _ParsedCredentials(typing.NamedTuple):
    config_file: Optional[str]
    s3_endpoint: Optional[str]
    s3_access_key: Optional[str]
    s3_secret_key: Optional[str]


@functools.lru_cache(maxsize=None)
def _get_s3cfg_values(config_paths=_s3_config_paths) -> _ParsedCredentials:
    """
    Gets S3 credentials from the .s3cfg file.

    Returns the 4-tuple config_file, s3_endpoint, s3_access_key, s3_secret_key,
    where each element may be None if not found.
    config_file is the config file path used. If it is None, no config file was found.
    """
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
        config_file=config_path,
        s3_endpoint=config["default"].get("host_base"),
        s3_access_key=config["default"].get("access_key"),
        s3_secret_key=config["default"].get("secret_key")
    )


class CURLStreamFile:
    """
    CURLStreamFile implements a file-like object around an HTTP download, the
    intention being to not buffer more than we have to. It is intended for
    tar-like files, where we start at the begining and read until the end of
    the file.

    It does implement `seek` and `tell`, but only for the purpose of
    implementing `read`, and only for the purpose of reading the entire file.
    It does support seeking to an arbitrary position, but is very inefficient
    in doing so as it requires re-opening the connection to the server.
    """

    def __init__(self,
                 uri: str,
                 begin: Optional[int] = None,
                 end: Optional[int] = None,
                 headers: Dict[str, Any] = None) -> None:
        self._uri = uri

        # NOTE: `256mb` buffer on the python IO object.
        cmd = [
            curl_path,
            "--header",
            "Accept-Encoding: identity",
            "-s",
            uri,
        ]

        if begin is not None or end is not None:
            if begin is None:
                begin = 0
            if end is None:
                end = ""
            cmd.extend(["--range", f"{begin}-{end}"])

        if headers is not None:
            for k, v in headers.items():
                cmd.extend(["--header", f"{k}: {v}"])

        self._curl = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            bufsize=256 * 1024 * 1024,
        )

        try:
            fcntl.fcntl(self._curl.stdout.fileno(), F_SETPIPE_SZ, PIPE_BUF_SZ)
        except PermissionError as e:
            logger.warning(
                f"Couldn't fcntl F_SETPIPE_SZ to {pipe_buf_sz}: {e.strerror}"
            )
        self._curr = 0 if begin is None else begin
        self._end = end
        self.closed = False

    def __del__(self):
        self.close()

    def _read_until(
            self, goal_position: int, ba: Union[bytearray, None] = None
    ) -> Union[bytes, int]:
        if ba is None:
            rq_sz = goal_position - self._curr
            if self._end is not None and self._curr + rq_sz > self._end:
                rq_sz = self._end - self._curr
                if rq_sz <= 0:
                    return bytes()
            ret_buff = self._curl.stdout.read(rq_sz)
            ret_buff_sz = len(ret_buff)
        else:
            rq_sz = len(ba)
            if self._end is not None and self._curr + rq_sz > self._end:
                rq_sz = self._end - self._curr
                if rq_sz <= 0:
                    return 0
                tmp_ba = bytearray(rq_sz)
                ret_buff_sz = self._curl.stdout.readinto(tmp_ba)
                ba[:ret_buff_sz] = tmp_ba[:ret_buff_sz]
                ret_buff = ba
            else:
                ret_buff_sz = self._curl.stdout.readinto(ba)
                ret_buff = ba
        if ret_buff_sz != rq_sz:
            self.closed = True
            err = self._curl.stderr.read()
            self._curl.terminate()
            if self._curl.returncode != 0:
                raise IOError(f"curl error: {self._curl.returncode}, {err}")
            else:
                raise IOError(f"Requested {rq_sz} != {ret_buff_sz}")
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
            raise IOError("CURLStreamFile closed.")
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
        if self._curl is not None:
            if self._curl.poll() is None:
                self._curl.stdout.close()
                self._curl.terminate()
                self._curl.wait()
            self._curl = None

    def readline(self):
        raise NotImplementedError("Unimplemented")

    """
    This seek() implementation should be avoided if you're seeking backwards,
    as it's not very efficient due to the need to restart the curl process.
    """

    def seek(self, position, whence=SEEK_SET):
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise ValueError("Unsupported `whence`")
        elif position > self._curr:
            # We're seeking forward, so we just read until we get there.
            self._read_until(position)
        else:
            # To seek backwards, we need to close out our existing process and
            # start a new one.
            self.close()

            # And we reinitialize ourself.
            self.__init__(self._uri, position, None)


class RequestsStreamFile:
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
            raise IOError(f"Requested {rq_sz} != {ret_buff_sz}")
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
            raise IOError("RequestsStreamFile closed.")
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
        raise NotImplementedError("Unimplemented")

    """
    This seek() implementation is effectively a no-op, and will throw an
    exception for anything other than a seek to the current position.
    """

    def seek(self, position, whence=SEEK_SET):
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise ValueError("Unsupported `whence`")
        else:
            raise ValueError("Seeking is unsupported")


def s3_upload(path: str,
              target_uri: str,
              s3_access_key_id: str,
              s3_secret_access_key: str,
              s3_endpoint: str = default_s3_endpoint):
    if s3_secret_access_key is None:
        raise TypeError("No secret key provided")
    if s3_access_key_id is None:
        raise TypeError("No access key provided")
    if s3_endpoint is None:
        raise TypeError("No S3 endpoint provided")
    client = boto3.session.Session.client(
        boto3.session.Session(),
        endpoint_url="https://" + s3_endpoint,
        service_name="s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key)
    path_uri = urlparse(target_uri)
    bucket = path_uri.netloc
    key = path_uri.path.lstrip('/')

    client.upload_file(path, bucket, key)


def s3_download(path_uri: str,
                s3_access_key_id: str,
                s3_secret_access_key: str,
                s3_endpoint: str = default_s3_endpoint) -> CURLStreamFile:
    if s3_secret_access_key is None:
        raise TypeError("No secret key provided")
    if s3_access_key_id is None:
        raise TypeError("No access key provided")
    if s3_endpoint is None:
        raise TypeError("No S3 endpoint provided")

    client = boto3.session.Session.client(
        boto3.session.Session(),
        endpoint_url="https://" + s3_endpoint,
        service_name="s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key)
    path_uri = urlparse(path_uri)
    bucket = path_uri.netloc
    key = path_uri.path.lstrip('/')

    url = client.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket,
                'Key': key},
        ExpiresIn=300)
    print(url)
    return CURLStreamFile(url)


def open_stream(
        path_uri: Union[str, os.PathLike],
        mode: str = "rb",
        s3_access_key_id: Optional[str] = None,
        s3_secret_access_key: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
) -> Union[RequestsStreamFile, CURLStreamFile, typing.BinaryIO]:
    """Open a file path, http(s):// URL, or s3:// URI.
    :param path_uri: File path, http(s):// URL, or s3:// URI to open.
    :param mode: Mode with which to open the stream.
        Supported values are:
        * "rb" for http(s)://,
        * "rb", "wb', and "ab" for s3://,
        * All standard modes for file paths.
    :param s3_access_key_id: S3 access key, corresponding to
        "aws_access_key_id" in boto3.
        If not specified, an s3:// URI is being opened, and ~/.s3cfg exists,
        ~/.s3cfg's "access_key" will be parsed as this credential.
    :param s3_secret_access_key: S3 secret key, corresponding to
        "aws_secret_access_key" in boto3.
        If not specified, an s3:// URI is being opened, and ~/.s3cfg exists,
        ~/.s3cfg's "secret_key" will be parsed as this credential.
    :param s3_endpoint: S3 endpoint.
        If not specified and a host_base was found
        alongside previously parsed credentials, that will be used.
        Otherwise, object.ord1.coreweave.com is the default.
    :return: An opened file-like object representing the target resource.
    """
    if isinstance(path_uri, os.PathLike):
        path_uri = os.fspath(path_uri)

    scheme, *location = path_uri.split("://", maxsplit=1)
    scheme = scheme.lower() if location else None

    normalized_mode = "".join(sorted(mode))

    if scheme in ("http", "https"):
        if normalized_mode != "br":
            raise ValueError(
                'Only the mode "rb" is valid when opening http(s):// streams.')
        if fcntl is not None and curl_path is not None:
            # We have fcntl and curl exists, so we can use the fast CURL-based loader.
            logger.debug(f"Using CURL for tensor streaming of {path_uri}")
            return CURLStreamFile(path_uri)
        else:
            # Fallback to slow requests-based loader.
            logger.debug(f"Using requests for tensor streaming {path_uri}")
            return RequestsStreamFile(path_uri)

    elif scheme == "s3":
        if normalized_mode not in ("br", "bw", "ab"):
            raise ValueError(
                'Only the modes "rb", "wb", and "ab" are valid'
                ' when opening s3:// streams.'
            )
        if not s3_access_key_id or not s3_secret_access_key:
            # Try to find default credentials if not specified
            try:
                parsed: _ParsedCredentials = _get_s3cfg_values()
            except ValueError as parse_error:
                raise ValueError(
                    "Attempted to access S3 bucket,"
                    " but credentials were not provided,"
                    " and the fallback .s3cfg file could not be parsed.") \
                    from parse_error

            if parsed.config_file is None:
                raise ValueError("Attempted to access S3 bucket,"
                                 " but credentials were not provided,"
                                 " and no default .s3cfg file could be found.")

            s3_access_key_id = s3_access_key_id or parsed.s3_access_key
            s3_secret_access_key = s3_secret_access_key or parsed.s3_secret_key
            s3_endpoint = s3_endpoint or parsed.s3_endpoint

            for required_credential, credential_name in (
                    (s3_access_key_id, "s3_access_key_id"),
                    (s3_secret_access_key, "s3_secret_access_key")
            ):
                if not required_credential:
                    raise ValueError(
                        "Attempted to access S3 bucket,"
                        f" but {credential_name} was not provided,"
                        " and could not be found in the default"
                        f" config file at {parsed.config_file}."
                    )

        # Regardless of whether the config needed to be parsed,
        # the endpoint gets a default value.
        s3_endpoint = s3_endpoint or default_s3_endpoint

        if 'w' in mode or 'a' in mode:
            class AutoUploadedTempFile(tempfile.NamedTemporaryFile):
                def close(self):
                    # Close, upload by name, and then delete the file.
                    #
                    # boto3's upload_fileobj could be used before closing the file,
                    # instead of closing it and then uploading it by name,
                    # but upload_fileobj is less performant than upload_file
                    # as of boto3's s3 library s3transfer, version 0.6.0.
                    # For details, see the implementation & comments:
                    # https://github.com/boto/s3transfer/blob/0.6.0/s3transfer/upload.py#L351
                    # TL;DR: s3transfer does multithreaded transfers
                    # that require multiple file handles to work properly,
                    # but Python cannot duplicate file handles such that
                    # they can be accessed in a thread-safe way,
                    # so they have to buffer it all in memory.
                    if self.closed:
                        # Makes close() idempotent.
                        # If the resulting object is used as a context manager,
                        # close() is called twice (once in the serializer code,
                        # once after, when leaving the context).
                        # Without this check, this would trigger two separate uploads.
                        return
                    try:
                        super().close()
                        s3_upload(self.name,
                                  path_uri,
                                  s3_access_key_id,
                                  s3_secret_access_key,
                                  s3_endpoint)
                    finally:
                        os.unlink(self.file)
            # delete must be False or the file will be deleted by the OS
            # as soon as it closes, before it can be uploaded
            # on platforms with primitive temporary file support (e.g. Windows)
            return AutoUploadedTempFile(mode="wb+", delete=False)
        else:
            return s3_download(path_uri,
                               s3_access_key_id,
                               s3_secret_access_key,
                               s3_endpoint)

    else:
        handle: typing.BinaryIO = open(path_uri, mode)
        handle.seek(0)
        return handle

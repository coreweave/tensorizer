import fcntl
import logging
import os
import subprocess
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

try:
    curl_path = shutil.which("curl")
except IOError as e:
    logger.warning(e.strerror)
    curl_path = None

s3_endpoint = "object.ord1.coreweave.com"
s3_access_key = None
s3_secret_key = None

def get_s3cfg_values():
    """
    Get the AWS credentials from the .s3cfg file.
    """
    import configparser
    config = configparser.ConfigParser()
    s3path = os.path.expanduser("~/.s3cfg")
    if not os.path.exists(s3path):
        return
    config.read(s3path)
    if "default" not in config:
        raise ValueError("No default section in ~/.s3cfg")
    if "access_key" not in config["default"]:
        raise ValueError("No access_key in ~/.s3cfg")
    else:
        s3_access_key = config["default"]["access_key"]
    if "secret_key" not in config["default"]:
        raise ValueError("No secret_key in ~/.s3cfg")
    else:
        s3_secret_key = config["default"]["secret_key"]
    if "host_base" in config["default"]:
        s3_endpoint = config["default"]["host_base"]



class CURLStreamFile(object):
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
                begin_pos = 0
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
        if self._curl is not None:
            if self._curl.poll() is None:
                self._curl.stdout.close()
                self._curl.terminate()
                self._curl.wait()
            self._curl = None

    def readline(self):
        raise Exception("Unimplemented")

    """
    This seek() implementation should be avoided if you're seeking backwards,
    as it's not very efficient due to the need to restart the curl process.
    """

    def seek(self, position, whence=SEEK_SET):
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise (Exception("Unsupported `whence`"))
        elif position > self._curr:
            # We're seeking forward, so we just read until we get there.
            self._read_until(position)
        else:
            # To seek backwards, we need to close out our existing process and
            # start a new one.
            self.close()

            # And we reinitialize ourself.
            self.__init__(self._uri, position, None)


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

def s3_upload(path: str,
              target_uri: str,
              aws_access_key_id: str,
              aws_secret_access_key: str,
              s3_endpoint: str = "object.ord1.coreweave.com"):
    if aws_secret_access_key is None:
        raise Exception("No secret key provided")
    if aws_access_key_id is None:
        raise Exception("No access key provided")
    if s3_endpoint is None:
        raise Exception("No S3 endpoint provided")
    client = boto3.session.Session.client(
        boto3.session.Session(),
        endpoint_url="https://" + s3_endpoint,
        service_name="s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key)
    path_uri = urlparse(target_uri)
    bucket = path_uri.netloc
    key = path_uri.path.lstrip('/')

    client.upload_file(path, bucket, key)

def s3_download(path_uri: str,
                aws_access_key_id: str,
                aws_secret_access_key: str,
                s3_endpoint: str = "object.ord1.coreweave.com") -> CURLStreamFile:
    if aws_secret_access_key is None:
        raise Exception("No secret key provided")
    if aws_access_key_id is None:
        raise Exception("No access key provided")
    if s3_endpoint is None:
        raise Exception("No S3 endpoint provided")

    client = boto3.session.Session.client(
        boto3.session.Session(),
        endpoint_url="https://" + s3_endpoint,
        service_name="s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key)
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
        path_uri: str,
        mode: str = "rb",
        aws_access_key_id: Optional[str] = s3_access_key,
        aws_secret_access_key: Optional[str] = s3_secret_key,
        s3_endpoint: Optional[str] = s3_endpoint,
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
    elif path_uri.startswith("s3://"):
        if 'w' in mode or 'a' in mode:
            tmp_path = tempfile.mktemp()
            handle = open(tmp_path, mode)
            old_close = handle.close
            handle.close = lambda: old_close() or s3_upload(tmp_path,
                                                            path_uri,
                                                            aws_access_key_id,
                                                            aws_secret_access_key,
                                                            s3_endpoint)
            return handle
        else:
            tmp_path = tempfile.mktemp()
            return s3_download(path_uri,
                               aws_access_key_id,
                               aws_secret_access_key,
                               s3_endpoint)

    else:
        handle: typing.BinaryIO = open(path_uri, mode)
        handle.seek(0)
        return handle

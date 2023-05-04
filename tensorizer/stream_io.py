import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import typing
import weakref
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import boto3

import tensorizer._wide_pipes as _wide_pipes

__all__ = ["open_stream", "CURLStreamFile"]

logger = logging.getLogger(__name__)

curl_path = shutil.which("curl")

default_s3_read_endpoint = "accel-object.ord1.coreweave.com"
default_s3_write_endpoint = "object.ord1.coreweave.com"

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
    ] = None
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


class CURLStreamFile:
    """
    CURLStreamFile implements a file-like object around an HTTP download, the
    intention being to not buffer more than we have to. It is intended for
    tar-like files, where we start at the beginning and read until the end of
    the file.

    It does implement `seek` and `tell`, but only for the purpose of
    implementing `read`, and only for the purpose of reading the entire file.
    It does support seeking to an arbitrary position, but is very inefficient
    in doing so as it requires re-opening the connection to the server.
    """

    def __init__(
        self,
        uri: str,
        begin: Optional[int] = None,
        end: Optional[int] = None,
        headers: Dict[str, Any] = None,
    ) -> None:
        self._uri = uri
        self._error_context = []

        if curl_path is None:
            RuntimeError(
                "cURL is a required dependency for streaming downloads"
                " and could not be found."
            )

        # NOTE: `256mb` buffer on the python IO object.
        cmd = [
            curl_path,
            "--header",
            "Accept-Encoding: identity",
            "-s",
            "-f",
            uri,
        ]

        if begin is not None or end is not None:
            cmd.extend(["--range", f"{begin or 0}-{end or ''}"])

        if headers is not None:
            for k, v in headers.items():
                cmd.extend(["--header", f"{k}: {v}"])

        self._curl = None
        with _wide_pipes.widen_new_pipes():  # Widen on Windows
            self._curl = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                bufsize=256 * 1024 * 1024,
            )

        _wide_pipes.widen_pipe(self._curl.stdout.fileno())  # Widen on Linux
        self._curr = 0 if begin is None else begin
        self._end = end
        self.closed = False

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
            "-I",  # Only read headers
            "-XGET",  # Use a GET request (in case HEAD is not supported)
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
        try:
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
            else:
                # stdout is normally closed by the Popen.communicate() method,
                # which we skip in favour of Popen.stdout.read()
                self._curl.stdout.close()
            self._curl = None

    def readline(self):
        raise NotImplementedError("Unimplemented")

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
        else:
            # To seek backwards, we need to close out our existing process and
            # start a new one.
            self.close()

            # And we reinitialize ourself.
            self.__init__(self._uri, position, None)


def _ensure_https_endpoint(endpoint: str):
    scheme, *location = endpoint.split("://", maxsplit=1)
    scheme = scheme.lower() if location else None
    if scheme is None:
        return "https://" + endpoint
    elif scheme == "https":
        return endpoint
    else:
        raise ValueError("Non-HTTPS endpoint URLs are not allowed.")


def s3_upload(
    path: str,
    target_uri: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_endpoint: str = default_s3_write_endpoint,
):
    if s3_secret_access_key is None:
        raise TypeError("No secret key provided")
    if s3_access_key_id is None:
        raise TypeError("No access key provided")
    if s3_endpoint is None:
        raise TypeError("No S3 endpoint provided")

    client = boto3.session.Session.client(
        boto3.session.Session(),
        endpoint_url=_ensure_https_endpoint(s3_endpoint),
        service_name="s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
    )
    path_uri = urlparse(target_uri)
    bucket = path_uri.netloc
    key = path_uri.path.lstrip("/")

    client.upload_file(path, bucket, key)


def s3_download(
    path_uri: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_endpoint: str = default_s3_read_endpoint,
) -> CURLStreamFile:
    if s3_secret_access_key is None:
        raise TypeError("No secret key provided")
    if s3_access_key_id is None:
        raise TypeError("No access key provided")
    if s3_endpoint is None:
        raise TypeError("No S3 endpoint provided")

    client = boto3.session.Session.client(
        boto3.session.Session(),
        endpoint_url=_ensure_https_endpoint(s3_endpoint),
        service_name="s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
    )
    path_uri = urlparse(path_uri)
    bucket = path_uri.netloc
    key = path_uri.path.lstrip("/")

    url = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=300,
    )
    return CURLStreamFile(url)


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
    Meant to replace .close() on a particular instance
    of a temporary file-like wrapper object, as an unbound
    callback to a weakref.finalize() registration on the wrapper.

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

    if file.closed:
        # Makes closure idempotent.

        # If the file object is used as a context
        # manager, close() is called twice (once in the
        # serializer code, once after, when leaving the
        # context).

        # Without this check, this would trigger two
        # separate uploads.
        return
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
) -> Union[CURLStreamFile, typing.BinaryIO]:
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
    """
    if isinstance(path_uri, os.PathLike):
        path_uri = os.fspath(path_uri)

    scheme, *location = path_uri.split("://", maxsplit=1)
    scheme = scheme.lower() if location else None

    normalized_mode = "".join(sorted(mode))

    if scheme in ("http", "https"):
        if normalized_mode != "br":
            raise ValueError(
                'Only the mode "rb" is valid when opening http(s):// streams.'
            )
        return CURLStreamFile(path_uri)

    elif scheme == "s3":
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

            guaranteed_closer = weakref.finalize(
                temp_file,
                _temp_file_closer,
                temp_file.file,
                temp_file.name,
                path_uri,
                s3_access_key_id,
                s3_secret_access_key,
                s3_endpoint,
            )
            temp_file.close = guaranteed_closer
            return temp_file
        else:
            s3_endpoint = s3_endpoint or default_s3_read_endpoint
            curl_stream_file = s3_download(
                path_uri, s3_access_key_id, s3_secret_access_key, s3_endpoint
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
        os.makedirs(os.path.dirname(path_uri), exist_ok=True)
        handle: typing.BinaryIO = open(path_uri, mode)
        handle.seek(0)
        return handle

##############################################################################
# serialization.py                                                   Wes Brown
# Fast torch module/model serialization/deserialization     (c) 2023 Coreweave
##############################################################################

# try to import UNIX only dependencies
try:
    import fcntl
    import resource
except ImportError:
    fcntl = None
    resource = None

import collections.abc
import concurrent.futures
import ctypes
import dataclasses
import hashlib
import io
import itertools
import logging
import mmap
import os
import queue
import struct
import threading
import time
import typing
import zlib
from collections import OrderedDict
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy
import torch

import tensorizer.stream_io as stream_io
import tensorizer.utils as utils
from tensorizer._NumpyTensor import _NumpyTensor

if torch.cuda.is_available():
    cudart = torch.cuda.cudart()
else:
    cudart = None

lz4 = None

__all__ = ["TensorSerializer", "TensorDeserializer", "TensorType"]

# Setup logger
logger = logging.getLogger(__name__)


# Whether the tensor is a parameter or a buffer on the model.
class TensorType(Enum):
    PARAM = 0
    BUFFER = 1
    STATE_DICT = 2


# If tensors with "opaque" dtypes (those that are not supported by numpy) are
# saved, then a tensorizer data version of 2 is required to (de)serialize the
# file. Otherwise, the file is compatible with tensorizer data version 1
TENSORIZER_VERSION = 2
NON_OPAQUE_TENSORIZER_VERSION = 1

TENSORIZER_MAGIC = b"|TZR|"

OPAQUE_DTYPE_SEP = "\0"


class HashType(Enum):
    CRC32 = 0
    SHA256 = 1


class TensorHash(typing.TypedDict):
    type: HashType
    hash: bytes


class TensorEntry(typing.TypedDict):
    name: str
    type: TensorType
    offset: int
    data_offset: int
    data_length: int
    dtype: str
    shape: List[int]
    hashes: List[TensorHash]
    raw_headers: bytes


class HashMismatchError(Exception):
    pass


class TensorDeserializer(collections.abc.Mapping):
    """
    Given a file-like object for read, deserialize tensors to a state_dict or
    a torch.nn.Module.

    See the docs_ for a usage walkthrough.

    .. _docs: https://github.com/coreweave/tensorizer/tree/main#basic-usage

    Args:
        file_obj: A file-like object to read from. It can also be a string
            representing a path to a file or an HTTP/HTTPS/S3 URI.
        device: The device to load the tensors to.
        filter_func: A function (tensor_name: str) -> bool that returns True
            if a tensor should be loaded, or False if it should be skipped.
            If None, all tensors are loaded.
        dtype: The dtype to cast the tensors as when loading them into a torch
            module. If None, the dtype will be inferred from the file.
        lazy_load: If True, tensors will be loaded and cached when keys are
            accessed. If False, all tensors will be loaded into memory up
            front.
        plaid_mode: If True, tensors will be loaded extremely fast into the
            target device. This is only supported on CUDA devices, and the
            buffers are going to be inconsistent due to the extreme
            naughtiness of reusing a backing buffer. This is only recommended
            for use with inference, and not training.
        verify_hash: If True, the hashes of each tensor will be verified
            against the hashes stored in the metadata. A `HashMismatchError`
            will be raised if any of the hashes do not match.

    Raises:
        HashMismatchError: If ``verify_hash=True`` and a deserialized tensor
            does not match its stored hash.

    Examples:
        Deserializing a pre-serialized_ 16-bit ``transformers`` model from S3::

            from tensorizer import TensorDeserializer, utils
            from transformers import AutoConfig, AutoModelForCausalLM
            import torch

            model_ref = "EleutherAI/gpt-neo-125M"

            # Create an empty torch.nn.Module with the right shape
            config = AutoConfig.from_pretrained(model_ref, dtype=torch.float16)
            with utils.no_init_or_tensor():
                model = AutoModelForCausalLM.from_config(config)

            # Public `tensorized` bucket hosted by CoreWeave; see the docs
            s3_uri = f"s3://tensorized/{model_ref}/fp16/model.tensors"

            deserializer = TensorDeserializer(s3_uri, plaid_mode=True)
            deserializer.load_into_module(model)

        ## From a private S3 bucket::

            from tensorizer import stream_io
            s3 = stream_io.open_stream(
                "s3://some-private-bucket/my-model.tensors",
                mode="rb",
                s3_access_key_id=...,
                s3_secret_access_key=...,
                s3_endpoint=...,
            )
            # Set up `model` as an empty torch.nn.Module in the shape of
            # my-model.tensors, then:
            deserializer = TensorDeserializer(s3, plaid_mode=True)
            deserializer.load_into_module(model)

        .. _pre-serialized: https://github.com/coreweave/tensorizer/tree/main#available-pre-tensorized-models-on-the-coreweave-cloud
    """

    def __init__(
        self,
        file_obj: Union[
            io.BufferedIOBase,
            io.RawIOBase,
            typing.BinaryIO,
            str,
            bytes,
            os.PathLike,
            int,
        ],
        device: Optional[Union[torch.device, str]] = None,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        lazy_load: bool = False,
        plaid_mode: bool = False,
        verify_hash: bool = False,
    ):
        # Whether to verify the hashes of the tensors when they are loaded.
        # This value is used when no verify_hash argument is passed to the
        # tensor loading methods.
        self._verify_hash = verify_hash

        if isinstance(file_obj, (str, bytes, os.PathLike, int)):
            self._file = stream_io.open_stream(file_obj, "rb")
        else:
            self._mode_check(file_obj)
            self._file = file_obj
        self.total_compressed_tensor_bytes = 0
        self.read_bytes = 0

        # If device is None, use the current device, otherwise use the given
        # device.
        device = utils.get_device() if device is None else torch.device(device)
        self._device: torch.device = device

        self._dtype: Optional[torch.dtype] = dtype

        self._plaid_mode: bool = plaid_mode

        # plaid_mode implies lazy_load
        self._lazy_load: bool = lazy_load or plaid_mode

        self._metadata: Dict[str, TensorEntry] = {}

        if self._plaid_mode and (
            not torch.cuda.is_available() or self._device.type == "cpu"
        ):
            raise ValueError("Plaid mode requires CUDA")

        # Read the magic
        magic = self._file.read(5)
        if magic != TENSORIZER_MAGIC:
            raise ValueError("Not a tensorizer file")

        # Read the version
        version = struct.unpack("<I", self._file.read(4))[0]

        # Check the version
        if version > TENSORIZER_VERSION:
            raise ValueError(
                f"This tensorizer version ({TENSORIZER_VERSION}) cannot read"
                f" file versioned {version}."
            )

        # Skip 32-byte hash (unused)
        self._file.read(32)

        # Read the total size of the file
        self.total_file_bytes = struct.unpack("<Q", self._file.read(8))[0]

        # Read total size of tensor data
        self.total_tensor_bytes = struct.unpack("<Q", self._file.read(8))[0]

        # Read the number of tensors
        self._tensors = struct.unpack("<Q", self._file.read(8))[0]

        # Read the metadata index of tensors. This is a list of offsets into the
        # file where the per-tensor data is stored. filter_func is a test that
        # determines the tensor names to read. If filter_func is None,
        # all tensors are read.
        self._load_metadatas(filter_func)

        self._prior_key: Optional[str] = None

        # We calculate the total tensor bytes here so that we can use mmap,
        # based on the total size of the tensors that we're going to read,
        # filtered by the filter_func.
        self.total_tensor_bytes = 0
        self._largest_tensor_bytes = 0
        for name, metadata in self._metadata.items():
            self.total_tensor_bytes += metadata["data_length"]
            if metadata["data_length"] > self._largest_tensor_bytes:
                self._largest_tensor_bytes = metadata["data_length"]

        # Allocate the buffer for the tensors. If we're not in lazy_load mode,
        # we'll allocate a single buffer for all the tensors. Otherwise, we'll
        # allocate a buffer for the largest tensor.
        self._buffer_addr = None
        self._is_memory_pinned = False
        if not self._lazy_load and not self._plaid_mode:
            start_allocate = time.time()

            # Check if our platform supports mmap.MAP_ANONYMOUS and
            # mmap.MAP_PRIVATE
            mmap_flags = 0
            mmap_flags |= getattr(mmap, "MAP_PRIVATE", 0)
            mmap_flags |= getattr(mmap, "MAP_ANONYMOUS", 0)
            mmap_args = {"flags": mmap_flags} if mmap_flags else {}
            self._buffer = mmap.mmap(-1, self.total_tensor_bytes, **mmap_args)

            # If we're on CUDA, we register the buffer with CUDA so that
            # it is pinned. This allows for Torch to internally use
            # cudaMemcpyAsync.
            if self._device.type == "cuda":
                # We need to use ctypes to get the address of the buffer
                # because mmap.mmap doesn't expose the buffer address.
                tb = ctypes.c_char * self.total_tensor_bytes
                ctb = tb.from_buffer(self._buffer)
                self._buffer_addr = ctypes.addressof(ctb)
                del ctb  # don't leave an open exported pointer into the mmap

                # Register the buffer with CUDA
                cudart.cudaHostRegister(
                    self._buffer_addr, self.total_tensor_bytes, 0
                )
                self._is_memory_pinned = True
            end_allocate = time.time()
            tensor_bytes_str = utils.convert_bytes(self.total_tensor_bytes)
            logger.info(
                f"Allocated {tensor_bytes_str} "
                f"for {len(self._metadata)} tensors "
                f"in {end_allocate - start_allocate:0.4f}"
            )
        else:
            self._buffer = bytearray(self._largest_tensor_bytes)

        # The number of bytes we've allocated so far. Tensors may be read
        # from the file in any order, so we need to keep track of how much
        # we've used so far so that we can index into the buffer correctly.
        self._allocated = 0

        # Our cache of tensors. This is a dict of name -> tensor. If lazy_load
        # is True, then the tensors are not loaded until they are accessed.
        self._cache: typing.OrderedDict[str, Union[torch.Tensor, None, bool]]

        # The offset in the file where the tensor data begins.
        self._tensors_begin = self._file.tell()

        if not self._lazy_load:
            # If we're not in lazy_load mode, we populate the cache with all
            # the tensors.
            self._cache = self._generate_state_dict()
        else:
            # We populate the cache with None values so that we can
            # differentiate between tensors that have not been loaded yet
            # and tensors that are not present in the file.
            self._cache = OrderedDict.fromkeys(self._metadata.keys())

    @staticmethod
    def _mode_check(file_obj: io.IOBase) -> None:
        try:
            readable = file_obj.readable()
        except AttributeError:
            # If file_obj doesn't implement the full io.IOBase interface
            # and checking is not possible, assume it's fine
            readable = True
        if isinstance(file_obj, io.TextIOBase) or not readable:
            mode = getattr(file_obj, "mode", "")
            raise ValueError(
                "TensorSerializer's file_obj must be readable "
                'and in binary mode (mode="rb"{})'.format(
                    mode and f', current mode="{mode}"'
                )
            )

    def __del__(self):
        self.close()

    def close(self):
        # Don't throw an attribute error if these aren't defined yet,
        # e.g. if __init__ threw an error before defining both
        buffer = getattr(self, "_buffer", None)
        if buffer is not None:
            if self._is_memory_pinned and self._buffer_addr:
                cudart.cudaHostUnregister(self._buffer_addr)
                self._is_memory_pinned = False
            if self._device.type != "cpu" and hasattr(buffer, "close"):
                # Don't close the mmap buffer for CPU tensors because
                # it would free their memory out from under them.
                # For GPU tensors we just need to wait until
                # they are transferred out of RAM.
                buffer.close()
        if getattr(self, "_file", None) is not None:
            self._file.close()

    def _read_string(self, io_obj=None):
        """
        Read a string from the file.
        """
        if io_obj is None:
            io_obj = self._file

        length = struct.unpack("<H", io_obj.read(2))[0]
        return io_obj.read(length).decode("utf-8")

    def _read_dtype(self, io_obj=None):
        """
        Read a dtype from the file.
        """
        if io_obj is None:
            io_obj = self._file

        length = struct.unpack("<B", io_obj.read(1))[0]
        return io_obj.read(length).decode("utf-8")

    def _read_metadata(self, metadata_stream: io.BytesIO) -> TensorEntry:
        name = self._read_string(metadata_stream)
        tensor_type = TensorType(
            struct.unpack("<B", metadata_stream.read(1))[0]
        )
        dtype = self._read_dtype(metadata_stream)
        shape_len = struct.unpack("<B", metadata_stream.read(1))[0]
        shape = self._read_shapes(
            metadata_stream.read(shape_len * 4), shape_len
        )
        offset = struct.unpack("<Q", metadata_stream.read(8))[0]
        data_offset = struct.unpack("<Q", metadata_stream.read(8))[0]
        data_length = struct.unpack("<Q", metadata_stream.read(8))[0]
        return TensorEntry(
            name=name,
            type=tensor_type,
            offset=offset,
            data_offset=data_offset,
            data_length=data_length,
            dtype=dtype,
            shape=shape,
        )

    def _load_metadatas(
        self, filter_func: Optional[Callable[[str], Union[bool, Any]]]
    ):
        """
        Read the metadata of tensors into self._metadata.
        """

        # Read metadata size.
        self._metadata_size = struct.unpack("<Q", self._file.read(8))[0]
        metadata_encoded = self._file.read(self._metadata_size)
        # Turn the metadata into a stream.
        metadata_stream = io.BytesIO(metadata_encoded)

        for i in range(self._tensors):
            metadata = self._read_metadata(metadata_stream)
            if filter_func is None or filter_func(metadata["name"]):
                self._metadata[metadata["name"]] = metadata

    @staticmethod
    def _read_shapes(obj, num_elems) -> List[int]:
        """
        Read the tensor shapes.
        """
        return list(struct.unpack(f"<{num_elems}I", obj))

    @property
    def total_bytes_read(self) -> int:
        if hasattr(self._file, "bytes_read"):
            return self._file.bytes_read
        if self._file.closed:
            return self.total_tensor_bytes
        else:
            return self._file.tell()

    # If our _file object has 'response_headers' attribute, we can use it
    # to determine if we were cached or not.
    @property
    def cache_status(self) -> Optional[str]:
        if hasattr(self._file, "response_headers"):
            return self._file.response_headers.get("x-cache-status", None)
        else:
            return None

    def __getitem__(self, name) -> torch.nn.Parameter:
        if self._plaid_mode:
            # In plaid_mode, we only have one valid tensor at a time, so
            # we need to make sure that prior tensors are not accessed.
            if self._prior_key is not None and self._prior_key != name:
                self._cache[self._prior_key] = False
            if self._cache[name] is False:
                raise RuntimeError(
                    f"Tensor {name} already overwritten in plaid_mode"
                )
            self._prior_key = name

        if name in self._cache and self._cache[name] is not None:
            return self._cache[name]

        # If we're in lazy_load mode, we populate the cache with the
        # tensor data and then convert it to a torch parameter. Most
        # of the time, access patterns are front to back, so seeking
        # forward in a stream works well even for HTTP/HTTPS streams.
        if name in self._metadata:
            self._file.seek(self._metadata[name]["offset"])
            tensor_arr = next(self.read_tensors(num_tensors=1))[3]
            self._cache[name] = self._to_torch_parameter(tensor_arr)
            return self._cache[name]
        else:
            raise KeyError(f"Tensor {name} not found")

    # To implement collections.abc.Mapping, this class needs to define:
    # 1. __getitem__(key)
    # 2. __iter__()
    # 3. __len__()
    #
    # It then inherits efficient default implementations for:
    # 1. get(key)
    # 2. keys()
    # 3. items()
    # 4. values()
    #
    # It also inherits __contains__(key), but it isn't efficient.
    # The default __contains__ implementation uses __getitem__,
    # so it loads tensor data unnecessarily. Instead, we can
    # check self._metadata with a simple dict lookup.
    #
    # Significant features of the defaults:
    # - values() -> collections.abc.ValuesView:
    #   - Doesn't need to load all the data up front
    #   - Returns a collection with a lazy __iter__
    #     - Only defers to our __getitem__ exactly when required
    #   - Implements __contains__ by iterating over itself with linear search
    # - items() -> collections.abc.ItemsView:
    #   - Also doesn't need to load data up front
    #   - Implements __iter__ as basically zip(keys(), values())
    #   - Implements __contains__ with __getitem__ since it knows where to
    #     look and needs the data anyway
    #
    # In summary, ignoring the (still efficient) __contains__, these are
    # simply lazy iterables over the respective elements from this parent
    # mapping.

    def __iter__(self):
        # iter() on a mapping returns an iterator of only keys
        yield from self._metadata

    def __len__(self):
        return len(self._metadata)

    def __contains__(self, key: str):
        return key in self._metadata

    def keys(self):
        # We override keys() because dict_keys can be slightly more efficient
        # than an extra collections.abc.KeysView wrapper.
        #
        # Technically this makes mapping.keys().mapping invalid on
        # Python 3.10+ but it is not intended to be supported anyway, so treat
        # it as not implemented.
        return self._metadata.keys()

    @staticmethod
    def _decode_hashes(b: memoryview) -> List[TensorHash]:
        """
        Decode the hashes from given bytes.
        """
        hashes: List[TensorHash] = []

        # Read the number of hashes.
        num_hashes = b[0]

        hash_idx = 1
        # Read the hashes.
        for i in range(num_hashes):
            # Read the hash type.
            hash_type = b[hash_idx]
            # Read the size of the hash.
            hash_size = b[hash_idx + 1]
            # Read the hash.
            hash_begin = hash_idx + 2
            hash_end = hash_begin + hash_size
            hash_bytes = bytes(b[hash_begin:hash_end])
            # Add the hash to the list.
            hash_entry = TensorHash(
                type=HashType(hash_type),
                hash=hash_bytes,
            )
            hash_idx = hash_end
            hashes.append(hash_entry)

        return hashes

    @staticmethod
    def _zero_hashes(b: memoryview) -> bytearray:
        """
        Zero out the encoded hashes in the given bytes,
        and return the data structure with the hashes zeroed out.
        This is used to prevent the hashes from being part of hash computation
        of the entire data structure.
        """
        # Read the number of hashes.
        num_hashes = b[0]
        zeroed_hashes = bytearray(b)

        hash_idx = 1
        # Read the hashes.
        for i in range(num_hashes):
            # Read the size of the hash.
            hash_size = b[hash_idx + 1]
            hash_begin = hash_idx + 2
            hash_end = hash_begin + hash_size
            zeroed_hashes[hash_begin:hash_end] = bytes(hash_size)
            hash_idx = hash_end
        return zeroed_hashes

    @staticmethod
    def _verify_hashes(
        name: str,
        hashes: List[TensorHash],
        headers: bytes,
        mv: Union[memoryview, bytes],
    ) -> None:
        """
        Verifies the hash of the tensor data.

        Args:
            hashes: The list of hashes to verify.
            headers: The headers of the tensor.
            mv: The memoryview of the tensor data.
        """
        for tensor_hash in hashes:
            hash_type = tensor_hash["type"]
            hash_body = tensor_hash["hash"]
            if hash_type == HashType.CRC32:
                crc = zlib.crc32(mv, zlib.crc32(headers))
                hash_crc = struct.unpack("<I", hash_body)[0]
                if crc != hash_crc:
                    raise HashMismatchError(
                        f"Tensor '{name}' failed CRC32 verification. "
                        f"Expected {hash_crc}, got {crc}."
                    )
            elif hash_type == HashType.SHA256:
                sha = hashlib.sha256(headers)
                sha.update(mv)
                sha_digest = sha.digest()
                if sha_digest != hash_body:
                    raise HashMismatchError(
                        f"Tensor '{name}' failed SHA256 verification. "
                        f"Expected {hash_body.hex()}, got {sha_digest.hex()}."
                    )
            else:
                raise ValueError(
                    f"Tensor '{name}' has an invalid hash type: {hash_type}"
                )

    def _read_numpytensors(
        self,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        num_tensors: int = -1,
        verify_hash: Optional[bool] = None,
    ) -> Iterator[Tuple[int, int, str, _NumpyTensor]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and a _NumpyTensor `tensor`.

        Note that this function does not seek to the beginning of the tensor
        data. It assumes that the file pointer is already at the beginning
        of the tensor data that it should read.

        It will read `num_tensors` tensors from the file, or all tensors
        if `num_tensors` is -1.

        The generator yields tuples of the form:
            (module_idx, tensor_type, name, arr)

        Args:
            filter_func: A function that takes a tensor name and returns
                True if the tensor should be returned, False otherwise.
            num_tensors: The number of tensors to read. If -1, all tensors
                will be read. If the zero-byte header is encountered before
                `num_tensors` tensors are read, the generator will stop
                yielding values.
            verify_hash: If True, the hashes of each tensor will be verified
                against the hashes stored in the metadata.
                A `HashMismatchError` will be raised if any of the hashes do
                not match. If ``None``, the value of the `verify_hash` argument
                passed to the `TensorDeserializer` constructor will be used.

        Raises:
            HashMismatchError: If ``verify_hash`` resolves to True and
            a deserialized tensor does not match its stored hash.
        """
        if verify_hash is None:
            verify_hash = self._verify_hash

        try:
            tensors_read = 0
            while num_tensors == -1 or tensors_read < num_tensors:
                header_sz = struct.unpack("<Q", self._file.read(8))[0]
                if header_sz == 0:
                    break
                # We read the entire header into memory rather than reading
                # it piecewise to avoid the overhead of many small reads,
                # especially for network streams.
                headers = self._file.read(header_sz - 8)
                header_len = len(headers)

                module_idx = struct.unpack("<H", headers[0:2])[0]

                tensor_type = TensorType(struct.unpack("<B", headers[2:3])[0])

                # Read the name.
                name_sz = struct.unpack("<H", headers[3:5])[0]
                idx = name_sz + 5
                name_bytes = headers[5:idx]
                name: str = name_bytes.decode("utf-8")

                # Read the dtype of the tensor.
                dtype_len = struct.unpack("<B", headers[idx : idx + 1])[0]
                dtype_end = idx + dtype_len + 1
                dtype = headers[idx + 1 : dtype_end].decode("utf-8")

                numpy_dtype, *torch_dtype = dtype.split(OPAQUE_DTYPE_SEP)
                if len(torch_dtype) == 0:
                    torch_dtype = None
                elif len(torch_dtype) == 1:
                    torch_dtype = torch_dtype[0]
                else:
                    raise ValueError(
                        "Can't deserialize a tensor with "
                        "multiple opaque dtype separators "
                        f"({OPAQUE_DTYPE_SEP!r}) in its dtype: "
                        f"{dtype!r}"
                    )

                # Read the shape amount, according to the serialized format.
                # The shape length is 1 byte after the dtype end.
                shape_len = struct.unpack(
                    "<B", headers[dtype_end : dtype_end + 1]
                )[0]
                # The shape elements are <I, so we read 4 bytes. _read_shapes
                # takes in the header object and the number of elements in
                # the shape.
                #
                # The amount of bytes for the shape is 4 * number of elements
                # in the shape. so, we need to read 4 * shape_len bytes after
                # the dtype end + 1 byte for the shape length. sort of
                # convoluted, but it works.
                shape_begin = dtype_end + 1
                shape_end = shape_begin + (4 * shape_len)
                shape_list = self._read_shapes(
                    headers[shape_begin:shape_end],
                    shape_len,
                )

                # Read our hashes in. We need to read the hashes size,
                # then read the hash bytes.
                hashes_sz_begin = shape_end
                with memoryview(headers) as hashes_mv:
                    hashes_sz = struct.unpack_from(
                        "<H", hashes_mv, hashes_sz_begin
                    )[0]
                    hashes_begin = hashes_sz_begin + 2
                    hashes_end = hashes_begin + hashes_sz
                    hashes_slice = hashes_mv[hashes_begin:hashes_end]
                    if name in self.keys():
                        hashes = self._decode_hashes(hashes_slice)
                        self._metadata[name]["hashes"] = hashes

                # Finally, get the tensor data length.
                data_length = struct.unpack("<q", headers[header_len - 8 :])[0]

                # Check if the name is in our pre-filtered list of keys
                # from the class-level filter_func, and then verify
                # that it passes the method-level filter_func.
                # Skip it if it fails either check.
                if name not in self.keys() or (
                    filter_func is not None and not filter_func(name)
                ):
                    self._file.seek(data_length, io.SEEK_CUR)
                    continue

                # We use memoryview to avoid copying the data.
                mv: memoryview
                if not self._plaid_mode and not self._lazy_load:
                    # In default mode, we've already allocated all the
                    # memory we need in a single buffer that contains
                    # all the tensors. We just need to slice out the
                    # memoryview for the current tensor.
                    mv = memoryview(self._buffer)[
                        self._allocated : self._allocated + data_length
                    ]
                    self._file.readinto(mv)
                    self._allocated += data_length
                elif self._plaid_mode:
                    # In plaid_mode, we don't allocate a buffer, we just
                    # reuse the same one. This is a filthy hack, as we're
                    # overwriting the buffer contents that is used to back
                    # the prior tensor. This works because we don't use
                    # the buffer contents after we yield the tensor, which
                    # is loaded straight into the GPU memory.
                    mv = memoryview(self._buffer)
                    self._file.readinto(mv[:data_length])
                else:
                    # In lazy_load mode, we allocate a new buffer for each
                    # tensor. This is a bit slower, but it's the only way
                    # to support lazy loading.
                    buffer = bytearray(data_length)
                    mv = memoryview(buffer)
                    self._file.readinto(mv)

                # Store our raw headers with hashes zeroed out
                # for model verification
                with memoryview(headers) as header_mv:
                    headers = b"".join(
                        (
                            struct.pack("<Q", header_sz),
                            header_mv[:hashes_begin],
                            self._zero_hashes(hashes_slice),
                            header_mv[hashes_end:],
                        )
                    )
                hashes_slice.release()
                self._metadata[name]["raw_headers"] = headers

                if verify_hash:
                    try:
                        self._verify_hashes(name, hashes, headers, mv)
                    except HashMismatchError:
                        # Necessary to prevent a BufferError on close()
                        mv.release()
                        raise

                tensor = _NumpyTensor.from_buffer(
                    numpy_dtype,
                    torch_dtype,
                    shape_list,
                    mv,
                )

                tensors_read += 1

                yield module_idx, tensor_type, name, tensor
        except EOFError:
            return

    def read_tensors(
        self,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        num_tensors: int = -1,
        verify_hash: Optional[bool] = None,
    ) -> Iterator[Tuple[int, int, str, torch.Tensor]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and torch `tensor`.

        Note that this function does not seek to the beginning of the tensor
        data. It assumes that the file pointer is already at the beginning
        of the tensor data that it should read.

        It will read `num_tensors` tensors from the file, or all tensors
        if `num_tensors` is -1.

        The generator yields tuples of the form:
            (module_idx, tensor_type, name, tensor)

        Args:
            filter_func: A function that takes a tensor name and returns
                True if the tensor should be returned, False otherwise.
            num_tensors: The number of tensors to read. If -1, all tensors
                will be read. If the zero-byte header is encountered before
                `num_tensors` tensors are read, the generator will stop
                yielding values.
            verify_hash: If True, the hashes of each tensor will be verified
                against the hashes stored in the metadata.
                A `HashMismatchError` will be raised if any of the hashes do
                not match. If ``None``, the value of the `verify_hash` argument
                passed to the `TensorDeserializer` constructor will be used.

        Yields:
            Tuples of the form (module_idx, tensor_type, name, tensor).

        Raises:
            HashMismatchError: If ``verify_hash`` resolves to True and
                a deserialized tensor does not match its stored hash.
        """

        data = self._read_numpytensors(
            filter_func=filter_func,
            num_tensors=num_tensors,
            verify_hash=verify_hash,
        )
        for module_idx, tensor_type, name, tensor in data:
            yield module_idx, tensor_type, name, tensor.to_tensor()

    def read_numpy_arrays(
        self,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        num_tensors: int = -1,
        allow_raw_data: bool = False,
        verify_hash: Optional[bool] = None,
    ) -> Iterator[Tuple[int, int, str, numpy.ndarray, bool, Optional[str]]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, the numpy `arr` that
        represents the tensor, a boolean representing if the returned datatype
        is opaque, and the name of the true datatype represented by the opaque
        data, if applicable.

        "Opaque data" refers to numpy arrays holding accurate raw binary data
        but an invalid dtype attribute, occurring when there is no numpy dtype
        corresponding to the original type that was serialized.
        These are only returned if `allow_raw_data` is set to `True`,
        otherwise, encountering such a datatype is an error,
        and the file should instead be deserialized with
        `TensorDeserializer.read_tensors()`.

        For example, if a ``torch.Tensor`` with the dtype ``torch.bfloat16``
        is serialized, then it can be accurately deserialized using
        `TensorDeserializer.read_tensors()`. Since there is no numpy type
        corresponding to ``torch.bfloat16``, attempting to deserialize the same
        file via `TensorDeserializer.read_numpy_arrays()` will raise a
        ``ValueError``.

        However, if `allow_raw_data` is set to ``True``, then
        `TensorDeserializer.read_numpy_arrays()` will return these
        arrays regardless, and the final two values of the yielded tuple,
        `is_opaque` and `torch_dtype`, will be ``True`` and a string
        representing the true non-numpy datatype represented by the data,
        respectively. Special handling is then required to use the returned
        data accurately.

        Note that this function does not seek to the beginning of the tensor
        data. It assumes that the file pointer is already at the beginning
        of the tensor data that it should read.

        It will read `num_tensors` tensors from the file, or all tensors
        if `num_tensors` is -1.

        The generator yields tuples of the form:
            (module_idx, tensor_type, name, arr, is_opaque, torch_dtype)

        See also: `TensorDeserializer.read_tensors`

        Args:
            filter_func: A function that takes a tensor name and returns
                True if the tensor should be returned, False otherwise.
            num_tensors: The number of tensors to read. If -1, all tensors
                will be read. If the zero-byte header is encountered before
                `num_tensors` tensors are read, the generator will stop
                yielding values.
            allow_raw_data: Whether to return numpy arrays containing
                uninterpretable opaque datatypes. If False and opaque
                datatypes are encountered, then a `ValueError` is raised.
                Defaults to False.
            verify_hash: If True, the hashes of each tensor will be verified
                against the hashes stored in the metadata.
                A `HashMismatchError` will be raised if any of the hashes do
                not match. If ``None``, the value of the `verify_hash` argument
                passed to the `TensorDeserializer` constructor will be used.

        Yields:
            Tuples of the form:
            (
                module_idx,
                tensor_type,
                name,
                arr,
                is_opaque,
                torch_dtype
            )
            If the `allow_raw_data` parameter is ``False`` (the default),
            the final two elements are always ``False`` and ``None``,
            respectively. Otherwise, ``is_opaque`` may be ``True``, and
            ``torch_dtype`` will then be a string representing the actual
            non-numpy datatype represented by the data in `arr`.

        Raises:
            ValueError: If an opaque datatype is encountered in the file
                and ``allow_raw_data=False``.
            HashMismatchError: If ``verify_hash`` resolves to True and
                a deserialized tensor does not match its stored hash.
        """
        data = self._read_numpytensors(
            filter_func=filter_func,
            num_tensors=num_tensors,
            verify_hash=verify_hash,
        )
        for module_idx, tensor_type, name, tensor in data:
            is_opaque = tensor.is_opaque
            arr = tensor.data
            torch_dtype = tensor.torch_dtype if is_opaque else None

            if is_opaque and not allow_raw_data:
                np_dtype = arr.dtype.str
                raise ValueError(
                    f"{name} has an opaque datatype: "
                    f"(Torch: {tensor.torch_dtype}, Numpy: {np_dtype}). "
                    "Set `allow_raw_data=True` to return as a numpy array "
                    f"with a datatype of {np_dtype}"
                )

            yield module_idx, tensor_type, name, arr, is_opaque, torch_dtype

    def _to_torch_parameter(
        self, tensor: Union[torch.Tensor, torch.nn.Parameter]
    ) -> torch.nn.Parameter:
        """
        Convert a tensor to a torch.nn.Parameter on a device, forcing
        gradient when appropriate. We also handle torch.nn.Parameter objects in
        a passthrough manner.
        """
        if isinstance(tensor, torch.nn.Parameter):
            tensor.data = tensor.data.to(self._device)
            if tensor.grad is not None:
                tensor.grad = tensor.grad.to(self._device)
            return tensor

        # Cast the tensor if a global dtype was given to the TensorDeserializer
        if (
            self._dtype is not None
            and tensor.dtype != torch.bool
            and torch.dtype != self._dtype
        ):
            tensor = tensor.to(self._dtype)

        gradient = tensor.dtype.is_complex or tensor.dtype.is_floating_point

        return torch.nn.Parameter(
            tensor.to(self._device), requires_grad=gradient
        )

    def _generate_state_dict(self) -> OrderedDict:
        """
        Load the tensors in this Tensorizer object into a state_dict. This
        is used to populate the cache in non-lazy_load cases.
        """
        if self._file.closed:
            raise IOError("IO closed, instantiate if you want to load again.")

        d = OrderedDict()
        for idx, typ, name, arr in self.read_tensors():
            d[name] = self._to_torch_parameter(arr)
        self.total_tensor_bytes = self._file.tell()
        self._file.close()
        return d

    def load_into_module(
        self,
        m: torch.nn.Module,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        verify_hash: Optional[bool] = None,
    ) -> int:
        """
        Given `m`, a torch.nn.Module, load the associate tensors in this
        Tensorizer object into the `torch.nn.Module`. Returns the number of
        tensors loaded into the module.

        Args:
            m: The module to load the tensors into.
            device: The device to load the tensors onto.
            dtype: The dtype to load the tensors as. Defaults to None, which
                means the dtype is not changed from the serialized dtype.
            filter_func: A function (tensor_name: str) -> bool that returns
                True if a tensor should be loaded, or False if it should be
                skipped.
            verify_hash: If True, the hashes of each tensor will be verified
                against the hashes stored in the metadata.
                A `HashMismatchError` will be raised if any of the hashes do
                not match. If ``None``, the value of the `verify_hash` argument
                passed to the `TensorDeserializer` constructor will be used.
                Only has an effect on tensors deserialized during this function
                call. I.e., if ``lazy_load`` is False, all tensors will have
                already been deserialized and checked (or not) prior to this
                call, so the value of this parameter will affect nothing.

        Raises:
            HashMismatchError: If ``verify_hash`` resolves to True and
                a deserialized tensor does not match its stored hash.
        """
        modules: typing.OrderedDict[str, torch.nn.Module] = OrderedDict()

        if verify_hash is None:
            verify_hash = self._verify_hash

        for name, module in m.named_modules():
            modules[name] = module

        tensor_ct = 0

        for name in self.keys():
            if filter_func is not None and not filter_func(name):
                continue
            obj_path, attr = name.rsplit(".", 1)
            module: torch.nn.Module = modules[obj_path]
            entry = self._metadata[name]

            # Swap out self._verify_hash for our function-local
            # verify_hash, since self.get() provides no mechanism
            # propagate this function-local hash to its implicit loads.
            global_verify_hash = self._verify_hash
            try:
                self._verify_hash = verify_hash
                param = self._to_torch_parameter(self.get(name))
            finally:
                self._verify_hash = global_verify_hash

            if entry["type"] is TensorType.PARAM:
                module.register_parameter(attr, param)
            elif entry["type"] is TensorType.BUFFER:
                module.register_buffer(attr, param)
            elif entry["type"] is TensorType.STATE_DICT:
                raise NotImplementedError(
                    "This was serialized using the write_state_dict() method,"
                    " and cannot be loaded using the load_into_module() method."
                    " Use the TensorDeserializer object directly as a"
                    " state_dict mapping instead."
                )
            tensor_ct += 1

        self._file.close()
        return tensor_ct

    def verify_module(
        self, m: torch.nn.Module
    ) -> Tuple[bool, List[Tuple[str, bool]]]:
        """
        Given `m`, a ``torch.nn.Module``, verify that the tensors in this
        `TensorDeserializer` match the tensors in the ``torch.nn.Module``.

        Returns a boolean indicating whether the verification passed, and
        a list of tuples of ``(tensor_name, bool)`` indicating whether the
        verification passed for each tensor.

        Args:
            m: A ground truth module object to compare deserialized tensors
                against. If any deserialized tensors differ from an entry
                with the same name in `m`, this function will report that
                verification failed.

        Returns:
             A 2-tuple ``(passed, results)`` where ``passed`` is a boolean
             reporting if all checks passed, i.e. the overall result
             of the verification, and ``results`` is a list of
             ``(tensor_name, bool)`` tuples listing each tensor that was checked
             and its individual verification result.
             ``results`` can be used to tell which tensor failed verification
             when ``passed`` is False.

        Raises:
            RuntimeError: If this function is called before tensor data and
                hashes have been loaded, for instance when instantiating the
                `TensorDeserializer` with ``lazy_load=True`` and then calling
                this function prior to loading the tensors into a module.
                If ``lazy_load=False``, this error case is impossible.
        """
        modules: typing.OrderedDict[
            str, Union[torch.nn.Module, torch.Tensor]
        ] = OrderedDict()

        results: List[Tuple[str, bool]] = []

        modules.update(m.state_dict())
        # Non-persistent buffers are serialized in tensorizer,
        # but aren't included in a state_dict() in PyTorch.
        modules.update(m.named_buffers())

        for name in self.keys():
            # Check if the module has this tensor.
            if name not in modules:
                results.append((name, False))
                continue
            module: torch.nn.Module = modules[name]
            entry = self._metadata[name]
            if "hashes" not in entry:
                raise RuntimeError(
                    f"No hashes found in metadata for {name}. This is usually"
                    " caused by a TensorDeserializer that was instantiated"
                    " with lazy_load=True, and not loaded into a module before"
                    " calling this."
                )
            numpy_tensor = _NumpyTensor.from_tensor(module)
            try:
                with memoryview(numpy_tensor.data).cast("B") as mv:
                    self._verify_hashes(
                        name,
                        entry["hashes"],
                        entry["raw_headers"],
                        mv,
                    )
                results.append((name, True))
            except HashMismatchError:
                results.append((name, False))

        absent_keys = set(self.keys()).difference(set(modules.keys()))
        for name in absent_keys:
            results.append((name, False))

        return all(result for name, result in results), results


class TensorSerializer:
    """
    Given a file-like object or path, serialize tensors from a torch.nn.Module
    to it.

    See the docs_ for a usage walkthrough.

    .. _docs: https://github.com/coreweave/tensorizer/tree/main#basic-usage

    Args:
        file_obj: A file-like object or path to a file to write to. The path
            can be a S3 URI.
        compress_tensors: If True, compress the tensors using lz4. This
            exists as an internal curiosity as it doesn't seem to make
            much of a difference in practice.

    Example:
        Serializing a 16-bit HuggingFace model to a private S3 bucket::

            from transformers import AutoModelForCausalLM
            from tensorizer import TensorSerializer
            import torch

            model_ref = "EleutherAI/gpt-neo-125M"
            s3_uri = f"s3://some-private-bucket/{model_ref}.tensors"

            model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            serializer = TensorSerializer(s3_uri)
            serializer.write_module(model)
            serializer.close()

        This example assumes the credentials for ``some-private-bucket``
        are configured in `~/.s3cfg`, where they will be auto-detected.
        To specify credentials manually, use a file-like object from
        `tensorizer.stream_io.open_stream()` in place of `s3_uri`.
    ..
    """

    def __init__(
        self,
        file_obj: Union[
            io.BufferedIOBase,
            io.RawIOBase,
            typing.BinaryIO,
            str,
            bytes,
            os.PathLike,
            int,
        ],
        compress_tensors: bool = False,
    ) -> None:
        if isinstance(file_obj, (str, bytes, os.PathLike, int)):
            self._file = stream_io.open_stream(file_obj, "wb+")
        else:
            self._mode_check(file_obj)
            self._file = file_obj

        # Get information about the file object's capabilities
        _fd_getter = getattr(self._file, "fileno", None)
        self._fd = _fd_getter() if callable(_fd_getter) else None
        _seekable_getter = getattr(self._file, "seekable", None)
        self._seekable = (
            _seekable_getter() if callable(_seekable_getter) else True
        )
        if not self._seekable:
            raise ValueError("file_obj must support seeking for serialization")

        # Decide on a pwrite implementation
        if hasattr(os, "pwrite") and self._fd is not None:
            # the os.pwrite syscall can't be used on some file-like objects
            # like io.BytesIO, as they aren't actual operating system constructs
            # It also may not be available, depending on the OS.
            self._pwrite = self._pwrite_syscall
            self._write_lock = None
            concurrent_writes_possible = True
        else:
            # The fallback implementation requires a lock, as a single
            # file offset must be shared between threads.
            self._pwrite = self._pwrite_fallback
            self._write_lock = threading.Lock()
            concurrent_writes_possible = False

        self._idx = 0
        self.total_compressed_tensor_bytes = 0
        self.compress_tensors = compress_tensors

        # This thread pool handles CPU-bound tasks like hashing.
        # Hashing from the Python standard library can benefit from
        # multithreading in spite of the GIL because CPython's hash function
        # implementations release the GIL during longer hash computations.
        self._computation_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count(),
            thread_name_prefix="TensorizerComputation-",
        )

        # There is no use spawning many writer threads when they share a lock.
        max_concurrent_writers = 4 if concurrent_writes_possible else 1

        # This thread pool handles straightforward write tasks, such as
        # tensor data writes.
        self._writer_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_writers,
            thread_name_prefix="TensorizerWriter-",
        )

        self._tensor_count_update_lock = threading.Lock()

        # This thread pool is specifically for writing tensor entry headers.
        # It is separate because these tasks each depend on the completion of
        # one or more hashing tasks from the _computation_pool, and may spend a
        # significant amount of time waiting without even attempting any I/O.
        # If it shared a worker count with the _writer_pool, these tasks could
        # stall other I/O operations that have no prerequisites and are ready,
        # since there is no way to yield its thread slot back to the pool.
        self._header_writer_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_writers,
            thread_name_prefix="TensorizerHeaderWriter-",
        )

        # Implementation detail for CPython: ThreadPoolExecutor objects
        # use an instance of queue.SimpleQueue as a FIFO work queue,
        # so the order that tasks are started (but not necessarily finished)
        # corresponds exactly to the order of ThreadPoolExecutor.submit() calls.
        # This guarantees that, for example, the order that the
        # _header_writer_pool waits on hashes from the _computation_pool
        # always matches the order that the _computation_pool itself begins
        # those hash operations, assuming corresponding tasks are submitted
        # to each pool in the same relative order.

        # Tracks work submitted to all pools to wait for pending work to finish.
        self._jobs: List[concurrent.futures.Future] = []

        if self.compress_tensors:
            import lz4.frame

            self.lz4_frame = lz4.frame
        else:
            self.lz4_frame = None

        # Write our magic bytes.
        self._file.write(TENSORIZER_MAGIC)

        # Write file header metadata
        self._file_header_loc = self._file.tell()
        self._file_header = self._FileHeader(
            version_number=NON_OPAQUE_TENSORIZER_VERSION,
            tensor_size=0,
            tensor_count=0,
        )
        self._file.write(self._file_header.to_bytes())

        # Reserve 256kb for metadata.
        metadata_size = 256 * 1024
        self._file.write(struct.pack("<Q", metadata_size))
        self._metadata_loc = self._file.tell()
        self._file.write(bytes(metadata_size))
        self._metadata_cur = self._metadata_loc
        self._metadata_end = self._metadata_loc + metadata_size

        self._tensor_index: List[TensorEntry] = []

    @dataclasses.dataclass
    class _FileHeader:
        format: ClassVar[struct.Struct] = struct.Struct(
            "<"  # Little-endian
            "I"  # Version number
            "32x"  # File hash (unused)
            "Q"  # Total size of tensor data (nominally, total file size)
            "8x"  # Nominally, total size of tensor data (actually unused)
            "Q"  # Total number of tensors
        )
        version_number: int
        tensor_size: int
        tensor_count: int

        def to_bytes(self):
            return self.format.pack(
                self.version_number,
                self.tensor_size,
                self.tensor_count,
            )

    @property
    def total_tensor_bytes(self):
        return self._file_header.tensor_size

    def _sync_prologue_state(self):
        """
        This is called after the tensor has been written to the file,
        and ensures that the file is in a consistent state.

        This must not be called while any I/O jobs are pending in
        ``self._jobs``, as it is not thread-safe, and the contents of
        ``self._file_header`` are only updated once each tensor writing job
        finishes.
        """
        curr = self._file.tell()

        # Write our zero-length field, that indicates that this is the last
        # tensor. This will be overwritten if another tensor is written.
        self._file.write(struct.pack("<Q", 0))

        # Write our new file header.
        self._file.seek(self._file_header_loc)
        self._file.write(self._file_header.to_bytes())

        # Reset our file pointer to the end of the file,
        # minus the zero-length field.
        self._file.seek(curr)

    def _pwrite(self, data, offset: int, verify: bool = True) -> int:
        """
        Thread-safe file write that leaves the file offset unchanged.

        Args:
            data: The data to write.
            offset: The position in the file at which to write.
            verify: Whether to throw an error if the number of bytes written
                doesn't match the length of `data`.

        Returns:
            The number of bytes written.
        Raises:
            OSError: ``verify=True`` and the number of bytes written
                did not match the length of `data`.
        """
        # The implementation for this function must be chosen during __init__
        # based on the capabilities of the platform and the file object used
        raise RuntimeError("pwrite was called before being initialized")

    def _pwrite_syscall(self, data, offset: int, verify: bool = True) -> int:
        # This implementation of pwrite uses a Unix syscall, and is safe to
        # run even between normal file writes.
        bytes_written = os.pwrite(self._fd, data, offset)
        if verify:
            self._verify_bytes_written(bytes_written, data)
        return bytes_written

    def _pwrite_fallback(self, data, offset: int, verify: bool = True) -> int:
        # This implementation of pwrite uses a lock shared with all writers
        # for the entire file object. It is not safe to run this
        # concurrently with any other code that could modify the file offset
        # except other calls to _pwrite_fallback.
        with self._write_lock:
            old_pos = self._file.tell()
            if old_pos != offset:
                self._file.seek(offset)
            bytes_written = self._file.write(data)
            self._file.seek(old_pos)
        if verify:
            self._verify_bytes_written(bytes_written, data)
        return bytes_written

    @staticmethod
    def _verify_bytes_written(bytes_written: int, data_written):
        # For typed buffers (e.g. arrays) the len() isn't the number of bytes
        expected_bytes_written = getattr(
            data_written, "nbytes", len(data_written)
        )
        if bytes_written != expected_bytes_written:
            raise OSError(
                f"pwrite failed to write correctly: {bytes_written} bytes were"
                f" written when {expected_bytes_written} bytes were requested"
            )

    @staticmethod
    def _mode_check(file_obj: io.IOBase) -> None:
        try:
            read_write = file_obj.writable() and file_obj.readable()
        except AttributeError:
            # If file_obj doesn't implement the full io.IOBase interface
            # and checking is not possible, assume it's fine
            read_write = True
        if isinstance(file_obj, io.TextIOBase) or not read_write:
            mode = getattr(file_obj, "mode", "")
            raise ValueError(
                "TensorSerializer's file_obj must be readable, writable, "
                'and in binary mode (mode="wb+"{})'.format(
                    mode and f', current mode="{mode}"'
                )
            )

    def __del__(self):
        self._shutdown_thread_pools()
        if getattr(self, "_file", None) is not None:
            self._file.close()

    def _shutdown_thread_pools(self):
        for pool in "_computation_pool", "_writer_pool", "_header_writer_pool":
            thread_pool = getattr(self, pool, None)
            if thread_pool is not None:
                for j in self._jobs:
                    j.cancel()
                thread_pool.shutdown(wait=False)

    def _synchronize_pools(self):
        for j in self._jobs:
            j.result(timeout=3600)
        self._jobs.clear()

    def close(self) -> None:
        """
        Finalizes the serialization and closes the file.
        """
        self._sync_prologue_state()

        final_sz = self._file.tell()
        self._file.close()
        self._shutdown_thread_pools()
        logger.info(f"Tensors completed serializing to {final_sz} bytes")
        # if self.compress_tensors:
        #     compression_ratio = (
        #         self.total_tensor_bytes / self.total_compressed_tensor_bytes
        #     )
        #     logger.info(f"Uncomp'd bytes: {self.total_tensor_bytes}")
        #     logger.info(f"Comp'd bytes: {self.total_compressed_tensor_bytes}")
        #     logger.info(f"Ratio: {compression_ratio:.2f}")

    @dataclasses.dataclass(init=False)
    class _TensorHeader:
        # Fields with ClassVar are shared across all instances,
        # other fields are calculated per-instance
        buffer: bytearray
        size: int

        start_segment: ClassVar[struct.Struct] = struct.Struct(
            "<"  # Little-endian
            "Q"  # Tensor header size
            "H"  # Module index.
            "B"  # Whether this is a parameter or a buffer
            "H"  # Parameter/buffer name length
        )

        # Variable length fields, can't be compiled into
        # a static struct definition without calculating sizes first
        variable_length_segment_template: ClassVar[str] = (
            "<"
            "{name_len:d}s"  # Parameter/buffer name UTF-8 bytes
            "B"  # Tensor dtype length
            "{dtype_len:d}s"  # Tensor dtype UTF-8 bytes
            "B"  # Tensor shape length
            "{shape_len:d}I"  # Tensor shape I array
        )
        variable_length_segment: struct.Struct
        variable_length_offset: ClassVar[int] = start_segment.size

        hash_header_segment: ClassVar[struct.Struct] = struct.Struct(
            "<"
            "H"  # Hash section length
            "B"  # Hash count (fixed for a particular tensorizer version)
        )
        hash_header_offset: int
        hash_count: ClassVar[int] = 2

        crc32_hash_segment: ClassVar[struct.Struct] = struct.Struct(
            "<"
            "B"  # CRC32 hash type (HashType enum value)
            "B"  # CRC32 hash length
            "I"  # CRC32 hash value
        )
        crc32_hash_offset: int

        sha256_hash_segment: ClassVar[struct.Struct] = struct.Struct(
            "<"
            "B"  # SHA256 hash type
            "B"  # SHA256 hash length
            "32s"  # SHA256 hash value
        )
        sha256_hash_offset: int

        data_length_segment: ClassVar[struct.Struct] = struct.Struct(
            "<q"  # Signed tensor data length
            # We write this signed so that we can use the signedness as an
            # indicator of possible tensor compression in the future.
        )
        data_length_offset: int

        data_offset: int

        # This isn't part of the tensor header,
        # but it shares much of the same information
        metadata_entry_segment_template: ClassVar[str] = (
            "<"
            "H"  # Name length
            "{name_len:d}s"  # Name
            "B"  # Whether this is a parameter or a buffer
            "B"  # Dtype length
            "{dtype_len:d}s"  # Dtype
            "B"  # Shape length
            "{shape_len:d}I"  # Shape
            "Q"  # Header start (relative to the file)
            "Q"  # Tensor data start (relative to the file)
            "Q"  # Tensor length
        )
        metadata_entry: bytes

        def __init__(
            self,
            module_index: int,
            tensor_type: TensorType,
            name: bytes,
            dtype: bytes,
            shape: Sequence[int],
            data_length: int,
            file_offset: int,
        ):
            # Calculate the variable length segment
            name_len = len(name)
            dtype_len = len(dtype)
            # NB: shape_len is the number of dimensions,
            # not the encoded byte length
            shape_len = len(shape)
            self.variable_length_segment = struct.Struct(
                self.variable_length_segment_template.format(
                    name_len=name_len,
                    dtype_len=dtype_len,
                    shape_len=shape_len,
                )
            )

            # Calculate offsets
            (
                self.variable_length_offset,
                self.hash_header_offset,
                self.crc32_hash_offset,
                self.sha256_hash_offset,
                self.data_length_offset,
                self.data_offset,
            ) = itertools.accumulate(
                (
                    self.start_segment.size,
                    self.variable_length_segment.size,
                    self.hash_header_segment.size,
                    self.crc32_hash_segment.size,
                    self.sha256_hash_segment.size,
                    self.data_length_segment.size,
                )
            )
            self.size = self.data_offset

            self.buffer = bytearray(self.size)
            self.start_segment.pack_into(
                self.buffer,
                0,  # Offset
                self.size,  # Tensor header size
                module_index,  # Module index.
                tensor_type.value,  # Whether this is a parameter or a buffer
                name_len,  # Parameter/buffer name length
            )
            self.variable_length_segment.pack_into(
                self.buffer,
                self.variable_length_offset,
                name,  # Parameter/buffer name UTF-8 bytes
                dtype_len,  # Tensor dtype length
                dtype,  # Tensor dtype UTF-8 bytes
                shape_len,  # Tensor shape length
                *shape,  # Tensor shape I array
            )
            self.hash_segment_size = (
                self.data_length_offset - self.hash_header_offset - 2
            )
            self.hash_header_segment.pack_into(
                self.buffer,
                self.hash_header_offset,
                self.hash_segment_size,  # Hash section length
                self.hash_count,  # Hash count
            )

            # Placeholders
            self.add_crc32(0)
            self.add_sha256(b"")

            self.data_length_segment.pack_into(
                self.buffer, self.data_length_offset, data_length
            )

            metadata_entry_segment: struct.Struct = struct.Struct(
                self.metadata_entry_segment_template.format(
                    name_len=name_len,
                    dtype_len=dtype_len,
                    shape_len=shape_len,
                )
            )

            self.metadata_entry = metadata_entry_segment.pack(
                name_len,  # Name length
                name,  # Name
                tensor_type.value,  # Whether this is a parameter or a buffer
                dtype_len,  # Dtype length
                dtype,  # Dtype
                shape_len,  # Shape length
                *shape,  # Shape
                file_offset,  # Header start (relative to the file)
                # Tensor data start (relative to the file):
                file_offset + self.data_offset,
                data_length,  # Tensor length
            )

        def add_crc32(self, value: int):
            self.crc32_hash_segment.pack_into(
                self.buffer,
                self.crc32_hash_offset,
                HashType.CRC32.value,  # Hash type
                4,  # CRC32 hash length
                value,  # Hash value
            )

        def add_sha256(self, value: bytes):
            self.sha256_hash_segment.pack_into(
                self.buffer,
                self.sha256_hash_offset,
                HashType.SHA256.value,  # Hash type
                32,  # SHA256 hash length
                value,  # Hash value
            )

    def write_tensor(
        self,
        idx,
        name,
        tensor_type: TensorType,
        tensor: Union[torch.Tensor, numpy.ndarray],
    ) -> None:
        """
        Serializes a tensor, laying things out so that it can be read in three
        calls from the input -- once for the size, once for the header, and
        once for the tensor itself.

        Args:
            idx: The index of the tensor in the module.
            name: The name of the tensor.
            tensor_type: The type of the tensor. This is used to determine
                how to interpret the tensor.
            tensor: The tensor to serialize.

        Serialization format:

         { uint64                           header_sz,
           uint16                           module_idx,
           uint8                            type,
           uint16                           name_sz,
           []char                           name,
           uint8                            dtype_sz,
           []char                           dtype_str,
           []{uint8                         shape_elem_type,
              union{uint8, uint16, uint32}  shape_elem_sz,
             }                              shape_elements,
           uint16                           hashes_sz,
           uint8                            num_hashes,
           []{uint8                         hash_type,
              uint8                         hash_sz,
              []char                        hash_str,
             }                              hashes,
           uint64                           tensor_sz,
           []byte                           tensor }
        """
        self._write_tensor(
            idx=idx, name=name, tensor_type=tensor_type, tensor=tensor
        )

    def _write_tensor(
        self,
        idx,
        name,
        tensor_type: TensorType,
        tensor: Union[torch.Tensor, numpy.ndarray],
        *,
        _synchronize: bool = True,
        _start_pos: Optional[int] = None,
    ) -> int:
        """
        Underlying implementation for `write_tensor()`,
        providing additional controls for asynchronous writes

        Args:
            idx: The index of the tensor in the module.
            name: The name of the tensor.
            tensor_type: The type of the tensor. This is used to determine
                how to interpret the tensor.
            tensor: The tensor to serialize.
            _synchronize: Whether to synchronize metadata after writing
                and ensure that all data is written to the file before
                the call returns. If false, data may continue to be written
                asynchronously even after this call returns.
            _start_pos:
                Where in the file to write the tensor entry. If not specified,
                writes starting at the current file offset.
        """
        if isinstance(tensor, torch.Tensor):
            numpy_tensor = _NumpyTensor.from_tensor(tensor)
        else:
            numpy_tensor = _NumpyTensor.from_array(tensor)

        dtype_name = numpy_tensor.numpy_dtype
        if numpy_tensor.is_opaque:
            # The datatype name needs to contain both the numpy dtype that the
            # data is serialized as and the original torch dtype.
            dtype_name += OPAQUE_DTYPE_SEP + numpy_tensor.torch_dtype
            self._file_header.version_number = TENSORIZER_VERSION

        tensor = numpy_tensor.data
        tensor_memory = numpy_tensor.data.data
        tensor_size = tensor.nbytes
        name_bytes = name.encode("utf-8")
        dtype_bytes = dtype_name.encode("utf-8")
        if len(dtype_bytes) >= 256:
            raise ValueError("dtype name length should be less than 256")
        shape = tensor.shape
        header_pos = self._file.tell() if _start_pos is None else _start_pos
        header = self._TensorHeader(
            idx,
            tensor_type,
            name_bytes,
            dtype_bytes,
            shape,
            tensor_size,
            header_pos,
        )

        tensor_pos = header_pos + header.data_offset

        # Add our tensor metadata to the index.
        metadata = header.metadata_entry
        # Check for overflow
        if self._metadata_cur + len(metadata) > self._metadata_end:
            raise RuntimeError("Metadata overflow")

        metadata_pos = self._metadata_cur
        self._metadata_cur += len(metadata)

        # This task is I/O-bound and has no prerequisites,
        # so it goes into the regular writer pool.
        def write_metadata():
            self._pwrite(metadata, metadata_pos)

        self._jobs.append(self._writer_pool.submit(write_metadata))

        # Calculate the hashes.

        # These two tasks are CPU-bound and don't block the GIL,
        # so they go into the computation thread pool.
        def compute_crc32():
            crc32 = zlib.crc32(header.buffer)
            return zlib.crc32(tensor_memory, crc32)

        def compute_sha256():
            sha256 = hashlib.sha256(header.buffer)
            sha256.update(tensor_memory)
            return sha256.digest()

        # This task is I/O-bound and dependent on the previous two tasks,
        # so it goes into the header writer pool.
        def commit_header(
            crc32_future: concurrent.futures.Future,
            sha256_future: concurrent.futures.Future,
        ):
            crc32 = crc32_future.result(3600)
            sha256 = sha256_future.result(3600)
            header.add_crc32(crc32)
            header.add_sha256(sha256)
            self._pwrite(header.buffer, header_pos)

        crc32_task = self._computation_pool.submit(compute_crc32)
        sha256_task = self._computation_pool.submit(compute_sha256)
        commit_header_task = self._header_writer_pool.submit(
            commit_header, crc32_task, sha256_task
        )
        self._jobs.extend((crc32_task, sha256_task, commit_header_task))

        # This task is I/O-bound and has no prerequisites,
        # so it goes into the regular writer pool.
        def write_tensor_data():
            bytes_written = self._pwrite(tensor_memory, tensor_pos)
            with self._tensor_count_update_lock:
                self._file_header.tensor_count += 1
                self._file_header.tensor_size += bytes_written

        self._jobs.append(self._writer_pool.submit(write_tensor_data))
        tensor_endpos = tensor_pos + tensor_size

        # Update our prologue.
        if _synchronize:
            self._synchronize_pools()
            # Move to the end of our serialized tensor to prepare
            # for the next one in the synchronized case.
            self._file.seek(tensor_endpos)
            self._sync_prologue_state()

        ds_size = tensor_endpos - header_pos
        ds_bytes = f"{ds_size:,} bytes"

        typ = {
            TensorType.PARAM: "p",
            TensorType.BUFFER: "b",
            TensorType.STATE_DICT: "sd",
        }[tensor_type]

        # if self.compress_tensors:
        #     comp_report = (
        #         f" - tensor:[raw: {tensor_raw_sz},"
        #         + f" compressed: {tensor_compressed_sz},"
        #         + f" ratio: {compression_ratio:.2f}]"
        #     )
        # else:
        comp_report = ""
        logger.info(
            f"{idx}:{typ}:{name} - {dtype_bytes.decode('utf-8')} - "
            f"{tensor.shape} -> {ds_bytes}{comp_report}"
        )
        return tensor_endpos

    @staticmethod
    def _async_bulk_device_to_host_transfer(
        tensors, max_read_ahead: Optional[int] = 32
    ) -> Tuple[Iterator[torch.Tensor], Callable]:
        """
        Transfers CUDA tensors to host memory asynchronously in bulk.

        Args:
            tensors: The list of tensors to transfer.
            max_read_ahead: The maximum number of tensors to queue.

        Returns:
            A tuple containing an iterator over CPU tensors,
            and a callback to cancel the transfer early.
        """
        if len(tensors) < max_read_ahead:
            transferred = queue.SimpleQueue()
        else:
            transferred = queue.Queue(maxsize=max_read_ahead)

        transfer_finished = False

        def _transfer():
            nonlocal transfer_finished
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                # This is in a separate CUDA stream because it shouldn't
                # affect any other GPU operations, even though each
                # of these transfers are synchronous
                for t in tensors:
                    if transfer_finished:
                        break
                    transferred.put(t.cpu().detach(), timeout=3600)
                else:
                    # Sentinel
                    transferred.put(None)
            transfer_finished = True

        transfer_thread = threading.Thread(target=_transfer, daemon=True)
        transfer_thread.start()

        def _interrupt_transfer():
            nonlocal transfer_finished
            if not transfer_finished:
                # Signal the worker thread to end on its next loop
                transfer_finished = True
                try:
                    # Unstick the worker thread so that
                    # it isn't waiting for an open spot
                    # that will never arrive
                    transferred.get_nowait()
                except queue.Empty:
                    pass

        return (
            iter(lambda: transferred.get(timeout=3600), None),
            _interrupt_transfer,
        )

    def write_module(
        self, m: torch.nn.Module, remove_tensors: bool = False
    ) -> None:
        """
        Serializes an entire ``torch.nn.Module`` instance at once,
        preparing it to be deserialized later with
        ``TensorDeserializer.load_into_module()``.

        This method contains several optimizations that make it
        much faster than serializing a module with several separate
        calls to `write_tensor()`. Thus, whenever possible,
        this method is preferred to serialize tensors in bulk.

        Args:
            m: The module to serialize.
            remove_tensors: Whether to delete each tensor from `m`
                after serializing it.
                Deleted tensors are replaced with ``None``.
        """
        tensors: List[torch.Tensor] = []
        size = 0
        chain = itertools.chain
        repeat = itertools.repeat
        for module_name, module in m.named_modules():
            parameters = module.named_parameters(recurse=False)
            buffers = module.named_buffers(recurse=False)
            for name, tensor in chain(parameters, buffers):
                tensors.append(tensor)
                size += len(module_name) + 1 + len(name)

        next_pos = self._file.tell()

        fallocate = getattr(os, "posix_fallocate", None)
        if fallocate and self._fd:
            size += sum(t.untyped_storage().size() for t in tensors)
            # Rough underestimate of header size
            header_min_size = 24
            size += header_min_size * len(tensors)
            try:
                fallocate(self._fd, next_pos, size)
            except OSError:
                pass

        cuda_tensors = [t for t in tensors if t.device.type == "cuda"]
        if cuda_tensors:
            transferred, interrupt_transfer = (
                self._async_bulk_device_to_host_transfer(cuda_tensors)
            )
        else:
            transferred = interrupt_transfer = None
        tensors.clear()

        try:
            for module_name, module in m.named_modules():
                parameters = module.named_parameters(recurse=False)
                buffers = module.named_buffers(recurse=False)

                for (name, tensor), tensor_type in chain(
                    zip(parameters, repeat(TensorType.PARAM)),
                    zip(buffers, repeat(TensorType.BUFFER)),
                ):
                    label = f"{module_name}.{name}"
                    if tensor.device.type == "cuda":
                        tensor = next(transferred)
                    next_pos = self._write_tensor(
                        self._idx,
                        label,
                        tensor_type,
                        tensor,
                        _synchronize=False,
                        _start_pos=next_pos,
                    )
                    if remove_tensors:
                        setattr(module, name, None)
                self._idx += 1
        except Exception:
            if interrupt_transfer is not None:
                interrupt_transfer()
            raise
        self._synchronize_pools()
        self._file.seek(next_pos)
        self._sync_prologue_state()

    def write_state_dict(self, state_dict: Dict):
        """
        Write the state_dict to the file in Tensorizer format.

        It is strongly recommended that you use write_module instead of
        this function, as it will also write out the parameter type,
        allowing for zero-copy loading of the module with
        TensorDeserializer.load_into_module.
        """
        idx = 0
        for name, param in state_dict.items():
            self.write_tensor(idx, name, TensorType.STATE_DICT, param)

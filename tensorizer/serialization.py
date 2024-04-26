##############################################################################
# serialization.py                                                   Wes Brown
# Fast torch module/model serialization/deserialization     (c) 2023 Coreweave
##############################################################################
import abc
import bisect
import collections
import collections.abc
import concurrent.futures
import contextlib
import dataclasses
import enum
import functools
import hashlib
import heapq
import io
import itertools
import logging
import operator
import os
import pathlib
import queue
import stat
import struct
import threading
import time
import types
import typing
import weakref
import zlib
from collections import OrderedDict
from enum import IntEnum
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy
import psutil
import redis
import torch

import tensorizer._crypt as _crypt
import tensorizer._crypt_info as _crypt_info
import tensorizer._linear_partition as _linear_partition
import tensorizer._syscalls as _syscalls
import tensorizer._tensor_path as _tensor_path
import tensorizer.stream_io as stream_io
import tensorizer.utils as utils
from tensorizer._crypt._cgroup_cpu_count import (
    effective_cpu_count as _effective_cpu_count,
)
from tensorizer._futuregroup import (
    _Future,
    _future_wait_and_raise,
    _FutureGroup,
)
from tensorizer._internal_utils import Chunked as _Chunked
from tensorizer._internal_utils import _variable_read
from tensorizer._NumpyTensor import OPAQUE_DTYPE_SEP, _NumpyTensor
from tensorizer._tensor_path import (
    _TensorPath,
    _TensorPathComponent,
    _TensorPathRegistry,
)

__all__ = [
    "TensorSerializer",
    "TensorDeserializer",
    "TensorType",
    "CryptographyError",
    "EncryptionParams",
    "DecryptionParams",
    "FilterFuncType",
]


@dataclasses.dataclass
class _PerfStats:
    file_readinto_ns: int = 0
    file_readinto_bytes: int = 0
    tensor_to_device_ns: int = 0
    tensor_to_device_bytes: int = 0
    lock: threading.Lock = dataclasses.field(
        init=False,
        repr=False,
        hash=False,
        compare=False,
        default_factory=threading.Lock,
    )


_enable_perf_stats: bool = bool(os.environ.get("TENSORIZER_ENABLE_PERF_STATS"))
_perf_stats = _PerfStats() if _enable_perf_stats else None

lz4 = None

# Setup logger
logger = logging.getLogger(__name__)


# Get CPU count
cpu_count: int = _effective_cpu_count()


class _SupportsBool(typing.Protocol):
    def __bool__(self) -> bool: ...


# Filter functions take either a single string (for a simple top-level dict key)
# or a tensor path (a sequence of path components, each a string or integer).
# Tensor paths are used for anything that is not a top-level string dict key.
FilterFuncType: "typing.TypeAlias" = Callable[
    [Union[str, Sequence[_TensorPathComponent]]], _SupportsBool
]


class CryptographyError(_crypt.CryptographyError):
    pass


def _require_libsodium() -> None:
    if not _crypt.available:
        raise RuntimeError(
            "libsodium shared object library not found or outdated."
            " libsodium is a required dependency when using tensor encryption."
            " Install an up-to-date version using the instructions at"
            " https://doc.libsodium.org/installation or through"
            ' a package manager (e.g. "apt-get install libsodium23")'
        )


def _requires_libsodium(func):
    if _crypt.available:
        return func
    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _require_libsodium()
            return func(*args, **kwargs)

        return wrapper


# Whether the tensor is a parameter or a buffer on the model.
class TensorType(IntEnum):
    PARAM = 0
    BUFFER = 1
    STATE_DICT = 2


# Current version
TENSORIZER_VERSION = 5

HEADERS_AT_TOP_TENSORIZER_VERSION = 5

# To serialize meta tensors into metadata-only tensors
# that deserialize back into zeroed-out buffers, data version 4 is required.
META_TENSOR_TENSORIZER_VERSION = 4

# To (de)serialize tensors with encryption, data version 3 is required.
ENCRYPTION_TENSORIZER_VERSION = 3

# If tensors with "opaque" dtypes (those that are not supported by numpy) are
# saved, then a tensorizer data version of 2 is required to (de)serialize the
# file.
OPAQUE_TENSORIZER_VERSION = 2

# Otherwise, the file is compatible with tensorizer data version 1.
NON_OPAQUE_TENSORIZER_VERSION = 1

TENSORIZER_MAGIC = b"|TZR|"

_TIMEOUT: typing.Final[int] = 3600


class HashType(IntEnum):
    CRC32 = 0
    SHA256 = 1


@dataclasses.dataclass(order=True)
class TensorHash:
    __slots__ = ("type", "hash")
    type: HashType
    hash: bytes


@dataclasses.dataclass(order=True)
class TensorEntry:
    __slots__ = (
        "name",
        "type",
        "dtype",
        "shape",
        "offset",
        "data_offset",
        "data_length",
        "hashes",
        "header_hashes",
    )
    name: _TensorPath
    type: TensorType
    dtype: str
    shape: Tuple[int, ...]
    offset: int
    data_offset: int
    data_length: int
    hashes: Optional[List[TensorHash]]
    header_hashes: Optional[Dict[HashType, Any]]

    @property
    def deserialized_length(self):
        if self.data_length > 0:
            return self.data_length
        element_size: int = numpy.dtype(self.dtype).itemsize
        num_elements: int = int(
            numpy.product(self.shape)
        )  # numpy.product([]) == 1.0
        return element_size * num_elements


class _FileFeatureFlags(enum.IntFlag):
    encrypted = enum.auto()


@dataclasses.dataclass
class _FileHeader:
    __slots__ = (
        "version_number",
        "feature_flags",
        "tensor_size",
        "tensor_count",
    )
    version_number_format: ClassVar[struct.Struct] = struct.Struct(
        "<I"  # Little-endian version number
    )
    format: ClassVar[struct.Struct] = struct.Struct(
        "<"
        "32s"  # File feature flags, in data version 4+, otherwise empty
        "Q"  # Total size of tensor data (nominally, total file size)
        "8x"  # Nominally, total size of tensor data (actually unused)
        "Q"  # Total number of tensors
    )
    version_number: int
    feature_flags: _FileFeatureFlags
    tensor_size: int
    tensor_count: int

    class InvalidVersionError(ValueError):
        version: int

        def __init__(self, *args, version: int):
            super().__init__(*args)
            self.version = version

    def to_bytes(self) -> bytes:
        if self.version_number >= META_TENSOR_TENSORIZER_VERSION:
            feature_flags: bytes = self.feature_flags.to_bytes(
                32, "little", signed=False
            )
        else:
            feature_flags: bytes = bytes(32)
        return self.version_number_format.pack(
            self.version_number
        ) + self.format.pack(feature_flags, self.tensor_size, self.tensor_count)

    @classmethod
    def from_io(
        cls, reader: io.BufferedIOBase, accepted_versions: Sequence[int]
    ) -> "_FileHeader":
        version_number = cls.version_number_format.unpack(
            reader.read(cls.version_number_format.size)
        )[0]
        if version_number not in accepted_versions:
            accepted_versions_str: str = ", ".join(
                map(str, sorted(set(accepted_versions)))
            )
            message = (
                "Unsupported version: this data stream uses tensorizer"
                f" data version {version_number}, which is not supported"
                " in this release of tensorizer, or"
                " for the serialization/deserialization features selected."
                f"\nSupported data versions: {accepted_versions_str}"
            )
            raise cls.InvalidVersionError(message, version=version_number)
        data = reader.read(cls.format.size)
        if len(data) < cls.format.size:
            raise ValueError(
                "File too small: ran out of data before reading a full header"
            )
        feature_flag_bytes, tensor_size, tensor_count = cls.format.unpack(data)
        feature_flag_int = int.from_bytes(
            feature_flag_bytes, "little", signed=False
        )
        feature_flags = _FileFeatureFlags(feature_flag_int)
        if not (0 <= feature_flags <= max(_FileFeatureFlags)):
            raise ValueError(
                f"Unsupported feature flags: {_FileFeatureFlags!r}"
            )
        return cls(version_number, feature_flags, tensor_size, tensor_count)


@dataclasses.dataclass(init=False)
class _TensorHeaderSerializer:
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
    hash_count: int

    crc32_hash_segment: ClassVar[struct.Struct] = struct.Struct(
        "<"
        "B"  # CRC32 hash type (HashType enum value)
        "B"  # CRC32 hash length
        "I"  # CRC32 hash value
    )
    crc32_hash_offset: int
    has_crc32: bool

    sha256_hash_segment: ClassVar[struct.Struct] = struct.Struct(
        "<"
        "B"  # SHA256 hash type
        "B"  # SHA256 hash length
        "32s"  # SHA256 hash value
    )
    sha256_hash_offset: int
    has_sha256: bool

    crypt_info: Optional[_crypt_info.CryptInfo]
    crypt_info_offset: int

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
        file_offset: int,  # location of header in file
        include_crc32: bool = True,
        include_sha256: bool = True,
        crypt_info: Optional[_crypt_info.CryptInfo] = None,
    ):
        self.module_index = module_index
        self.tensor_type = tensor_type
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data_length = data_length
        self.file_offset = file_offset
        self.include_crc32 = include_crc32
        self.include_sha256 = include_sha256

        # Calculate the variable length segment
        self.name_len = len(name)
        self.dtype_len = len(dtype)
        # NB: shape_len is the number of dimensions,
        # not the encoded byte length
        self.shape_len = len(shape)
        self.crypt_info = crypt_info
        if crypt_info is None:
            crypt_info_len = 0
        else:
            crypt_info_len = crypt_info.sized_size
        self.variable_length_segment = struct.Struct(
            self.variable_length_segment_template.format(
                name_len=self.name_len,
                dtype_len=self.dtype_len,
                shape_len=self.shape_len,
            )
        )
        crc32_len = sha256_len = self.hash_count = 0
        self.has_crc32 = include_crc32
        self.has_sha256 = include_sha256
        if include_crc32:
            crc32_len = self.crc32_hash_segment.size
            self.hash_count += 1
        if include_sha256:
            sha256_len = self.sha256_hash_segment.size
            self.hash_count += 1

        # Calculate offsets
        (
            self.variable_length_offset,
            self.hash_header_offset,
            self.crc32_hash_offset,
            self.sha256_hash_offset,
            self.crypt_info_offset,
            self.data_length_offset,
            self.size,
        ) = itertools.accumulate(
            (
                self.start_segment.size,
                self.variable_length_segment.size,
                self.hash_header_segment.size,
                crc32_len,
                sha256_len,
                crypt_info_len,
                self.data_length_segment.size,
            )
        )

    def build(self, tensor_data_offset: int):
        # tensor_data_offset: location of tensor data in file
        self.data_offset = tensor_data_offset

        self.buffer = bytearray(self.size)
        self.start_segment.pack_into(
            self.buffer,
            0,  # Offset
            self.size,  # Tensor header size
            self.module_index,  # Module index.
            self.tensor_type.value,  # Whether this is a parameter or a buffer
            self.name_len,  # Parameter/buffer name length
        )
        self.variable_length_segment.pack_into(
            self.buffer,
            self.variable_length_offset,
            self.name,  # Parameter/buffer name UTF-8 bytes
            self.dtype_len,  # Tensor dtype length
            self.dtype,  # Tensor dtype UTF-8 bytes
            self.shape_len,  # Tensor shape length
            *self.shape,  # Tensor shape I array
        )

        after_hashes = self.crypt_info_offset
        self.hash_segment_size = after_hashes - self.hash_header_offset - 2
        self.hash_header_segment.pack_into(
            self.buffer,
            self.hash_header_offset,
            self.hash_segment_size,  # Hash section length
            self.hash_count,  # Hash count
        )

        # Placeholders
        if self.include_crc32:
            self.add_crc32(0)
        if self.include_sha256:
            self.add_sha256(b"")
        if self.crypt_info is not None:
            self.crypt_info.sized_pack_into(self.buffer, self.crypt_info_offset)

        self.data_length_segment.pack_into(
            self.buffer, self.data_length_offset, self.data_length
        )

        metadata_entry_segment = self.get_metadata_entry_segment()

        self.metadata_entry = metadata_entry_segment.pack(
            self.name_len,  # Name length
            self.name,  # Name
            self.tensor_type.value,  # Whether this is a parameter or a buffer
            self.dtype_len,  # Dtype length
            self.dtype,  # Dtype
            self.shape_len,  # Shape length
            *self.shape,  # Shape
            self.file_offset,  # Header start (relative to the file)
            # Tensor data start (relative to the file):
            self.data_offset,
            self.data_length,  # Tensor length
        )

    def get_metadata_entry_segment(self) -> struct.Struct:
        return struct.Struct(
            self.metadata_entry_segment_template.format(
                name_len=self.name_len,
                dtype_len=self.dtype_len,
                shape_len=self.shape_len,
            )
        )

    def _hashable_segment_views(self):
        # Skip areas where we store hashes and crypt_info
        yield memoryview(self.buffer)[: self.hash_header_offset]
        yield memoryview(self.buffer)[self.data_length_offset :]

    def compute_crc32(self) -> int:
        crc32 = 0
        for view in self._hashable_segment_views():
            with view:
                crc32 = zlib.crc32(view, crc32)
        return crc32

    def compute_sha256(self):
        sha256 = hashlib.sha256()
        for view in self._hashable_segment_views():
            with view:
                sha256.update(view)
        return sha256

    def add_crc32(self, value: int):
        if not self.has_crc32:
            raise ValueError(
                "Cannot add CRC32 to header defined without a CRC32 field"
            )
        self.crc32_hash_segment.pack_into(
            self.buffer,
            self.crc32_hash_offset,
            HashType.CRC32.value,  # Hash type
            4,  # CRC32 hash length
            value,  # Hash value
        )

    def add_sha256(self, value: bytes):
        if not self.has_sha256:
            raise ValueError(
                "Cannot add SHA256 to header defined without a SHA256 field"
            )
        self.sha256_hash_segment.pack_into(
            self.buffer,
            self.sha256_hash_offset,
            HashType.SHA256.value,  # Hash type
            32,  # SHA256 hash length
            value,  # Hash value
        )

    def update_crypt_info(self):
        if self.crypt_info is not None:
            self.crypt_info.sized_pack_into(self.buffer, self.crypt_info_offset)


@dataclasses.dataclass(init=False)
class _TensorHeaderDeserializer:
    buffer: bytearray
    module_idx: int
    tensor_type: TensorType
    name: _TensorPath
    dtype: str
    shape: Tuple[int, ...]
    hashes: List[TensorHash]
    crypt_info: Optional[_crypt_info.CryptInfo]
    data_length: int

    _hashable_segments: Sequence[slice]

    header_len_segment: ClassVar[struct.Struct] = struct.Struct("<Q")
    tensor_info_segment: ClassVar[struct.Struct] = struct.Struct(
        "<HB"  # Module index, tensor type
    )

    read_name = partial(_variable_read, length_fmt="H", data_fmt="s")
    read_dtype = partial(_variable_read, length_fmt="B", data_fmt="s")
    read_shape = partial(_variable_read, length_fmt="B", data_fmt="I")
    read_hash_block = partial(_variable_read, length_fmt="H", data_fmt="s")
    read_crypt_info_block = partial(
        _variable_read, length_fmt="Q", data_fmt="s"
    )

    data_length_segment: ClassVar[struct.Struct] = struct.Struct("<q")

    @classmethod
    def from_io(
        cls,
        reader: io.BufferedIOBase,
        zero_hashes: bool = True,
        check_crypt_info: bool = False,
    ) -> Optional["_TensorHeaderDeserializer"]:
        # We read the entire header into memory rather than reading
        # it piecewise to avoid the overhead of many small reads,
        # especially for network streams.
        header_len_bytes = reader.read(cls.header_len_segment.size)
        offset = cls.header_len_segment.size
        header_len: int = cls.header_len_segment.unpack(header_len_bytes)[0]
        if header_len == 0:
            return None
        buffer = bytearray(header_len)
        buffer[:offset] = header_len_bytes
        with memoryview(buffer) as mv:
            reader.readinto(mv[offset:])
        return cls(
            buffer, zero_hashes=zero_hashes, check_crypt_info=check_crypt_info
        )

    def __init__(
        self,
        buffer: bytearray,
        zero_hashes: bool = True,
        check_crypt_info: bool = False,
    ):
        self.buffer = buffer
        offset = self.header_len_segment.size
        self.module_idx, tensor_type = self.tensor_info_segment.unpack_from(
            buffer, offset
        )
        self.tensor_type = TensorType(tensor_type)
        offset += self.tensor_info_segment.size

        # Read the name.
        name_slice, offset = self.read_name(buffer, offset)
        with name_slice:
            self.name: _TensorPath = _TensorPath.deserialize_(name_slice)
        if self.tensor_type != TensorType.STATE_DICT and not self.name.is_str_:
            raise ValueError(
                "Cannot deserialize structured tensor paths"
                " from non-state-dicts"
            )

        # Read the dtype of the tensor.
        dtype_slice, offset = self.read_dtype(buffer, offset)
        with dtype_slice:
            self.dtype: str = str(dtype_slice, "utf-8")

        # Read the shape.
        self.shape, offset = self.read_shape(buffer, offset)

        # Read our hashes in.
        hash_start = offset
        hashes_slice, offset = self.read_hash_block(buffer, offset)
        with hashes_slice:
            self.hashes = self._decode_hashes(hashes_slice)
            if zero_hashes:
                self._zero_hashes(hashes_slice)

        if check_crypt_info:
            crypt_info_slice, offset = self.read_crypt_info_block(
                buffer, offset
            )
            with crypt_info_slice:
                self.crypt_info = _crypt_info.CryptInfo.unpack_from(
                    crypt_info_slice
                )
        else:
            self.crypt_info = None
            self._hashable_segments = (slice(None, None),)

        # Finally, get the tensor data length.
        data_length_start = offset = len(buffer) - self.data_length_segment.size
        self.data_length = self.data_length_segment.unpack_from(buffer, offset)[
            0
        ]
        self._hashable_segments = (
            slice(None, hash_start),
            slice(data_length_start, None),
        )

    def _hashable_segment_views(self):
        for segment_slice in self._hashable_segments:
            yield memoryview(self.buffer)[segment_slice]

    def compute_crc32(self) -> int:
        crc32 = 0
        for view in self._hashable_segment_views():
            with view:
                crc32 = zlib.crc32(view, crc32)
        return crc32

    def compute_sha256(self):
        sha256 = hashlib.sha256()
        for view in self._hashable_segment_views():
            with view:
                sha256.update(view)
        return sha256

    def compute_hashes(self) -> Dict[HashType, Any]:
        hashes = {}
        for hash_type in self.hashes:
            if hash_type.type in hashes:
                continue
            elif hash_type.type == HashType.CRC32:
                hashes[hash_type.type] = self.compute_crc32()
            elif hash_type.type == HashType.SHA256:
                hashes[hash_type.type] = self.compute_sha256()
        return hashes

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
    def _zero_hashes(b: memoryview) -> None:
        """
        Zero out the encoded hashes in the given bytes.
        This is used to prevent the hashes from being part of hash computation
        of the entire data structure.
        """
        # Read the number of hashes.
        num_hashes = b[0]

        hash_idx = 1
        # Read the hashes.
        for i in range(num_hashes):
            # Read the size of the hash.
            hash_size = b[hash_idx + 1]
            hash_begin = hash_idx + 2
            hash_end = hash_begin + hash_size
            b[hash_begin:hash_end] = bytes(hash_size)
            hash_idx = hash_end


class _MetadataDeserializer(dict):
    _total_len_segment: ClassVar[struct.Struct] = struct.Struct("<Q")
    _read_name = partial(_variable_read, length_fmt="H", data_fmt="s")
    _tensor_type_segment: ClassVar[struct.Struct] = struct.Struct("<B")
    _read_dtype = partial(_variable_read, length_fmt="B", data_fmt="s")
    _read_shape = partial(_variable_read, length_fmt="B", data_fmt="I")
    _location_segment: ClassVar[struct.Struct] = struct.Struct(
        "<"
        "Q"  # Tensor header offset
        "Q"  # Tensor data offset
        "Q"  # Tensor data length
    )

    @classmethod
    def from_io(
        cls, reader: io.BufferedIOBase, count: int
    ) -> Tuple["_MetadataDeserializer", _TensorPathRegistry, bytes]:
        raw = reader.read(cls._total_len_segment.size)
        total_len: int = cls._total_len_segment.unpack(raw)[0]
        if total_len == 0:
            return cls(), _TensorPathRegistry(), raw
        else:
            encoded_metadata: bytes = reader.read(total_len)
            raw += encoded_metadata
            return cls.from_buffer(encoded_metadata, count) + (raw,)

    @classmethod
    def from_buffer(
        cls, buffer: bytes, count: int
    ) -> Tuple["_MetadataDeserializer", _TensorPathRegistry]:
        offset = 0
        entries = cls()
        registry = _TensorPathRegistry()
        for i in range(count):
            entry, offset = cls._read_entry(buffer, offset, registry)
            entries[entry.name] = entry
        return entries, registry

    @classmethod
    def _read_entry(
        cls, buffer: bytes, offset: int, registry: _TensorPathRegistry
    ) -> Tuple[TensorEntry, int]:
        # Read the name.
        name_slice, offset = cls._read_name(buffer, offset)
        with name_slice:
            name: _TensorPath = _TensorPath.deserialize_(name_slice)
            registry.register_path(name)

        tensor_type = TensorType(buffer[offset])
        offset += 1

        # Read the dtype of the tensor.
        dtype_slice, offset = cls._read_dtype(buffer, offset)
        with dtype_slice:
            dtype: str = str(dtype_slice, "utf-8")

        # Read the shape.
        shape, offset = cls._read_shape(buffer, offset)

        (
            header_offset,
            data_offset,
            data_length,
        ) = cls._location_segment.unpack_from(buffer, offset)
        offset += cls._location_segment.size

        return (
            TensorEntry(
                name=name,
                type=tensor_type,
                dtype=dtype,
                shape=shape,
                offset=header_offset,
                data_offset=data_offset,
                data_length=data_length,
                # The following fields are only available in the per-tensor headers
                hashes=None,
                header_hashes=None,
            ),
            offset,
        )


class HashMismatchError(Exception):
    pass


@dataclasses.dataclass(init=False)
class EncryptionParams:
    """
    Defines encryption parameters for a TensorSerializer.

    There are three ways to use this class, mainly using its factory functions:

    #. Using `EncryptionParams.random()`

    This will generate a random encryption key.
    This is the fastest and most secure option, but you must
    save it somewhere to be able to use it for decryption later.

    #. Using `EncryptionParams.from_string()`

    This will generate a reproducible encryption key from an arbitrary string,
    using the Argon2 (Argon2id, RFC 9106) password hashing algorithm.

    The resulting key has resistance against brute-force attacks that attempt
    to guess the input string, achieved by making each attempt
    expensive to compute, both in CPU time and RAM usage.

    The difficulty to compute the key may be adjusted using additional
    parameters to ``from_string()``.
    See `EncryptionParams.from_string()`'s documentation for more details.

    #. Using `EncryptionParams(key=...)` directly

    You can supply an exact key to use for encryption by directly invoking
    the `EncryptionParams` constructor. This must be a `bytes` object of the
    correct length to be used as an XSalsa20 cipher key (32 bytes).

    This allows bringing your own key derivation algorithm or random key source,
    but is more complicated and risky to use than the other options.
    Do not use this with an insecure key.

    Examples:

        Using `EncryptionParams.from_string()` with
        an environment variable::

            source: str = os.getenv("SUPER_SECRET_STRONG_PASSWORD")
            encryption_params = EncryptionParams.from_string(source)

            # Use this to encrypt something:
            serializer = TensorSerializer(
                "model.tensors", encryption=encryption_params
            )
            serializer.write_module(...)
            serializer.close()

            # Then decrypt it again
            decryption_params = DecryptionParams.from_string(source)
            deserializer = TensorDeserializer(
                "model.tensors", encryption=decryption_params
            )
            deserializer.load_into_module(...)
            deserializer.close()


        Using `EncryptionParams.random()`::

            encryption_params = EncryptionParams.random()

            # Use this to encrypt something:
            serializer = TensorSerializer(
                "model.tensors", encryption=encryption_params
            )
            serializer.write_module(...)
            serializer.close()

            # Then decrypt it again
            key: bytes = encryption_params.key
            decryption_params = DecryptionParams.from_key(key)
            deserializer = TensorDeserializer(
                "model.tensors", encryption=decryption_params
            )
            deserializer.load_into_module(...)
            deserializer.close()
    """

    __slots__ = ("key", "_algorithm")

    class _Algorithm(abc.ABC):
        __slots__ = ()

        @abc.abstractmethod
        def chunk(self) -> _crypt_info.KeyDerivationChunk: ...

    @dataclasses.dataclass
    class _FromStringPWHashAlgorithm(_Algorithm):
        __slots__ = ("pwhash_params",)
        pwhash_params: "_crypt.PWHash"

        def chunk(self) -> _crypt_info.KeyDerivationChunk:
            return _crypt_info.PWHashKeyDerivationChunk(
                opslimit=self.pwhash_params.opslimit,
                memlimit=self.pwhash_params.memlimit,
                alg=self.pwhash_params.alg,
                salt=self.pwhash_params.salt,
            )

    key: bytes
    _algorithm: Optional[_Algorithm]

    @_requires_libsodium
    def __init__(self, key: bytes):
        if not isinstance(key, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Encryption key must be a cryptographically secure bytestring."
                " To derive an encryption key from an arbitrary string,"
                " use EncryptionParams.from_string()"
            )

        if len(key) != _crypt.ChunkedEncryption.KEY_BYTES:
            raise ValueError(
                "Invalid encryption key length,"
                f" should be {_crypt.ChunkedEncryption.KEY_BYTES} bytes;"
                f" got {len(key)} bytes instead."
                " To generate a valid encryption key from any string"
                " or bytes object,"
                " use EncryptionParams.from_string()"
            )
        self.key = bytes(key)
        self._algorithm = None

    @classmethod
    @_requires_libsodium
    def random(cls) -> "EncryptionParams":
        """
        Generates a random encryption key with no associated source string.

        This is the fastest and most secure option, but you must
        save the resulting key (from ``EncryptionParams.key``)
        somewhere to be able to use it for decryption later.

        Returns:
            An `EncryptionParams` instance to pass to a `TensorSerializer`.
        """
        return cls(_crypt.random_bytes(_crypt.ChunkedEncryption.KEY_BYTES))

    @staticmethod
    @_requires_libsodium
    def _derive_salt(
        salt: Union[str, bytes, bytearray, memoryview, None],
        encoding,
        fallback_size: int = 32,
    ) -> bytes:
        if salt is None:
            return _crypt.random_bytes(fallback_size)
        elif isinstance(salt, (bytes, bytearray, memoryview)):
            return salt
        elif isinstance(salt, str):
            return salt.encode(encoding)
        else:
            raise TypeError("Invalid object type provided for salt")

    @_requires_libsodium
    def _crypt_info_chunk(self) -> Optional[_crypt_info.CryptInfoChunk]:
        if self._algorithm is None:
            return None
        else:
            return self._algorithm.chunk()

    if _crypt.available:

        class OpsLimit(IntEnum):
            """
            For discussion on this parameter, see the libsodium documentation:
            https://libsodium.gitbook.io/doc/password_hashing/default_phf#key-derivation

            Summary quote:
                opslimit represents the maximum amount of computations
                to perform. Raising this number will make the function
                require more CPU cycles to compute a key.

            The preset levels are related to each other as follows:
            `MIN` < `INTERACTIVE` < `MODERATE` < `SENSITIVE`
            (`MIN` is easiest to compute, `SENSITIVE` is hardest).
            """

            MIN = _crypt.PWHash.OPSLIMIT_MIN
            INTERACTIVE = _crypt.PWHash.OPSLIMIT_INTERACTIVE
            MODERATE = _crypt.PWHash.OPSLIMIT_MODERATE
            SENSITIVE = _crypt.PWHash.OPSLIMIT_SENSITIVE

        class MemLimit(IntEnum):
            """
            For discussion on this parameter, see the libsodium documentation:
            https://libsodium.gitbook.io/doc/password_hashing/default_phf#key-derivation

            Summary quote:
                memlimit is the maximum amount of RAM in bytes
                that the function will use.

            The preset levels are related to each other as follows:
            `MIN` < `INTERACTIVE` < `MODERATE` < `SENSITIVE`
            (`MIN` is easiest to compute, `SENSITIVE` is hardest).
            """

            MIN = _crypt.PWHash.MEMLIMIT_MIN
            INTERACTIVE = _crypt.PWHash.MEMLIMIT_INTERACTIVE
            MODERATE = _crypt.PWHash.MEMLIMIT_MODERATE
            SENSITIVE = _crypt.PWHash.MEMLIMIT_SENSITIVE

    else:

        class OpsLimit(IntEnum):
            def __getattribute__(self, item):
                super().__getattribute__(self, item)
                _require_libsodium()

        class MemLimit(IntEnum):
            def __getattribute__(self, item):
                super().__getattribute__(self, item)
                _require_libsodium()

    @property
    def salt(self) -> bytes:
        """
        Cryptographic salt used for key derivation.
        This is stored within a serialized model, and will be retrieved
        automatically during deserialization, so it does not normally
        need to be manually saved.

        However, manually saving this value will allow recalculating an exact
        binary key separately from the deserialization process.

        Returns:
            The cryptographic salt used for key derivation.
        Raises:
            ValueError: If no salt is being used for key derivation.
        """
        if isinstance(self._algorithm, self._FromStringPWHashAlgorithm):
            return bytes(self._algorithm.pwhash_params.salt)
        elif self._algorithm is None:
            raise ValueError(
                "An exact binary key is being used."
                " Exact binary keys do not use key derivation algorithms,"
                " and thus have no salt."
            )
        else:
            raise ValueError(
                "The key derivation algorithm in use does not use a salt."
            )

    @classmethod
    @_requires_libsodium
    def from_string(
        cls,
        source: Union[str, bytes],
        opslimit: Union[OpsLimit, int, None] = None,
        memlimit: Union[MemLimit, int, None] = None,
        salt: Union[str, bytes, None] = None,
        *,
        encoding="utf-8",
    ) -> "EncryptionParams":
        """
        Generates an encryption key from any string.

        This method uses the Argon2 (Argon2id, RFC 9106) password hashing
        algorithm to create a key from an input string.

        The key has resistance against brute-force attacks that attempt
        to guess the input string, achieved by making each attempt
        expensive to compute, both in CPU time and RAM usage.

        The computational difficulty can be increased or decreased
        via the `opslimit` and `memlimit` parameters.
        Higher computational difficulty gives more security
        for weak input strings, but may impact performance.
        The default setting is a "moderate" profile taken from ``libsodium``.

        Presets (as well as minimum values) are available through the
        `EncryptionParams.OpsLimit` and `EncryptionParams.MemLimit` enums.

        Rough estimates of performance impact (on a 3.20 GHz processor)::

            from tensorizer import EncryptionParams

            OpsLimit = EncryptionParams.OpsLimit
            MemLimit = EncryptionParams.MemLimit
            s = "X" * 40

            EncryptionParams.from_string(  # Takes about 0.05 ms, 8 KiB RAM
                s, opslimit=OpsLimit.MIN, memlimit=MemLimit.MIN
            )
            EncryptionParams.from_string(  # Takes about 90 ms, 64 MiB RAM
                s, opslimit=OpsLimit.INTERACTIVE, memlimit=MemLimit.INTERACTIVE
            )
            EncryptionParams.from_string(  # Takes about 500 ms, 256 MiB RAM
                s, opslimit=OpsLimit.MODERATE, memlimit=MemLimit.MODERATE
                # Default: equivalent to opslimit=None, memlimit=None
            )
            EncryptionParams.from_string(  # Takes about 3.0 seconds, 1 GiB RAM
                s, opslimit=OpsLimit.SENSITIVE, memlimit=MemLimit.SENSITIVE
            )

        Performance Tuning:
            If possible, use `EncryptionParams.random()` instead of this method,
            and save the generated key to use for decryption.

            If that is not possible, save the binary key generated during
            `EncryptionParams.from_string()` (from the ``.key`` attribute),
            and use that key for decryption (via `DecryptionParams.from_key()`)
            to remove the cost of re-computing the key at deserialization time.

            If that is not possible, use a strong input string.
            For input strings that are already very strong and high-entropy,
            where brute-force attacks on the input string are no more likely
            to succeed than brute-force attacks on a 256-bit key itself,
            (e.g. very long, randomly generated strings),
            `opslimit` and `memlimit` may be tuned down to minimize
            their performance impact.

            If that is not possible, test different values of `opslimit`
            and `memlimit` to determine an acceptable tradeoff between
            performance and security for your use case.

        See Also:
            ``libsodium`` documentation for ``pwhash``,
            the Argon2id implementation used in ``from_string()``:
            https://libsodium.gitbook.io/doc/password_hashing/default_phf#key-derivation

        See Also:
            RFC 9106 for details on Argon2,
            https://datatracker.ietf.org/doc/html/rfc9106

        Args:
            source: The source string from which to derive a key.
            opslimit:
                Difficulty of the key derivation algorithm on CPU resources.
                For details, refer to the `libsodium documentation`_.
                Can be provided as a preset ``EncryptionParams.OpsLimit`` value,
                or a custom integer. If None (the default),
                uses ``EncryptionParams.OpsLimit.MODERATE``.
            memlimit:
                Amount of RAM required by the key derivation algorithm.
                For details, refer to the `libsodium documentation`_.
                Can be provided as a preset ``EncryptionParams.MemLimit`` value,
                or a custom integer. If None (the default),
                uses ``EncryptionParams.MemLimit.MODERATE``.
            salt: A non-secret cryptographic salt to be stored
                in the serialized file.
                If None (the default), a secure random salt is used.
                This normally does not need to be chosen manually, however,
                this can allow reproducing an exact binary key separately
                from deserialization.
            encoding: The encoding to use to convert `source` to ``bytes``
                if provided as a ``str``. Defaults to UTF-8.

        Returns:
            An `EncryptionParams` instance to pass to a `TensorSerializer`.

        .. _libsodium documentation: https://libsodium.gitbook.io/doc/password_hashing/default_phf#key-derivation
        """
        if not source:
            raise ValueError("Source cannot be empty")
        if isinstance(opslimit, cls.MemLimit) or not isinstance(
            opslimit, (cls.OpsLimit, int, type(None))
        ):
            raise TypeError(
                "opslimit parameter: expected EncryptionParams.OpsLimit,"
                f" int, or None; {opslimit.__class__.__name__} found"
            )
        if isinstance(memlimit, cls.OpsLimit) or not isinstance(
            memlimit, (cls.MemLimit, int, type(None))
        ):
            raise TypeError(
                "memlimit parameter: expected EncryptionParams.MemLimit,"
                f" int, or None; {memlimit.__class__.__name__} found"
            )
        if isinstance(source, str):
            source = source.encode(encoding)
        if opslimit is None:
            opslimit = cls.OpsLimit.MODERATE
        if memlimit is None:
            memlimit = cls.MemLimit.MODERATE
        pwhash_params = _crypt.PWHash(
            salt=salt, opslimit=opslimit, memlimit=memlimit
        )
        encryption_params = cls(key=pwhash_params.hash(source))
        encryption_params._algorithm = cls._FromStringPWHashAlgorithm(
            pwhash_params=pwhash_params
        )
        return encryption_params


@dataclasses.dataclass(init=False)
class DecryptionParams:
    """
    Defines decryption parameters for a TensorDeserializer.

    There are two ways to use this class, using its factory functions:

    #. Using `DecryptionParams.from_string()`

    This will decrypt tensors using the specified key string.
    This may be used if `EncryptionParams.from_string()`
    was used during encryption.

    #. Using `DecryptionParams.from_key()`

    This will decrypt tensors using an exact binary key.
    This may always be used with the ``key`` from an `EncryptionParams` object,
    regardless of whether the key was generated with
    `EncryptionParams.from_string()` or `EncryptionParams.random()`.

    Examples:

        Using `DecryptionParams.from_string()` with
        an environment variable::

            source: str = os.getenv("SUPER_SECRET_STRONG_PASSWORD")
            encryption_params = EncryptionParams.from_string(source)

            # Use this to encrypt something:
            serializer = TensorSerializer(
                "model.tensors", encryption=encryption_params
            )
            serializer.write_module(...)
            serializer.close()

            # Then decrypt it again
            decryption_params = DecryptionParams.from_string(source)
            deserializer = TensorDeserializer(
                "model.tensors", encryption=decryption_params
            )
            deserializer.load_into_module(...)
            deserializer.close()


        Using `DecryptionParams.from_key()`::

            encryption_params = EncryptionParams.random()

            # Use this to encrypt something:
            serializer = TensorSerializer(
                "model.tensors", encryption=encryption_params
            )
            serializer.write_module(...)
            serializer.close()

            # Then decrypt it again
            key: bytes = encryption_params.key
            decryption_params = DecryptionParams.from_key(key)
            deserializer = TensorDeserializer(
                "model.tensors", encryption=decryption_params
            )
            deserializer.load_into_module(...)
            deserializer.close()
    """

    key: Optional[bytes]
    source: Optional[bytes]

    def __init__(self):
        self.key = None
        self.source = None

    @classmethod
    @_requires_libsodium
    def from_string(
        cls, source: Union[str, bytes], *, encoding="utf-8"
    ) -> "DecryptionParams":
        """
        Reverses the process of `EncryptionParams.from_string()`.
        Encryption algorithm parameters such as ``opslimit``, ``memlimit``,
        and ``salt`` are automatically inferred from the file during decryption.

        Tensors encrypted with `EncryptionParams.random()` or a custom binary
        key cannot be decrypted using this method.
        Use `DecryptionParams.from_key() instead.

        Using `DecryptionParams.from_key()` is always faster than this method.
        Use it whenever possible (when you already know the exact binary key).

        Args:
            source: Source string to use for decryption.
            encoding: The encoding to use to convert `source` to ``bytes``
                if provided as a ``str``. Defaults to UTF-8.

        Returns:

        """
        if not source:
            raise ValueError("Source cannot be empty")
        if isinstance(source, str):
            source = source.encode(encoding)
        elif not isinstance(source, bytes):
            raise TypeError("Invalid source type: must be str or bytes")
        params = cls()
        params.source = source
        return params

    @classmethod
    @_requires_libsodium
    def from_key(cls, key: bytes) -> "DecryptionParams":
        if not key:
            raise ValueError("Key cannot be empty")
        elif len(key) != _crypt.ChunkedEncryption.KEY_BYTES:
            raise ValueError(
                "Invalid decryption key length,"
                f" should be {_crypt.ChunkedEncryption.KEY_BYTES} bytes;"
                f" got {len(key)} bytes instead."
                " DecryptionParams.from_key() should be used with a key"
                " read from EncryptionParams.key."
                " To decrypt with an arbitrary string instead of a binary key,"
                " use DecryptionParams.from_string() instead."
            )
        params = cls()
        params.key = key
        return params


class _KeyDerivation:
    __slots__ = ("source", "_cache")
    source: Union[str, bytes]
    _cache: Dict[Any, bytes]

    def __init__(self, source: Union[str, bytes]):
        self.source = source
        self._cache = {}

    def _derive_key(self, method: _crypt_info.KeyDerivationChunk) -> bytes:
        if isinstance(method, _crypt_info.PWHashKeyDerivationChunk):
            if method.alg != _crypt.PWHash.ALG_ARGON2ID13:
                raise CryptographyError(
                    f"Unsupported pwhash algorithm ({method.alg})"
                )
            return EncryptionParams.from_string(
                source=self.source,
                opslimit=method.opslimit,
                memlimit=method.memlimit,
                salt=method.salt,
            ).key
        else:
            raise CryptographyError("Unknown key derivation method")

    @staticmethod
    def _hash_key(chunk: _crypt_info.KeyDerivationChunk) -> tuple:
        if isinstance(chunk, _crypt_info.PWHashKeyDerivationChunk):
            return (
                chunk.derivation_method,
                chunk.opslimit,
                chunk.memlimit,
                chunk.alg,
                bytes(chunk.salt),
            )
        else:
            raise CryptographyError("Unknown key derivation method")

    def key(self, method: _crypt_info.KeyDerivationChunk) -> bytes:
        hash_key = self._hash_key(method)
        cached = self._cache.get(hash_key, None)
        if cached is not None:
            return cached
        else:
            key: bytes = self._derive_key(method)
            self._cache[hash_key] = key
            return key

    def clear_cache(self) -> None:
        self._cache.clear()


class TensorDeserializer(
    collections.abc.Mapping, contextlib.AbstractContextManager
):
    """
    Given a file-like object for read, deserialize tensors to a state_dict or
    a torch.nn.Module.

    See the docs_ for a usage walkthrough.

    .. _docs: https://github.com/coreweave/tensorizer/tree/main#basic-usage

    Args:
        file_obj: A file-like object to read from. It can also be a string
            representing a path to a file or an HTTP/HTTPS/S3 URI.
        device: The device to load the tensors to.
        filter_func: A function ``(tensor_name: str) -> bool`` that returns True
            if a tensor should be loaded, or False if it should be skipped.
            If None, all tensors are loaded.
            For objects that were serialized from lists or nested mappings,
            the tensor name will be given as a ``Sequence`` of path components
            (each ``str`` or ``int``) for anything that is not a plain top-level
            ``str`` key in the original serialized object. In this case,
            the filter function should be compatible with the signature
            ``(tensor_path: str | Sequence[str | int] -> bool)`` instead.
        dtype: The dtype to cast the tensors as when loading them into a torch
            module. If None, the dtype will be inferred from the file.
        lazy_load: If True, tensors will be loaded and cached when keys are
            accessed. If False, all tensors will be loaded into memory up
            front.
        plaid_mode: left for backwards compatibility; has no effect
        plaid_mode_buffers: left for backwards compatibility; has no effect
        num_readers: Number of threads from which to read the file_obj.
            The default (None) uses a dynamic number of threads.
            Set this value to 1 to disable concurrent reading.
        verify_hash: If True, the hashes of each tensor will be verified
            against the hashes stored in the metadata. A `HashMismatchError`
            will be raised if any of the hashes do not match.
        encryption: A `DecryptionParams` object holding a password or key
            to use for decryption. ``None`` (the default) means no decryption.

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

            deserializer = TensorDeserializer(s3_uri)
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
            deserializer = TensorDeserializer(s3)
            deserializer.load_into_module(model)

        .. _pre-serialized: https://github.com/coreweave/tensorizer/tree/main#available-pre-tensorized-models-on-the-coreweave-cloud
    """

    @dataclasses.dataclass
    class _CopiedData:
        __slots__ = ("header", "numpy_tensor", "parameter")
        header: _TensorHeaderDeserializer
        numpy_tensor: _NumpyTensor
        parameter: torch.nn.Parameter

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
        filter_func: Optional[FilterFuncType] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        lazy_load: bool = False,
        plaid_mode: Optional[bool] = None,  # pylint: disable=unused-argument
        plaid_mode_buffers: Optional[
            int
        ] = None,  # pylint: disable=unused-argument
        num_readers: Optional[int] = None,
        verify_hash: bool = False,
        encryption: Optional[DecryptionParams] = None,
    ):
        # Whether to verify the hashes of the tensors when they are loaded.
        # This value is used when no verify_hash argument is passed to the
        # tensor loading methods.
        self._cleanup = contextlib.ExitStack()
        # If __init__ fails, there is no way to invoke cleanup functions
        # other than __del__, so instead, enter the cleanup context
        # pre-emptively and cancel it if __init__ is successful
        with self._cleanup:
            # If device is None, use the current device, otherwise use the given
            # device.
            device = (
                utils.get_device() if device is None else torch.device(device)
            )
            self._device: torch.device = device
            is_cuda: bool = self._device.type == "cuda"
            if is_cuda and not torch.cuda.is_available():
                raise RuntimeError(
                    "Cannot deserialize to CUDA device"
                    " because CUDA is not available"
                )
            if is_cuda:
                self._preload_cuda()

            self._verify_hash = verify_hash
            if encryption is not None and not isinstance(
                encryption, DecryptionParams
            ):
                raise TypeError(
                    "encryption parameter: expected DecryptionParams instance"
                    f" or None, {encryption.__class__.__name__} found"
                )
            self._encryption = encryption
            self._encrypted = encryption is not None
            self._key_derivation = None
            if self._encrypted:
                _require_libsodium()
                self._decryption_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=cpu_count,
                    thread_name_prefix="TensorizerDecryption",
                )
                if self._encryption.key is None:
                    if self._encryption.source is None:
                        raise ValueError(
                            "Invalid DecryptionParams,"
                            " no key or source string provided"
                        )
                    else:
                        self._key_derivation = _KeyDerivation(
                            source=self._encryption.source
                        )
                        self._cleanup.callback(self._key_derivation.clear_cache)
            else:
                self._decryption_pool = None

            self._file_spec = file_obj
            if isinstance(self._file_spec, (str, bytes, os.PathLike, int)):
                self._file = stream_io.open_stream(self._file_spec, "rb")
            else:
                self._mode_check(self._file_spec)
                self._file = self._file_spec
            self._cleanup.callback(self._file.close)
            self.total_compressed_tensor_bytes = 0
            self.read_bytes = 0
            self._ephemeral_bytes_read: int = 0
            self._ephemeral_bytes_read_lock = threading.Lock()
            self._last_yielded_key: Optional[str] = None

            self._dtype: Optional[torch.dtype] = dtype

            self._lazy_load: bool = lazy_load

            self._metadata: Dict[_TensorPath, TensorEntry] = {}

            # Read the magic
            magic = self._file.read(5)
            if magic != TENSORIZER_MAGIC:
                raise ValueError("Not a tensorizer file")

            # Read the file header and check for a compatible data version
            accepted_versions = (
                NON_OPAQUE_TENSORIZER_VERSION,
                OPAQUE_TENSORIZER_VERSION,
                ENCRYPTION_TENSORIZER_VERSION,
                META_TENSOR_TENSORIZER_VERSION,
                TENSORIZER_VERSION,
            )
            encryption_ver: int = ENCRYPTION_TENSORIZER_VERSION
            if self._encrypted:
                accepted_versions = tuple(
                    filter(lambda v: v >= encryption_ver, accepted_versions)
                )
            try:
                self._file_header = _FileHeader.from_io(
                    self._file, accepted_versions=accepted_versions
                )
            except _FileHeader.InvalidVersionError as e:
                if self._encrypted and e.version in (
                    NON_OPAQUE_TENSORIZER_VERSION,
                    OPAQUE_TENSORIZER_VERSION,
                ):
                    raise CryptographyError(
                        "Tensor decryption was requested,"
                        " but the file provided comes from a tensorizer version"
                        " predating encryption, so it must not be encrypted."
                        " Either set encryption=None on the TensorDeserializer,"
                        " or ensure that the correct file was provided."
                    ) from e
                else:
                    raise

            version_number: int = self._file_header.version_number
            self._has_crypt_info: bool = (
                version_number == encryption_ver
                or version_number >= META_TENSOR_TENSORIZER_VERSION
                and _FileFeatureFlags.encrypted in self._file_flags
            )
            if self._encrypted and not self._has_crypt_info:
                raise CryptographyError(
                    "Tensor decryption was requested,"
                    " but the file provided is not flagged as encrypted."
                    " Either set encryption=None on the TensorDeserializer,"
                    " or ensure that the correct file was provided."
                )
            elif self._has_crypt_info and not self._encrypted:
                raise CryptographyError(
                    "Tensor is encrypted, but decryption was not requested"
                )

            # Read the metadata index of tensors.
            # This is a list of offsets into the file where the per-tensor data
            # is stored.
            self._metadata: Dict[_TensorPath, TensorEntry]
            self._metadata, structure, self._metadata_raw = (
                _MetadataDeserializer.from_io(
                    self._file, self._file_header.tensor_count
                )
            )
            if not self._metadata:
                raise ValueError("Tensor index in the file is empty")

            self._headers: Optional[
                Dict[_TensorPath, _TensorHeaderDeserializer]
            ]
            if version_number >= HEADERS_AT_TOP_TENSORIZER_VERSION:
                metadata_ordered = sorted(
                    self._metadata.values(), key=operator.attrgetter("offset")
                )
                self._headers = {}
                for entry in metadata_ordered:
                    if self._file.tell() > entry.offset:
                        raise ValueError("Header offsets overlap or are wrong")
                    self._file.seek(entry.offset)
                    header = _TensorHeaderDeserializer.from_io(
                        self._file,
                        zero_hashes=True,
                        check_crypt_info=self._has_crypt_info,
                    )
                    if header is None:
                        raise KeyError("Unexpected empty header")
                    self._headers[entry.name] = header
            else:
                self._headers = None

            # filter_func is a test that determines the tensor names to read.
            # If filter_func is None, all tensors are read.
            if filter_func is not None:
                self._metadata = {
                    name: entry
                    for name, entry in self._metadata.items()
                    if filter_func(name.normalize_())
                }
                # Remove keys from structure that aren't in self._metadata
                structure.filter(self._metadata.__contains__)
            self._structure: Dict[Union[str, int], Any] = structure.dict()

            dynamic_num_readers: bool = num_readers is None
            if not dynamic_num_readers and not isinstance(num_readers, int):
                raise TypeError(
                    "num_readers: expected int or None,"
                    f" got {num_readers.__class__.__name__}"
                )

            if dynamic_num_readers:
                num_readers = self._choose_dynamic_num_readers(
                    self._metadata.values()
                )

            if num_readers < 1:
                raise ValueError("num_readers must be positive")
            elif num_readers > len(self._metadata):
                num_readers = len(self._metadata)

            if num_readers > 1:
                self._reopen = self._reopen_func()
                if self._reopen is None:
                    if dynamic_num_readers:
                        num_readers = 1
                    else:
                        raise ValueError(
                            "Cannot reopen this type of file to enable parallel"
                            " reading with num_readers > 1. File paths, URIs,"
                            " and HTTP(S) or S3 streams returned from"
                            " tensorizer.stream_io.open_stream, plus some"
                            " open files are capable of being reopened."
                            " Other file-like objects and special files"
                            " (e.g. BytesIO, pipes, sockets) are not supported."
                        )
            else:
                self._reopen = None

            response_headers = getattr(self._file, "response_headers", None)
            if (
                num_readers > 1
                and response_headers is not None
                and response_headers.get("accept-ranges") != "bytes"
            ):
                # The server does not indicate support for range requests
                if dynamic_num_readers:
                    num_readers = 1
                else:
                    raise RuntimeError(
                        "The server streaming the file to deserialize does not"
                        " support HTTP range requests, necessary for"
                        " parallel reading with num_readers > 1."
                        " Set num_readers = 1,"
                        " or use a different HTTP(S) endpoint."
                    )
            self._etag = (
                None
                if response_headers is None
                else response_headers.get("etag") or None
            )

            self._num_readers = num_readers
            self._reader_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_readers,
                thread_name_prefix="TensorDeserializerCopy",
            )
            self._cleanup.callback(self._reader_pool.shutdown, wait=False)

            self.total_tensor_bytes = sum(
                entry.deserialized_length for entry in self._metadata.values()
            )
            num_tensors = len(self._metadata)

            if logger.isEnabledFor(logging.DEBUG):
                # Skip creating this string unless debug logging is enabled
                logger.debug(
                    f"Deserializing {self.total_tensor_bytes} bytes"
                    f" from {num_tensors}"
                    f" tensor{'s' * (num_tensors != 1)} using"
                    f" lazy_load={self._lazy_load},"
                    f" verify_hash={self._verify_hash},"
                    f" encrypted={bool(self._encrypted)},"
                    f" device={self._device},"
                    f" dtype={self._dtype},"
                    f" data_version={self._file_header.version_number}"
                )

            self._keys_enumerated: Dict[str, int] = {
                k: i for i, k in enumerate(self._metadata.keys())
            }

            # The number of bytes we've allocated so far. Tensors may be read
            # from the file in any order, so we need to keep track of how much
            # we've used so far so that we can index into the buffer correctly.
            self._allocated = 0

            # Our cache of tensors. This is a dict of name -> tensor.
            # If lazy_load is True, then the tensors are not loaded until they
            # are accessed.
            self._cache: typing.OrderedDict[
                str, Optional[TensorDeserializer._CopiedData]
            ]

            # The offset in the file where the tensor data begins.
            self._tensors_begin = self._file.tell()

            if not self._lazy_load:
                # If we're not in lazy_load mode, we populate the cache with all
                # the tensors.
                self._generate_state_dict()
            else:
                # We populate the cache with None values so that we can
                # differentiate between tensors that have not been loaded yet
                # and tensors that are not present in the file.
                self._cache = OrderedDict.fromkeys(self._metadata.keys())

            # Once __init__ finishes successfully,
            # cancel the currently entered cleanup context — except when eagerly
            # loading to the GPU, because after eager loading finishes and data
            # has been moved from the CPU to the GPU, there is no need to keep
            # around the CPU buffers anymore, and we can clean up early.
            if self._lazy_load or not is_cuda:
                self._cleanup = self._cleanup.pop_all()

    @property
    def _file_flags(self) -> _FileFeatureFlags:
        return self._file_header.feature_flags

    @_file_flags.setter
    def _file_flags(self, val: _FileFeatureFlags):
        self._file_header.feature_flags = val

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
                "TensorDeserializer's file_obj must be readable "
                'and in binary mode (mode="rb"{})'.format(
                    mode and f', current mode="{mode}"'
                )
            )

    def _reopen_func(self) -> Optional[Callable]:
        spec = self._file_spec
        if isinstance(self._file, stream_io.CURLStreamFile):
            return self._file._fork
        elif isinstance(spec, (str, bytes, os.PathLike)):
            return partial(stream_io.open_stream, spec, mode="rb")
        # Other types of files can be reopened under certain conditions
        fd: Optional[int] = None
        if hasattr(self._file, "fileno"):
            # This should cover any case where _file_spec is an int, as well
            fd = self._file.fileno()
            if not isinstance(fd, int) or fd < 0:
                fd = None
        if fd is not None:
            # If it is a regular file, we can try to re-open it
            true_stat = os.stat(fd)
            if not stat.S_ISREG(true_stat.st_mode):
                return None
            # Check for a symlink at /proc/self/fd/N
            # Could also try the path at self._file.name here
            for path in map(
                pathlib.Path, (f"/proc/self/fd/{fd:d}", f"/dev/fd/{fd:d}")
            ):
                try:
                    maybe_stat: os.stat_result = path.stat()
                    break
                except OSError:
                    continue
            else:
                return None
            # Make sure it refers to the correct file
            if not os.path.samestat(true_stat, maybe_stat):
                return None
            # Double check that this one is still a regular file
            if not stat.S_ISREG(maybe_stat.st_mode):
                return None

            def reopen_file(begin, end):
                if self._file.closed:
                    # If the file is closed, its file descriptor is
                    # almost certainly gone, so don't try to open its
                    # fd-based symlink.
                    raise OSError(
                        "Could not reopen file: original file was closed"
                    )
                reopened = open(path, mode="rb")
                # Since file descriptor integers can be reassigned,
                # or in case of any other shenanigans, double check
                # that this is still the correct file after the open() call.
                try:
                    reopened_stat = os.stat(reopened.fileno())
                    if os.path.samestat(true_stat, reopened_stat):
                        reopened.seek(begin)
                        return reopened
                    else:
                        raise OSError(
                            "Could not reopen file: got different file"
                        )
                except Exception:
                    reopened.close()
                    raise

            return reopen_file
        return None

    def _choose_dynamic_num_readers(
        self, tensors: Iterable[TensorEntry]
    ) -> int:
        if isinstance(getattr(self._file, "raw", self._file), io.FileIO):
            num_readers = 8
        elif self._verify_hash:
            num_readers = 4
        else:
            num_readers = 2
        if self._device.type == "cuda":
            free_ram = psutil.virtual_memory().available
            allowed_ram = free_ram - (10 << 20)
            ram_cost = heapq.nlargest(
                num_readers, (t.deserialized_length for t in tensors)
            )
            num_readers = len(ram_cost) or 1
            if num_readers > 1:
                # Get the total RAM cost for each potential value of num_readers
                ram_cost[:] = itertools.accumulate(ram_cost)
                # Find the index (reader count) for the amount of RAM
                # closest to, but still below, allowed_ram
                num_readers = bisect.bisect_left(ram_cost, allowed_ram) or 1
        return num_readers

    def __del__(self):
        self.close()

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def close(self):
        self._cleanup.close()
        self._encryption = None
        self._key_derivation = None

    @property
    def total_bytes_read(self) -> int:
        if hasattr(self._file, "bytes_read"):
            return self._file.bytes_read + self._ephemeral_bytes_read
        if self._file.closed:
            # Caution: This case is an underestimate because it doesn't include
            # any metadata read, unlike the other cases.
            return self.total_tensor_bytes
        elif self._num_readers > 1 and self._last_yielded_key is not None:
            last_yielded = self._metadata[self._last_yielded_key]
            return last_yielded.data_offset + last_yielded.data_length
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

    @staticmethod
    def _preload_cuda() -> Callable:
        called: bool = TensorDeserializer._preload_cuda_called
        TensorDeserializer._preload_cuda_called = True
        if not called and torch.cuda.is_available():

            def _attempt_preload():
                # noinspection PyBroadException
                try:
                    torch.empty((1,), device="cuda")
                except Exception:
                    pass

            preload_thread = threading.Thread(target=_attempt_preload)
            preload_thread.start()
            return preload_thread.join
        else:
            return lambda timeout=None: None

    _preload_cuda_called: ClassVar[bool] = False

    def _read_single_tensor(
        self, expected_path: _TensorPath
    ) -> torch.nn.Parameter:
        expected_name: Union[tuple, str] = expected_path.normalize_()
        this_one_tensor_filter: FilterFuncType = partial(
            operator.eq, expected_name
        )
        tensors = tuple(self.read_tensors(filter_func=this_one_tensor_filter))
        num_tensors = len(tensors)
        if num_tensors == 0:
            raise RuntimeError(f"Tensor not found: {expected_name!r}")
        elif num_tensors > 1:
            raise RuntimeError(
                f"Found too many tensors: expected 1, found {num_tensors}"
            )
        *_, name, tensor = tensors[0]
        if name != expected_name:
            raise RuntimeError(
                "Encountered unexpected tensor:"
                f" expected {expected_name!r}, found {name!r}"
            )
        return tensor

    def _load_prefixed(
        self,
        prefix: Iterable[_TensorPathComponent],
        use_dict_proxies: bool,
        strip_prefix: bool = True,
        filter_func: Optional[FilterFuncType] = None,
    ) -> Union[dict, list, None]:
        prefix: Tuple[_TensorPathComponent, ...] = tuple(prefix)
        prefix_len: int = len(prefix)

        keys = [
            k
            for k in self._metadata.keys()
            if k[:prefix_len] == prefix
            and (filter_func is None or filter_func(k.normalize_()))
        ]
        if not keys:
            return None

        with contextlib.closing(self._bulk_load(keys)) as loader:
            unstructured: Dict[_TensorPath, torch.Tensor] = {}
            for data in loader:
                data: TensorDeserializer._CopiedData
                unstructured[data.header.name] = data.parameter
                del data

        restructured = _tensor_path.restructure(unstructured, use_dict_proxies)

        if strip_prefix:
            for i in range(prefix_len):
                if not restructured:
                    restructured = None
                    break
                elif isinstance(restructured, list):
                    restructured = restructured[0]
                else:
                    restructured = restructured[prefix[i]]
        return restructured

    def tree(
        self,
        prefix: Optional[Iterable[_TensorPathComponent]] = None,
        *,
        default: Any = None,
        filter_func: Optional[FilterFuncType] = None,
    ) -> Union[typing.Mapping, Sequence, torch.Tensor, Any]:
        """
        Loads a (sub)tree of the serialized object's structure
        as a mix of potentially-nested mappings and sequences,
        or a leaf tensor. If no leaf tensors match the prefix,
        `default` is returned.

        Structures other than simple one-level mappings can be serialized via
        ``TensorSerializer.write_state_dict``.

        When accessing nested structures without this function
        (e.g. when using ``__getitem__``), all layers are accessed as
        read-only mappings instead of a mix of mappings and sequences.
        In that scenario, sequences are represented as mappings with
        integer keys instead of string keys.
        This function, conversely, allows loading sequences as actual
        ``Sequence`` type objects.

        Args:
            prefix: Path to a nested layer to load directly, as a sequence.
                If ``None`` (the default) or an empty sequence, loads
                all layers starting from the root.
            default: Object to return if no leaf tensors match the prefix.
            filter_func: An optional callable with the signature
                ``(tensor_name: str | Sequence[str | int]) -> bool``
                that takes the full path of each tensor in the subtree that
                would be loaded, and can apply additional filtering logic to
                decide whether each tensor should be loaded
                on a per-tensor basis (more granular than a full subtree).
                The function should return ``False`` to skip loading the tensor,
                or ``True`` to load the tensor.
                Note that the callable will always receive full tensor paths
                relative to the root of the deserializer, including the
                subtree's full prefix, if any.

        Returns:
            An object composed of potentially-nested mappings, sequences,
            and tensors representing the subtree beginning at the specified
            prefix.

        Examples:
            Serialize and deserialize nested objects::

                serializer = TensorSerializer(path)
                nested_structure = {
                    "model": {
                        "layer": [
                            {"weight": torch.tensor(...)},
                            {"weight": torch.tensor(...)},
                        ],
                    },
                    "lm_head": torch.tensor(...),
                }
                serializer.write_state_dict(nested_structure)
                serializer.close()

                deserializer = TensorDeserializer(path)

                # All the layers can be accessed as nested mappings
                assert list(deserializer.keys()) == ["model", "lm_head"]
                assert list(deserializer["model"].keys()) == ["layer"]

                # Note: when accessed without .tree(),
                # the list turns into a mapping with int keys
                assert list(deserializer["model"]["layer"].keys()) == [0, 1]

                assert list(
                    deserializer["model"]["layer"][0].keys()
                ) == ["weight"]

                # The layers can be accessed with their
                # original structure as well

                from typing import Sequence, Mapping

                tree = deserializer.tree()
                assert isinstance(tree, Mapping)
                assert isinstance(tree["model"]["layer"], Sequence)

                # You can load the structure of a specific prefix
                subtree = deserializer.tree(("model", "layer"))
                assert isinstance(subtree, Sequence) and len(subtree) == 2

            Serialize and deserialize a simple list::

                serializer = TensorSerializer(path)
                items = [
                    torch.tensor(...), torch.tensor(...), torch.tensor(...)
                ]
                serializer.write_state_dict(items)
                serializer.close()

                deserializer = TensorDeserializer(path)

                # The deserializer object is a mapping with int keys
                # that resembles the original sequence
                assert list(deserializer.keys()) == [0, 1, 2]
                for i in range(3):
                    print(deserializer[i].sum())
                for tensor in deserializer.values():
                    # Needs .values() to iterate over the contents
                    print(tensor.sum())

                # The result of deserializer.tree() is an actual sequence
                from typing import Sequence
                deserialized_sequence = deserializer.tree()
                assert isinstance(deserialized_sequence, Sequence)
                for i in range(3):
                    print(deserialized_sequence[i].sum())
                for tensor in deserialized_sequence:
                    # Not a mapping; doesn't need .values()
                    print(tensor.sum())
        """
        if prefix is None:
            prefix = ()
        if isinstance(prefix, (str, int)):
            raise TypeError(
                "Invalid prefix: expected sequence,"
                f" got {prefix.__class__.__name__!r}"
                " (for a single-element prefix, use a single-element sequence)"
            )
        prefixed = self._load_prefixed(
            prefix, use_dict_proxies=False, filter_func=filter_func
        )
        return prefixed if prefixed is not None else default

    def __getitem__(self, name) -> Union[torch.nn.Parameter, typing.Mapping]:
        path: tuple = (name,)
        maybe_copied_data = self._cache.get(path)
        if maybe_copied_data is not None:
            return maybe_copied_data.parameter

        # If we're in lazy_load mode, we populate the cache with the
        # tensor data and then convert it to a torch parameter. Most
        # of the time, access patterns are front to back, so seeking
        # forward in a stream works well even for HTTP/HTTPS streams.
        if name not in self._structure:
            raise KeyError(f"Tensor {name} not found")
        branch = self._structure[name]
        if isinstance(branch, _TensorPath):
            return self._read_single_tensor(_TensorPath(path))
        else:
            return self._load_prefixed(
                path, use_dict_proxies=True
            ) or types.MappingProxyType({})

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
        yield from self._structure

    def __len__(self):
        return len(self._structure)

    def __contains__(self, key: str):
        return key in self._structure

    def keys(self):
        # We override keys() because dict_keys can be slightly more efficient
        # than an extra collections.abc.KeysView wrapper.
        #
        # Technically this makes mapping.keys().mapping invalid on
        # Python 3.10+ but it is not intended to be supported anyway, so treat
        # it as not implemented.
        return self._structure.keys()

    def _verify_hashes(
        self,
        name: _TensorPath,
        hashes: Iterable[TensorHash],
        header_hashes: Dict[HashType, Any],
        mv: Union[memoryview, bytes],
    ) -> None:
        """
        Verifies the hash of the tensor data.

        Args:
            hashes: The list of hashes to verify.
            header_hashes: The pre-computed hashes for the tensor's headers.
            mv: The memoryview of the tensor data.
        """
        metadata_entry = self._metadata.get(name)
        if metadata_entry and metadata_entry.data_length == 0:
            # The tensor was zero-filled, so there is nothing to compare against
            mv = b""
        for tensor_hash in hashes:
            hash_type = tensor_hash.type
            hash_body = tensor_hash.hash
            if hash_type == HashType.CRC32:
                crc = zlib.crc32(mv, header_hashes[hash_type])
                hash_crc = struct.unpack("<I", hash_body)[0]
                if crc != hash_crc:
                    raise HashMismatchError(
                        f"Tensor '{name}' failed CRC32 verification. "
                        f"Expected {hash_crc}, got {crc}."
                    )
            elif hash_type == HashType.SHA256:
                sha = header_hashes[hash_type].copy()
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

    @staticmethod
    def _get_encryption_method(
        crypt_info: _crypt_info.CryptInfo,
    ) -> _crypt_info.CryptInfoChunk:
        encryption_method = crypt_info.find_chunks(
            (
                _crypt_info.XSalsa20ParallelChunk,
                _crypt_info.XSalsa20SequentialChunk,
            )
        )
        if not encryption_method:
            raise CryptographyError("No known encryption method found in file")
        elif len(encryption_method) > 1:
            raise CryptographyError(
                "Could not interpret encryption method of the file"
            )
        return encryption_method[0]

    def _derive_encryption_key(
        self, crypt_info: _crypt_info.CryptInfo
    ) -> bytes:
        if self._encryption.key is not None:
            return self._encryption.key
        # Requires key derivation from a source string
        if self._key_derivation is None:
            raise ValueError("Invalid DecryptionParams")
        # Check for a KeyDerivationChunk
        kd = crypt_info.find_chunks(_crypt_info.KeyDerivationChunk)
        if not kd:
            raise CryptographyError(
                "Source string was provided, but the tensor was"
                " not originally encrypted using a source string"
                " (e.g. EncryptionParams.from_string())."
                " DecryptionParams.from_key() must be used instead"
            )
        elif len(kd) > 1:
            raise CryptographyError(
                "Could not interpret encryption key derivation"
                " method of the file"
            )
        method = typing.cast(_crypt_info.KeyDerivationChunk, kd[0])
        return self._key_derivation.key(method)

    @classmethod
    def _get_decryption_manager(
        cls,
        decryption_pool: Optional[concurrent.futures.ThreadPoolExecutor],
        encryption_method: _crypt_info.CryptInfoChunk,
        key: bytes,
        buffer,
    ) -> Union["_crypt.ChunkedEncryption", "_crypt.SequentialEncryption"]:
        if isinstance(encryption_method, _crypt_info.XSalsa20ParallelChunk):
            if encryption_method.num_macs == 1:
                return _crypt.SequentialEncryption(
                    key=key,
                    buffer=buffer,
                    nonce=encryption_method.nonce,
                    mac=encryption_method.macs[0],
                    intent=_crypt.SequentialEncryption.INTENT.DECRYPTION,
                )
            else:
                nonces = _crypt.ChunkedEncryption.sequential_nonces(
                    initial_nonce=encryption_method.nonce,
                    count=encryption_method.num_macs,
                )
                return _crypt.ChunkedEncryption(
                    key=key,
                    buffer=buffer,
                    chunk_size=encryption_method.chunk_size,
                    nonces=nonces,
                    macs=encryption_method.macs,
                    executor=decryption_pool,
                    intent=_crypt.ChunkedEncryption.INTENT.DECRYPTION,
                )
        elif isinstance(encryption_method, _crypt_info.XSalsa20SequentialChunk):
            return _crypt.SequentialEncryption(
                key=key,
                buffer=buffer,
                nonce=encryption_method.nonce,
                mac=encryption_method.mac,
                intent=_crypt.SequentialEncryption.INTENT.DECRYPTION,
            )
        else:
            raise CryptographyError("Unknown encryption method")

    @classmethod
    def _stream_decrypt(
        cls,
        file_,
        decryption_pool: Optional[concurrent.futures.ThreadPoolExecutor],
        encryption_method: _crypt_info.CryptInfoChunk,
        key: bytes,
        buffer,
    ):
        try:
            with cls._get_decryption_manager(
                decryption_pool, encryption_method, key, buffer
            ) as crypto:
                if isinstance(crypto, _crypt.ChunkedEncryption):
                    fs = []
                    for chunk in range(crypto.num_chunks):
                        with crypto.chunk_view(chunk) as view:
                            file_.readinto(view)
                        fs.append(crypto.decrypt_chunk(chunk))
                    crypto.wait_or_raise(fs, timeout=_TIMEOUT)
                else:
                    file_.readinto(buffer)
                    crypto.decrypt()
        except _crypt.CryptographyError as e:
            raise CryptographyError("Tensor decryption failed") from e
        finally:
            del crypto

    def _read_numpytensors(
        self,
        filter_func: Optional[FilterFuncType] = None,
        num_tensors: int = -1,
        verify_hash: Optional[bool] = None,
    ) -> Iterator[_CopiedData]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and a _NumpyTensor `tensor`.

        The generator yields tuples of the form:
            (module_idx, tensor_type, name, arr)

        Args:
            filter_func: A function that takes a tensor name and returns
                True if the tensor should be returned, False otherwise.
            num_tensors: The number of tensors to read. If -1, all tensors
                will be read. Otherwise, yields until `num_tensors` tensors
                are read, or the file ends.
            verify_hash: If True, the hashes of each tensor will be verified
                against the hashes stored in the metadata.
                A `HashMismatchError` will be raised if any of the hashes do
                not match. If ``None``, the value of the `verify_hash` argument
                passed to the `TensorDeserializer` constructor will be used.

        Raises:
            HashMismatchError: If `verify_hash` resolves to True and
            a deserialized tensor does not match its stored hash.
        """
        if num_tensors < 0 and num_tensors != -1:
            raise ValueError("num_tensors must be -1 or non-negative")
        elif num_tensors == 0:
            return
        keys_to_read = self._metadata.keys()
        if self._last_yielded_key is not None:
            start = self._keys_enumerated[self._last_yielded_key]
            keys_to_read = itertools.islice(keys_to_read, start + 1, None)
        if filter_func is not None:
            keys_to_read = (
                k for k in keys_to_read if filter_func(k.normalize_())
            )
        if num_tensors != -1:
            keys_to_read = itertools.islice(keys_to_read, num_tensors)

        bulk_loader = self._bulk_load(keys_to_read, verify_hash)
        with contextlib.closing(bulk_loader):
            yield from bulk_loader

    def read_tensors(
        self,
        filter_func: Optional[FilterFuncType] = None,
        num_tensors: int = -1,
        verify_hash: Optional[bool] = None,
    ) -> Iterator[Tuple[int, int, str, torch.Tensor]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and torch `tensor`.

        The generator yields tuples of the form:
            (module_idx, tensor_type, name, tensor)

        Args:
            filter_func: A function that takes a tensor name and returns
                True if the tensor should be returned, False otherwise.
            num_tensors: The number of tensors to read. If -1, all tensors
                will be read. Otherwise, yields until `num_tensors` tensors
                are read, or the file ends.
            verify_hash: If True, the hashes of each tensor will be verified
                against the hashes stored in the metadata.
                A `HashMismatchError` will be raised if any of the hashes do
                not match. If ``None``, the value of the `verify_hash` argument
                passed to the `TensorDeserializer` constructor will be used.
        Yields:
            Tuples of the form (module_idx, tensor_type, name, tensor).

        Raises:
            HashMismatchError: If `verify_hash` resolves to True and
                a deserialized tensor does not match its stored hash.
        """
        copied_data = self._read_numpytensors(
            filter_func=filter_func,
            num_tensors=num_tensors,
            verify_hash=verify_hash,
        )
        for data in copied_data:
            yield data.header.module_idx, data.header.tensor_type, data.header.name.normalize_(), data.parameter

    def read_numpy_arrays(
        self,
        filter_func: Optional[FilterFuncType] = None,
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

        The generator yields tuples of the form:
            (module_idx, tensor_type, name, arr, is_opaque, torch_dtype)

        See also: `TensorDeserializer.read_tensors`

        Args:
            filter_func: A function that takes a tensor name and returns
                True if the tensor should be returned, False otherwise.
            num_tensors: The number of tensors to read. If -1, all tensors
                will be read. Otherwise, yields until `num_tensors` tensors
                are read, or the file ends.
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
            HashMismatchError: If `verify_hash` resolves to True and
                a deserialized tensor does not match its stored hash.
        """
        if self._device.type != "cpu":
            raise RuntimeError(
                "read_numpy_arrays is only valid when deserializing to the CPU"
            )
        copied_data = self._read_numpytensors(
            filter_func=filter_func,
            num_tensors=num_tensors,
            verify_hash=verify_hash,
        )
        for data in copied_data:
            module_idx = data.header.module_idx
            tensor_type = data.header.tensor_type
            name = data.header.name
            numpy_tensor = data.numpy_tensor

            is_opaque = numpy_tensor.is_opaque
            arr = numpy_tensor.data
            torch_dtype = numpy_tensor.torch_dtype if is_opaque else None

            if is_opaque and not allow_raw_data:
                np_dtype = arr.dtype.str
                raise ValueError(
                    f"{name} has an opaque datatype: "
                    f"(Torch: {numpy_tensor.torch_dtype}, Numpy: {np_dtype}). "
                    "Set `allow_raw_data=True` to return as a numpy array "
                    f"with a datatype of {np_dtype}"
                )

            yield module_idx, tensor_type, name.normalize_(), arr, is_opaque, torch_dtype

    def _to_torch_parameter(
        self, tensor: Union[torch.Tensor, torch.nn.Parameter]
    ) -> torch.nn.Parameter:
        """
        Convert a tensor to a torch.nn.Parameter on a device, forcing
        gradient when appropriate. We also handle torch.nn.Parameter objects in
        a passthrough manner.
        """
        original_device: torch.device = tensor.device
        if isinstance(tensor, torch.nn.Parameter):
            tensor.data = tensor.data.to(self._device)
            if tensor.grad is not None:
                tensor.grad = tensor.grad.to(self._device)
            return tensor

        # Cast the tensor if a global dtype was given to the TensorDeserializer
        if (
            self._dtype is not None
            and tensor.dtype != torch.bool
            and tensor.dtype != self._dtype
        ):
            target_dtype = self._dtype
        else:
            target_dtype = None

        gradient = tensor.dtype.is_complex or tensor.dtype.is_floating_point

        start = time.perf_counter_ns() if _perf_stats else 0
        tensor_on_device = tensor.to(device=self._device, dtype=target_dtype)
        end = time.perf_counter_ns() if _perf_stats else 0

        result = torch.nn.Parameter(
            tensor_on_device,
            requires_grad=gradient,
        )
        if _perf_stats and original_device != result.device:
            duration = end - start
            bytes_transferred = result.element_size() * result.nelement()
            with _perf_stats.lock:
                _perf_stats.tensor_to_device_ns += duration
                _perf_stats.tensor_to_device_bytes += bytes_transferred
        return result

    def _generate_state_dict(self) -> None:
        """
        Load the tensors in this Tensorizer object into a state_dict. This
        is used to populate the cache in non-lazy_load cases.
        """
        if self._file.closed:
            raise IOError("IO closed, instantiate if you want to load again.")

        self._cache = OrderedDict()
        keys = tuple(self._metadata.keys())
        bulk_loader = self._bulk_load(keys)
        with contextlib.closing(bulk_loader):
            for _ in bulk_loader:
                # Just run this for the caching side effect
                pass

        self.total_tensor_bytes = sum(
            self._metadata[name].data_length for name in keys
        )
        self._file.close()

    def _bulk_load(
        self,
        keys: Iterable[Union[str, _TensorPath]],
        verify_hash: Optional[bool] = None,
    ) -> Generator[_CopiedData, None, None]:
        # For each key in keys, identify the ones that are not in self._cache,
        # and run those through _bulk_load_uncached.
        # Results are stored in self._cache
        # Results are then yielded in order with the keys provided

        keys: List[_TensorPath] = list(map(_TensorPath.wrap_, keys))
        if len(set(keys)) != len(keys):
            raise ValueError("Keys must not have any duplicates")

        # Make a stack of items to yield, in-order
        keys.reverse()
        remaining: List[Optional[TensorDeserializer._CopiedData]]
        remaining = list(map(self._cache.get, keys))
        # Also allow looking up entries by key
        # This would just use one dict if they had efficient .last() methods
        indices: Dict[_TensorPath, int] = {k: i for i, k in enumerate(keys)}
        # Begin loading any that are not already cached
        uncached: List[_TensorPath] = [
            k for k, v in zip(keys, remaining) if v is None
        ]
        loader = self._bulk_load_uncached(uncached, verify_hash)
        with contextlib.closing(loader):
            while remaining:
                if remaining[-1] is not None:
                    # Pop the next value, if ready
                    # self._last_yielded_key is used by
                    # self._read_numpytensors() when num_tensors > 0
                    self._last_yielded_key = remaining[-1].header.name
                    yield remaining.pop()
                else:
                    # Otherwise, wait on something new from the loader
                    item = next(loader)
                    key = item.header.name
                    remaining[indices[key]] = self._cache[key] = item

    def _bulk_load_uncached(
        self, keys: Sequence[_TensorPath], verify_hash: Optional[bool] = None
    ) -> Generator[_CopiedData, None, None]:
        if not keys:
            return

        # Ensure all keys are present and in sorted
        # order relative to self._metadata.keys().
        # This will raise a KeyError if a requested key
        # isn't in self._metadata.keys().
        keys = sorted(keys, key=self._keys_enumerated.__getitem__)
        if any(itertools.starmap(operator.eq, zip(keys, keys[1:]))):
            raise ValueError("Keys must not have any duplicates")

        if verify_hash is None:
            verify_hash = self._verify_hash

        # Main route for multiple keys

        # Each reader will sequentially read a segment of the file
        # Segments are chosen to minimize the maximum length of any segment
        tensor_info: Tuple[TensorEntry, ...] = tuple(
            map(self._metadata.__getitem__, keys)
        )
        tensors_per_reader: List[Tuple[TensorEntry, ...]]
        effective_num_readers: int = min(self._num_readers, len(keys))
        if effective_num_readers == 1:
            tensors_per_reader = [tensor_info]
        elif effective_num_readers == len(tensor_info):
            tensors_per_reader = [(t,) for t in tensor_info]
        else:
            tensor_sizes: List[int] = [t.data_length for t in tensor_info]
            reader_slices: Iterable[slice] = _linear_partition.partition(
                tensor_sizes, effective_num_readers
            )
            tensors_per_reader = [tensor_info[s] for s in reader_slices]
            del tensor_sizes, reader_slices
        effective_num_readers = len(tensors_per_reader)

        copy_result = Union[Exception, TensorDeserializer._CopiedData]
        transfer_out_queue: "queue.SimpleQueue[copy_result]"
        transfer_out_queue = queue.SimpleQueue()

        futures: List[concurrent.futures.Future] = []
        barrier = threading.Barrier(effective_num_readers)

        # Essentially a mutable atomic flag; bytearray operations are atomic,
        # So the flag is set by appending an element, and checked by its length
        halt: bytearray = bytearray()

        for thread_idx, tensor_items in enumerate(tensors_per_reader):
            future = self._reader_pool.submit(
                self._copy_thread,
                thread_idx,
                halt,
                barrier,
                verify_hash,
                tensor_items,
                transfer_out_queue,
            )
            futures.append(future)

        try:
            for _ in range(len(keys)):
                copied_data: copy_result = transfer_out_queue.get(timeout=3600)
                if isinstance(copied_data, Exception):
                    raise copied_data
                yield copied_data
        except BaseException:
            # error occurred; halt
            halt.append(1)
            raise

    def _copy_thread(
        unsafe_self,
        thread_idx: int,
        halt: bytearray,
        barrier: threading.Barrier,
        verify_hash: bool,
        tensor_items: Sequence[TensorEntry],
        transfer_out_queue: "queue.SimpleQueue[Union[Exception, _CopiedData]]",
    ):
        # Need to get rid of self or more safely have thread-local storage

        try:
            is_cuda = unsafe_self._device.type == "cuda"
            cuda_stream = None
            if is_cuda:
                cuda_stream = torch.cuda.Stream(unsafe_self._device)

            # Allocating pinned memory seems to block creating new threads, so
            # ensure all threads are created before we go
            barrier.wait(timeout=_TIMEOUT)

            if len(tensor_items) == 0:
                return

            begin_offset = tensor_items[0].offset
            end_offset = (
                tensor_items[-1].data_offset + tensor_items[-1].data_length
            )
        except Exception as e:
            barrier.abort()
            transfer_out_queue.put(e)
            del transfer_out_queue
            return

        file_ = None
        readinto_duration = readinto_bytes = 0

        shared_buffer_tensor: Optional[torch.Tensor] = None
        shared_buffer_mv: Optional[memoryview] = None
        try:
            if thread_idx != 0:
                file_ = unsafe_self._reopen(begin=begin_offset, end=end_offset)

                old_etag = unsafe_self._etag
                if old_etag and hasattr(file_, "response_headers"):
                    new_etag = file_.response_headers.get("etag") or None
                    if new_etag is not None and new_etag != old_etag:
                        # This might indicate that a different version of the
                        # file was retrieved on the second attempt. ETag values
                        # are not guaranteed to be stable for unchanged files,
                        # though, so this isn't an error, just interesting info
                        logger.info(
                            "ETag in re-opened file doesn't match"
                            f" (original: {old_etag}, new: {new_etag})"
                        )
            else:
                file_ = unsafe_self._file
                file_.seek(begin_offset)

            # create CPU-pinned memory buffer
            # TODO: experiment with mmap(MMAP_LOCKED | MMAP_ANONYMOUS | MMAP_PRIVATE)

            if is_cuda:
                total_tensor_bytes: int = max(
                    t.deserialized_length for t in tensor_items
                )
                shared_buffer_tensor = torch.empty(
                    (total_tensor_bytes,),
                    device="cpu",
                    dtype=torch.uint8,
                    pin_memory=True,
                )
                shared_buffer_mv: memoryview = (
                    shared_buffer_tensor.numpy().data.cast("B")
                )

            tensor_sizes_by_name: Dict[_TensorPath, int] = {
                t.name: t.deserialized_length for t in tensor_items
            }

            # then for each tensor in tensor_items
            tensors_read = 0
            while tensors_read < len(tensor_items):
                if halt:
                    break

                if unsafe_self._headers is None:
                    header = _TensorHeaderDeserializer.from_io(
                        file_,
                        zero_hashes=True,
                        check_crypt_info=unsafe_self._has_crypt_info,
                    )
                    if header is None:
                        raise KeyError("Unexpected empty header")

                    # Skip it if this tensor is not one we're supposed to load
                    if header.name not in tensor_sizes_by_name:
                        file_.seek(header.data_length, io.SEEK_CUR)
                        continue
                else:
                    header = unsafe_self._headers[
                        tensor_items[tensors_read].name
                    ]

                numpy_dtype, *torch_dtype = header.dtype.split(OPAQUE_DTYPE_SEP)
                if not torch_dtype:
                    torch_dtype = None
                elif len(torch_dtype) == 1:
                    torch_dtype = torch_dtype[0]
                else:
                    raise ValueError(
                        "Can't deserialize a tensor with "
                        "multiple opaque dtype separators "
                        f"({OPAQUE_DTYPE_SEP!r}) in its dtype: "
                        f"{header.dtype!r}"
                    )

                unsafe_self._metadata[header.name].hashes = header.hashes

                header_hashes = header.compute_hashes()
                unsafe_self._metadata[header.name].header_hashes = header_hashes

                is_encrypted: bool = (
                    header.crypt_info is not None
                    and header.crypt_info.num_chunks != 0
                )
                has_data: bool = header.data_length > 0
                if unsafe_self._encrypted and not is_encrypted and has_data:
                    raise CryptographyError(
                        "Tensor is not encrypted, but decryption was requested"
                    )
                elif is_encrypted and not unsafe_self._encrypted:
                    raise CryptographyError(
                        "Tensor is encrypted, but decryption was not requested"
                    )
                elif unsafe_self._encrypted and is_encrypted:
                    encryption_method = unsafe_self._get_encryption_method(
                        header.crypt_info
                    )
                    key = unsafe_self._derive_encryption_key(header.crypt_info)
                else:
                    key = None
                    encryption_method = None

                needed_buffer_size = tensor_sizes_by_name[header.name]
                is_meta = needed_buffer_size > 0 and header.data_length == 0
                assert is_meta or needed_buffer_size == header.data_length

                if is_cuda:
                    if is_meta:
                        shared_buffer_tensor[:needed_buffer_size].zero_()
                    mv: memoryview = shared_buffer_mv[:needed_buffer_size]
                else:
                    # Not in CUDA, no pinned memory.
                    # Allocate a new buffer for each tensor
                    # because that's what we're going to be using long-term
                    buffer_tensor = torch.empty(
                        (needed_buffer_size,), device="cpu", dtype=torch.uint8
                    )
                    if is_meta:
                        buffer_tensor.zero_()
                    mv: memoryview = buffer_tensor.numpy().data.cast("B")
                    del buffer_tensor

                if not is_meta:
                    start = time.perf_counter_ns() if _perf_stats else 0
                    file_.seek(unsafe_self._metadata[header.name].data_offset)

                    if unsafe_self._encrypted and mv.nbytes > 0:
                        TensorDeserializer._stream_decrypt(
                            file_,
                            unsafe_self._decryption_pool,  # decryption_pool safe to be shared
                            encryption_method,
                            key,
                            mv,
                        )
                    else:
                        file_.readinto(mv)

                    if verify_hash:
                        unsafe_self._verify_hashes(
                            header.name, header.hashes, header_hashes, mv
                        )

                    readinto_duration += (
                        time.perf_counter_ns() - start if _perf_stats else 0
                    )
                    readinto_bytes += mv.nbytes

                # create a tensor around it and maybe torch.to('cuda')
                numpy_tensor = _NumpyTensor.from_buffer(
                    numpy_dtype,
                    torch_dtype,
                    header.shape,
                    mv,
                )
                del mv
                tensor = numpy_tensor.to_tensor()

                stream_context = (
                    torch.cuda.stream(cuda_stream)
                    if is_cuda
                    else contextlib.nullcontext()
                )
                with stream_context:
                    parameter = unsafe_self._to_torch_parameter(tensor)
                    if cuda_stream is not None:
                        cuda_stream.synchronize()

                # put it on transfer_out_queue
                transfer_out_queue.put(
                    TensorDeserializer._CopiedData(
                        header, numpy_tensor, parameter
                    )
                )
                tensors_read += 1
        except Exception as e:
            del shared_buffer_tensor, shared_buffer_mv
            transfer_out_queue.put(e)
            del transfer_out_queue
        finally:
            if file_ is not None and file_ is not unsafe_self._file:
                bytes_read = getattr(file_, "bytes_read", 0)
                file_.close()
                if bytes_read:
                    with unsafe_self._ephemeral_bytes_read_lock:
                        unsafe_self._ephemeral_bytes_read += bytes_read
            if _perf_stats and (readinto_duration or readinto_bytes):
                with _perf_stats.lock:
                    _perf_stats.file_readinto_ns += readinto_duration
                    _perf_stats.file_readinto_bytes += readinto_bytes

    def load_into_module(
        self,
        m: torch.nn.Module,
        filter_func: Optional[FilterFuncType] = None,
        verify_hash: Optional[bool] = None,
    ) -> int:
        """
        Given `m`, a torch.nn.Module, load the associate tensors in this
        Tensorizer object into the `torch.nn.Module`. Returns the number of
        tensors loaded into the module.

        Args:
            m: The module to load the tensors into.
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
            HashMismatchError: If `verify_hash` resolves to True and
                a deserialized tensor does not match its stored hash.
        """
        modules: typing.OrderedDict[str, torch.nn.Module] = OrderedDict()

        if verify_hash is None:
            verify_hash = self._verify_hash

        for name, module in m.named_modules():
            modules[name] = module

        keys = tuple(
            k
            for k in self._metadata.keys()
            if filter_func is None or filter_func(k.normalize_())
        )

        tensor_ct = len(keys)

        buffer_type = TensorType.BUFFER
        param_type = TensorType.PARAM
        state_dict_type = TensorType.STATE_DICT

        bulk_loader = self._bulk_load(keys, verify_hash=verify_hash)
        with contextlib.closing(bulk_loader):
            for copied_data in bulk_loader:
                path: _TensorPath = copied_data.header.name
                entry = self._metadata[path]
                if entry.type is state_dict_type:
                    raise NotImplementedError(
                        "This was serialized using"
                        " TensorSerializer.write_state_dict(), so it cannot be"
                        " loaded using TensorDeserializer.load_into_module()."
                        " Use the TensorDeserializer object directly as a"
                        " state_dict mapping instead."
                    )
                elif (
                    entry.type is not buffer_type
                    and entry.type is not param_type
                ):
                    raise RuntimeError(f"Invalid tensor type: {entry.type}")
                elif not path.is_str_:
                    raise NotImplementedError(
                        "Cannot deserialize structured tensor keys as a module;"
                        " try using the TensorDeserializer directly"
                        " as a state_dict mapping instead."
                    )
                tensor = copied_data.parameter
                name: str = path.normalize_()
                obj_path, attr = name.rsplit(".", 1)
                module: torch.nn.Module = modules[obj_path]

                if entry.type is param_type:
                    module.register_parameter(attr, tensor)
                elif entry.type is buffer_type:
                    module.register_buffer(attr, tensor)

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

        for path, entry in self._metadata.items():
            name = path.normalize_()
            # Check if the module has this tensor.
            if name not in modules:
                results.append((name, False))
                continue
            module: torch.nn.Module = modules[name]
            if entry.hashes is None:
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
                        entry.hashes,
                        entry.header_hashes,
                        mv,
                    )
                results.append((name, True))
            except HashMismatchError:
                results.append((name, False))

        absent_keys = set(self.keys()).difference(set(modules.keys()))
        for name in absent_keys:
            results.append((name, False))

        return all(result for name, result in results), results

    def to_redis(
        self, redis_client: redis.Redis, key_prefix: str, force: bool = False
    ) -> None:
        """
        Given a redis client and a key_prefix, write the tensors in this
        Tensorizer object to the redis client under the given key prefixes.

        Args:
            redis_client: A redis client to write the tensors to.
            key_prefix: A key prefix to use for the tensors.
            force: If True, overwrite existing keys in redis.
        """
        header_entry = b"|TZR|"
        header_entry += self._file_header.to_bytes()
        header_entry += self._metadata_raw
        redis_client.set(f"{key_prefix}:header:0", header_entry)

        for name, entry in self._metadata.items():
            offset = entry.offset
            data_offset = entry.data_offset
            header_size = data_offset - offset
            self._file.seek(offset)
            header_entry = self._file.read(header_size)
            # Check if the key already exists
            name_str: str = name.serialized_().decode("utf-8")
            if not force and redis_client.exists(
                f"{key_prefix}:{name_str}:{offset}"
            ):
                continue
            redis_client.set(f"{key_prefix}:{name_str}:{offset}", header_entry)
            data_entry = self._file.read(entry.data_length)
            redis_client.set(
                f"{key_prefix}:{name_str}:{data_offset}", data_entry
            )


class TensorSerializer:
    """
    Given a file-like object or path, serialize tensors from a torch.nn.Module
    to it.

    See the docs_ for a usage walkthrough.

    .. _docs: https://github.com/coreweave/tensorizer/tree/main#basic-usage

    Args:
        file_obj: A file-like object or path to a file to write to. The path
            can be a S3 URI.
        encryption: An `EncryptionParams` object holding a password or key
            to use for encryption. If None, no encryption will be used.
        limit_cpu_concurrency: If not ``None`` (the default), try to limit
            CPU-bound thread pools to at most this many worker threads each.
            There are multiple thread pools, so the number of total threads
            will exceed this number. The default limits are based on the CPU
            count or the active cgroups CPU resource limit if applicable.
        compress_tensors: Not implemented. Specifying this option does nothing.
            Previously, if True, compress the tensors using lz4. This
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
        *,
        encryption: Optional[EncryptionParams] = None,
        limit_cpu_concurrency: Optional[int] = None,
        max_tensors: Optional[int] = None,
    ) -> None:
        if isinstance(file_obj, (str, bytes, os.PathLike, int)):
            self._file = stream_io.open_stream(file_obj, "wb+")
        else:
            self._mode_check(file_obj)
            self._file = file_obj

        if limit_cpu_concurrency is not None:
            if not isinstance(limit_cpu_concurrency, int):
                raise TypeError(
                    "limit_cpu_concurrency parameter: expected int or None,"
                    f" {limit_cpu_concurrency.__class__.__name__} found"
                )
            if limit_cpu_concurrency < 1:
                raise ValueError(
                    "limit_cpu_concurrency parameter: must be positive"
                    f" (or None for unbound), got {limit_cpu_concurrency}"
                )
        self.cpu_limit: int = limit_cpu_concurrency or cpu_count

        if encryption is not None and not isinstance(
            encryption, EncryptionParams
        ):
            raise TypeError(
                "encryption parameter: expected EncryptionParams instance"
                f" or None, {encryption.__class__.__name__} found"
            )
        self._encryption = encryption
        self._encrypted = encryption is not None
        self._used_nonces: Optional[Set[bytes]]
        if self._encrypted:
            _require_libsodium()
            self._crypt_chunk_size = 2 << 20
            self._used_nonces = set()
        else:
            self._crypt_chunk_size = None
            self._used_nonces = None
        self._path_registry: _TensorPathRegistry = _TensorPathRegistry()

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

        self._pools = []
        # This thread pool handles CPU-bound tasks like hashing.
        # Hashing from the Python standard library can benefit from
        # multithreading in spite of the GIL because CPython's hash function
        # implementations release the GIL during longer hash computations.
        self._computation_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.cpu_limit,
            thread_name_prefix="TensorizerComputation",
        )
        self._pools.append(self._computation_pool)

        # There is no use spawning many writer threads when they share a lock.
        max_concurrent_writers = 4 if concurrent_writes_possible else 1

        # This thread pool handles straightforward write tasks, such as
        # tensor data writes.
        self._writer_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_writers,
            thread_name_prefix="TensorizerWriter",
        )
        self._pools.append(self._writer_pool)

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
            thread_name_prefix="TensorizerHeaderWriter",
        )
        self._pools.append(self._header_writer_pool)

        if self._encrypted:
            self._encryption_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_concurrent_writers,
                thread_name_prefix="TensorizerEncryption",
            )
            self._pools.append(self._encryption_pool)

            self._decryption_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_concurrent_writers,
                thread_name_prefix="TensorizerDecryption",
            )
            self._pools.append(self._decryption_pool)
        else:
            self._encryption_pool = self._decryption_pool = None

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
        self._jobs: List[_Future] = []
        # Tracks work submitted to the decryption pool to prevent conflicting,
        # overlapping in-place operations on tensors using shared storage.
        self._decryption_jobs: typing.MutableMapping[
            int, concurrent.futures.Future
        ] = (weakref.WeakValueDictionary() if self._encrypted else {})

        if self.compress_tensors:
            import lz4.frame

            self.lz4_frame = lz4.frame
        else:
            self.lz4_frame = None

        # Write our magic bytes.
        self._write(TENSORIZER_MAGIC)

        # Write file header metadata
        version_number = HEADERS_AT_TOP_TENSORIZER_VERSION
        feature_flags = _FileFeatureFlags(0)
        if self._encrypted:
            feature_flags |= _FileFeatureFlags.encrypted
        self._file_header_loc = self._file.tell()
        self._file_header = _FileHeader(
            version_number=version_number,
            feature_flags=feature_flags,
            tensor_size=0,
            tensor_count=0,
        )
        self._write(self._file_header.to_bytes())

        self._metadata_start = self._file.tell()
        self._metadata_cur = (
            self._metadata_start + 8
        )  # position of next metadata entry we'd write. Leave 8 bytes for metadata length field
        self._metadata_end: Optional[int] = None
        self._header_end: Optional[int] = None
        if max_tensors:
            # Estimate 256 bytes per metadata entry and 1024 bytes per header entry
            self._metadata_end = self._metadata_cur + max_tensors * 256
            # this is less about header_end itself but ensuring that tensor_start is on a 4096-byte aligned boundary
            self._header_end = (
                (self._metadata_end + max_tensors * 1024) + 4095
            ) & ~4095

        self._header_cur = (
            self._metadata_end
        )  # is the start of where to write header data, or None
        self._tensor_cur = (
            self._header_end
        )  # is the start of where to write tensor data. or None

    @property
    def total_tensor_bytes(self):
        return self._file_header.tensor_size

    def _flush(self):
        if hasattr(self._file, "flush"):
            # Don't keep anything in the buffered I/O object's write buffer,
            # because os.pwrite calls won't update it, and an os.pwrite
            # followed by an outdated buffer flush may overwrite good data
            # with blank bytes
            self._file.flush()

    def _sync_prologue_state(self):
        """
        This is called after the tensor has been written to the file,
        and ensures that the file is in a consistent state.

        This must not be called while any I/O jobs are pending in
        ``self._jobs``, as the contents of ``self._file_header``
        are only updated once each tensor writing job finishes.
        """
        # Write our zero-length field, that indicates that this is the last
        # tensor. This will be overwritten if another tensor is written.
        self._pwrite(struct.pack("<Q", 0), self._tensor_cur)

        # Write our new file header.
        self._pwrite(self._file_header.to_bytes(), self._file_header_loc)

        # Reset our file pointer to the end of the file,
        # minus the zero-length field.
        self._file.seek(self._tensor_cur)
        self._flush()

    def _pwrite(
        self, data, offset: int, verify: Union[bool, int] = True
    ) -> int:
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

    @staticmethod
    def _mv_suffix(data: "collections.abc.Buffer", start: int):
        if not isinstance(data, memoryview):
            data = memoryview(data)
        try:
            if data.ndim != 1:
                data = data.cast("B")
            return data[start:]
        finally:
            del data

    def _pwrite_syscall(
        self, data, offset: int, verify: Union[bool, int] = True
    ) -> int:
        # This implementation of pwrite uses a Unix syscall, and is safe to
        # run even between normal file writes.
        bytes_written: int = 0
        expected_bytes_written: int = (
            self._buffer_size(data) if isinstance(verify, bool) else verify
        )
        bytes_just_written: int = os.pwrite(self._fd, data, offset)
        bytes_written += bytes_just_written
        while bytes_written < expected_bytes_written and bytes_just_written > 0:
            # Writes larger than ~2 GiB may not complete in a single pwrite call
            offset += bytes_just_written
            with self._mv_suffix(data, bytes_written) as mv:
                bytes_just_written = os.pwrite(self._fd, mv, offset)
            bytes_written += bytes_just_written
        if isinstance(verify, int) or verify:
            self._verify_bytes_written(bytes_written, expected_bytes_written)
        return bytes_written

    def _write(self, data, expected_bytes_written: Optional[int] = None) -> int:
        # Thread-unsafe non-parallel write at the current file position
        # Calls `.write()` on `self._file` one or more times,
        # until all data is written or no more is being written,
        # as unbuffered I/O objects are not guaranteed to write
        # all data in a single call to `.write()`.
        if expected_bytes_written is None:
            expected_bytes_written = self._buffer_size(data)
        bytes_written: int = 0
        bytes_just_written: int = self._file.write(data)
        bytes_written += bytes_just_written
        if bytes_just_written > expected_bytes_written:
            raise ValueError("Wrote more data than expected")
        while bytes_written < expected_bytes_written and bytes_just_written > 0:
            with self._mv_suffix(data, bytes_written) as mv:
                bytes_just_written = self._file.write(mv)
            bytes_written += bytes_just_written
        return bytes_written

    def _pwrite_fallback(
        self, data, offset: int, verify: Union[bool, int] = True
    ) -> int:
        # This implementation of pwrite uses a lock shared with all writers
        # for the entire file object. It is not safe to run this
        # concurrently with any other code that could modify the file offset
        # except other calls to _pwrite_fallback.
        expected_bytes_written: int = (
            verify if isinstance(verify, int) else self._buffer_size(data)
        )
        with self._write_lock:
            old_pos = self._file.tell()
            if old_pos != offset:
                self._file.seek(offset)
            bytes_written = self._write(data, expected_bytes_written)
            self._file.seek(old_pos)
        if isinstance(verify, int) or verify:
            self._verify_bytes_written(bytes_written, expected_bytes_written)
        return bytes_written

    @staticmethod
    def _buffer_size(buffer: Union[memoryview, Any]) -> int:
        # For typed buffers (e.g. arrays) the len() isn't the number of bytes
        return getattr(buffer, "nbytes", len(buffer))

    @staticmethod
    def _verify_bytes_written(bytes_written: int, expected_bytes_written: int):
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
        for j in getattr(self, "_jobs", ()):
            j.cancel()
        for thread_pool in getattr(self, "_pools", ()):
            thread_pool.shutdown(wait=False)

    def _synchronize_pools(self):
        _future_wait_and_raise(self._jobs, _TIMEOUT)
        self._jobs.clear()
        self._decryption_jobs.clear()

    def close(self) -> None:
        """
        Finalizes the serialization and closes the file.
        """
        self._sync_prologue_state()

        final_sz = self._file.tell()
        self._file.close()
        self._shutdown_thread_pools()
        logger.debug(f"Tensors completed serializing to {final_sz} bytes")
        # if self.compress_tensors:
        #     compression_ratio = (
        #         self.total_tensor_bytes / self.total_compressed_tensor_bytes
        #     )
        #     logger.info(f"Uncomp'd bytes: {self.total_tensor_bytes}")
        #     logger.info(f"Comp'd bytes: {self.total_compressed_tensor_bytes}")
        #     logger.info(f"Ratio: {compression_ratio:.2f}")

    def _new_nonces(self, count: int) -> Tuple[bytes, ...]:
        if count < 0:
            raise ValueError("Invalid nonce count")
        elif count == 0:
            return ()
        elif self._used_nonces is None:
            raise RuntimeError(
                "Tried to create cryptographic nonces while"
                " encryption is disabled"
            )
        nonces = tuple(
            _crypt.ChunkedEncryption.sequential_nonces(
                initial_nonce=_crypt.ChunkedEncryption.random_nonce(),
                count=count,
            )
        )

        if self._used_nonces.intersection(nonces):
            raise RuntimeError("Illegal nonce reuse")
        self._used_nonces.update(nonces)
        return nonces

    def write_tensor(
        self,
        idx,
        name,
        tensor_type: TensorType,
        tensor: Union[torch.Tensor, numpy.ndarray],
    ) -> None:
        """
        Serializes a tensor. Header data is appended to the metadata block at the top of the file, and
        tensor data is appended to the bottom of the file.

        Args:
            idx: The index of the tensor in the module.
            name: The name of the tensor.
            tensor_type: The type of the tensor. This is used to determine
                how to interpret the tensor.
            tensor: The tensor to serialize.

        Serialization format:

         in header block near top of file:
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
           uint64                           tensor_sz}
           .....
           affer all headers, bottom of file:
           {[]byte                           tensor }
        """
        if isinstance(tensor, numpy.ndarray):
            tensor = torch.from_numpy(tensor)

        write_spec = self._WriteSpec(
            module_index=idx, name=name, tensor_type=tensor_type, tensor=tensor
        )
        self._bulk_write([write_spec], incremental=True)

    class _WriteSpec:
        def __init__(
            self,
            module_index: int,
            name: Union[str, _TensorPath],
            tensor_type: TensorType,
            tensor: torch.Tensor,
        ):
            self.tensor = tensor
            self.min_file_version = 0
            self.user_owns_tensor_data = True

            # Every parameter to _TensorHeaderSerializer() exists as an attribute except self.file_offset
            # defaulting to the simplest possible case:
            #   CPU-based
            #   contiguous
            #   not hashing
            #   not encrypted
            #   not meta
            #   not opaque
            self.module_index = module_index
            self.tensor_type: TensorType = tensor_type
            self.name = _TensorPath.wrap_(name)
            self.dtype: Optional[str] = None  # _prepare_for_write_numpy_tensor
            self.shape = tensor.size()
            self.data_length = tensor.nbytes
            # self.file_offset  # intentionally omitted, handled by _write_headers()
            self.include_crc32 = True
            self.include_sha256 = True
            self.crypt_info: Optional[_crypt_info.CryptInfo] = (
                None  # _prepare_for_write_encryption
            )

            # Additional payloads that get set and used during the prepare_for_write procedures
            self.numpy_tensor: Optional[_NumpyTensor] = (
                None  # $et in _prepare_for_write_numpy_tensor
            )
            self.header: Optional[_TensorHeaderSerializer] = (
                None  # $et in _prepare_for_write_headers
            )
            self.metadata_pos = -1  # Set in _prepare_for_write_headers
            self.encryptor: Optional[_crypt.ChunkedEncryption] = (
                None  # $et in _do_encryption if encrypted
            )

            # self.tensor_data_task is a future for processing some contents of self.tensor
            # e.g. cuda transfer, make_contiguous, hashing, encryption, writing, or decryption.
            # They are often chained from one step of the process to the next
            self.tensor_data_task: Optional[_Future] = None

        def set_min_file_version_number(self, version_number):
            self.min_file_version = max(self.min_file_version, version_number)

    def _maybe_fallocate(self, tensors: Sequence[_WriteSpec]):
        if not _syscalls.has_fallocate() or not self._fd:
            return

        next_pos = self._file.tell()
        size = sum(len(t.name.serialized_()) for t in tensors)
        size += sum(
            t.tensor.element_size()
            * t.tensor.nelement()
            * (not t.tensor.is_meta)
            for t in tensors
        )
        # Rough underestimate of header size
        header_min_size = 24
        size += header_min_size * len(tensors)
        _syscalls.try_fallocate(
            self._fd, next_pos, size, suppress_all_errors=True
        )

    def _bulk_write(self, write_specs: Iterable[_WriteSpec], incremental=False):
        write_specs = list(write_specs)

        if not incremental:
            # TODO: make into a future
            self._maybe_fallocate(write_specs)

        for w in write_specs:
            self._path_registry.register_path(w.name)

        cuda_executor = self._prepare_for_write_cuda(write_specs)
        try:
            self._prepare_for_write_contiguous(write_specs)
            self._prepare_for_write_meta(write_specs)
            self._prepare_for_write_numpy_tensor(write_specs)
            self._prepare_for_write_opaque(write_specs)
            if self._encrypted:
                self._prepare_for_write_encryption(write_specs)
            self._prepare_for_write_headers(write_specs)
            self._prepare_for_write_hashes(write_specs)

            if self._encrypted:
                self._do_encryption(write_specs)
            self._do_commit_headers(write_specs)
            self._do_commit_tensor_data(write_specs)
            if self._encrypted:
                self._maybe_decrypt_data(write_specs)

            self._file_header.version_number = max(
                self._file_header.version_number,
                max(w.min_file_version for w in write_specs),
            )

            self._synchronize_pools()
            self._sync_prologue_state()
        except Exception as e:
            for j in self._jobs:
                j.cancel()
            if cuda_executor is not None:
                cuda_executor.shutdown(wait=False)
            raise e

        if cuda_executor is not None:
            cuda_executor.shutdown(wait=True)

    def write_module(
        self,
        m: torch.nn.Module,
        *,
        include_non_persistent_buffers: bool = True,
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
            include_non_persistent_buffers: Whether to serialize buffers
                registered with ``persistent=False``.
                Set to ``False`` to match the behaviour of
                ``torch.nn.Module.state_dict()``,
                which saves only persistent buffers.
                The default may change to ``False`` in a later version.
        """

        modules = tuple(m.named_modules())

        def extract_tensors() -> Iterator[TensorSerializer._WriteSpec]:
            chain = itertools.chain
            repeat = itertools.repeat
            for idx, (module_name, module) in enumerate(modules):
                module: torch.nn.Module
                parameters = module.named_parameters(recurse=False)
                buffers = module.named_buffers(recurse=False)
                for (name, tensor), tensor_type in chain(
                    zip(parameters, repeat(TensorType.PARAM)),
                    zip(buffers, repeat(TensorType.BUFFER)),
                ):
                    label = f"{module_name}.{name}"
                    yield TensorSerializer._WriteSpec(
                        module_index=idx,
                        name=label,
                        tensor_type=tensor_type,
                        tensor=tensor,
                    )

        def persistent_buffers() -> Set[str]:
            persistent_buffers_set: Set[str] = {
                name
                for name, _ in m.named_buffers(
                    recurse=True, remove_duplicate=False
                )
            }
            if hasattr(m, "_non_persistent_buffers_set"):
                # Direct access to the _non_persistent_buffers_set attribute
                # is an order of magnitude faster than generating
                # a state_dict, but this is a private interface, and thus
                # not guaranteed to remain stable between torch versions

                for module_name, module in modules:
                    # noinspection PyProtectedMember
                    persistent_buffers_set.difference_update(
                        f"{module_name}.{name}"
                        for name in module._non_persistent_buffers_set
                    )
            else:
                # Filtering down to only the buffers that appear
                # in the state_dict() representation is the supported way
                # to access the persistent buffer list, but is much slower
                persistent_buffers_set.intersection_update(
                    m.state_dict().keys()
                )

            return persistent_buffers_set

        all_tensors = extract_tensors()

        if not include_non_persistent_buffers:
            persistent = persistent_buffers()
            all_tensors = (
                spec
                for spec in all_tensors
                if spec.tensor_type != TensorType.BUFFER
                or str(spec.name) in persistent
            )

        self._bulk_write(all_tensors)

    def write_state_dict(self, state_dict: Union[Dict, List, Tuple]):
        """
        Write the state_dict to the file in Tensorizer format.

        The values are accessed at deserialization time by using
        a ``TensorDeserializer`` object as a mapping.

        The object passed to this function can be a ``dict``, a ``list``,
        or any combination of dictionaries and lists nested together,
        as long as the dictionaries have only string keys, as in JSON.
        Other recognized mapping and sequence types (like tuples)
        are converted to dictionaries and lists.

        All leaf values must be instances of ``torch.Tensor``.

        When deserialized, the original structure of mappings and sequences
        can be accessed via the ``TensorDeserializer.tree`` method,
        or the ``TensorDeserializer`` can still be used as a mapping,
        which will provide uniform access to all layers as read-only mapping
        proxies, with either ``str`` or ``int`` keys, for mapping and sequence
        layers, respectively.

        It is strongly recommended that you use write_module instead of
        this function, as it will also write out the parameter type,
        and allow for zero-copy loading of the module with
        ``TensorDeserializer.load_into_module``.

        See Also:
            `TensorDeserializer.tree`

        Examples:
            Serialize a normal state dict::

                serializer = TensorSerializer(path)
                state_dict = model.state_dict()
                serializer.write_state_dict(state_dict)
                serializer.close()

                deserializer = TensorDeserializer(path)
                assert deserializer.keys() == state_dict.keys()

            Serialize any dictionary of tensors::

                serializer = TensorSerializer(path)
                state_dict = {
                    "model.layer.0.weight": torch.tensor(...),
                    "model.layer.1.weight": torch.tensor(...),
                    "lm_head": torch.tensor(...),
                }
                serializer.write_state_dict(state_dict)
                serializer.close()

                deserializer = TensorDeserializer(path)
                assert deserializer.keys() == state_dict.keys()

            Serialize nested objects::

                serializer = TensorSerializer(path)
                nested_structure = {
                    "model": {
                        "layer": [
                            {"weight": torch.tensor(...)},
                            {"weight": torch.tensor(...)},
                        ],
                    },
                    "lm_head": torch.tensor(...),
                }
                serializer.write_state_dict(nested_structure)
                serializer.close()

                deserializer = TensorDeserializer(path)

                # All the layers can be accessed as nested mappings
                assert list(deserializer.keys()) == ["model", "lm_head"]
                assert list(deserializer["model"].keys()) == ["layer"]

                # Note: when accessed without .tree(),
                # the list turns into a mapping with int keys
                assert list(deserializer["model"]["layer"].keys()) == [0, 1]

                assert list(
                    deserializer["model"]["layer"][0].keys()
                ) == ["weight"]

                # The layers can be accessed with their
                # original structure as well

                from typing import Sequence, Mapping

                tree = deserializer.tree()
                assert isinstance(tree, Mapping)
                assert isinstance(tree["model"]["layer"], Sequence)

                # You can load the structure of a specific prefix
                subtree = deserializer.tree(("model", "layer"))
                assert isinstance(subtree, Sequence) and len(subtree) == 2

            Serialize a simple list::

                serializer = TensorSerializer(path)
                items = [
                    torch.tensor(...), torch.tensor(...), torch.tensor(...)
                ]
                serializer.write_state_dict(items)
                serializer.close()

                deserializer = TensorDeserializer(path)

                # The deserializer object is a mapping with int keys
                # that resembles the original sequence
                assert list(deserializer.keys()) == [0, 1, 2]
                for i in range(3):
                    print(deserializer[i].sum())
                for tensor in deserializer.values():
                    # Needs .values() to iterate over the contents
                    print(tensor.sum())

                # The result of deserializer.tree() is an actual sequence
                from typing import Sequence
                deserialized_sequence = deserializer.tree()
                assert isinstance(deserialized_sequence, Sequence)
                for i in range(3):
                    print(deserialized_sequence[i].sum())
                for tensor in deserialized_sequence:
                    # Not a mapping; doesn't need .values()
                    print(tensor.sum())
        """
        idx = 0
        self._bulk_write(
            TensorSerializer._WriteSpec(
                module_index=idx,
                name=name,
                tensor_type=TensorType.STATE_DICT,
                tensor=param,
            )
            for name, param in _tensor_path.flatten_structure(
                torch.Tensor, state_dict
            )
        )

    def _prepare_for_write_cuda(
        self, write_specs: Sequence[_WriteSpec]
    ) -> Optional[concurrent.futures.ThreadPoolExecutor]:
        cuda_specs = [w for w in write_specs if w.tensor.device.type == "cuda"]
        if not cuda_specs:
            return None

        class CudaTransfer:
            def __init__(self, max_size):
                self.executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="TransferThread",
                    initializer=self._allocate_staging_tensor,
                    initargs=(max_size,),
                )

            def submit(self, write_spec) -> concurrent.futures.Future:
                return self.executor.submit(self._transfer, write_spec)

            def _allocate_staging_tensor(self, max_size: int):
                self._stream = torch.cuda.Stream()
                self._staging_tensor = torch.empty(
                    (max_size,),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )

            def _transfer(self, write_spec):
                nbytes = (
                    write_spec.tensor.element_size()
                    * write_spec.tensor.nelement()
                )
                staging_tensor_view = (
                    self._staging_tensor.narrow(0, 0, nbytes)
                    .view(write_spec.tensor.dtype)
                    .view(write_spec.shape)
                )
                with torch.cuda.stream(self._stream):
                    staging_tensor_view.copy_(write_spec.tensor)
                write_spec.user_owns_tensor_data = False
                write_spec.tensor = staging_tensor_view.clone().detach()

        max_tensor_size = max(
            [w.tensor.element_size() * w.tensor.nelement() for w in cuda_specs]
        )
        cuda_transfer = CudaTransfer(max_tensor_size)

        for w in cuda_specs:
            assert w.tensor_data_task is None
            w.tensor_data_task = cuda_transfer.submit(w)
            self._jobs.append(w.tensor_data_task)

        return cuda_transfer.executor

    def _prepare_for_write_contiguous(self, write_specs: Sequence[_WriteSpec]):
        def make_contiguous(write_spec, dependency):
            if dependency is not None:
                dependency.result(_TIMEOUT)
            write_spec.tensor = write_spec.tensor.contiguous()
            write_spec.data_length = write_spec.tensor.nbytes
            write_spec.user_owns_tensor_data = False

        for w in write_specs:
            # if there is a tensor_data_task it is a cuda tensor
            if w.tensor_data_task is not None or w.tensor.is_contiguous():
                continue
            w.tensor_data_task = self._computation_pool.submit(
                make_contiguous, w, w.tensor_data_task
            )

    def _prepare_for_write_numpy_tensor(
        self, write_specs: Sequence[_WriteSpec]
    ):
        for w in write_specs:
            # all futures are resolved here. This step is not multi-threaded.
            if w.tensor_data_task is not None:
                w.tensor_data_task.result(_TIMEOUT)
                w.tensor_data_task = None
            w.numpy_tensor = _NumpyTensor.from_tensor(w.tensor)
            w.dtype = w.numpy_tensor.numpy_dtype
            if w.numpy_tensor.data.data.nbytes != w.tensor.nbytes:
                raise ValueError(
                    f"Cannot serialize tensor {w.name!r}:"
                    f" buffer size of underlying memory ({w.numpy_tensor.data.data.nbytes})"
                    f" doesn't match reported size ({w.tensor.nbytes})"
                )

    def _prepare_for_write_opaque(
        self, write_specs: Sequence[_WriteSpec]
    ) -> None:
        for w in write_specs:
            if not w.numpy_tensor.is_opaque:  # type: ignore
                continue
            # The datatype name needs to contain both the numpy dtype that the
            # data is serialized as and the original torch dtype.
            w.dtype += OPAQUE_DTYPE_SEP + w.numpy_tensor.torch_dtype  # type: ignore
            w.set_min_file_version_number(OPAQUE_TENSORIZER_VERSION)

    @staticmethod
    def _do_clone(write_spec, dependency: Optional[concurrent.futures.Future]):
        if dependency is not None:
            dependency.result(_TIMEOUT)
        write_spec.tensor = write_spec.tensor.clone().detach()
        write_spec.numpy_tensor = _NumpyTensor.from_tensor(write_spec.tensor)

    def _prepare_for_write_encryption(
        self, write_specs: Sequence[_WriteSpec]
    ) -> None:
        assert self._encrypted and self._encryption is not None

        # If any tensors are shared, so we need to clone all but one of them before encrypting
        write_specs_by_addr: Dict[int, List[TensorSerializer._WriteSpec]] = (
            collections.defaultdict(list)
        )
        for w in write_specs:
            if w.tensor.device.type != "cpu":
                continue
            address = w.tensor.untyped_storage().data_ptr()
            write_specs_by_addr[address].append(w)

        for shared_write_specs in write_specs_by_addr.values():
            if len(shared_write_specs) == 1:
                continue

            clone_dependencies = _FutureGroup(
                [
                    w.tensor_data_task
                    for w in shared_write_specs
                    if w.tensor_data_task is not None
                ]
            )

            clone_tasks = []
            for w in shared_write_specs[1:]:
                clone_tasks.append(
                    self._computation_pool.submit(
                        self._do_clone, w, clone_dependencies
                    )
                )
                w.user_owns_tensor_data = False

            shared_write_specs[0].tensor_data_task = _FutureGroup(
                clone_tasks + clone_dependencies.futures
            )
            for w in shared_write_specs[1:]:
                w.tensor_data_task = _FutureGroup(clone_tasks)

        for w in write_specs:
            assert w.numpy_tensor is not None
            w.include_crc32 = False

            if w.data_length == 0:
                # All headers are expected to have crypt_info segments, so add
                # an empty one
                w.crypt_info = _crypt_info.CryptInfo()
                continue

            if w.tensor_data_task is not None:
                w.tensor_data_task.result(_TIMEOUT)
                w.tensor_data_task = None

            tensor_memory: memoryview = w.numpy_tensor.tensor_memory
            chunked = _Chunked(
                total_size=tensor_memory.nbytes,
                chunk_size=self._crypt_chunk_size,
            )
            nonces = self._new_nonces(chunked.count)
            w.encryptor = _crypt.ChunkedEncryption(
                key=self._encryption.key,
                buffer=tensor_memory,
                chunk_size=self._crypt_chunk_size,
                nonces=nonces,
                executor=self._computation_pool,
            )

            key_derivation_chunk = self._encryption._crypt_info_chunk()
            encryption_algorithm_chunk = _crypt_info.XSalsa20ParallelChunk(
                chunk_size=self._crypt_chunk_size,
                nonce=nonces[0],
                macs=w.encryptor.macs,
            )
            chunks: Sequence[Any]
            if key_derivation_chunk is not None:
                chunks = (key_derivation_chunk, encryption_algorithm_chunk)
            else:
                chunks = (encryption_algorithm_chunk,)
            w.crypt_info = _crypt_info.CryptInfo(chunks)

    def _prepare_for_write_headers(
        self, write_specs: Sequence[_WriteSpec]
    ) -> None:
        # We first need to construct the headers so that we know the size of each
        for w in write_specs:
            dtype_bytes = w.dtype.encode("utf-8")  # type: ignore
            if len(dtype_bytes) >= 256:
                raise ValueError("dtype name length should be less than 256")

            w.header = _TensorHeaderSerializer(
                w.module_index,
                w.tensor_type,
                w.name.serialized_(),  # name as bytes
                dtype_bytes,
                w.shape,
                w.data_length,
                0,  # bogus file_offset. This gets filled in in build()
                include_crc32=w.include_crc32,
                include_sha256=w.include_sha256,
                crypt_info=w.crypt_info,
            )

        # Specify the offsets for each metadata entry
        file_offset = (
            self._metadata_cur
        )  # position of next metadata entry to write

        ## metadata
        for w in write_specs:
            w.metadata_pos = file_offset
            file_offset += w.header.get_metadata_entry_segment().size

        self._metadata_cur = file_offset
        if self._metadata_end is None:
            self._metadata_end = self._metadata_cur
        elif file_offset > self._metadata_end:
            raise RuntimeError("Metadata block is full. Increase max_tensors")

        ## headers
        if self._header_cur is not None:
            if self._header_cur < file_offset:
                raise RuntimeError("Somehow wrote past metadata block")
            file_offset = self._header_cur

        for w in write_specs:
            w.header.file_offset = file_offset
            file_offset += w.header.size

        self._header_cur = file_offset
        if self._header_end is None:
            self._header_end = self._header_cur
        elif self._header_cur > self._header_end:
            raise RuntimeError("Header block is full. Increase max_tensors")

        ## tensors
        if self._tensor_cur is None:
            # The block of tensor data starts on a page-aligned boundary
            self._tensor_cur = (file_offset + 4095) & ~4095
        else:
            if self._tensor_cur < file_offset:
                raise RuntimeError("Somehow wrote past header block")
            # Each tensor itself begins on an 8-byte aligned boundary
            file_offset = (self._tensor_cur + 7) & ~7

        # file_offset is now where we should start writing tensor data
        for w in write_specs:
            w.header.build(file_offset)  # type: ignore
            file_offset += w.data_length

        self._tensor_cur = file_offset

    def _prepare_for_write_meta(
        self, write_specs: Sequence[_WriteSpec]
    ) -> None:
        for w in write_specs:
            if not w.tensor.is_meta:
                continue
            w.tensor = torch.empty(
                (0,) * w.tensor.ndim, device="cpu", dtype=w.tensor.dtype
            )
            w.data_length = 0
            w.user_owns_tensor_data = False

    def _prepare_for_write_hashes(
        self, write_specs: Sequence[_WriteSpec]
    ) -> None:
        def compute_crc32(
            write_spec: TensorSerializer._WriteSpec,
            dependency: Optional[_Future],
        ):
            if dependency is not None:
                dependency.result(_TIMEOUT)
            header_crc32 = write_spec.header.compute_crc32()
            crc32 = zlib.crc32(
                write_spec.numpy_tensor.tensor_memory, header_crc32
            )
            write_spec.header.add_crc32(crc32)

        def compute_sha256(
            write_spec: TensorSerializer._WriteSpec,
            dependency: Optional[_Future],
        ):
            if dependency is not None:
                dependency.result(_TIMEOUT)
            sha256 = write_spec.header.compute_sha256()
            sha256.update(write_spec.numpy_tensor.tensor_memory)
            write_spec.header.add_sha256(sha256.digest())

        for w in write_specs:
            old_tensor_data_task = w.tensor_data_task

            hash_tasks = []
            if w.include_crc32:
                crc32_task = self._computation_pool.submit(
                    compute_crc32, w, old_tensor_data_task
                )
                hash_tasks.append(crc32_task)
            if w.include_sha256:
                sha256_task = self._computation_pool.submit(
                    compute_sha256, w, old_tensor_data_task
                )
                hash_tasks.append(sha256_task)

            if hash_tasks:
                w.tensor_data_task = _FutureGroup(hash_tasks)
                self._jobs.extend(hash_tasks)

    def _do_encryption(self, write_specs: Sequence[_WriteSpec]) -> None:
        def encrypt(write_spec, dependency: _Future):
            if dependency is not None:
                dependency.result(_TIMEOUT)
            try:
                write_spec.encryptor.encrypt_all(wait=True, timeout=_TIMEOUT)
            except _crypt.CryptographyError as e:
                raise CryptographyError("Tensor encryption failed") from e
            write_spec.header.update_crypt_info()

        for w in write_specs:
            if not w.data_length:
                continue
            w.tensor_data_task = self._encryption_pool.submit(
                encrypt, w, w.tensor_data_task
            )
            self._jobs.append(w.tensor_data_task)

    def _do_commit_headers(self, write_specs: Sequence[_WriteSpec]) -> None:
        # TODO: this is lots of tiny writes. Buffer them for performance
        def commit_header(write_spec, dependency: _Future):
            if dependency is not None:
                dependency.result(_TIMEOUT)
            self._pwrite(
                write_spec.header.metadata_entry,
                write_spec.metadata_pos,
                verify=len(write_spec.header.metadata_entry),
            )
            self._pwrite(
                write_spec.header.buffer,
                write_spec.header.file_offset,
                verify=write_spec.header.size,
            )

        metadata_size = (
            self._metadata_cur - self._metadata_start - 8
        )  # 8 bytes for metadata length field
        metadata_size_task = self._header_writer_pool.submit(
            self._pwrite,
            struct.pack("<Q", metadata_size),
            self._metadata_start,
            verify=8,
        )
        self._jobs.append(metadata_size_task)

        for w in write_specs:
            commit_header_task = self._header_writer_pool.submit(
                commit_header, w, w.tensor_data_task
            )
            # Note this does _not_ set w.tensor_data_task, as committing headers is safe
            self._jobs.append(commit_header_task)

    def _do_commit_tensor_data(self, write_specs: Sequence[_WriteSpec]):
        def commit_tensor_data(
            write_spec: TensorSerializer._WriteSpec,
            dependency: Optional[_Future],
        ):
            if dependency is not None:
                dependency.result(_TIMEOUT)
            if write_spec.header.data_length == 0:
                bytes_written = 0
            else:
                bytes_written = self._pwrite(
                    write_spec.numpy_tensor.tensor_memory,
                    write_spec.header.data_offset,  # type: ignore
                    verify=write_spec.header.data_length,  # type: ignore
                )

            with self._tensor_count_update_lock:  # TODO: get rid of this?
                self._file_header.tensor_count += 1
                self._file_header.tensor_size += bytes_written

        for w in write_specs:
            w.tensor_data_task = self._writer_pool.submit(
                commit_tensor_data, w, w.tensor_data_task
            )
            self._jobs.append(w.tensor_data_task)

    def _maybe_decrypt_data(self, write_specs: Sequence[_WriteSpec]):
        def decrypt(
            write_spec: TensorSerializer._WriteSpec,
            dependency: Optional[_Future],
        ):
            try:
                if dependency is not None:
                    dependency.result(_TIMEOUT)
            finally:
                # Try to decrypt again even if writing to disk failed
                # to avoid exiting with the tensor memory in a modified state
                fs = write_spec.encryptor.decrypt_all(wait=False)  # type: ignore
                try:
                    _crypt.ChunkedEncryption.wait_or_raise(
                        fs,
                        timeout=_TIMEOUT,
                        return_when=concurrent.futures.ALL_COMPLETED,
                    )
                except _crypt.CryptographyError as e:
                    try:
                        original_exc = (
                            dependency.exception(timeout=0)
                            if dependency is not None
                            else None
                        )
                    except (
                        concurrent.futures.TimeoutError,
                        concurrent.futures.CancelledError,
                    ):
                        original_exc = None
                    raise CryptographyError(
                        "Restoring encrypted tensor data in memory failed"
                    ) from (original_exc if original_exc is not None else e)

        for w in write_specs:
            if not w.user_owns_tensor_data:
                continue
            w.tensor_data_task = self._decryption_pool.submit(
                decrypt, w, w.tensor_data_task
            )
            self._jobs.append(w.tensor_data_task)


def _get_perf_stats():
    if _perf_stats is None:
        raise RuntimeError("Performance stats are not enabled")
    with _perf_stats.lock:
        return dict(
            tensor_to_device_secs=_perf_stats.tensor_to_device_ns * 1e-9,
            tensor_to_device_bytes=_perf_stats.tensor_to_device_bytes,
            file_readinto_secs=_perf_stats.file_readinto_ns * 1e-9,
            file_readinto_bytes=_perf_stats.file_readinto_bytes,
        )

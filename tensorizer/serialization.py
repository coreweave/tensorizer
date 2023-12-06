##############################################################################
# serialization.py                                                   Wes Brown
# Fast torch module/model serialization/deserialization     (c) 2023 Coreweave
##############################################################################
import abc
import collections.abc
import concurrent.futures
import contextlib
import ctypes
import dataclasses
import enum
import functools
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
import weakref
import zlib
from collections import OrderedDict
from enum import Enum
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
import redis
import torch

import tensorizer._crypt as _crypt
import tensorizer._crypt_info as _crypt_info
import tensorizer.stream_io as stream_io
import tensorizer.utils as utils
from tensorizer._crypt._cgroup_cpu_count import (
    effective_cpu_count as _effective_cpu_count,
)
from tensorizer._internal_utils import Chunked as _Chunked
from tensorizer._internal_utils import _variable_read
from tensorizer._NumpyTensor import _NumpyTensor
from tensorizer.stream_io import CURLStreamFile

if torch.cuda.is_available():
    cudart = torch.cuda.cudart()
else:
    cudart = None

lz4 = None

__all__ = [
    "TensorSerializer",
    "TensorDeserializer",
    "TensorType",
    "CryptographyError",
    "EncryptionParams",
    "DecryptionParams",
]

# Setup logger
logger = logging.getLogger(__name__)


# Get CPU count
cpu_count: int = _effective_cpu_count()


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
class TensorType(Enum):
    PARAM = 0
    BUFFER = 1
    STATE_DICT = 2


# If tensors with "opaque" dtypes (those that are not supported by numpy) are
# saved, then a tensorizer data version of 2 is required to (de)serialize the
# file. Otherwise, the file is compatible with tensorizer data version 1
TENSORIZER_VERSION = 3
OPAQUE_TENSORIZER_VERSION = 2
NON_OPAQUE_TENSORIZER_VERSION = 1

TENSORIZER_MAGIC = b"|TZR|"

OPAQUE_DTYPE_SEP = "\0"

_TIMEOUT: typing.Final[int] = 3600


class HashType(Enum):
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
    name: str
    type: TensorType
    dtype: str
    shape: Tuple[int, ...]
    offset: int
    data_offset: int
    data_length: int
    hashes: Optional[List[TensorHash]]
    header_hashes: Optional[Dict[HashType, Any]]


@dataclasses.dataclass
class _FileHeader:
    __slots__ = ("version_number", "tensor_size", "tensor_count")
    version_number_format: ClassVar[struct.Struct] = struct.Struct(
        "<I"  # Little-endian version number
    )
    format: ClassVar[struct.Struct] = struct.Struct(
        "<"
        "32x"  # File hash (unused)
        "Q"  # Total size of tensor data (nominally, total file size)
        "8x"  # Nominally, total size of tensor data (actually unused)
        "Q"  # Total number of tensors
    )
    version_number: int
    tensor_size: int
    tensor_count: int

    class InvalidVersionError(ValueError):
        version: int

        def __init__(self, *args, version: int):
            super().__init__(*args)
            self.version = version

    def to_bytes(self) -> bytes:
        return self.version_number_format.pack(
            self.version_number
        ) + self.format.pack(self.tensor_size, self.tensor_count)

    @classmethod
    def from_io(
        cls, reader: io.BufferedIOBase, accepted_versions: Sequence[int]
    ) -> "_FileHeader":
        version_number = cls.version_number_format.unpack(
            reader.read(cls.version_number_format.size)
        )[0]
        if version_number not in accepted_versions:
            message = (
                "Unsupported version: this data stream uses tensorizer"
                f" data version {version_number}, which is not supported"
                " in this release of tensorizer, or"
                " for the serialization/deserialization features selected."
                f"\nSupported data versions: {tuple(accepted_versions)}"
            )
            raise cls.InvalidVersionError(message, version=version_number)
        data = reader.read(cls.format.size)
        if len(data) < cls.format.size:
            raise ValueError(
                "File too small: ran out of data before reading a full header"
            )
        return cls(version_number, *cls.format.unpack(data))


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

    @classmethod
    def decode(cls):
        pass

    def __init__(
        self,
        module_index: int,
        tensor_type: TensorType,
        name: bytes,
        dtype: bytes,
        shape: Sequence[int],
        data_length: int,
        file_offset: int,
        include_crc32: bool = True,
        include_sha256: bool = True,
        crypt_info: Optional[_crypt_info.CryptInfo] = None,
    ):
        # Calculate the variable length segment
        name_len = len(name)
        dtype_len = len(dtype)
        # NB: shape_len is the number of dimensions,
        # not the encoded byte length
        shape_len = len(shape)
        self.crypt_info = crypt_info
        if crypt_info is None:
            crypt_info_len = 0
        else:
            crypt_info_len = crypt_info.sized_size
        self.variable_length_segment = struct.Struct(
            self.variable_length_segment_template.format(
                name_len=name_len,
                dtype_len=dtype_len,
                shape_len=shape_len,
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
            self.data_offset,
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

        after_hashes = self.crypt_info_offset
        self.hash_segment_size = after_hashes - self.hash_header_offset - 2
        self.hash_header_segment.pack_into(
            self.buffer,
            self.hash_header_offset,
            self.hash_segment_size,  # Hash section length
            self.hash_count,  # Hash count
        )

        # Placeholders
        if include_crc32:
            self.add_crc32(0)
        if include_sha256:
            self.add_sha256(b"")
        if crypt_info is not None:
            crypt_info.sized_pack_into(self.buffer, self.crypt_info_offset)

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

    def _hashable_segment_views(self):
        if self.crypt_info is None:
            yield memoryview(self.buffer)
        else:
            yield memoryview(self.buffer)[: self.crypt_info_offset]
            # Skip crypt_info
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
    name: str
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
            self.name: str = str(name_slice, "utf-8")

        # Read the dtype of the tensor.
        dtype_slice, offset = self.read_dtype(buffer, offset)
        with dtype_slice:
            self.dtype: str = str(dtype_slice, "utf-8")

        # Read the shape.
        self.shape, offset = self.read_shape(buffer, offset)

        # Read our hashes in.
        hashes_slice, offset = self.read_hash_block(buffer, offset)
        with hashes_slice:
            self.hashes = self._decode_hashes(hashes_slice)
            if zero_hashes:
                self._zero_hashes(hashes_slice)

        if check_crypt_info:
            crypt_info_start = offset
            crypt_info_slice, offset = self.read_crypt_info_block(
                buffer, offset
            )
            self._hashable_segments = (
                slice(None, crypt_info_start),
                slice(offset, None),
            )
            with crypt_info_slice:
                self.crypt_info = _crypt_info.CryptInfo.unpack_from(
                    crypt_info_slice
                )
        else:
            self.crypt_info = None
            self._hashable_segments = (slice(None, None),)

        # Finally, get the tensor data length.
        offset = len(buffer) - self.data_length_segment.size
        self.data_length = self.data_length_segment.unpack_from(buffer, offset)[
            0
        ]

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
    ) -> Tuple["_MetadataDeserializer", bytes]:
        raw = reader.read(cls._total_len_segment.size)
        total_len: int = cls._total_len_segment.unpack(raw)[0]
        if total_len == 0:
            return cls(), raw
        else:
            encoded_metadata: bytes = reader.read(total_len)
            raw += encoded_metadata
            return cls.from_buffer(encoded_metadata, count), raw

    @classmethod
    def from_buffer(cls, buffer: bytes, count: int) -> "_MetadataDeserializer":
        offset = 0
        entries = cls()
        for i in range(count):
            entry, offset = cls._read_entry(buffer, offset)
            entries[entry.name] = entry
        return entries

    @classmethod
    def _read_entry(cls, buffer: bytes, offset: int) -> Tuple[TensorEntry, int]:
        # Read the name.
        name_slice, offset = cls._read_name(buffer, offset)
        with name_slice:
            name: str = str(name_slice, "utf-8")

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
        def chunk(self) -> _crypt_info.KeyDerivationChunk:
            ...

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

        class OpsLimit(enum.IntEnum):
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

        class MemLimit(enum.IntEnum):
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

        class OpsLimit(enum.IntEnum):
            def __getattribute__(self, item):
                super().__getattribute__(self, item)
                _require_libsodium()

        class MemLimit(enum.IntEnum):
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
        plaid_mode_buffers: The number of buffers to use in plaid mode. This
            is only used if ``plaid_mode=True``. These buffers are used to
            pipeline the loading and processing of tensors.
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
        plaid_mode_buffers: Optional[int] = None,
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

            if isinstance(file_obj, (str, bytes, os.PathLike, int)):
                self._file = stream_io.open_stream(file_obj, "rb")
            else:
                self._mode_check(file_obj)
                self._file = file_obj
            self._cleanup.callback(self._file.close)
            self.total_compressed_tensor_bytes = 0
            self.read_bytes = 0

            # If device is None, use the current device, otherwise use the given
            # device.
            device = (
                utils.get_device() if device is None else torch.device(device)
            )
            self._device: torch.device = device
            is_cuda = self._device.type == "cuda"
            if is_cuda and not torch.cuda.is_available():
                raise RuntimeError(
                    "Cannot deserialize to CUDA device"
                    " because CUDA is not available"
                )

            self._dtype: Optional[torch.dtype] = dtype

            self._plaid_mode: bool = plaid_mode

            self._lazy_load: bool = lazy_load

            self._metadata: Dict[str, TensorEntry] = {}

            if self._plaid_mode and not is_cuda:
                raise ValueError("Plaid mode requires CUDA")

            # Read the magic
            magic = self._file.read(5)
            if magic != TENSORIZER_MAGIC:
                raise ValueError("Not a tensorizer file")

            # Read the file header
            if self._encrypted:
                accepted_versions = (TENSORIZER_VERSION,)
            else:
                accepted_versions = (
                    NON_OPAQUE_TENSORIZER_VERSION,
                    OPAQUE_TENSORIZER_VERSION,
                    TENSORIZER_VERSION,
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

            self._has_crypt_info: bool = (
                self._file_header.version_number >= TENSORIZER_VERSION
            )

            # The total size of the file.
            # WARNING: this is not accurate. This field isn't used in the
            # deserializer, but has been available as a public attribute,
            # so it is kept how it was for compatibility until the next
            # major version.
            self.total_file_bytes = self._file_header.tensor_size

            # Read the metadata index of tensors.
            # This is a list of offsets into the file where the per-tensor data
            # is stored.
            self._metadata, self._metadata_raw = _MetadataDeserializer.from_io(
                self._file, self._file_header.tensor_count
            )
            if not self._metadata:
                raise ValueError("Tensor index in the file is empty")
            # filter_func is a test that determines the tensor names to read.
            # If filter_func is None, all tensors are read.
            if filter_func is not None:
                self._metadata = {
                    name: entry
                    for name, entry in self._metadata.items()
                    if filter_func(name)
                }

            # We calculate the total tensor bytes here so that we can use mmap,
            # based on the total size of the tensors that we're going to read,
            # filtered by the filter_func.
            tensor_sizes = {
                name: entry.data_length
                for name, entry in self._metadata.items()
            }
            self.total_tensor_bytes = sum(tensor_sizes.values())
            if not self._plaid_mode and plaid_mode_buffers is not None:
                raise ValueError(
                    "Cannot specify plaid_mode_buffers when plaid_mode=False"
                )
            if plaid_mode_buffers is not None:
                self._plaid_mode_buffer_count = plaid_mode_buffers
            elif self._verify_hash:
                self._plaid_mode_buffer_count = 8
            elif isinstance(self._file, CURLStreamFile):
                self._plaid_mode_buffer_count = 1
            else:
                self._plaid_mode_buffer_count = 2
            single_largest_tensor = max(tensor_sizes.values(), default=0)
            # Round up to the nearest multiple of the page size
            # Just so that more reads happen on page boundaries
            single_largest_tensor -= single_largest_tensor % -mmap.PAGESIZE
            # Sizes for plaid mode buffers, only allocated if plaid mode
            # is actually being used
            self._plaid_mode_buffers = (
                single_largest_tensor,
            ) * self._plaid_mode_buffer_count

            self._buffers = {}

            # Allocate the buffer for the tensors.
            # Check if our platform supports mmap.MAP_ANONYMOUS and
            # mmap.MAP_PRIVATE
            mmap_flags = 0
            mmap_flags |= getattr(mmap, "MAP_PRIVATE", 0)
            mmap_flags |= getattr(mmap, "MAP_ANONYMOUS", 0)
            mmap_args = {"flags": mmap_flags} if mmap_flags else {}
            anonymous_mmap = partial(mmap.mmap, -1, **mmap_args)

            start_allocate = time.monotonic()
            if self._plaid_mode:
                # Allocate a buffer big enough to fit any tensor,
                # and pin it later
                total_plaid_mode_buffer_size = sum(self._plaid_mode_buffers)
                self._buffer = anonymous_mmap(total_plaid_mode_buffer_size)
                # Track which buffers overlap for later concurrent access
                self._buffer_ids = {}
                # Sub-buffers alternate between which segment they use
                with memoryview(self._buffer) as mv:
                    starts = (
                        0,
                        *tuple(itertools.accumulate(self._plaid_mode_buffers))[
                            :-1
                        ],
                    )
                    for (name, size), start in zip(
                        tensor_sizes.items(), itertools.cycle(starts)
                    ):
                        end = start + size
                        self._buffers[name] = mv[start:end]
                        self._buffer_ids[name] = start
            elif not self._lazy_load:
                # Eager loading mode
                # Allocate a single buffer for all the tensors, and pin it later
                self._buffer = anonymous_mmap(self.total_tensor_bytes)
                # Mark sub-buffer locations
                with memoryview(self._buffer) as mv:
                    sub_buffer_start = 0
                    for name, size in tensor_sizes.items():
                        sub_buffer_end = sub_buffer_start + size
                        self._buffers[name] = mv[
                            sub_buffer_start:sub_buffer_end
                        ]
                        sub_buffer_start = sub_buffer_end
            else:
                # Lazy loading mode
                # mmap objects usually reserve memory without committing it
                # so any of these allocation strategies could be lazy,
                # But the other modes pin memory which forces pre-allocation.
                # The allocation strategy used here is to reserve memory with a
                # separate mmap for each potential tensor, which allows granular
                # de-allocation/garbage collection and easy madvise calls
                self._buffer = None
                for name, size in tensor_sizes.items():
                    self._buffers[name] = anonymous_mmap(size)

            # Register cleanup callbacks for buffers and views
            if is_cuda:
                # Buffers shouldn't be cleaned up for CPU tensors
                # because they will still be in use after deserialization.
                # For CUDA tensors, it is just a staging area.
                if hasattr(self._buffer, "close"):
                    self._cleanup.enter_context(self._buffer)
                for name, buffer in self._buffers.items():
                    self._cleanup.enter_context(buffer)

            # If we're on CUDA, and not performing lazy memory allocation,
            # register the buffer with CUDA so that it is pinned.
            # This allows for PyTorch to internally use cudaMemcpyAsync.
            if is_cuda and (self._plaid_mode or not self._lazy_load):
                # We need to use ctypes to get the address of the buffer
                # because mmap.mmap doesn't expose the buffer address.
                tb = ctypes.c_char * len(self._buffer)
                ctb = tb.from_buffer(self._buffer)
                buffer_addr = ctypes.addressof(ctb)
                # Don't leave an open exported pointer into the mmap
                del ctb

                # Register the buffer with CUDA
                cudart.cudaHostRegister(buffer_addr, len(self._buffer), 0)
                self._cleanup.callback(cudart.cudaHostUnregister, buffer_addr)
            end_allocate = time.monotonic()
            tensor_bytes_str = utils.convert_bytes(self.total_tensor_bytes)
            logger.debug(
                f"Allocated {tensor_bytes_str} "
                f"for {len(self._metadata)} tensors "
                f"in {end_allocate - start_allocate:0.4f}"
            )

            # The number of bytes we've allocated so far. Tensors may be read
            # from the file in any order, so we need to keep track of how much
            # we've used so far so that we can index into the buffer correctly.
            self._allocated = 0

            # Our cache of tensors. This is a dict of name -> tensor.
            # If lazy_load is True, then the tensors are not loaded until they
            # are accessed.
            self._cache: typing.OrderedDict[
                str, Union[torch.Tensor, None, bool]
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
            return self._file.bytes_read
        if self._file.closed:
            # Caution: This case is an underestimate because it doesn't include
            # any metadata read, unlike the other two cases.
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

    def _read_single_tensor(
        self, expected_name: str, *args, **kwargs
    ) -> torch.Tensor:
        tensors = tuple(self.read_tensors(*args, **kwargs, num_tensors=1))
        num_tensors = len(tensors)
        if num_tensors == 0:
            raise RuntimeError("Tensor not found")
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

    def __getitem__(self, name) -> torch.nn.Parameter:
        if name in self._cache and self._cache[name] is not None:
            return self._cache[name]

        # If we're in lazy_load mode, we populate the cache with the
        # tensor data and then convert it to a torch parameter. Most
        # of the time, access patterns are front to back, so seeking
        # forward in a stream works well even for HTTP/HTTPS streams.
        if name in self._metadata:
            self._file.seek(self._metadata[name].offset)
            tensor = self._read_single_tensor(name)
            self._cache[name] = self._to_torch_parameter(tensor)
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
    def _verify_hashes(
        name: str,
        hashes: Iterable[TensorHash],
        header_hashes: Dict[HashType, Any],
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

    def _get_decryption_manager(
        self, encryption_method: _crypt_info.CryptInfoChunk, key: bytes, buffer
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
                    executor=self._decryption_pool,
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

    def _stream_decrypt(
        self, encryption_method: _crypt_info.CryptInfoChunk, key: bytes, buffer
    ):
        try:
            with self._get_decryption_manager(
                encryption_method, key, buffer
            ) as crypto:
                if isinstance(crypto, _crypt.ChunkedEncryption):
                    fs = []
                    for chunk in range(crypto.num_chunks):
                        with crypto.chunk_view(chunk) as view:
                            self._file.readinto(view)
                        fs.append(crypto.decrypt_chunk(chunk))
                    crypto.wait_or_raise(fs, timeout=_TIMEOUT)
                else:
                    self._file.readinto(buffer)
                    crypto.decrypt()
        except _crypt.CryptographyError as e:
            raise CryptographyError("Tensor decryption failed") from e
        finally:
            del crypto

    @staticmethod
    @contextlib.contextmanager
    def _release_on_exc(mv: memoryview):
        try:
            yield mv
        except GeneratorExit:
            del mv
            raise
        except BaseException:
            mv.release()
            del mv
            raise

    def _read_numpytensors(
        self,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        num_tensors: int = -1,
        verify_hash: Optional[bool] = None,
        raw: bool = False,
    ) -> Iterator[Tuple[int, int, str, Union[_NumpyTensor, memoryview]]]:
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
                header = _TensorHeaderDeserializer.from_io(
                    self._file,
                    zero_hashes=True,
                    check_crypt_info=self._has_crypt_info,
                )

                if header is None:
                    break

                # Check if the name is in our pre-filtered list of keys
                # from the class-level filter_func, and then verify
                # that it passes the method-level filter_func.
                # Skip it if it fails either check.
                if header.name not in self.keys() or (
                    filter_func is not None and not filter_func(header.name)
                ):
                    self._file.seek(header.data_length, io.SEEK_CUR)
                    continue

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

                self._metadata[header.name].hashes = header.hashes

                header_hashes = header.compute_hashes()
                self._metadata[header.name].header_hashes = header_hashes

                is_encrypted: bool = (
                    header.crypt_info is not None
                    and header.crypt_info.num_chunks != 0
                )
                if self._encrypted and not is_encrypted:
                    raise CryptographyError(
                        "Tensor is not encrypted, but decryption was requested"
                    )
                elif is_encrypted and not self._encrypted:
                    raise CryptographyError(
                        "Tensor is encrypted, but decryption was not requested"
                    )
                elif self._encrypted or is_encrypted:
                    assert self._encrypted and is_encrypted
                    encryption_method = self._get_encryption_method(
                        header.crypt_info
                    )
                    key = self._derive_encryption_key(header.crypt_info)
                else:
                    key = None
                    encryption_method = None

                # We use memoryview to avoid copying the data.
                mv: memoryview
                if not self._plaid_mode and not self._lazy_load:
                    # In default mode, we've already allocated all the
                    # memory we need in a single buffer that contains
                    # all the tensors. We just need to slice out the
                    # memoryview for the current tensor.
                    mv = self._buffers[header.name]
                    self._allocated += header.data_length
                elif self._plaid_mode:
                    # In plaid_mode, we don't allocate a buffer, we just
                    # reuse the same one. This is a filthy hack, as we're
                    # overwriting the buffer contents that is used to back
                    # the prior tensor. This works because we don't use
                    # the buffer contents after we yield the tensor, which
                    # is loaded straight into the GPU memory.
                    mv = self._buffers[header.name]
                else:
                    # In lazy_load mode, we allocate a new buffer for each
                    # tensor. This is a bit slower, but it's the only way
                    # to support lazy loading.
                    buffer = self._buffers[header.name]
                    if len(buffer) != header.data_length:
                        raise RuntimeError("Header data length mismatch")
                    mv = memoryview(buffer)

                if not self._encrypted or mv.nbytes == 0:
                    self._file.readinto(mv)
                elif self._encrypted and mv.nbytes > 0:
                    with self._release_on_exc(mv):
                        self._stream_decrypt(encryption_method, key, mv)

                if verify_hash:
                    with self._release_on_exc(mv):
                        # Releasing on an exception is necessary to prevent
                        # a BufferError on close()
                        self._verify_hashes(
                            header.name, header.hashes, header_hashes, mv
                        )

                if raw:
                    tensor = mv
                else:
                    tensor = _NumpyTensor.from_buffer(
                        numpy_dtype,
                        torch_dtype,
                        header.shape,
                        mv,
                    )

                tensors_read += 1

                yield header.module_idx, header.tensor_type, header.name, tensor
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
            and tensor.dtype != self._dtype
        ):
            target_dtype = self._dtype
        else:
            target_dtype = None

        gradient = tensor.dtype.is_complex or tensor.dtype.is_floating_point

        return torch.nn.Parameter(
            tensor.to(device=self._device, dtype=target_dtype),
            requires_grad=gradient,
        )

    def _generate_state_dict(self) -> None:
        """
        Load the tensors in this Tensorizer object into a state_dict. This
        is used to populate the cache in non-lazy_load cases.
        """
        if self._file.closed:
            raise IOError("IO closed, instantiate if you want to load again.")

        self._cache = OrderedDict()
        keys = tuple(self.keys())
        bulk_loader = self._bulk_load(keys)
        with contextlib.closing(bulk_loader):
            for _ in bulk_loader:
                # Just run this for the caching side effect
                pass
        # for idx, typ, name, arr in self.read_tensors():
        #     d[name] = self._to_torch_parameter(arr)
        self.total_tensor_bytes = self._file.tell()
        self._file.close()

    @contextlib.contextmanager
    def _optimize_plaid_mode_buffers(
        self, keys: Iterable[str]
    ) -> Generator[None, None, None]:
        """
        Optimize sub-buffers to alternate between which segment of the shared
        plaid mode buffer they use, so that more operations can overlap at once.
        This can't be done during ``__init__`` because the alternating pattern
        has to match the exact list of keys given, which could be smaller
        than the total list of keys due to a filter_func.

        Temporarily replaces self._buffers while the context manager is active.
        Releases the created sub-buffers when the context manager is exited.

        Args:
            keys: The list of keys being loaded.
        """
        if not self._plaid_mode:
            yield
            return
        with contextlib.ExitStack() as exit_stack:
            tensor_sizes = [
                (key, self._metadata[key].data_length) for key in keys
            ]
            optimized_buffers = self._buffers.copy()
            with memoryview(self._buffer) as mv:
                starts = (
                    0,
                    *tuple(itertools.accumulate(self._plaid_mode_buffers))[:-1],
                )
                for (name, size), start in zip(
                    tensor_sizes, itertools.cycle(starts)
                ):
                    end = start + size
                    optimized_buffers[name] = exit_stack.enter_context(
                        mv[start:end]
                    )

            unoptimized_buffers = self._buffers
            self._buffers = optimized_buffers
            try:
                yield
            finally:
                self._buffers = unoptimized_buffers

    class _AtomicCountdown:
        __slots__ = ("_count", "_condition", "_cancelled", "_initial")

        def __init__(self, count: int, initial: Optional[int] = None):
            if count <= 0:
                raise ValueError("Invalid count.")
            self._condition = threading.Condition()
            self._initial = self._count = count
            if initial is not None:
                self._count = initial
            self._cancelled = False

        def _is_done(self):
            return self._count == 0 or self._cancelled

        def wait(self, timeout: float = None) -> bool:
            """
            Waits for the internal counter to hit zero,
            but doesn't decrement the counter itself.
            """
            with self._condition:
                self._condition.wait_for(self._is_done, timeout=timeout)
                return not self._cancelled

        def trigger(self) -> None:
            """
            Decrements the internal counter and then returns.
            Does not wait for the counter to reach zero.
            """
            with self._condition:
                if self._count > 0:
                    self._count -= 1
                if self._count == 0:
                    self._condition.notify_all()

        def reset(self, count: Optional[int] = None) -> None:
            """Resets the internal counter."""
            if count is not None and count <= 0:
                raise ValueError("Invalid count.")
            with self._condition:
                self._count = self._initial if count is None else count
                self._cancelled = False

        def cancel(self) -> None:
            with self._condition:
                self._cancelled = True
                self._condition.notify_all()

    def _bulk_load(
        self, keys: Iterable[str], verify_hash: Optional[bool] = None
    ) -> Generator[torch.nn.Parameter, None, None]:
        keys: Tuple[str, ...] = tuple(keys)
        # Quick route for no keys
        if not keys:
            return
        if verify_hash is None:
            verify_hash = self._verify_hash

        for key in keys:
            if key not in self:
                raise KeyError(f"Invalid key: {key}")

        # Quick route for all cached keys
        if all(self._cache.get(key) is not None for key in keys):
            yield from map(self._cache.get, keys)
            return

        # Splice cached values with freshly loaded values
        # when some are cached and some are not by using a recursive call
        if any(self._cache.get(key) is not None for key in keys):
            uncached = {
                i: k for i, k in enumerate(keys) if self._cache.get(k) is None
            }
            loader = self._bulk_load(uncached.values(), verify_hash)
            with contextlib.closing(loader):
                for i, k in enumerate(keys):
                    if i in uncached:
                        yield next(loader)
                    else:
                        yield self._cache[k]
                if tuple(loader):
                    raise RuntimeError("Loaded too many tensors")
            return

        # If this function is later exposed through a public method, add another
        # case here to optimize for unsorted keys. Otherwise, unsorted
        # keys become disastrously slow. It could sort the list internally,
        # read them all, unsort them, and then yield them all in their original
        # order.

        # Quick route for a single key
        if len(keys) == 1:
            old_verify_hash, self._verify_hash = self._verify_hash, verify_hash
            try:
                item = self.get(keys[0])
            finally:
                self._verify_hash = old_verify_hash
            yield item
            return

        # Main route for multiple keys
        transfer_in_queue = queue.SimpleQueue()
        transfer_out_queue = queue.SimpleQueue()
        sentinel = object()

        max_num_hash_tasks = 2
        tasks_per_tensor = 1
        if verify_hash:
            tasks_per_tensor += max_num_hash_tasks

        atomic_countdown: typing.Type = TensorDeserializer._AtomicCountdown
        if self._plaid_mode:
            countdowns = tuple(
                atomic_countdown(tasks_per_tensor, initial=0)
                for _ in range(self._plaid_mode_buffer_count)
            )
            countdown_cycle = itertools.cycle(countdowns)
        else:
            countdown_cycle = itertools.cycle((None,))

        class Cancelled(BaseException):
            # Derive from BaseException to avoid being caught by any
            # generic "except Exception" clauses; these need to force an exit
            pass

        def cancel_thread(thread: threading.Thread):
            thread_id = ctypes.c_long(thread.ident)
            exc = ctypes.py_object(Cancelled)
            return (
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, exc) == 1
            )

        def receive_and_check(timeout: int) -> torch.nn.Parameter:
            outcome = transfer_out_queue.get(timeout=timeout)
            if outcome is sentinel:
                raise RuntimeError("Loading failed")
            return outcome

        def transfer() -> None:
            countdown = None
            is_cuda = self._device.type == "cuda"
            try:
                stream = torch.cuda.Stream() if is_cuda else None
                with (
                    torch.cuda.stream(stream)
                    if is_cuda
                    else contextlib.nullcontext()
                ):
                    while True:
                        next_tensor, countdown = transfer_in_queue.get(
                            timeout=_TIMEOUT
                        )
                        next_tensor: torch.Tensor
                        countdown: Optional[atomic_countdown]
                        if next_tensor is sentinel:
                            break
                        try:
                            transfer_out_queue.put(
                                self._to_torch_parameter(next_tensor),
                                timeout=_TIMEOUT,
                            )
                        finally:
                            if countdown is not None:
                                countdown.trigger()
                    if stream is not None:
                        stream.synchronize()
            except (Cancelled, Exception) as e:
                transfer_out_queue.put_nowait(sentinel)
                if not isinstance(e, Cancelled):
                    # Don't print error tracebacks for Cancelled exceptions
                    # since those are just side effects of other exceptions
                    raise
            finally:
                if countdown is not None:
                    countdown.cancel()

        def ready_buffers(
            key_list: Iterable[str],
        ) -> Generator[str, None, None]:
            # Prime buffers using madvise slightly before they are needed,
            # and return their names.
            # This function is a good candidate to adapt for other types of
            # just-in-time buffer allocation in the future.
            can_madvise = (
                self._buffers
                and hasattr(next(iter(self._buffers.values())), "madvise")
                and hasattr(mmap, "MADV_WILLNEED")
            )
            if not can_madvise:
                # If madvise is not possible, just yield the existing buffers
                yield from key_list
                return
            out_queue = collections.deque(maxlen=2)
            for buf_name in key_list:
                self._buffers[buf_name].madvise(mmap.MADV_WILLNEED)
                if len(out_queue) == out_queue.maxlen:
                    yield out_queue.popleft()
                out_queue.append(buf_name)
            yield from out_queue

        stop: bool = False

        if verify_hash:
            computation_threads = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(keys) * max_num_hash_tasks, cpu_count),
                thread_name_prefix="TensorizerComputation",
            )
        else:
            computation_threads = None

        def check_hash(
            metadata: TensorEntry,
            data,
            hashes: Iterable[TensorHash],
            countdown: Optional[atomic_countdown],
        ) -> None:
            with memoryview(data).cast("B") as mv:
                try:
                    self._verify_hashes(
                        metadata.name, hashes, metadata.header_hashes, mv
                    )
                finally:
                    if countdown is not None:
                        countdown.trigger()

        def check_hashes(
            tensor_name: str,
            tensor: torch.Tensor,
            countdown: Optional[atomic_countdown],
        ) -> None:
            entry: TensorEntry = self._metadata[tensor_name]
            storage = tensor.untyped_storage()
            data = ctypes.cast(
                storage.data_ptr(),
                ctypes.POINTER(ctypes.c_ubyte * storage.nbytes()),
            ).contents
            for h in entry.hashes:
                checks.append(
                    computation_threads.submit(
                        check_hash, entry, data, (h,), countdown
                    )
                )

        checks: List[concurrent.futures.Future] = []

        def read_into_buffers() -> None:
            buffers = ready_buffers(keys)
            try:
                with contextlib.closing(buffers):
                    for name, countdown in zip(buffers, countdown_cycle):
                        countdown: Optional[atomic_countdown]
                        if stop:
                            break
                        metadata: TensorEntry = self._metadata[name]
                        self._file.seek(metadata.offset)
                        if countdown is not None and not countdown.wait(
                            _TIMEOUT
                        ):
                            break
                        tensor = self._read_single_tensor(
                            name, verify_hash=False
                        )
                        if stop:
                            break
                        hashing_required = computation_threads is not None
                        if countdown is not None:
                            countdown.reset(
                                1 + len(metadata.hashes) * hashing_required
                            )
                        if hashing_required:
                            check_hashes(name, tensor, countdown)
                        transfer_in_queue.put_nowait((tensor, countdown))
            except (Cancelled, Exception) as e:
                transfer_out_queue.put_nowait(sentinel)
                if not isinstance(e, Cancelled):
                    # Don't print error tracebacks for Cancelled exceptions
                    # since those are just side effects of other exceptions
                    raise

        transfer_thread = threading.Thread(
            target=transfer, name="TensorizerTransfer", daemon=True
        )
        read_thread = threading.Thread(target=read_into_buffers, daemon=True)

        with self._optimize_plaid_mode_buffers(keys):
            transfer_thread.start()
            read_thread.start()

            try:
                for key in keys[:-1]:
                    self._cache[key] = receive_and_check(_TIMEOUT)
                    yield self._cache[key]
                # Stop before yielding the final tensor
                # to catch up on hash verification
                read_thread.join(timeout=_TIMEOUT)
                if computation_threads is not None:
                    # At this point, all checks have been added to `checks`
                    computation_threads.shutdown(wait=True)
                    # At this point, all checks have finished
                    for check in checks:
                        # This will raise if any of the checks failed
                        check.result(timeout=_TIMEOUT)
                    checks.clear()
                self._cache[keys[-1]] = receive_and_check(_TIMEOUT)
                transfer_in_queue.put_nowait((sentinel, None))
                transfer_thread.join(timeout=_TIMEOUT)
                yield self._cache[keys[-1]]
            except Exception:
                stop = True
                if computation_threads is not None:
                    computation_threads.shutdown(wait=False)
                if transfer_thread.is_alive():
                    transfer_in_queue.put_nowait((sentinel, None))
                    # A graceful exit is preferred if it can happen
                    # in a reasonable amount of time
                    transfer_thread.join(timeout=4)
                    if transfer_thread.is_alive():
                        cancel_thread(transfer_thread)
                        transfer_thread.join(timeout=_TIMEOUT)
                if read_thread.is_alive():
                    # A graceful exit is again preferred, but not necessary
                    read_thread.join(timeout=2)
                    if read_thread.is_alive():
                        cancel_thread(read_thread)
                        read_thread.join(timeout=_TIMEOUT)
                for check in checks:
                    check.cancel()
                raise

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

        keys = tuple(
            k for k in self.keys() if filter_func is None or filter_func(k)
        )

        tensor_ct = len(keys)

        bulk_loader = self._bulk_load(keys, verify_hash=verify_hash)
        with contextlib.closing(bulk_loader):
            for name, tensor in zip(keys, bulk_loader):
                obj_path, attr = name.rsplit(".", 1)
                module: torch.nn.Module = modules[obj_path]
                entry = self._metadata[name]

                if entry.type is TensorType.PARAM:
                    module.register_parameter(attr, tensor)
                elif entry.type is TensorType.BUFFER:
                    module.register_buffer(attr, tensor)
                elif entry.type is TensorType.STATE_DICT:
                    raise NotImplementedError(
                        "This was serialized using"
                        " TensorSerializer.write_state_dict(), so it cannot be"
                        " loaded using TensorDeserializer.load_into_module()."
                        " Use the TensorDeserializer object directly as a"
                        " state_dict mapping instead."
                    )
                else:
                    raise RuntimeError(f"Invalid tensor type: {entry.type}")

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

        for name in self._metadata:
            entry = self._metadata[name]
            offset = self._metadata[name].offset
            data_offset = self._metadata[name].data_offset
            header_size = data_offset - offset
            self._file.seek(offset)
            header_entry = self._file.read(header_size)
            # Check if the key already exists
            if not force and redis_client.exists(
                f"{key_prefix}:{name}:{offset}"
            ):
                continue
            redis_client.set(f"{key_prefix}:{name}:{offset}", header_entry)
            data_entry = self._file.read(entry.data_length)
            redis_client.set(f"{key_prefix}:{name}:{data_offset}", data_entry)


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
    ) -> None:
        if isinstance(file_obj, (str, bytes, os.PathLike, int)):
            self._file = stream_io.open_stream(file_obj, "wb+")
        else:
            self._mode_check(file_obj)
            self._file = file_obj

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
            max_workers=cpu_count,
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
        self._jobs: List[concurrent.futures.Future] = []
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
        self._file.write(TENSORIZER_MAGIC)

        # Write file header metadata
        if not self._encrypted:
            # Can't tell if OPAQUE_TENSORIZER_VERSION is needed
            # until a tensor is written later with an opaque dtype,
            # so assume it is non-opaque until then.
            version_number = NON_OPAQUE_TENSORIZER_VERSION
        else:
            # File encryption requires a newer tensorizer version
            version_number = TENSORIZER_VERSION
        self._file_header_loc = self._file.tell()
        self._file_header = _FileHeader(
            version_number=version_number,
            tensor_size=0,
            tensor_count=0,
        )
        self._file.write(self._file_header.to_bytes())

        # Reserve 256 KiB for metadata.
        metadata_size = 256 * 1024
        self._file.write(struct.pack("<Q", metadata_size))
        self._metadata_loc = self._file.tell()
        self._file.write(bytes(metadata_size))
        self._flush()
        self._metadata_cur = self._metadata_loc
        self._metadata_end = self._metadata_loc + metadata_size

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

    def _pwrite_syscall(
        self, data, offset: int, verify: Union[bool, int] = True
    ) -> int:
        # This implementation of pwrite uses a Unix syscall, and is safe to
        # run even between normal file writes.
        bytes_written = os.pwrite(self._fd, data, offset)
        if isinstance(verify, int):
            self._verify_bytes_written(bytes_written, verify)
        elif verify:
            self._verify_bytes_written(bytes_written, self._buffer_size(data))
        return bytes_written

    def _pwrite_fallback(
        self, data, offset: int, verify: Union[bool, int] = True
    ) -> int:
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
        if isinstance(verify, int):
            self._verify_bytes_written(bytes_written, verify)
        elif verify:
            self._verify_bytes_written(bytes_written, self._buffer_size(data))
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
        for j in self._jobs:
            j.result(timeout=_TIMEOUT)
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
        _temporary_buffer: bool = False,
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
            if not tensor.is_contiguous():
                _temporary_buffer = True
            numpy_tensor = _NumpyTensor.from_tensor(tensor.contiguous())
        else:
            if (
                isinstance(tensor, numpy.ndarray)
                and not tensor.flags.c_contiguous
                and hasattr(numpy, "ascontiguousarray")
            ):
                numpy_tensor = _NumpyTensor.from_array(
                    numpy.ascontiguousarray(tensor)
                )
                _temporary_buffer = True
            else:
                numpy_tensor = _NumpyTensor.from_array(tensor)

        dtype_name = numpy_tensor.numpy_dtype
        if numpy_tensor.is_opaque:
            # The datatype name needs to contain both the numpy dtype that the
            # data is serialized as and the original torch dtype.
            dtype_name += OPAQUE_DTYPE_SEP + numpy_tensor.torch_dtype
            self._file_header.version_number = max(
                OPAQUE_TENSORIZER_VERSION,
                self._file_header.version_number,
            )

        tensor: numpy.ndarray = numpy_tensor.data
        tensor_memory: memoryview = numpy_tensor.data.data
        tensor_size: int = tensor.nbytes
        if tensor_memory.nbytes != tensor_size:
            raise ValueError(
                f"Cannot serialize tensor {name!r}:"
                f" buffer size of underlying memory ({tensor_memory.nbytes})"
                f" doesn't match reported size ({tensor_size})"
            )
        name_bytes = name.encode("utf-8")
        dtype_bytes = dtype_name.encode("utf-8")
        if len(dtype_bytes) >= 256:
            raise ValueError("dtype name length should be less than 256")
        shape = tensor.shape
        header_pos = self._file.tell() if _start_pos is None else _start_pos

        if self._encrypted:
            chunks = _Chunked(
                total_size=tensor_memory.nbytes,
                chunk_size=self._crypt_chunk_size,
            )
            nonces = self._new_nonces(chunks.count)
            encryptor = _crypt.ChunkedEncryption(
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
                macs=encryptor.macs,
            )
            if key_derivation_chunk is not None:
                chunks = (key_derivation_chunk, encryption_algorithm_chunk)
            else:
                chunks = (encryption_algorithm_chunk,)
            crypt_info = _crypt_info.CryptInfo(chunks)
        else:
            encryptor = None
            if self._file_header.version_number == TENSORIZER_VERSION:
                crypt_info = _crypt_info.CryptInfo()
            else:
                crypt_info = None

        include_crc32: bool = not self._encrypted

        header = _TensorHeaderSerializer(
            idx,
            tensor_type,
            name_bytes,
            dtype_bytes,
            shape,
            tensor_size,
            header_pos,
            include_crc32=include_crc32,
            include_sha256=True,
            crypt_info=crypt_info,
        )

        tensor_pos = header_pos + header.data_offset

        # Add our tensor metadata to the index.
        metadata = header.metadata_entry
        # Check for overflow
        if self._metadata_cur + len(metadata) > self._metadata_end:
            raise RuntimeError("Metadata overflow")

        metadata_pos = self._metadata_cur
        metadata_len = len(metadata)
        self._metadata_cur += metadata_len

        # This task is I/O-bound and has no prerequisites,
        # so it goes into the regular writer pool.
        def write_metadata():
            self._pwrite(metadata, metadata_pos, verify=metadata_len)

        self._jobs.append(self._writer_pool.submit(write_metadata))

        # Calculate the hashes.

        # These two tasks are CPU-bound and don't block the GIL,
        # so they go into the computation thread pool.
        def compute_crc32(prerequisite: Optional[concurrent.futures.Future]):
            if prerequisite is not None:
                prerequisite.result(_TIMEOUT)
            crc32 = header.compute_crc32()
            return zlib.crc32(tensor_memory, crc32)

        def compute_sha256(prerequisite: Optional[concurrent.futures.Future]):
            if prerequisite is not None:
                prerequisite.result(_TIMEOUT)
            sha256 = header.compute_sha256()
            sha256.update(tensor_memory)
            return sha256.digest()

        # This task is I/O-bound and dependent on the previous two tasks,
        # so it goes into the header writer pool.
        def commit_header(
            crc32_future: Optional[concurrent.futures.Future],
            sha256_future: Optional[concurrent.futures.Future],
            encrypt_future: Optional[concurrent.futures.Future],
        ):
            crc32 = sha256 = None
            if crc32_future is not None:
                crc32 = crc32_future.result(_TIMEOUT)
            if sha256_future is not None:
                sha256 = sha256_future.result(_TIMEOUT)
            if encrypt_future is not None:
                encrypt_future.result(_TIMEOUT)
            # These must be written only after all other futures complete
            # to prevent a race condition from other threads hashing
            # a partially-filled-in hash section
            if crc32_future is not None:
                header.add_crc32(crc32)
            if sha256_future is not None:
                header.add_sha256(sha256)
            if encrypt_future is not None:
                header.update_crypt_info()
            self._pwrite(header.buffer, header_pos, verify=header.data_offset)

        hash_tasks = []
        if self._encrypted and not _temporary_buffer:
            # If multiple tensors share memory, and were encrypted in-place,
            # then this must not start hashing until any previous decryption
            # tasks have restored this memory to its original state
            mem_pointer = tensor.__array_interface__["data"][0]
            pending_decryption = self._decryption_jobs.get(mem_pointer, None)
        else:
            mem_pointer = None
            pending_decryption = None
        if include_crc32:
            crc32_task = self._computation_pool.submit(
                compute_crc32, pending_decryption
            )
            hash_tasks.append(crc32_task)
        else:
            crc32_task = None
        sha256_task = self._computation_pool.submit(
            compute_sha256, pending_decryption
        )
        hash_tasks.append(sha256_task)
        self._jobs.extend(hash_tasks)

        def encrypt(prerequisites: Iterable[concurrent.futures.Future]):
            fs = concurrent.futures.wait(prerequisites, timeout=_TIMEOUT)
            for f in fs.done:
                # Raise exceptions
                f.result()
            for f in fs.not_done:
                # Raise timeouts
                f.result(0)
            try:
                encryptor.encrypt_all(
                    wait=True,
                    timeout=_TIMEOUT,
                )
            except _crypt.CryptographyError as e:
                raise CryptographyError("Tensor encryption failed") from e

        # This task is I/O-bound, so it goes into the regular writer pool.
        def write_tensor_data(
            prerequisite: Optional[concurrent.futures.Future], size: int
        ):
            if prerequisite is not None:
                prerequisite.result(_TIMEOUT)
            bytes_written = self._pwrite(tensor_memory, tensor_pos, verify=size)
            with self._tensor_count_update_lock:
                self._file_header.tensor_count += 1
                self._file_header.tensor_size += bytes_written

        def decrypt(prerequisite: concurrent.futures.Future):
            try:
                prerequisite.result(_TIMEOUT)
            finally:
                # Try to decrypt again even if writing to disk failed
                # to avoid exiting with the tensor memory in a modified state
                fs = encryptor.decrypt_all(wait=False)
                try:
                    _crypt.ChunkedEncryption.wait_or_raise(
                        fs,
                        timeout=_TIMEOUT,
                        return_when=concurrent.futures.ALL_COMPLETED,
                    )
                except _crypt.CryptographyError as e:
                    try:
                        original_exc = prerequisite.exception(timeout=0)
                    except (
                        concurrent.futures.TimeoutError,
                        concurrent.futures.CancelledError,
                    ):
                        original_exc = None
                    raise CryptographyError(
                        "Restoring encrypted tensor data in memory failed"
                    ) from (original_exc if original_exc is not None else e)

        # Encrypt the tensor memory in-place before writing
        if self._encrypted:
            encrypt_task = self._encryption_pool.submit(encrypt, hash_tasks)
            self._jobs.append(encrypt_task)
        else:
            encrypt_task = None

        commit_header_task = self._header_writer_pool.submit(
            commit_header, crc32_task, sha256_task, encrypt_task
        )
        self._jobs.append(commit_header_task)

        # Write the potentially-encrypted tensor memory to the file
        write_task = self._writer_pool.submit(
            write_tensor_data, encrypt_task, tensor_size
        )
        self._jobs.append(write_task)
        # Decrypt the memory after writing is finished, if it was encrypted
        if self._encrypted and not _temporary_buffer:
            decrypt_task = self._decryption_pool.submit(decrypt, write_task)
            self._jobs.append(decrypt_task)
            assert mem_pointer is not None
            self._decryption_jobs[mem_pointer] = decrypt_task

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
        logger.debug(
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
                    transferred.put(t.cpu().detach(), timeout=_TIMEOUT)
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
            iter(lambda: transferred.get(timeout=_TIMEOUT), None),
            _interrupt_transfer,
        )

    class _WriteSpec(typing.NamedTuple):
        idx: int
        name: str
        tensor_type: TensorType
        tensor: torch.Tensor
        callback: Optional[Callable]

    def _bulk_write(self, tensors: Iterable[_WriteSpec]):
        tensors = collections.deque(tensors)
        next_pos = self._file.tell()

        fallocate = getattr(os, "posix_fallocate", None)
        if fallocate and self._fd:
            size = sum(len(t.name) for t in tensors)
            size += sum(
                t.tensor.element_size() * t.tensor.nelement() for t in tensors
            )
            # Rough underestimate of header size
            header_min_size = 24
            size += header_min_size * len(tensors)
            try:
                fallocate(self._fd, next_pos, size)
            except OSError:
                pass

        cuda_tensors = [
            t.tensor for t in tensors if t.tensor.device.type == "cuda"
        ]
        if cuda_tensors:
            (
                transferred,
                interrupt_transfer,
            ) = self._async_bulk_device_to_host_transfer(cuda_tensors)
        else:
            transferred = interrupt_transfer = None
        del cuda_tensors

        if self._encrypted:
            shared = []
            seen_addresses = set()
            for t in reversed(tensors):
                if t.tensor.device.type == "cuda":
                    shared.append(False)
                else:
                    address = t.tensor.data_ptr()
                    shared.append(address in seen_addresses)
                    seen_addresses.add(address)
            del seen_addresses
        else:
            shared = [False] * len(tensors)

        try:
            while tensors:
                idx, name, tensor_type, tensor, callback = tensors.popleft()
                is_shared = shared.pop()
                self._idx = idx
                if tensor.device.type == "cuda":
                    tensor = next(transferred)
                    temp_tensor = True
                elif is_shared and self._encrypted:
                    # Un-shares tensor memory in preparation for in-place
                    # operations on the buffer that would otherwise conflict
                    # with one another. Full support for shared-memory tensors
                    # (e.g. if they were only written once) could make
                    # this unnecessary, once implemented.
                    # Another option would be to reuse the same encrypted
                    # weights and decrypt them at the end. This would require
                    # confirming that the tensor data regions are actually
                    # identical, and don't just overlap.
                    tensor = tensor.clone().detach()
                    temp_tensor = True
                else:
                    temp_tensor = False
                next_pos = self._write_tensor(
                    idx,
                    name,
                    tensor_type,
                    tensor,
                    _synchronize=False,
                    _start_pos=next_pos,
                    _temporary_buffer=temp_tensor,
                )
                if callback is not None:
                    callback()
        except Exception:
            if interrupt_transfer is not None:
                interrupt_transfer()
            raise
        self._synchronize_pools()
        self._file.seek(next_pos)
        self._sync_prologue_state()

    def write_module(
        self,
        m: torch.nn.Module,
        remove_tensors: bool = False,
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
            remove_tensors: Whether to delete each tensor from `m`
                after serializing it.
                Deleted tensors are replaced with ``None``.
            include_non_persistent_buffers: Whether to serialize buffers
                registered with ``persistent=False``.
                Set to ``False`` to match the behaviour of
                ``torch.nn.Module.state_dict()``,
                which saves only persistent buffers.
                The default may change to ``False`` in a later version.
        """

        modules = tuple(m.named_modules())

        def extract_tensors():
            chain = itertools.chain
            repeat = itertools.repeat
            callback = None
            for idx, (module_name, module) in enumerate(modules):
                module: torch.nn.Module
                parameters = module.named_parameters(recurse=False)
                buffers = module.named_buffers(recurse=False)
                for (name, tensor), tensor_type in chain(
                    zip(parameters, repeat(TensorType.PARAM)),
                    zip(buffers, repeat(TensorType.BUFFER)),
                ):
                    label = f"{module_name}.{name}"
                    if remove_tensors:
                        callback = partial(setattr, module, name, None)
                    yield TensorSerializer._WriteSpec(
                        idx=idx,
                        name=label,
                        tensor_type=tensor_type,
                        tensor=tensor,
                        callback=callback,
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
                spec for spec in all_tensors if spec.name in persistent
            )

        self._bulk_write(all_tensors)

    def write_state_dict(self, state_dict: Dict):
        """
        Write the state_dict to the file in Tensorizer format.

        It is strongly recommended that you use write_module instead of
        this function, as it will also write out the parameter type,
        allowing for zero-copy loading of the module with
        TensorDeserializer.load_into_module.
        """
        idx = 0
        self._bulk_write(
            TensorSerializer._WriteSpec(
                idx=idx,
                name=name,
                tensor_type=TensorType.STATE_DICT,
                tensor=param,
                callback=None,
            )
            for name, param in state_dict.items()
        )

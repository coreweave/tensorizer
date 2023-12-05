import abc
import dataclasses
import io
import struct
import typing
import weakref
from functools import partial
from typing import ClassVar, List, Optional, Sequence, Union

from tensorizer._internal_utils import _unpack_memoryview_from, _variable_read


class CryptInfoChunk(abc.ABC):
    _chunk_types: ClassVar[
        typing.MutableMapping[int, typing.Type["CryptInfoChunk"]]
    ] = weakref.WeakValueDictionary()
    chunk_type: ClassVar[int]
    _length_segment: ClassVar[struct.Struct] = struct.Struct("<Q")
    _chunk_type_segment: ClassVar[struct.Struct] = struct.Struct("<H")

    __slots__ = ()

    @classmethod
    @abc.abstractmethod
    def unpack_from(cls, buffer, offset: int = 0) -> "CryptInfoChunk":
        chunk_type = CryptInfoChunk._chunk_type_segment.unpack_from(
            buffer, offset
        )[0]
        return CryptInfoChunk._chunk_types[chunk_type].unpack_from(
            buffer, offset + CryptInfoChunk._chunk_type_segment.size
        )

    @abc.abstractmethod
    def pack_into(self, buffer, offset: int = 0) -> int:
        CryptInfoChunk._chunk_type_segment.pack_into(
            buffer, offset, self.chunk_type
        )
        return offset + CryptInfoChunk._chunk_type_segment.size

    def pack(self) -> bytes:
        buffer = io.BytesIO(bytes(self.size))
        self.pack_into(buffer.getbuffer(), 0)
        return buffer.getvalue()

    def sized_pack(self) -> bytes:
        buffer = io.BytesIO(bytes(self.sized_size))
        self.sized_pack_into(buffer.getbuffer(), 0)
        return buffer.getvalue()

    def sized_pack_into(self, buffer, offset: int = 0) -> int:
        length_offset = offset
        offset += CryptInfoChunk._length_segment.size
        ret = self.pack_into(buffer, offset)
        CryptInfoChunk._length_segment.pack_into(
            buffer, length_offset, ret - offset
        )
        return ret

    @property
    def sized_size(self) -> int:
        return self.size + CryptInfoChunk._length_segment.size

    @property
    @abc.abstractmethod
    def size(self) -> int:
        return CryptInfoChunk._chunk_type_segment.size

    # noinspection PyMethodOverriding
    def __init_subclass__(
        cls, /, *, chunk_type: Optional[int] = None, **kwargs
    ):
        super().__init_subclass__(**kwargs)
        if chunk_type is not None:
            cls.chunk_type = chunk_type
            CryptInfoChunk._chunk_types[chunk_type] = cls


class KeyDerivationChunk(CryptInfoChunk, abc.ABC, chunk_type=1):
    _derivation_methods: ClassVar[
        typing.MutableMapping[int, typing.Type["KeyDerivationChunk"]]
    ] = weakref.WeakValueDictionary()
    derivation_method: ClassVar[int]
    _derivation_method_segment: ClassVar[struct.Struct] = struct.Struct("<H")

    __slots__ = ()

    # noinspection PyMethodOverriding
    def __init_subclass__(
        cls, /, *, derivation_method: Optional[int] = None, **kwargs
    ):
        super().__init_subclass__(**kwargs)
        if derivation_method is not None:
            cls.derivation_method = derivation_method
            KeyDerivationChunk._derivation_methods[derivation_method] = cls

    @classmethod
    @abc.abstractmethod
    def unpack_from(cls, buffer, offset: int = 0) -> "KeyDerivationChunk":
        derivation_method = (
            KeyDerivationChunk._derivation_method_segment.unpack_from(
                buffer, offset
            )[0]
        )
        return KeyDerivationChunk._derivation_methods[
            derivation_method
        ].unpack_from(
            buffer, offset + KeyDerivationChunk._derivation_method_segment.size
        )

    @abc.abstractmethod
    def pack_into(self, buffer, offset: int = 0) -> int:
        offset = super().pack_into(buffer, offset)
        KeyDerivationChunk._derivation_method_segment.pack_into(
            buffer, offset, self.derivation_method
        )
        return offset + KeyDerivationChunk._derivation_method_segment.size

    @property
    @abc.abstractmethod
    def size(self) -> int:
        return KeyDerivationChunk._derivation_method_segment.size + super().size


@dataclasses.dataclass(frozen=True)
class PWHashKeyDerivationChunk(KeyDerivationChunk, derivation_method=1):
    opslimit: int
    memlimit: int
    alg: int
    salt: Union[bytes, bytearray, memoryview]

    __slots__ = ("opslimit", "memlimit", "alg", "salt")

    _algorithm_segment: ClassVar[struct.Struct] = struct.Struct(
        "<"  # Little-endian
        "Q"  # Opslimit (unsigned long long)
        "Q"  # Memlimit (size_t)
        "i"  # Algorithm identifier (int)
    )
    _salt_segment_template: ClassVar[str] = (
        "<H{salt_len:d}s"  # Salt length and bytes
    )
    read_salt = partial(_variable_read, length_fmt="H", data_fmt="s")

    @property
    def _salt_segment(self):
        return struct.Struct(
            self._salt_segment_template.format(salt_len=len(self.salt))
        )

    @classmethod
    def unpack_from(cls, buffer, offset: int = 0) -> "PWHashKeyDerivationChunk":
        opslimit, memlimit, alg = cls._algorithm_segment.unpack_from(
            buffer, offset
        )
        offset += cls._algorithm_segment.size
        salt = cls.read_salt(buffer, offset)[0]
        return cls(opslimit=opslimit, memlimit=memlimit, alg=alg, salt=salt)

    def pack_into(self, buffer, offset: int = 0) -> int:
        offset = super().pack_into(buffer, offset)
        self._algorithm_segment.pack_into(
            buffer, offset, self.opslimit, self.memlimit, self.alg
        )
        offset += self._algorithm_segment.size
        salt_segment = self._salt_segment
        salt_segment.pack_into(buffer, offset, len(self.salt), self.salt)
        offset += salt_segment.size
        return offset

    @property
    def size(self) -> int:
        return self._algorithm_segment.size + 2 + len(self.salt) + super().size


@dataclasses.dataclass
class XSalsa20ParallelChunk(CryptInfoChunk, chunk_type=2):
    chunk_size: int
    nonce: Union[bytes, bytearray, memoryview]
    num_macs: int = dataclasses.field(init=False)
    macs: Sequence[Union[bytes, bytearray, memoryview]]

    __slots__ = ("chunk_size", "nonce", "macs", "__dict__")

    NONCE_BYTES: ClassVar[int] = 24
    MAC_BYTES: ClassVar[int] = 16
    CHUNK_QUANTUM: ClassVar[int] = 64
    MINIMUM_CHUNK_SIZE: ClassVar[int] = 1024

    _header_segment: ClassVar[struct.Struct] = struct.Struct(
        "<"  # Little-endian
        "Q"  # Chunk size
        f"{NONCE_BYTES:d}s"  # Initial nonce
        "Q"  # Number of MACs
    )

    _mac_segment: ClassVar[struct.Struct] = struct.Struct(f"<{MAC_BYTES:d}s")

    def __post_init__(self):
        if len(self.nonce) != self.NONCE_BYTES:
            raise ValueError("Invalid nonce size")
        if not (
            isinstance(self.chunk_size, int)
            and (self.chunk_size % self.CHUNK_QUANTUM == 0)
            and self.chunk_size >= self.MINIMUM_CHUNK_SIZE
        ):
            raise ValueError("Invalid chunk size")
        self.num_macs = len(self.macs)
        for mac in self.macs:
            if len(mac) != self.MAC_BYTES:
                raise ValueError("Invalid MAC size")

    @classmethod
    def unpack_from(cls, buffer, offset: int = 0) -> "XSalsa20ParallelChunk":
        chunk_size, nonce, num_macs = (
            XSalsa20ParallelChunk._header_segment.unpack_from(buffer, offset)
        )
        offset += XSalsa20ParallelChunk._header_segment.size
        macs = []
        for i in range(num_macs):
            macs.append(
                _unpack_memoryview_from(
                    XSalsa20ParallelChunk._mac_segment.size, buffer, offset
                )
            )
            offset += XSalsa20ParallelChunk._mac_segment.size
        return cls(chunk_size, nonce, macs)

    def pack_into(self, buffer, offset: int = 0) -> int:
        offset = super().pack_into(buffer, offset)
        XSalsa20ParallelChunk._header_segment.pack_into(
            buffer, offset, self.chunk_size, self.nonce, self.num_macs
        )
        offset += XSalsa20ParallelChunk._header_segment.size
        for mac in self.macs:
            XSalsa20ParallelChunk._mac_segment.pack_into(buffer, offset, mac)
            del mac
            offset += XSalsa20ParallelChunk._mac_segment.size
        return offset

    @property
    def size(self) -> int:
        return (
            XSalsa20ParallelChunk._header_segment.size
            + XSalsa20ParallelChunk._mac_segment.size * self.num_macs
            + super().size
        )


@dataclasses.dataclass
class XSalsa20SequentialChunk(CryptInfoChunk, chunk_type=3):
    nonce: Union[bytes, bytearray, memoryview]
    mac: Union[bytes, bytearray, memoryview]

    __slots__ = ("nonce", "mac")

    NONCE_BYTES: ClassVar[int] = 24
    MAC_BYTES: ClassVar[int] = 16

    _contents_segment: ClassVar[struct.Struct] = struct.Struct(
        "<"  # Little-endian
        f"{NONCE_BYTES:d}s"  # Nonce
        f"{MAC_BYTES:d}s"  # MAC
    )

    def __post_init__(self):
        if len(self.nonce) != self.NONCE_BYTES:
            raise ValueError("Invalid nonce size")
        if len(self.mac) != self.MAC_BYTES:
            raise ValueError("Invalid MAC size")

    @classmethod
    def unpack_from(cls, buffer, offset: int = 0) -> "XSalsa20SequentialChunk":
        nonce, mac = XSalsa20SequentialChunk._contents_segment.unpack_from(
            buffer, offset
        )
        return cls(nonce, mac)

    def pack_into(self, buffer, offset: int = 0) -> int:
        offset = super().pack_into(buffer, offset)
        XSalsa20SequentialChunk._contents_segment.pack_into(
            buffer, offset, self.nonce, self.mac
        )
        return offset + XSalsa20SequentialChunk._contents_segment.size

    @property
    def size(self) -> int:
        return XSalsa20SequentialChunk._contents_segment.size + super().size


@dataclasses.dataclass
class CryptInfo:
    num_chunks: int = dataclasses.field(init=False)
    chunks: Sequence[CryptInfoChunk] = ()

    _length_segment: ClassVar[struct.Struct] = struct.Struct(
        "<Q"  # Little-endian segment length
    )

    _chunk_length_segment: ClassVar[struct.Struct] = _length_segment

    _count_segment: ClassVar[struct.Struct] = struct.Struct(
        "<H"  # Little-endian entry count
    )

    def __post_init__(self):
        self.num_chunks = len(self.chunks)

    @property
    def sized_size(self) -> int:
        return self._length_segment.size + self.size

    @property
    def size(self) -> int:
        return self._count_segment.size + sum(c.sized_size for c in self.chunks)

    def find_chunks(
        self,
        typ: Union[
            typing.Type[CryptInfoChunk],
            typing.Tuple[typing.Type[CryptInfoChunk], ...],
        ],
    ) -> Sequence[CryptInfoChunk]:
        return tuple(c for c in self.chunks if isinstance(c, typ))

    def pack_into(self, buffer, offset: int = 0) -> int:
        CryptInfo._count_segment.pack_into(buffer, offset, self.num_chunks)
        offset += CryptInfo._count_segment.size
        for chunk in self.chunks:
            offset = chunk.sized_pack_into(buffer, offset)
        return offset

    def sized_pack_into(self, buffer, offset: int = 0) -> int:
        length_offset = offset
        offset += CryptInfo._length_segment.size
        ret = self.pack_into(buffer, offset)
        CryptInfo._length_segment.pack_into(buffer, length_offset, ret - offset)
        return ret

    @classmethod
    def unpack_from(cls, buffer, offset: int = 0) -> "CryptInfo":
        num_chunks: int = CryptInfo._count_segment.unpack_from(buffer, offset)[
            0
        ]
        offset += CryptInfo._count_segment.size
        if num_chunks < 0:
            raise ValueError(
                "Invalid CryptInfo chunk count, cannot be negative"
            )
        chunks: List[CryptInfoChunk] = []
        with memoryview(buffer) as mv:
            for i in range(num_chunks):
                chunk_size: int = CryptInfo._chunk_length_segment.unpack_from(
                    buffer, offset
                )[0]
                offset += CryptInfo._chunk_length_segment.size
                chunk_end: int = offset + chunk_size
                with mv[offset:chunk_end] as chunk_mv:
                    # Blocks out-of-bounds accesses
                    chunks.append(CryptInfoChunk.unpack_from(chunk_mv))
                offset = chunk_end
        return cls(chunks)

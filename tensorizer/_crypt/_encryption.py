import concurrent.futures
import ctypes
import dataclasses
import enum
import io
import mmap
import typing
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from ctypes import (
    POINTER,
    c_char_p,
    c_int,
    c_size_t,
    c_ubyte,
    c_uint64,
    c_ulonglong,
    c_void_p,
)
from typing import (
    ClassVar,
    Final,
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import libnacl
from libnacl import nacl

try:
    from ._cgroup_cpu_count import effective_cpu_count
    from ._exceptions import CryptographyError
except ImportError:
    from _cgroup_cpu_count import effective_cpu_count
    from _exceptions import CryptographyError

IN, OUT, INOUT = 1, 2, 3

c_ubyte_p = POINTER(c_ubyte)


def error_handler(failure_message: str = "Cryptographic operation failed"):
    def _errcheck(result, func, args):
        del args  # Break any references to ctypes buffers
        if result == -1:
            raise CryptographyError(failure_message)
        elif result == 0:
            return None
        else:
            raise RuntimeError("Unknown return code")

    return _errcheck


errcheck = error_handler()
encrypt_errcheck = error_handler("Encryption failed")
decrypt_errcheck = error_handler("Decryption failed: MAC doesn't match data")


def init_asymmetric():
    # https://libsodium.gitbook.io/doc/public-key_cryptography/authenticated_encryption
    prototype = ctypes.CFUNCTYPE(
        c_int,
        c_ubyte_p,
        c_ubyte_p,
        c_ubyte_p,
        c_ulonglong,
        c_ubyte_p,
        c_ubyte_p,
        c_ubyte_p,
    )

    crypto_box_detached_paramflags = (
        (INOUT, "c"),
        (INOUT, "mac"),
        (IN, "m"),
        (IN, "mlen"),
        (IN, "n"),
        (IN, "pk"),
        (IN, "sk"),
    )

    crypto_box_open_detached_paramflags = (
        (INOUT, "m"),
        (IN, "c"),
        (IN, "mac"),
        (IN, "clen"),
        (IN, "n"),
        (IN, "pk"),
        (IN, "sk"),
    )

    crypto_box_detached = prototype(
        ("crypto_box_detached", nacl), crypto_box_detached_paramflags
    )
    crypto_box_open_detached = prototype(
        ("crypto_box_open_detached", nacl), crypto_box_open_detached_paramflags
    )

    crypto_box_detached.errcheck = encrypt_errcheck
    crypto_box_open_detached.errcheck = decrypt_errcheck

    return crypto_box_detached, crypto_box_open_detached


def init_symmetric():
    # https://libsodium.gitbook.io/doc/secret-key_cryptography/secretbox
    prototype = ctypes.CFUNCTYPE(
        c_int,
        c_ubyte_p,
        c_ubyte_p,
        c_ubyte_p,
        c_ulonglong,
        c_ubyte_p,
        c_ubyte_p,
    )

    crypto_secretbox_detached_paramflags = (
        (INOUT, "c"),
        (INOUT, "mac"),
        (IN, "m"),
        (IN, "mlen"),
        (IN, "n"),
        (IN, "k"),
    )

    crypto_secretbox_open_detached_paramflags = (
        (INOUT, "m"),
        (IN, "c"),
        (IN, "mac"),
        (IN, "clen"),
        (IN, "n"),
        (IN, "k"),
    )

    crypto_secretbox_detached = prototype(
        ("crypto_secretbox_detached", nacl),
        crypto_secretbox_detached_paramflags,
    )
    crypto_secretbox_open_detached = prototype(
        ("crypto_secretbox_open_detached", nacl),
        crypto_secretbox_open_detached_paramflags,
    )

    crypto_secretbox_detached.errcheck = encrypt_errcheck
    crypto_secretbox_open_detached.errcheck = decrypt_errcheck

    return crypto_secretbox_detached, crypto_secretbox_open_detached


def init_sodium_increment():
    # https://libsodium.gitbook.io/doc/helpers#incrementing-large-numbers
    prototype = ctypes.CFUNCTYPE(
        None,
        c_ubyte_p,
        c_size_t,
    )

    paramflags = (
        (INOUT, "n"),
        (IN, "nlen"),
    )

    sodium_increment = prototype(
        ("sodium_increment", nacl),
        paramflags,
    )

    return sodium_increment


def init_randombytes_buf():
    # https://libsodium.gitbook.io/doc/generating_random_data
    prototype = ctypes.CFUNCTYPE(
        None,
        c_ubyte_p,
        c_size_t,
    )

    paramflags = (
        (INOUT, "buf"),
        (IN, "size"),
    )

    randombytes_buf = prototype(
        ("randombytes_buf", nacl),
        paramflags,
    )

    return randombytes_buf


def init_crypto_pwhash():
    # https://libsodium.gitbook.io/doc/password_hashing/default_phf
    prototype = ctypes.CFUNCTYPE(
        c_int,
        c_ubyte_p,
        c_ulonglong,
        c_char_p,
        c_ulonglong,
        c_ubyte_p,
        c_ulonglong,
        c_size_t,
        c_int,
    )

    paramflags = (
        (INOUT, "out"),
        (IN, "outlen"),
        (IN, "passwd"),
        (IN, "passwdlen"),
        (IN, "salt"),
        (IN, "opslimit"),
        (IN, "memlimit"),
        (IN, "alg"),
    )

    crypto_pwhash = prototype(
        ("crypto_pwhash", nacl),
        paramflags,
    )

    crypto_pwhash.errcheck = error_handler(
        "Password hashing failed. May have run out of memory."
    )

    return crypto_pwhash


def init_crypto_onetimeauth_poly1305():
    # https://libsodium.gitbook.io/doc/advanced/poly1305
    prototype = ctypes.CFUNCTYPE(
        c_int,
        c_ubyte_p,
        c_ubyte_p,
        c_ulonglong,
        c_ubyte_p,
    )

    paramflags = (
        (INOUT, "out"),
        (IN, "in_buf"),  # Actually called "in", but "in" is a Python keyword
        (IN, "inlen"),
        (IN, "k"),
    )

    crypto_onetimeauth_poly1305 = prototype(
        ("crypto_onetimeauth_poly1305", nacl),
        paramflags,
    )

    crypto_onetimeauth_poly1305.errcheck = errcheck

    return crypto_onetimeauth_poly1305


def init_crypto_core_hsalsa20():
    prototype = ctypes.CFUNCTYPE(
        c_int,
        c_ubyte_p,
        c_ubyte_p,
        c_ubyte_p,
        c_ubyte_p,
    )

    paramflags = (
        (INOUT, "out"),
        (IN, "in_buf"),  # Actually called "in", but "in" is a Python keyword
        (IN, "k"),
        (IN, "c", None),
    )

    crypto_core_hsalsa20 = prototype(
        ("crypto_core_hsalsa20", nacl),
        paramflags,
    )

    crypto_core_hsalsa20.errcheck = errcheck

    return crypto_core_hsalsa20


def init_crypto_stream_salsa20_xor_ic():
    # https://libsodium.gitbook.io/doc/advanced/stream_ciphers/salsa20
    prototype = ctypes.CFUNCTYPE(
        c_int,
        c_ubyte_p,
        c_ubyte_p,
        c_ulonglong,
        c_ubyte_p,
        c_uint64,
        c_ubyte_p,
    )

    paramflags = (
        (INOUT, "c"),
        (IN, "m"),
        (IN, "mlen"),
        (IN, "n"),
        (IN, "ic"),
        (IN, "k"),
    )

    crypto_stream_salsa20_xor_ic = prototype(
        ("crypto_stream_salsa20_xor_ic", nacl),
        paramflags,
    )

    crypto_stream_salsa20_xor_ic.errcheck = errcheck

    return crypto_stream_salsa20_xor_ic


def init_sodium_memzero():
    prototype = ctypes.CFUNCTYPE(
        None,
        c_void_p,
        c_size_t,
    )

    paramflags = (
        (IN, "pnt"),
        (IN, "len"),
    )

    sodium_memzero = prototype(
        ("sodium_memzero", nacl),
        paramflags,
    )

    return sodium_memzero


crypto_box_detached, crypto_box_open_detached = init_asymmetric()
crypto_secretbox_detached, crypto_secretbox_open_detached = init_symmetric()
sodium_increment = init_sodium_increment()
randombytes_buf = init_randombytes_buf()
crypto_pwhash = init_crypto_pwhash()
crypto_onetimeauth_poly1305 = init_crypto_onetimeauth_poly1305()
crypto_core_hsalsa20 = init_crypto_core_hsalsa20()
crypto_stream_salsa20_xor_ic = init_crypto_stream_salsa20_xor_ic()
sodium_memzero = init_sodium_memzero()


class Constants(type):
    @staticmethod
    def _get_constant(name, typ) -> int:
        prototype = ctypes.CFUNCTYPE(typ)
        paramflags = ()
        getter = prototype((name, nacl), paramflags)
        return getter()

    def __new__(cls, name: str, bases: tuple, dct: dict) -> NamedTuple:
        annotations = dct.pop("__annotations__", {})
        entries = {}
        for constant_name, constant_type in dct.items():
            if constant_name.startswith("_"):
                continue
            entries[constant_name] = cls._get_constant(
                constant_name.lower(), constant_type
            )
        constant_class = typing.cast(
            type, NamedTuple(name, **{k: annotations[k] for k in entries})
        )
        return constant_class(**entries)


class Const(metaclass=Constants):
    # Dynamically pulls constants from libsodium
    # Syntax:
    # libsodium name: Python type = C type

    crypto_stream_salsa20_NONCEBYTES: int = c_size_t
    crypto_stream_salsa20_KEYBYTES: int = c_size_t

    crypto_onetimeauth_poly1305_BYTES: int = c_size_t
    crypto_onetimeauth_poly1305_KEYBYTES: int = c_size_t

    crypto_pwhash_SALTBYTES: int = c_size_t
    crypto_pwhash_PASSWD_MIN: int = c_size_t
    crypto_pwhash_PASSWD_MAX: int = c_size_t
    crypto_pwhash_BYTES_MIN: int = c_size_t
    crypto_pwhash_BYTES_MAX: int = c_size_t

    crypto_pwhash_OPSLIMIT_MIN: int = c_ulonglong
    crypto_pwhash_OPSLIMIT_INTERACTIVE: int = c_ulonglong
    crypto_pwhash_OPSLIMIT_MODERATE: int = c_ulonglong
    crypto_pwhash_OPSLIMIT_SENSITIVE: int = c_ulonglong
    crypto_pwhash_OPSLIMIT_MAX: int = c_ulonglong

    crypto_pwhash_MEMLIMIT_MIN: int = c_size_t
    crypto_pwhash_MEMLIMIT_INTERACTIVE: int = c_size_t
    crypto_pwhash_MEMLIMIT_MODERATE: int = c_size_t
    crypto_pwhash_MEMLIMIT_SENSITIVE: int = c_size_t
    crypto_pwhash_MEMLIMIT_MAX: int = c_size_t

    crypto_pwhash_ALG_ARGON2ID13: int = c_int


if (
    Const.crypto_stream_salsa20_KEYBYTES
    != Const.crypto_onetimeauth_poly1305_KEYBYTES
):
    # The definitions of these two algorithms should set these
    # both at 32 bytes.
    raise RuntimeError(
        "crypto_stream_salsa20_KEYBYTES doesn't match"
        " crypto_onetimeauth_poly1305_KEYBYTES"
    )


SALSA20_BLOCK_SIZE: Final[int] = 64

Buffer = Union[bytes, bytearray, memoryview]
MutableBuffer = Union[bytearray, memoryview]


def as_ucstr(buffer):
    if isinstance(buffer, (bytearray, memoryview, mmap.mmap)):
        return (c_ubyte * len(buffer)).from_buffer(buffer)
    else:
        return ctypes.cast(buffer, c_ubyte_p)


def increment_inplace(buffer: MutableBuffer):
    sodium_increment(as_ucstr(buffer), len(buffer))


def random_bytes(size: int) -> bytes:
    if not isinstance(size, int):
        raise TypeError("Invalid size")
    buf = io.BytesIO(bytes(size))
    buf_view = buf.getbuffer()
    assert len(buf_view) == size
    randombytes_buf(buf=as_ucstr(buf_view), size=size)
    del buf_view
    return buf.getvalue()


@dataclasses.dataclass(init=False)
class PWHash:
    __slots__ = ("opslimit", "memlimit", "alg", "salt")

    OPSLIMIT_MIN: ClassVar[int] = Const.crypto_pwhash_OPSLIMIT_MIN
    OPSLIMIT_INTERACTIVE: ClassVar[int] = (
        Const.crypto_pwhash_OPSLIMIT_INTERACTIVE
    )
    OPSLIMIT_MODERATE: ClassVar[int] = Const.crypto_pwhash_OPSLIMIT_MODERATE
    OPSLIMIT_SENSITIVE: ClassVar[int] = Const.crypto_pwhash_OPSLIMIT_SENSITIVE
    OPSLIMIT_MAX: ClassVar[int] = Const.crypto_pwhash_OPSLIMIT_MAX

    MEMLIMIT_MIN: ClassVar[int] = Const.crypto_pwhash_MEMLIMIT_MIN
    MEMLIMIT_INTERACTIVE: ClassVar[int] = (
        Const.crypto_pwhash_MEMLIMIT_INTERACTIVE
    )
    MEMLIMIT_MODERATE: ClassVar[int] = Const.crypto_pwhash_MEMLIMIT_MODERATE
    MEMLIMIT_SENSITIVE: ClassVar[int] = Const.crypto_pwhash_MEMLIMIT_SENSITIVE
    MEMLIMIT_MAX: ClassVar[int] = Const.crypto_pwhash_MEMLIMIT_MAX

    ALG_ARGON2ID13: ClassVar[int] = Const.crypto_pwhash_ALG_ARGON2ID13

    opslimit: int
    memlimit: int
    alg: int

    salt: bytes

    @staticmethod
    def _range_check(value, lower, upper, name: str) -> None:
        if not (lower <= value <= upper):
            raise ValueError(
                f"Invalid {name}, must be in the range [{lower}, {upper})"
            )

    def __init__(
        self,
        salt: Union[bytes, bytearray, memoryview, None] = None,
        opslimit=OPSLIMIT_MODERATE,
        memlimit=MEMLIMIT_MODERATE,
    ):
        if salt is None:
            salt: bytes = random_bytes(Const.crypto_pwhash_SALTBYTES)
        elif not isinstance(salt, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Invalid type for salt,"
                " expected bytes, bytearray, or memoryview;"
                f" got {type(salt).__name__}"
            )
        if len(salt) != Const.crypto_pwhash_SALTBYTES:
            raise ValueError(
                "Invalid salt size,"
                f" {len(salt)} != {Const.crypto_pwhash_SALTBYTES}"
            )
        self._range_check(
            opslimit,
            Const.crypto_pwhash_OPSLIMIT_MIN,
            Const.crypto_pwhash_OPSLIMIT_MAX,
            "opslimit",
        )
        self._range_check(
            memlimit,
            Const.crypto_pwhash_MEMLIMIT_MIN,
            Const.crypto_pwhash_MEMLIMIT_MAX,
            "memlimit",
        )
        self.opslimit = opslimit
        self.memlimit = memlimit
        self.alg = Const.crypto_pwhash_ALG_ARGON2ID13
        self.salt = salt

    def hash(
        self,
        passwd: bytes,
        output_size: int = Const.crypto_stream_salsa20_KEYBYTES,
    ) -> bytes:
        if not isinstance(passwd, bytes):
            raise TypeError(
                "Invalid type for password,"
                f" expected bytes, got {type(passwd).__name__}"
            )
        if not isinstance(output_size, int):
            raise TypeError(
                "Invalid type for output_size"
                f" expected int, got {type(output_size).__name__}"
            )
        self._range_check(
            len(passwd),
            Const.crypto_pwhash_PASSWD_MIN,
            Const.crypto_pwhash_PASSWD_MAX,
            "password length",
        )
        self._range_check(
            output_size,
            Const.crypto_pwhash_BYTES_MIN,
            Const.crypto_pwhash_BYTES_MAX,
            "output length",
        )
        buf = io.BytesIO(bytes(output_size))
        buf_view = buf.getbuffer()
        assert len(buf_view) == output_size
        crypto_pwhash(
            out=as_ucstr(buf_view),
            outlen=output_size,
            passwd=passwd,
            passwdlen=len(passwd),
            salt=as_ucstr(self.salt),
            opslimit=self.opslimit,
            memlimit=self.memlimit,
            alg=self.alg,
        )
        del buf_view
        return buf.getvalue()


def poly1305(out: MutableBuffer, in_buf: Buffer, key: Buffer):
    if len(out) != Const.crypto_onetimeauth_poly1305_BYTES:
        raise ValueError("Invalid output buffer size for onetimeauth")
    if len(key) < Const.crypto_onetimeauth_poly1305_KEYBYTES:
        # The key can be larger, but only the prefix will be used
        raise ValueError("Invalid key buffer size for onetimeauth")
    return crypto_onetimeauth_poly1305(
        out=as_ucstr(out),
        in_buf=as_ucstr(in_buf),
        inlen=len(in_buf),
        k=as_ucstr(key),
    )


def salsa20_subkey(
    nonce: Buffer, key: Buffer, out: Optional[MutableBuffer] = None
) -> Optional[MutableBuffer]:
    if out is None:
        out = bytearray(Const.crypto_stream_salsa20_KEYBYTES)
    elif len(out) != Const.crypto_stream_salsa20_KEYBYTES:
        raise ValueError("Invalid key buffer size for hsalsa20")
    crypto_core_hsalsa20(
        out=as_ucstr(out), in_buf=as_ucstr(nonce), k=as_ucstr(key)
    )
    return out


def salsa20_inplace(
    buffer: MutableBuffer, nonce: Buffer, key: Buffer, block: int = 0
):
    if len(key) != Const.crypto_stream_salsa20_KEYBYTES:
        raise ValueError(
            "Invalid key buffer size for crypto_stream_salsa20_xor_ic"
        )
    if len(nonce) != Const.crypto_stream_salsa20_NONCEBYTES:
        raise ValueError(
            "Invalid nonce buffer size for crypto_stream_salsa20_xor_ic"
        )
    if block < 0:
        raise ValueError("Invalid block index")

    buf = as_ucstr(buffer)
    return crypto_stream_salsa20_xor_ic(
        c=buf,
        m=buf,
        mlen=len(buffer),
        n=as_ucstr(nonce),
        ic=block,
        k=as_ucstr(key),
    )


def secretbox_inplace_open_detached_noverify(
    buffer: MutableBuffer,
    mac_out: MutableBuffer,
    nonce: Buffer,
    key: Buffer,
):
    # Based on crypto_secretbox_open_detached from:
    # https://github.com/jedisct1/libsodium/blob/master/src/libsodium/crypto_secretbox/crypto_secretbox_easy.c
    # But outputs the MAC instead of verifying it
    block_0 = bytearray(SALSA20_BLOCK_SIZE)
    subkey: bytearray = salsa20_subkey(nonce, key)
    subnonce: Buffer = nonce[16:]

    zero_bytes: int = libnacl.crypto_secretbox_ZEROBYTES
    c_len = len(buffer)
    m_len_0 = min(c_len, SALSA20_BLOCK_SIZE - zero_bytes)
    with memoryview(buffer) as buf_mv:
        block_0[zero_bytes : zero_bytes + m_len_0] = buf_mv[:m_len_0]
        salsa20_inplace(block_0, subnonce, subkey)
        poly1305(mac_out, buf_mv, block_0)
        with memoryview(block_0) as block_mv:
            buf_mv[:m_len_0] = block_mv[zero_bytes:]
        sodium_memzero(as_ucstr(block_0), len(block_0))
        if c_len > m_len_0:
            salsa20_inplace(buf_mv[m_len_0:], subnonce, subkey, block=1)
    sodium_memzero(as_ucstr(subkey), len(subkey))


class AsymmetricParams(NamedTuple):
    pk: bytes
    sk: bytes
    nonce: bytes
    mac: Buffer

    @classmethod
    def random(cls):
        pk, sk = libnacl.crypto_box_keypair()
        nonce = libnacl.randombytes(libnacl.crypto_box_NONCEBYTES)
        mac = bytearray(libnacl.crypto_box_MACBYTES)
        return cls(pk, sk, nonce, mac)


class SymmetricParams(NamedTuple):
    k: bytes
    nonce: bytes
    mac: Buffer

    @classmethod
    def random(cls):
        k = libnacl.randombytes(libnacl.crypto_secretbox_KEYBYTES)
        nonce = libnacl.randombytes(libnacl.crypto_secretbox_NONCEBYTES)
        mac = bytearray(libnacl.crypto_secretbox_MACBYTES)
        return cls(k, nonce, mac)


class _EncryptionManager(AbstractContextManager):
    key: Buffer
    buffer: MutableBuffer
    total_size: int
    emit_mac: bool
    _owns_buffer_view: bool

    class INTENT(enum.Flag):
        ENCRYPTION = 1
        DECRYPTION = 2

        @classmethod
        def both(cls) -> "_EncryptionManager.INTENT":
            return cls.ENCRYPTION | cls.DECRYPTION

    intent: INTENT

    __slots__ = (
        "key",
        "buffer",
        "total_size",
        "emit_mac",
        "_owns_buffer_view",
        "intent",
    )

    KEY_BYTES: ClassVar[int] = libnacl.crypto_secretbox_KEYBYTES
    NONCE_BYTES: ClassVar[int] = libnacl.crypto_secretbox_NONCEBYTES
    MAC_BYTES: ClassVar[int] = libnacl.crypto_secretbox_MACBYTES

    def __init__(
        self,
        key: Buffer,
        buffer: MutableBuffer,
        emit_mac: bool,
        intent: INTENT,
    ):
        if len(key) != self.KEY_BYTES:
            raise ValueError(
                f"Invalid key length: {len(key):d} != {self.KEY_BYTES}"
            )

        try:
            with memoryview(buffer) as mv:
                buffer_read_only = mv.readonly
        except TypeError as e:
            raise TypeError(
                f"Invalid buffer type: {buffer.__class__.__name__}"
            ) from e
        if buffer_read_only:
            raise TypeError(
                "Immutable buffers cannot be used for"
                " in-place encryption or decryption"
            )

        if not intent:
            raise ValueError("Invalid intent")

        self.key = key
        if isinstance(buffer, memoryview):
            self.buffer = buffer.cast("B")
            self._owns_buffer_view = True
        else:
            self.buffer = buffer
            self._owns_buffer_view = False
        self.total_size = len(self.buffer)
        self.emit_mac = emit_mac
        self.intent = intent

    def close(self) -> None:
        if self._owns_buffer_view:
            self.buffer.release()

    def __exit__(self, __exc_type, __exc_value, __traceback):
        self.close()

    @staticmethod
    def is_mutable(buffer: Buffer) -> bool:
        return not (
            isinstance(buffer, bytes)
            or isinstance(buffer, memoryview)
            and buffer.readonly
        )

    @staticmethod
    def random_nonce() -> bytes:
        return random_bytes(_EncryptionManager.NONCE_BYTES)

    @staticmethod
    def _empty_mac() -> bytearray:
        return bytearray(_EncryptionManager.MAC_BYTES)

    @staticmethod
    def _check_nonce(nonce: Buffer) -> None:
        if len(nonce) != _EncryptionManager.NONCE_BYTES:
            raise ValueError(
                f"Invalid nonce length: {len(nonce):d} !="
                f" {_EncryptionManager.NONCE_BYTES}"
            )

    @property
    def _decrypt_intent(self) -> bool:
        return bool(self.intent & self.INTENT.DECRYPTION)

    @property
    def _encrypt_intent(self) -> bool:
        return bool(self.intent & self.INTENT.ENCRYPTION)

    def _check_mac(self, mac: Buffer) -> None:
        if len(mac) != _EncryptionManager.MAC_BYTES:
            raise ValueError(
                "Invalid MAC length:"
                f" {len(mac):d} != {_EncryptionManager.MAC_BYTES}"
            )
        mac_is_mutable: bool = _EncryptionManager.is_mutable(mac)
        if not mac_is_mutable:
            if self._encrypt_intent:
                raise TypeError(
                    f"Immutable MAC (type: {mac.__class__.__name__})"
                    " cannot be used for encryption"
                )
            if self.emit_mac:
                raise TypeError(
                    f"Immutable MAC (type: {mac.__class__.__name__})"
                    " cannot be used for decryption with delayed"
                    " verification"
                )

    def _inplace_encrypt(
        self,
        buffer: MutableBuffer,
        mac: MutableBuffer,
        nonce: Buffer,
        release_on_exc: bool = False,
    ) -> int:
        if not self.is_mutable(mac):
            raise TypeError(
                f"Immutable MAC (type: {mac.__class__.__name__})"
                " cannot be used for encryption"
            )

        ret = len(buffer)
        buf = as_ucstr(buffer)
        try:
            crypto_secretbox_detached(
                c=buf,
                mac=as_ucstr(mac),
                m=buf,
                mlen=len(buffer),
                n=as_ucstr(nonce),
                k=as_ucstr(self.key),
            )
        except BaseException:
            if release_on_exc and isinstance(buffer, memoryview):
                buffer.release()
            del buffer
            raise
        finally:
            del buf
        return ret

    def _inplace_decrypt(
        self,
        buffer: MutableBuffer,
        mac: Buffer,
        nonce: Buffer,
        release_on_exc: bool = False,
    ) -> int:
        ret = len(buffer)
        try:
            if not self.emit_mac:
                buf = as_ucstr(buffer)
                try:
                    crypto_secretbox_open_detached(
                        m=buf,
                        c=buf,
                        mac=as_ucstr(mac),
                        clen=len(buffer),
                        n=as_ucstr(nonce),
                        k=as_ucstr(self.key),
                    )
                finally:
                    del buf
            else:
                secretbox_inplace_open_detached_noverify(
                    buffer, mac, nonce, self.key
                )
        except BaseException:
            if release_on_exc and isinstance(buffer, memoryview):
                buffer.release()
            del buffer
            raise
        return ret


class SequentialEncryption(_EncryptionManager):
    nonce: Buffer
    mac: Buffer

    __slots__ = ("nonce", "mac")

    def __init__(
        self,
        key: Buffer,
        buffer: MutableBuffer,
        nonce: Optional[Buffer] = None,
        mac: Optional[Buffer] = None,
        *,
        intent: "SequentialEncryption.INTENT" = (
            _EncryptionManager.INTENT.both()
        ),
    ):
        super().__init__(key, buffer, False, intent)
        decrypt_only: bool = self._decrypt_intent and not self._encrypt_intent
        if nonce is None:
            if decrypt_only:
                raise TypeError("Nonce cannot be None for decryption")
            nonce = self.random_nonce()
        self.nonce = nonce
        if mac is None:
            if decrypt_only:
                raise TypeError("MAC cannot be None for decryption")
            mac = self._empty_mac()
        self.mac = mac

        self._check_nonce(nonce)
        self._check_mac(mac)

    def encrypt(self) -> None:
        self._inplace_encrypt(self.buffer, self.mac, self.nonce)

    def decrypt(self) -> None:
        self._inplace_decrypt(self.buffer, self.mac, self.nonce)


class ChunkedEncryption(_EncryptionManager):
    chunk_size: int
    nonces: Tuple[Buffer, ...]
    macs: Tuple[Buffer, ...]
    executor: ThreadPoolExecutor
    _owns_executor: bool
    num_chunks: int
    remainder: int

    __slots__ = (
        "chunk_size",
        "nonces",
        "macs",
        "executor",
        "_owns_executor",
        "num_chunks",
        "remainder",
    )

    @staticmethod
    def sequential_nonces(
        initial_nonce: Buffer, count: int
    ) -> Iterator[Buffer]:
        if count < 0:
            raise ValueError(f"Invalid nonce count: {count}")
        elif count == 0:
            return
        elif count == 1 and isinstance(initial_nonce, bytes):
            yield initial_nonce
            return
        nonce = io.BytesIO(initial_nonce)
        del initial_nonce
        yield nonce.getvalue()
        for _ in range(1, count):
            increment_inplace(nonce.getbuffer())
            yield nonce.getvalue()

    def __init__(
        self,
        key: Buffer,
        buffer: MutableBuffer,
        chunk_size: int,
        nonces: Optional[Iterable[Buffer]] = None,
        macs: Optional[Iterable[Buffer]] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        num_threads: Optional[int] = None,
        *,
        automatic_verification: bool = True,
        intent: "ChunkedEncryption.INTENT" = _EncryptionManager.INTENT.both(),
    ):
        super().__init__(
            key=key,
            buffer=buffer,
            emit_mac=not automatic_verification,
            intent=intent,
        )
        self.chunk_size = chunk_size
        self.remainder = self.total_size % chunk_size
        self.num_chunks = (self.total_size // chunk_size) + (self.remainder > 0)

        decrypt_only: bool = self._decrypt_intent and not self._encrypt_intent
        if nonces is None:
            if decrypt_only:
                raise TypeError("Nonces cannot be None for decryption")
            initial_nonce = self.random_nonce()
            self.nonces = tuple(
                self.sequential_nonces(initial_nonce, self.num_chunks)
            )
        else:
            self.nonces = tuple(nonces)
        if macs is None:
            if decrypt_only:
                raise TypeError("MACs cannot be None for decryption")
            self.macs = tuple(self._empty_mac() for _ in range(self.num_chunks))
        else:
            self.macs = tuple(macs)

        nonce_count = len(self.nonces)
        mac_count = len(self.macs)

        if nonce_count != self.num_chunks:
            raise ValueError(
                f"Number of nonces provided ({nonce_count:d}) does not match"
                f" the number of chunks ({self.num_chunks:d})"
            )
        elif mac_count != self.num_chunks:
            raise ValueError(
                f"Number of MACs provided ({mac_count:d}) does not match"
                f" the number of chunks ({self.num_chunks:d})"
            )

        for nonce in self.nonces:
            self._check_nonce(nonce)
        for mac in self.macs:
            self._check_mac(mac)

        duplicate_nonce_count = nonce_count - len(set(self.nonces))
        if duplicate_nonce_count != 0:
            raise ValueError(
                f"Nonces must be unique ({duplicate_nonce_count} duplicates)"
            )

        if executor is not None:
            self._owns_executor = False
            if num_threads is not None:
                raise ValueError("Cannot specify both executor and num_threads")
            self.executor = executor
        else:
            self._owns_executor = True
            if num_threads is None:
                num_threads = effective_cpu_count()
            self.executor = ThreadPoolExecutor(
                max_workers=min(num_threads, self.num_chunks),
                thread_name_prefix="ChunkedEncryption",
            )

    def close(self) -> None:
        super().close()
        if self._owns_executor:
            self.executor.shutdown(wait=False)

    def _check_chunk_index(self, chunk_index: int) -> None:
        if not isinstance(chunk_index, int):
            raise TypeError("chunk_index must be an int")
        if not (0 <= chunk_index < self.num_chunks):
            raise IndexError(f"chunk_index {chunk_index:d} is out of bounds")

    def _chunk_bounds(self, chunk_index: int) -> slice:
        self._check_chunk_index(chunk_index)
        begin = chunk_index * self.chunk_size
        if chunk_index == self.num_chunks - 1:
            end = self.total_size
        else:
            end = (chunk_index + 1) * self.chunk_size
        return slice(begin, end)

    def chunk_view(self, chunk_index: int) -> memoryview:
        bounds = self._chunk_bounds(chunk_index)
        with memoryview(self.buffer) as mv:
            return mv[bounds]

    def concatenated_macs(self) -> bytearray:
        return bytearray().join(self.macs)

    def coalesce_macs(self, nonce: Buffer) -> bytearray:
        if len(nonce) != self.NONCE_BYTES:
            raise ValueError(
                f"Invalid nonce length: {len(nonce):d} != {self.NONCE_BYTES}"
            )
        final_mac = bytearray(self.MAC_BYTES)
        self._inplace_encrypt(self.concatenated_macs(), final_mac, nonce)
        return final_mac

    def _transform_chunk(
        self, transform_func, chunk_index: int
    ) -> concurrent.futures.Future:
        view = self.chunk_view(chunk_index)
        mac = self.macs[chunk_index]
        nonce = self.nonces[chunk_index]
        future: concurrent.futures.Future = self.executor.submit(
            transform_func, view, mac, nonce, True
        )
        future.add_done_callback(lambda _: view.release())
        return future

    def _transform_all(
        self, func, wait: bool, timeout: Optional[float]
    ) -> Iterable[concurrent.futures.Future]:
        if not wait and timeout is not None:
            raise ValueError("Cannot specify a timeout if wait=False")
        futures = []
        for i in range(self.num_chunks):
            futures.append(func(i))
        if wait:
            self.wait_or_raise(futures, timeout)
        return futures

    @staticmethod
    def wait_or_raise(
        futures: Iterable[concurrent.futures.Future],
        timeout: Optional[float],
        return_when: str = concurrent.futures.FIRST_EXCEPTION,
    ) -> None:
        fs = concurrent.futures.wait(
            futures,
            timeout=timeout,
            return_when=return_when,
        )
        del futures
        raised = tuple(f for f in fs.done if f.exception() is not None)
        if raised:
            # Cancel other futures
            for f in fs.not_done:
                f.cancel()

            # Wait for cancelled futures to finish to ensure
            # that all buffers are released

            # noinspection PyTypeChecker
            cancel_timeout = None if timeout is None else timeout / 2
            fs = concurrent.futures.wait(
                fs.not_done,
                timeout=cancel_timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            ).not_done

            try:
                for f in raised:
                    f.result()  # Raise exceptions from threads
                for f in fs:
                    f.result(0)  # Raise exceptions from timeouts
            finally:
                del fs, raised
        elif fs.not_done:
            for f in fs.not_done:
                f.result(0)

    def encrypt_chunk(self, chunk_index: int) -> concurrent.futures.Future:
        return self._transform_chunk(self._inplace_encrypt, chunk_index)

    def encrypt_all(
        self, wait: bool = False, timeout: Optional[float] = None
    ) -> Iterable[concurrent.futures.Future]:
        return self._transform_all(self.encrypt_chunk, wait, timeout)

    def decrypt_chunk(self, chunk_index: int) -> concurrent.futures.Future:
        return self._transform_chunk(self._inplace_decrypt, chunk_index)

    def decrypt_all(
        self, wait: bool = False, timeout: Optional[float] = None
    ) -> Iterable[concurrent.futures.Future]:
        return self._transform_all(self.decrypt_chunk, wait, timeout)

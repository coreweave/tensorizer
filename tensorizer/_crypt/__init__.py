"""
Internal cryptographic library for tensorizer.
These functions are only meant to be used by tensorizer itself,
and are not guaranteed to have a stable interface across versions.
"""

__all__ = (
    "available",
    "ChunkedEncryption",
    "Const",
    "CryptographyError",
    "PWHash",
    "random_bytes",
)

from ._exceptions import CryptographyError

try:
    from ._encryption import (
        ChunkedEncryption,
        Const,
        PWHash,
        SequentialEncryption,
        random_bytes,
    )

    available: bool = True


except (OSError, AttributeError):
    available: bool = False

    def __getattr__(name):
        if name in __all__:
            raise RuntimeError(
                "Encryption module was not initialized,"
                " make sure a recent version of libsodium is installed"
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

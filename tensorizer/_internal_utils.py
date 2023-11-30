import dataclasses
import struct
import typing
from typing import Tuple, Union

_Buffer = Union[bytes, bytearray, memoryview]  # type: typing.TypeAlias


@dataclasses.dataclass(init=False)
class Chunked:
    __slots__ = ("count", "total_size", "chunk_size", "remainder")
    count: int
    total_size: int
    chunk_size: int
    remainder: int

    def __init__(self, total_size: int, chunk_size: int):
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.remainder = total_size % chunk_size
        self.count = total_size // chunk_size + (self.remainder != 0)


def _variable_read(
    data: bytes, offset: int = 0, length_fmt: str = "B", data_fmt: str = "s"
) -> Tuple[Union[memoryview, Tuple], int]:
    """
    Reads a variable-length field preceded by a length from a buffer.

    Returns:
        A tuple of the data read, and the offset in the buffer
        following the end of the field.
    """
    assert length_fmt in ("B", "H", "I", "Q")
    if length_fmt == "B":
        length: int = data[offset]
        offset += 1
    else:
        length_struct = struct.Struct("<" + length_fmt)
        length: int = length_struct.unpack_from(data, offset)[0]
        offset += length_struct.size
    if data_fmt == "s":
        # When the data is read as bytes, just return a memoryview
        end = offset + length
        return _unpack_memoryview_from(length, data, offset), end
    else:
        data_struct = struct.Struct(f"<{length:d}{data_fmt}")
        data = data_struct.unpack_from(data, offset)
        offset += data_struct.size
        return data, offset


def _unpack_memoryview_from(
    length: int, buffer: _Buffer, offset: int
) -> memoryview:
    # Grabbing a memoryview with bounds checking.
    # Bounds checking is normally provided by the struct module,
    # but it can't return memoryviews.
    with memoryview(buffer) as mv:
        end = offset + length
        view = mv[offset:end]
        if len(view) < length:
            view.release()
            mv.release()
            # Simulate a struct.error message for consistency
            raise struct.error(
                "unpack_from requires a buffer of at least"
                f" {length:d} bytes for unpacking {length:d} bytes at offset"
                f" {offset:d} (actual buffer size is {len(buffer):d})"
            )
        return view

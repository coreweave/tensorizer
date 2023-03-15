##############################################################################
# serialization.py                                                   Wes Brown
# Fast torch module/model serialization/deserialization     (c) 2023 Coreweave
##############################################################################
import hashlib
import zlib

# try to import UNIX only dependencies
try:
    import fcntl
    import resource
except ImportError:
    fcntl = None
    resource = None

from enum import Enum
import io
import os
import tensorizer.stream_io as stream_io
import tensorizer.utils as utils
import mmap
# Python doesn't know about this.
MADV_PAGEOUT = 21
import numpy
import struct
import torch
import typing
import logging
import tempfile
import regex

from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Iterator, Dict

lz4 = None

# Setup logger
logger = logging.getLogger(__name__)


# Whether the tensor is a parameter or a buffer on the model.
class TensorType(Enum):
    PARAM = 0
    BUFFER = 1
    STATEDICT = 2


TENSORIZER_VERSION = 1
TENSORIZER_MAGIC = b"|TZR|"


class TensorEntry(typing.TypedDict):
    name: str
    type: TensorType
    offset: int
    data_offset: int
    data_length: int
    length: int
    dtype: str
    shape: List[int]


class HashType(Enum):
    CRC32 = 0
    SHA256 = 1

class TensorHash(typing.TypedDict):
    type: HashType
    hash: bytes


class TensorDeserializer:
    """
    Given a file-like object for read, deserialize tensors to a state_dict or a
    torch.nn.Module.
    """

    def __init__(
            self,
            file_obj: Union[io.BufferedIOBase, io.RawIOBase, typing.BinaryIO, str],
            device: Union[torch.device, str, None] = None,
            pattern: Union[regex.Pattern, str, None] = None,
            dtype: Union[torch.dtype, str, None] = None,
            preload: bool = False,
            use_mmap: bool = False,
            oneshot: bool = False):
        if isinstance(file_obj, str):
            self._file = stream_io.open_stream(file_obj, "rb+")
        else:
            self._file = file_obj
        self.total_compressed_tensor_bytes = 0
        self.read_bytes = 0
        self._buffer = bytearray(1024 * 1024)
        self._state_dict = None
        self._idx = 0
        self._metadata: Dict[str, TensorEntry] = {}

        # Read the magic
        magic = self._file.read(5)
        if magic != TENSORIZER_MAGIC:
            raise ValueError("Not a tensorizer file")

        # Read the version
        version = struct.unpack("<I", self._file.read(4))[0]

        # Check the version
        if version > TENSORIZER_VERSION:
            raise ValueError(
                f"Cannot read version {version} with version {TENSORIZER_VERSION}"
            )

        # Skip 32-byte hash (unused)
        self._file.read(32)

        # Read the total size of the file
        self.total_file_bytes = struct.unpack("<Q", self._file.read(8))[0]

        # Read total size of tensor data
        self.total_tensor_bytes = struct.unpack("<Q", self._file.read(8))[0]

        # Read the number of tensors
        self._tensors = struct.unpack("<Q", self._file.read(8))[0]

        # The pattern is a regex that matches the tensor names to read and expose.
        if isinstance(pattern, str):
            pattern = regex.compile(pattern)
        self._pattern = pattern

        # Read the metadata index of tensors. This is a list of offsets into the
        # file where the per-tensor data is stored. pattern is a regex that
        # matches the tensor names to read. If pattern is None, all tensors are
        # read.
        self._read_metadatas(pattern)

        # If device is None, use the current device, otherwise use the given
        # device.
        self._device = device

        # If dtype is not None, convert all tensors to this dtype if possible.
        self._dtype = dtype

        if oneshot and use_mmap:
            raise ValueError("Cannot use mmap with oneshot")
        elif oneshot and preload:
            raise ValueError("Cannot use preload with oneshot")
        elif oneshot and device == "cpu":
            raise ValueError("Cannot use cpu device with oneshot")
        self._oneshot: bool = oneshot
        self._prior_key: Optional[str] = None

        # We calculate the total tensor bytes here so that we can use mmap, based
        # on the total size of the tensors that we're going to read, filtered by
        # the pattern.
        self.total_tensor_bytes = 0
        for name, metadata in self._metadata.items():
            self.total_tensor_bytes += metadata["data_length"]

        self._mmap = None
        self._mmap_allocated = 0
        if use_mmap:
            self._mmap_file = tempfile.TemporaryFile("wb+")
            #self._mmap_file.truncate(self.total_tensor_bytes)
            self._mmap_file.write(b'\0' * self.total_tensor_bytes)
            self._mmap_file.seek(0)
            self._mmap = mmap.mmap(self._mmap_file.fileno(),
                                   self.total_tensor_bytes)

        # Our cache of tensors. This is a dict of name -> tensor. If preload is
        # True, we load all filtered tensors into memory. Otherwise, we load tensors
        # on demand.
        self._cache: typing.OrderedDict[str, Union[torch.Tensor, None, bool]]

        # The offset in the file where the tensor data begins.
        self._tensors_begin = self._file.tell()

        if oneshot and preload:
            raise ValueError("Cannot use preload with oneshot")
        elif preload:
            self._cache = self._generate_state_dict(device=device,
                                                    pattern=pattern,
                                                    dtype=dtype)
        else:
            self._cache = OrderedDict.fromkeys(self._metadata.keys())

    def __del__(self):
        self.close()

    def close(self):
        # Don't throw an attribute error if these aren't defined yet,
        # e.g. if __init__ threw an error before defining both
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap_file.close()
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
        tensor_type = TensorType(struct.unpack("<B", metadata_stream.read(1))[0])
        dtype = self._read_dtype(metadata_stream)
        shape_len = struct.unpack("<B", metadata_stream.read(1))[0]
        shape = self._read_shapes(metadata_stream.read(shape_len * 4), shape_len)
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

    def _read_metadatas(self, pattern: Union[regex.Pattern, str, None]):
        """
        Read the metadata of tensors into self._metadata.
        """
        if isinstance(pattern, str):
            pattern = regex.compile(pattern)

        # Read metadata size.
        self._metadata_size = struct.unpack("<Q", self._file.read(8))[0]
        metadata_encoded = self._file.read(self._metadata_size)
        # Turn the metadata into a stream.
        metadata_stream = io.BytesIO(metadata_encoded)

        for i in range(self._tensors):
            metadata = self._read_metadata(metadata_stream)
            if pattern is None or pattern.match(metadata["name"]):
                self._metadata[metadata["name"]] = metadata

    @staticmethod
    def _read_shapes(obj, num_elems) -> List[int]:
        """
        Read the tensor shapes.
        """
        bstr = obj
        shape = []
        for i in range(num_elems):
            shape.append(struct.unpack("<I", bstr[0:4])[0])
            bstr = bstr[4:]
        return shape

    @property
    def total_bytes_read(self) -> int:
        if self._file.closed:
            return self.total_tensor_bytes
        else:
            return self._file.tell()

    def __getitem__(self, name) -> torch.nn.Parameter:
        if self._oneshot:
            if self._prior_key is not None and self._prior_key != name:
                self._cache[self._prior_key] = False
            if self._cache[name] is False:
                raise RuntimeError(f"Tensor {name} already overwritten in oneshot mode")
            self._prior_key = name

        if name in self._cache and self._cache[name] is not None:
            return self._cache[name]

        if name in self._metadata:
            tensor_arr = next(self.read_tensors(num_tensors=1))[3]
            self._cache[name] = self._to_torch_parameter(tensor_arr,
                                                         self._dtype,
                                                         self._device)
            return self._cache[name]
        else:
            raise KeyError(f"Tensor {name} not found")

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def keys(self):
        return self._metadata.keys()

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def _decode_hashes(self, b: bytes) -> List[TensorHash]:
        """
        Decode the hashes from given bytes.
        """
        hashes: List[TensorHash] = []

        # Read the number of hashes.
        num_hashes = struct.unpack("<B", b[0:1])[0]

        hash_idx = 1
        # Read the hashes.
        for i in range(num_hashes):
            # Read the hash type.
            hash_type = struct.unpack("<B", b[hash_idx:hash_idx+1])[0]
            # Read the size of the hash.
            hash_size = struct.unpack("<B", b[hash_idx+1:hash_idx+2])[0]
            # Read the hash.
            hash_begin = hash_idx + 2
            hash_end = hash_begin + hash_size
            hash_bytes = b[hash_begin:hash_end]
            # Add the hash to the list.
            hash_entry: TensorHash = {
                "type": hash_type,
                "hash": hash_bytes,
            }
            hash_idx = hash_end
            hashes.append(hash_entry)

        return hashes


    def read_tensors(
            self,
            pattern: Union[regex.Pattern, str, None] = None,
            num_tensors: int = -1,
    ) -> Iterator[Tuple[int, int, str, numpy.ndarray]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and the numpy `arr` that
        represents the tensor.

        Note that this function does not seek to the beginning of the tensor
        data. It assumes that the file pointer is already at the beginning
        of the tensor data that it should read.
        """
        if isinstance(pattern, str):
            pattern = regex.compile(pattern)

        try:
            tensors_read = 0
            while num_tensors == -1 or tensors_read < num_tensors:
                header_sz = struct.unpack("<Q", self._file.read(8))[0]
                if header_sz == 0:
                    break
                headers = self._file.read(header_sz - 8)
                header_len = len(headers)
                module_idx = struct.unpack("<H", headers[0:2])[0]
                tensor_type = TensorType(struct.unpack("<B", headers[2:3])[0])
                name_sz = struct.unpack("<H", headers[3:5])[0]
                idx = name_sz + 5
                name_bytes = headers[5:idx]
                name: str = name_bytes.decode("utf-8")
                dtype_len = struct.unpack("<B", headers[idx: idx + 1])[0]
                dtype_end = idx + dtype_len + 1
                dtype = headers[idx + 1: dtype_end]
                # read the shape amount, according to the serialized format. the shape
                # length is 1 byte after the dtype end.
                shape_len = struct.unpack(
                    "<B", headers[dtype_end: dtype_end + 1]
                )[0]
                # the shape elements are <I, so we read 4 bytes. _read_shapes takes in
                # the header object and the number of elements in the shape.
                #
                # the amount of bytes for the shape is 4 * the number of elements in
                # the shape. so, we need to read 4 * shape_len bytes after the
                # dtype end + 1 byte for the shape length. sort of convoluted, but it
                # works.
                shape_begin = dtype_end + 1
                shape_end = shape_begin + (4 * shape_len)
                shape_list = self._read_shapes(
                    headers[shape_begin:shape_end],
                    shape_len,
                )

                # Read our hashes in. We need to read the hashes size, then read the
                # hash bytes.
                #
                # TODO: Actually verify the hashes on request.
                hashes_begin = shape_end
                hashes_sz = struct.unpack("<H",
                                          headers[hashes_begin: hashes_begin + 2])[0]
                hashes_end = hashes_begin + hashes_sz
                hashes = self._decode_hashes(headers[hashes_begin + 2: hashes_end])

                data_length = struct.unpack("<q", headers[header_len - 8:])[0]
                # Check if the name matches the pattern, drop if it doesn't.
                if pattern is not None and not pattern.match(name):
                    self._file.seek(data_length, io.SEEK_CUR)
                    continue

                mv: memoryview
                if self._mmap is not None:
                    mmap_offset = self._mmap_allocated
                    mv = memoryview(self._mmap)[mmap_offset:data_length + mmap_offset]
                    self._file.readinto(mv)
                    self._mmap_allocated += data_length
                elif self._oneshot:
                    if data_length > len(self._buffer):
                        self._buffer = bytearray(data_length)
                    mv = memoryview(self._buffer)
                    self._file.readinto(mv[:data_length])
                else:
                    buffer = bytearray(data_length)
                    mv = memoryview(buffer)
                    self._file.readinto(mv)

                arr = numpy.ndarray.__new__(
                    numpy.memmap,
                    shape_list,
                    dtype=dtype,
                    buffer=mv,
                    offset=0,
                )
                tensors_read += 1
                yield module_idx, tensor_type, name, arr
        except EOFError:
            return

    @staticmethod
    def _to_torch_parameter(arr: numpy.ndarray,
                            dtype: Optional[str] = None,
                            device=utils.get_device()) -> torch.nn.Parameter:
        """
        Convert a numpy array to a torch tensor on a device.
        """
        if dtype is not None and arr.dtype != "bool" and arr.dtype != dtype:
            arr = arr.astype(dtype)
        gradient = arr.dtype.kind in ("f", "c")

        return torch.nn.Parameter(
            torch.as_tensor(arr, device=device), requires_grad=gradient
        )

    def _generate_state_dict(
            self,
            device=utils.get_device(),
            dtype: Optional[str] = None,
            pattern: Union[regex.Pattern, str, None] = None,
    ) -> OrderedDict:
        """
        Load the tensors in this Tensorizer object into a state_dict.

        :param device: The device to load the tensors onto.
        :param dtype: The dtype to load the tensors as. Defaults to None, which
            means the dtype is not changed from the serialized dtype.
        :param pattern: A regex pattern to match against the tensor names, if
            None, all tensors are loaded. If the pattern doesn't match, the
            tensor is skipped.
        :return:
        """
        if self._file.closed:
            raise IOError("IO closed, instantiate if you want to load again.")

        d = OrderedDict()
        for idx, typ, name, arr in self.read_tensors(pattern=pattern):
            d[name] = self._to_torch_parameter(arr, dtype, device)
        self.total_tensor_bytes = self._file.tell()
        self._file.close()
        return d

    def load_into_module(
            self,
            m: torch.nn.Module,
            device=utils.get_device(),
            dtype: Optional[str] = None,
            pattern: Union[regex.Pattern, str, None] = None,
    ) -> int:
        """
        Given `m`, a torch.nn.Module, load the associate tensors in this
        Tensorizer object into the `torch.nn.Module`. Returns the number of tensors
        loaded into the module.
        """
        if self._file.closed:
            raise IOError("IO closed, instantiate if you want to load again.")

        modules: typing.OrderedDict[str, torch.nn.Module] = OrderedDict()
        for name, module in m.named_modules():
            modules[name] = module

        tensor_ct = 0
        for idx, typ, name, arr in self.read_tensors(pattern=pattern):
            param = self._to_torch_parameter(arr, dtype, device)
            obj_path, attr = name.rsplit(".", 1)
            module: torch.nn.Module = modules[obj_path]
            if typ is TensorType.PARAM:
                module.register_parameter(attr, param)
            elif typ is TensorType.BUFFER:
                module.register_buffer(attr, param)
            elif typ is TensorType.STATEDICT:
                raise NotImplementedError(
                    "This was serialized using the write_state_dict() method, and"
                    " cannot be loaded using the load_tensors() method. Use the"
                    " state_dict() method instead.")
            tensor_ct += 1
        self.total_tensor_bytes = self._file.tell()
        self._file.close()
        return tensor_ct


class TensorSerializer:
    """
    Given a file-like object or path, serialize tensors from a torch.nn.Module
    to it.
    """

    def __init__(
            self,
            file_obj: Union[io.BufferedIOBase, io.RawIOBase, typing.BinaryIO,
            str, bytes, os.PathLike, int],
            compress_tensors: bool = False) -> None:
        if isinstance(file_obj, (str, bytes, os.PathLike, int)):
            self._file = stream_io.open_stream(file_obj, "wb+")
        else:
            self._file = file_obj
        self._tensors = 0
        self.total_tensor_bytes = 0
        self.total_compressed_tensor_bytes = 0
        self.compress_tensors = compress_tensors
        self.read_bytes = 0
        self._buffer = bytearray(1024 * 1024)
        self._state_dict = None
        self._idx = 0
        if self.compress_tensors:
            import lz4.frame
            self.lz4_frame = lz4.frame
        else:
            self.lz4_frame = None

        # Write our magic bytes.
        self._file.write(TENSORIZER_MAGIC)

        # Write the version number.
        self._file.write(struct.pack("<I", TENSORIZER_VERSION))

        # Reserve 32 bytes for the hash. (Unused for now)
        self._hash_loc = self._file.tell()
        self._file.write(struct.pack("<Q", 0) * 4)

        # Reserve 8 bytes for the total size of the file.
        self._size_loc = self._file.tell()
        self._file.write(struct.pack("<Q", 0))

        # Reserve 8 bytes for the total size of tensor data.
        self._tensor_size_loc = self._file.tell()
        self._file.write(struct.pack("<Q", 0))

        # Reserve the next 8 bytes for the total number of tensors.
        self._tensor_ct_loc = self._file.tell()
        self._file.write(struct.pack("<Q", 0))

        # Reserve 256kb for metadata.
        metadata_size = 256 * 1024
        self._file.write(struct.pack("<Q", metadata_size))
        self._metadata_loc = self._file.tell()
        self._file.write(struct.pack("<Q", 0) * (metadata_size // 8))
        self._metadata_cur = self._metadata_loc
        self._metadata_end = self._metadata_loc + metadata_size

        self._tensor_index: List[TensorEntry] = []

    def __del__(self):
        self._file.close()

    @staticmethod
    def _dump_shape(obj) -> bytes:
        """
        Returns shape of the tensor as an encoded byte string.
        """
        bstr = struct.pack("<B", len(obj))
        for i in obj:
            bstr += struct.pack("<I", i)
        return bstr

    def close(self) -> None:
        """
        Finalizes the serialization and closes the file.
        """
        self._sync_prologue_state()

        final_sz = self._file.tell()
        self._file.close()
        logger.info(f"Tensors completed serializing to {final_sz} bytes")
        if self.compress_tensors:
            compression_ratio = (
                    self.total_tensor_bytes / self.total_compressed_tensor_bytes
            )
            logger.info(f"Uncomp'd bytes: {self.total_tensor_bytes}")
            logger.info(f"Comp'd bytes: {self.total_compressed_tensor_bytes}")
            logger.info(f"Ratio: {compression_ratio:.2f}")

    def _sync_prologue_state(self):
        """
        This is called after the tensor has been written to the file, and
        ensures that the file is in a consistent state.
        :return:
        """
        curr = self._file.tell()

        # Write our zero-length field, that indicates that this is the last
        # tensor. This will be overwritten if another tensor is written.
        self._file.write(struct.pack("<Q", 0))
        # Write the total number of tensors.
        self._file.seek(self._tensor_ct_loc)
        self._file.write(struct.pack("<Q", self._tensors))
        # Write our total file size.
        self._file.seek(self._size_loc)
        self._file.write(struct.pack("<Q", curr))
        # Write the total bytes of tensor data written.
        self._file.seek(self._size_loc)
        self._file.write(struct.pack("<Q", self.total_tensor_bytes))

        # Reset our file pointer to the end of the file, minus the zero-length
        # field.
        self._file.seek(curr)

    def write_tensor(
            self,
            idx,
            name,
            tensor_type: TensorType,
            tensor: Union[torch.Tensor, numpy.ndarray]) -> None:
        """
        Serializes a tensor, laying things out so that it can be read in three
        calls from the input -- once for the size, once for the header, and
        once for the tensor itself.

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

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().detach().numpy()

        if len(str(tensor.dtype)) >= 256:
            raise ValueError("dtype length should be less than 256")

        # Reserve room for our tensor header size.
        ds_header_begin = self._file.tell()
        self._file.write(struct.pack("<Q", 0))

        # Module index.
        self._file.write(struct.pack("<H", idx))

        # Whether this is a parameter or a buffer
        self._file.write(struct.pack("<B", tensor_type.value))

        # Parameter/buffer name
        name_bytes = bytes(name, "utf-8")
        self._file.write(struct.pack("<H", len(name_bytes)))
        self._file.write(name_bytes)

        # Write out our tensor dtype
        dtype_bytes = bytes(tensor.dtype.str, "utf-8")
        dtype_len = len(dtype_bytes)
        self._file.write(struct.pack("<B", dtype_len))
        self._file.write(dtype_bytes)

        # ... and shape
        shape_bytes = self._dump_shape(tensor.shape)
        self._file.write(shape_bytes)

        # Reserve room for our tensor hashes.
        hash_loc = self._file.tell()
        # Reserve the length of the hash data structures.
        self._file.write(struct.pack("<H", 0))
        # Write the number of hashes we're going to write.
        self._file.write(struct.pack("<B", 2))
        # Reserve room for the hashes.
        # .. first, CRC32.
        self._file.write(struct.pack("<B", HashType.CRC32.value))
        # ... length of CRC32.
        self._file.write(struct.pack("<B", 4))
        # ... and reserve for the actual CRC32.
        crc32_loc = self._file.tell()
        self._file.write(struct.pack("<I", 0))
        # .. second, SHA256.
        self._file.write(struct.pack("<B", HashType.SHA256.value))
        # ... length of SHA256.
        self._file.write(struct.pack("<B", 32))
        # ... and reserve for the actual SHA256.
        sha256_loc = self._file.tell()
        self._file.write(struct.pack("<32B", *[0] * 32))
        hash_end = self._file.tell()
        hashes_sz = hash_end - hash_loc - 2

        # Reserve room for our 64-bit tensor length
        tensor_size_loc = self._file.tell()
        self._file.write(struct.pack("<Q", 0))
        tensor_startpos = self._file.tell()

        # Write the total number of bytes for our hash data structures.
        self._file.seek(hash_loc)
        self._file.write(struct.pack("<H", hashes_sz))
        self._file.seek(tensor_startpos)

        tensor_raw_sz = 0
        tensor_compressed_sz = 0
        compression_ratio = 0
        if self.compress_tensors:
            # NOTE: This compression feature is not complete, as we do not
            #       yet decompress. This was judged to *not* be worthwhile.
            #       This is left here as an example for future adventurers that
            #       may want to do model compression.
            # Create a write buffer to compress our tensor serialization.
            tensor_buffer = tempfile.TemporaryFile()
            tensor.tofile(tensor_buffer)
            tensor_raw_sz = tensor_buffer.tell()
            self.total_tensor_bytes += tensor_raw_sz
            tensor_buffer.seek(0)
            tensor_compressed = self.lz4_frame.compress(tensor_buffer.read())
            tensor_compressed_sz = len(tensor_compressed)
            compression_ratio = (tensor_raw_sz * 1.0) / tensor_compressed_sz
            if compression_ratio > 2:
                self.total_compressed_tensor_bytes += tensor_compressed_sz
            else:
                self.total_compressed_tensor_bytes += tensor_raw_sz

        # Serialize our tensors
        tensor.tofile(self._file)
        tensor_endpos = self._file.tell()

        # Go back and write our tensor length out
        self._file.seek(tensor_size_loc, io.SEEK_SET)
        # We write this signed, so that we can use the signedness as an
        # indicator of possible tensor compression in the future.
        tensor_size = tensor_endpos - tensor_startpos
        self._file.write(struct.pack("<q", tensor_size))
        self.total_tensor_bytes += tensor_size

        # Write our data structure header size.
        self._file.seek(ds_header_begin)
        ds_header_size = tensor_startpos - ds_header_begin
        self._file.write(struct.pack("<Q", ds_header_size))

        # Add our tensor metadata to the index.
        self._file.seek(self._metadata_cur)
        self._file.write(struct.pack("<H", len(name_bytes)))
        self._file.write(name_bytes)
        self._file.write(struct.pack("<B", tensor_type.value))
        self._file.write(struct.pack("<B", dtype_len))
        self._file.write(dtype_bytes)
        self._file.write(shape_bytes)
        self._file.write(struct.pack("<Q", ds_header_begin))
        self._file.write(struct.pack("<Q", tensor_startpos))
        self._file.write(struct.pack("<Q", tensor_endpos - tensor_startpos))
        self._metadata_cur = self._file.tell()

        # Check for overflow
        if self._file.tell() > self._metadata_end:
            raise RuntimeError("Metadata overflow")

        # Read our header and tensor back in to calculate the hashes.
        self._file.seek(ds_header_begin)
        ds_header_bytes = self._file.read(ds_header_size)
        tensor_bytes = self._file.read(tensor_endpos - tensor_startpos)

        # Write our hashes out.
        crc32 = zlib.crc32(ds_header_bytes + tensor_bytes)
        self._file.seek(crc32_loc)
        self._file.write(struct.pack("<I", crc32))

        sha256 = hashlib.sha256(ds_header_bytes + tensor_bytes).digest()
        self._file.seek(sha256_loc)
        self._file.write(sha256)

        # Move to the end of our serialized tensor to prepare for the next one.
        self._file.seek(tensor_endpos)
        self._tensors += 1

        # Update our prolog and epilog.
        self._sync_prologue_state()

        ds_size = self._file.tell() - ds_header_begin
        ds_bytes = f"{ds_size:,} bytes"

        typ = {TensorType.PARAM: "p",
               TensorType.BUFFER: "b",
               TensorType.STATEDICT: "sd"}[tensor_type]

        if self.compress_tensors:
            comp_report = (
                    f" - tensor:[raw: {tensor_raw_sz},"
                    + f" compressed: {tensor_compressed_sz},"
                    + f" ratio: {compression_ratio:.2f}]"
            )
        else:
            comp_report = ""
        logger.info(
            f"{idx}:{typ}:{name} - {dtype_bytes.decode('utf-8')} - "
            f"{tensor.shape} -> {ds_bytes}{comp_report}"
        )

    def write_module(self,
                     m: torch.nn.Module,
                     remove_tensors: bool = False) -> None:
        for module_name, module in m.named_modules():
            for name, param in module.named_parameters(recurse=False):
                label = module_name + "." + name
                self.write_tensor(self._idx, label, TensorType.PARAM, param)
                if remove_tensors:
                    setattr(module, name, None)
            for name, buf in module.named_buffers(recurse=False):
                label = module_name + "." + name
                self.write_tensor(self._idx, label, TensorType.BUFFER, buf)
                if remove_tensors:
                    setattr(module, name, None)
            self._idx += 1

    def write_state_dict(self, state_dict: Dict):
        """
        Write the state_dict to the file in Tensorizer format.
        """
        idx = 0
        for name, param in state_dict.items():
            self.write_tensor(idx, name, TensorType.STATEDICT, param)
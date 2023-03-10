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

from enum import Enum
import io
import os
import tensorizer.stream_io as stream_io
import tensorizer.utils as utils
import numpy
import struct
import torch
import typing
import logging
import tempfile
import regex

from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Iterator, Callable, Dict, Any

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
            preload: bool = False):
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

        # Read total size of file
        self.total_tensor_bytes = struct.unpack("<Q", self._file.read(8))[0]

        # Read the number of tensors
        self._tensors = struct.unpack("<Q", self._file.read(8))[0]

        # Read the metadata index of tensors.
        self._read_metadata()

        self._device = device
        if preload:
            self._cache = self._generate_state_dict(device=device,
                                                    pattern=pattern,
                                                    dtype=dtype)
        else:
            filtered_keys = []
            for key in self._metadata.keys():
                if pattern is None or pattern.match(key):
                    filtered_keys.append(key)
            self._cache = OrderedDict(zip(filtered_keys,
                                          [None] * len(filtered_keys)))

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

    def _read_metadata(self):
        """
        Read the metadata of tensors into self._metadata.
        """
        # Read metadata size.
        self._metadata_size = struct.unpack("<Q", self._file.read(8))[0]
        metadata_encoded = self._file.read(self._metadata_size)
        # Turn the metadata into a stream.
        metadata_stream = io.BytesIO(metadata_encoded)

        for i in range(self._tensors):
            name = self._read_string(metadata_stream)
            tensor_type = TensorType(struct.unpack("<B", metadata_stream.read(1))[0])
            dtype = self._read_dtype(metadata_stream)
            shape_len = struct.unpack("<B", metadata_stream.read(1))[0]
            shape = self._read_shapes(metadata_stream.read(shape_len * 4), shape_len)
            offset = struct.unpack("<Q", metadata_stream.read(8))[0]
            data_offset = struct.unpack("<Q", metadata_stream.read(8))[0]
            data_length = struct.unpack("<Q", metadata_stream.read(8))[0]
            self._metadata[name] = TensorEntry(
                name=name,
                type=tensor_type,
                offset=offset,
                data_offset=data_offset,
                data_length=data_length,
                dtype=dtype,
                shape=shape,
            )

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

    def __getitem__(self, name):
        if name in self._cache and self._cache[name] is not None:
            return self._cache[name]

        if name in self._metadata:
            self._file.seek(self._metadata[name]['offset'])
            tensor_arr = next(self.read_tensors(num_tensors=1))[3]
            self._cache[name] = self._to_torch_parameter(tensor_arr)
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

    def read_tensors(
            self,
            pattern: Union[regex.Pattern, str, None] = None,
            num_tensors: int = -1,
    ) -> Iterator[Tuple[int, int, str, numpy.ndarray]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and the numpy `arr` that
        represents the tensor.
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
                # the shape elements are <I so we read 4 bytes. _read_shapes takes in
                # the header object and the number of elements in the shape.
                #
                # the amount of bytes for the shape is 4 * the number of elements in
                # the shape. so, we need to read 4 * shape_len bytes after the
                # dtype end + 1 byte for the shape length. sort of convoluted, but it
                # works.
                shape_list = self._read_shapes(
                    headers[dtype_end + 1: (dtype_end + 1) + (4 * shape_len)],
                    shape_len,
                )
                data_length = struct.unpack("<q", headers[header_len - 8:])[0]
                if data_length > len(self._buffer):
                    self._buffer = bytearray(data_length)
                with memoryview(self._buffer) as mem:
                    self._file.readinto(mem[:data_length])
                # Check if the name matches the pattern, drop if it doesn't.
                if pattern is not None and not pattern.match(name):
                    continue

                arr = numpy.ndarray.__new__(
                    numpy.memmap,
                    shape_list,
                    dtype=dtype,
                    buffer=self._buffer,
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
        gradient = True
        if arr.dtype not in ["float", "complex"]:
            gradient = False
            if dtype is not None and arr.dtype != dtype:
                arr = arr.astype(dtype)
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

    def load_tensors(
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
        Returns shape of the tensor
        """
        bstr = struct.pack("<B", len(obj))
        for i in obj:
            bstr += struct.pack("<I", i)
        return bstr

    def close(self) -> None:
        """
        Finalizes the serialization with a zero-length field, and closes out
        the file.
        """
        self._file.write(struct.pack("<Q", 0))
        # Write the total number of tensors.
        self._file.seek(self._tensor_ct_loc)
        self._file.write(struct.pack("<Q", self._tensors))
        # Write the total size of the file.
        self._file.seek(self._size_loc)
        self._file.write(struct.pack("<Q", self._file.tell()))

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
        dtype_bytes = bytes(str(tensor.dtype), "utf-8")
        dtype_len = len(dtype_bytes)
        self._file.write(struct.pack("<B", dtype_len))
        self._file.write(dtype_bytes)

        # ... and shape
        shape_bytes = self._dump_shape(tensor.shape)
        self._file.write(shape_bytes)

        # Reserve room for our 64-bit tensor length
        tensor_size_loc = self._file.tell()
        self._file.write(struct.pack("<Q", 0))

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
        tensor_startpos = self._file.tell()
        tensor.tofile(self._file)
        tensor_endpos = self._file.tell()

        # Go back and write our tensor length out
        self._file.seek(tensor_size_loc, io.SEEK_SET)
        # We write this signed, so that we can use the signedness as an
        # indicator of possible tensor compression in the future.
        self._file.write(struct.pack("<Q", tensor_endpos - tensor_startpos))

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

        # Move to the end of our serialized tensor to prepare for the next one.
        self._file.seek(tensor_endpos)
        self._tensors += 1
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
        self.total_tensor_bytes = self._file.tell()

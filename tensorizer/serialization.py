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
import ctypes
import hashlib
import io
import logging
import mmap
import os
import struct
import tempfile
import time
import typing
import zlib
from enum import Enum

import numpy
import torch

import tensorizer.stream_io as stream_io
import tensorizer.utils as utils

if torch.cuda.is_available():
    cudart = torch.cuda.cudart()
else:
    cudart = None

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

lz4 = None

__all__ = ["TensorSerializer", "TensorDeserializer", "TensorType"]

# Setup logger
logger = logging.getLogger(__name__)


# Whether the tensor is a parameter or a buffer on the model.
class TensorType(Enum):
    PARAM = 0
    BUFFER = 1
    STATE_DICT = 2


TENSORIZER_VERSION = 1
TENSORIZER_MAGIC = b"|TZR|"


class TensorEntry(typing.TypedDict):
    name: str
    type: TensorType
    offset: int
    data_offset: int
    data_length: int
    dtype: str
    shape: List[int]


class HashType(Enum):
    CRC32 = 0
    SHA256 = 1


class TensorHash(typing.TypedDict):
    type: HashType
    hash: bytes


def _convert_dtype_to_numpy(dtype: Union[numpy.dtype, str, torch.dtype]):
    if isinstance(dtype, numpy.dtype):
        return dtype
    elif isinstance(dtype, str):
        return numpy.dtype(dtype)
    elif isinstance(dtype, torch.dtype):
        # Converted from PyTorch's own mapping used in their testing:
        # https://github.com/pytorch/pytorch/blob/v2.0.0/torch/testing/_internal/common_utils.py#L1009
        numpy_dtype = {
            torch.bool: numpy.bool_,
            torch.uint8: numpy.uint8,
            torch.int8: numpy.int8,
            torch.int16: numpy.int16,
            torch.int32: numpy.int32,
            torch.int64: numpy.int64,
            torch.float16: numpy.float16,
            torch.float32: numpy.float32,
            torch.float64: numpy.float64,
            torch.complex64: numpy.complex64,
            torch.complex128: numpy.complex128,
        }.get(dtype)

        if numpy_dtype is None:
            raise TypeError(
                "The provided torch.dtype class could not be converted to a"
                " numpy.dtype"
            )
        else:
            # Convert it from a dtype class to a dtype object instance
            return numpy.dtype(numpy_dtype)
    else:
        raise TypeError("Could not interpret provided dtype as a numpy.dtype")


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
        dtype: The dtype to load the tensors as. If None, the dtype will be
            inferred from the file.
        lazy_load: If True, tensors will be loaded and cached when keys are
            accessed. If False, all tensors will be loaded into memory up
            front.
        plaid_mode: If True, tensors will be loaded extremely fast into the
            target device. This is only supported on CUDA devices, and the
            buffers are going to be inconsistent due to the extreme
            naughtiness of reusing a backing buffer. This is only recommended
            for use with inference, and not training.

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
        device: Union[torch.device, str, None] = None,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        dtype: Union[numpy.dtype, str, None] = None,
        *,
        lazy_load: bool = False,
        plaid_mode: bool = False,
    ):
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
        self._device = device

        # If dtype is not None, convert all tensors to this dtype if possible.
        if dtype is None:
            self._dtype = None
        else:
            self._dtype = _convert_dtype_to_numpy(dtype)

        self._metadata: Dict[str, TensorEntry] = {}

        if plaid_mode and (
            not torch.cuda.is_available() or self._device.type == "cpu"
        ):
            raise ValueError("Plaid mode requires CUDA")

        # plaid_mode implies lazy_load
        self._lazy_load = lazy_load or plaid_mode

        # Read the magic
        magic = self._file.read(5)
        if magic != TENSORIZER_MAGIC:
            raise ValueError("Not a tensorizer file")

        # Read the version
        version = struct.unpack("<I", self._file.read(4))[0]

        # Check the version
        if version > TENSORIZER_VERSION:
            raise ValueError(
                f"This {TENSORIZER_VERSION} version cannot read"
                f"file versioned {version}."
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

        self._plaid_mode: bool = plaid_mode
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
        if not lazy_load and not plaid_mode:
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

        if not lazy_load:
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
        if self._file.closed:
            return self.total_tensor_bytes
        else:
            return self._file.tell()

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
    def _decode_hashes(b: bytes) -> List[TensorHash]:
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
            hash_type = struct.unpack("<B", b[hash_idx : hash_idx + 1])[0]
            # Read the size of the hash.
            hash_size = struct.unpack("<B", b[hash_idx + 1 : hash_idx + 2])[0]
            # Read the hash.
            hash_begin = hash_idx + 2
            hash_end = hash_begin + hash_size
            hash_bytes: bytes = b[hash_begin:hash_end]
            # Add the hash to the list.
            hash_entry = TensorHash(
                type=HashType(hash_type),
                hash=hash_bytes,
            )
            hash_idx = hash_end
            hashes.append(hash_entry)

        return hashes

    def read_tensors(
        self,
        filter_func: Optional[Callable[[str], Union[bool, Any]]] = None,
        num_tensors: int = -1,
    ) -> Iterator[Tuple[int, int, str, numpy.ndarray]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and the numpy `arr` that
        represents the tensor.

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
        """

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
                dtype = headers[idx + 1 : dtype_end]

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

                # Read our hashes in. We need to read the hashes size, then
                # read the hash bytes.
                #
                # TODO: Actually verify the hashes on request.
                hashes_begin = shape_end
                hashes_sz_slice = headers[hashes_begin : hashes_begin + 2]
                hashes_sz = struct.unpack("<H", hashes_sz_slice)[0]
                hashes_end = hashes_begin + hashes_sz

                hashes_slice = headers[hashes_begin + 2 : hashes_end]
                hashes = self._decode_hashes(hashes_slice)

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

                # Convert the memoryview to a numpy array. We use the
                # memmap class to avoid copying the data.
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

    def _to_torch_parameter(
        self, arr: Union[numpy.ndarray, torch.nn.Parameter]
    ) -> torch.nn.Parameter:
        """
        Convert a numpy array to a torch.nn.Parameter on a device, forcing
        gradient when appropriate. We also handle torch.nn.Parameter objects in
        a passthrough manner.
        """
        if isinstance(arr, torch.nn.Parameter):
            arr.data = arr.data.to(self._device)
            if arr.grad is not None:
                arr.grad = arr.grad.to(self._device)
            return arr

        if (
            self._dtype is not None
            and arr.dtype != "bool"
            and arr.dtype != self._dtype
        ):
            arr = arr.astype(self._dtype)
        gradient = arr.dtype.kind in ("f", "c")

        return torch.nn.Parameter(
            torch.from_numpy(arr).to(self._device), requires_grad=gradient
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
        """
        modules: typing.OrderedDict[str, torch.nn.Module] = OrderedDict()

        for name, module in m.named_modules():
            modules[name] = module

        tensor_ct = 0

        for name in self.keys():
            if filter_func is not None and not filter_func(name):
                continue
            obj_path, attr = name.rsplit(".", 1)
            module: torch.nn.Module = modules[obj_path]
            entry = self._metadata[name]
            param = self._to_torch_parameter(self.get(name))
            if entry["type"] is TensorType.PARAM:
                module.register_parameter(attr, param)
            elif entry["type"] is TensorType.BUFFER:
                module.register_buffer(attr, param)
            elif entry["type"] is TensorType.STATE_DICT:
                raise NotImplementedError(
                    "This was serialized using the write_state_dict() method,"
                    " and cannot be loaded using the load_tensors() method."
                    " Use the TensorDeserializer object directly as a"
                    " state_dict mapping instead."
                )
            tensor_ct += 1

        self._file.close()
        return tensor_ct


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
        self._tensors = 0
        self._idx = 0
        self.total_tensor_bytes = 0
        self.total_compressed_tensor_bytes = 0
        self.compress_tensors = compress_tensors
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
        if getattr(self, "_file", None) is not None:
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
        bytes_to_hash = self._file.read(ds_header_size)
        bytes_to_hash += self._file.read(tensor_endpos - tensor_startpos)

        # Write our hashes out.
        crc32 = zlib.crc32(bytes_to_hash)
        self._file.seek(crc32_loc)
        self._file.write(struct.pack("<I", crc32))

        sha256 = hashlib.sha256(bytes_to_hash).digest()
        self._file.seek(sha256_loc)
        self._file.write(sha256)

        # Move to the end of our serialized tensor to prepare for the next one.
        self._file.seek(tensor_endpos)
        self._tensors += 1

        # Update our prolog and epilog.
        self._sync_prologue_state()

        ds_size = self._file.tell() - ds_header_begin
        ds_bytes = f"{ds_size:,} bytes"

        typ = {
            TensorType.PARAM: "p",
            TensorType.BUFFER: "b",
            TensorType.STATE_DICT: "sd",
        }[tensor_type]

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

    def write_module(
        self, m: torch.nn.Module, remove_tensors: bool = False
    ) -> None:
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

        It is strongly recommended that you use write_module instead of
        this function, as it will also write out the parameter type,
        allowing for zero-copy loading of the module with
        TensorDeserializer.load_into_module.
        """
        idx = 0
        for name, param in state_dict.items():
            self.write_tensor(idx, name, TensorType.STATE_DICT, param)

##############################################################################
# tensorizer.py                                                      Wes Brown
# Fast LLM model serialization and deserialization          (c) 2022 Coreweave
##############################################################################

import fcntl
import subprocess
import resource
import io
from io import SEEK_SET, SEEK_END
import numpy
import struct
import torch
import typing
from dotmap import DotMap
import pickle
import pickletools
import time
import logging
import sys
import pathlib

thisPath = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(thisPath + "/../../transformers/src")
sys.path.append(thisPath + "/../../")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import no_init_weights, PreTrainedModel
from lm_node.unitrim import Unitrimmer
from tokenizers import Tokenizer

import json
import os
from urllib import request
import tempfile

from collections import OrderedDict
from typing import Tuple, Union, List, Iterator, Callable

lz4 = None

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
)
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# =============================================================================
# From `pipe(7)` manpage:
#
# Pipe capacity
# A pipe has a limited capacity. If the pipe is full, then a write(2) will
# block or fail, depending on whether the O_NONBLOCK flag is set (see below).
# Different implementations have different limits for the pipe capacity.
#
# Applications should not rely on a particular capacity: an application should
# be designed so that a reading process consumes data as soon as it is
# available, so that a writing process does not remain blocked.
#
# In Linux versions before 2.6.11, the capacity of a pipe was the same as the
# system page size (e.g., 4096 bytes on i386). Since Linux 2.6.11, the pipe
# capacity is 16 pages (i.e., 65,536 bytes in a system with a page size of
# 4096 bytes). Since Linux 2.6.35, the default pipe capacity is 16 pages, but
# the capacity can be queried and set using the fcntl(2) F_GETPIPE_SZ and
# F_SETPIPE_SZ operations. See fcntl(2) for more information.
#
# =============================================================================
# From `fcntl(2)` manpage:
#
# Changing the capacity of a pipe
#
# F_SETPIPE_SZ (int; since Linux 2.6.35)
# Change the capacity of the pipe referred to by fd to be at least arg bytes.
# An unprivileged process can adjust the pipe capacity to any value between the
# system page size and the limit defined in /proc/sys/fs/pipe−max−size
# (see proc(5)). Attempts to set the pipe capacity below the page size are
# silently rounded up to the page size. Attempts by an unprivileged process to
# set the pipe capacity above the limit in /proc/sys/fs/pipe−max−size yield the
# error EPERM; a privileged process (CAP_SYS_RESOURCE) can override the limit.
#
# When allocating the buffer for the pipe, the kernel may use a capacity larger
# than arg, if that is convenient for the implementation. (In the current
# implementation, the allocation is the next higher power-of-two page-size
# multiple of the requested size.) The actual capacity (in bytes) that is set
# is returned as the function result.
#
# Attempting to set the pipe capacity smaller than the amount of buffer space
# currently used to store data produces the error EBUSY.
#
# Note that because of the way the pages of the pipe buffer are employed when
# data is written to the pipe, the number of bytes that can be written may be
# less than the nominal size, depending on the size of the writes.
#
# F_GETPIPE_SZ (void; since Linux 2.6.35)
# Return (as the function result) the capacity of the pipe referred to by fd.
#
# =============================================================================
# Constant for `F_SETPIPE_SZ`, as python3's `fcntl` module doesn't have this
# defined -- despite the documentation saying that they're there.
#
# TODO: Make this work or fail gracefully on non-Linux systems. Not sure if
#       this is really relevant, as I don't even know if CUDA is available on
#       non-Linux systems in a production sense.
F_SETPIPE_SZ = 1031

# Whether the tensor is a parameter or a buffer on the model.
TENSOR_PARAM = 0
TENSOR_BUFFER = 1


# Silly function to convert to human bytes
def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    step_unit = 1000.0

    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit


def no_init(loading_code: Callable[[], PreTrainedModel]) -> PreTrainedModel:
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    with no_init_weights():
        result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def no_init_or_tensor(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy
    original_empty = torch.empty
    torch.empty = lambda *a, **b: None

    with no_init_weights():
        result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]
    torch.empty = original_empty

    return result


class CURLStreamFile(object):
    """
    CURLStreamFile implements a file-like object around an HTTP download, the
    intention being to not buffer more than we have to.
    """

    def __init__(self, uri: str) -> None:
        # NOTE: `256mb` buffer on the python IO object.
        self._curl = subprocess.Popen(
            [
                "/usr/bin/curl",
                "--header",
                "Accept-Encoding: identity",
                "-s",
                uri,
            ],
            stdout=subprocess.PIPE,
            bufsize=256 * 1024 * 1024,
        )
        # Read our max-fd-size, fall back to 1mb if invalid.
        pipe_buf_sz = 1024 * 1024
        try:
            pipe_file = open("/proc/sys/fs/pipe-max-size", "r")
            pipe_buf_sz = int(pipe_file.read())
            logger.debug(f"pipe-max-size: {pipe_buf_sz}")
        except IOError as e:
            logger.warning(
                f"Could not read /proc/sys/fs/pipe-max-size: {e.strerror}"
            )
        try:
            fcntl.fcntl(self._curl.stdout.fileno(), F_SETPIPE_SZ, pipe_buf_sz)
        except PermissionError as e:
            logger.warning(
                f"Couldn't fcntl F_SETPIPE_SZ to {pipe_buf_sz}: {e.strerror}"
            )
        self._curr = 0
        self.closed = False

    def _read_until(
        self, goal_position: int, ba: Union[bytearray, None] = None
    ) -> Union[bytes, int]:
        if ba is None:
            rq_sz = goal_position - self._curr
            ret_buff = self._curl.stdout.read(rq_sz)
            ret_buff_sz = len(ret_buff)
        else:
            rq_sz = len(ba)
            ret_buff_sz = self._curl.stdout.readinto(ba)
            ret_buff = ba
        if ret_buff_sz != rq_sz:
            self.closed = True
            err = self._curl.stderr.read()
            self._curl.terminate()
            if self._curl.returncode != 0:
                raise (IOError(f"curl error: {self._curl.returncode}, {err}"))
            else:
                raise (IOError(f"Requested {rq_sz} != {ret_buff_sz}"))
        self._curr += ret_buff_sz
        if ba is None:
            return ret_buff
        else:
            return ret_buff_sz

    def tell(self) -> int:
        return self._curr

    def readinto(self, ba: bytearray) -> int:
        goal_position = self._curr + len(ba)
        return self._read_until(goal_position, ba)

    def read(self, size=None) -> bytes:
        if self.closed:
            raise (IOError("CURLStreamFile closed."))
        if size is None:
            return self._curl.stdout.read()
        goal_position = self._curr + size
        return self._read_until(goal_position)

    @staticmethod
    def writable() -> bool:
        return False

    @staticmethod
    def fileno() -> int:
        return -1

    def close(self):
        self.closed = True
        self._curl.terminate()

    def readline(self):
        raise Exception("Unimplemented")

    """
    This seek() implementation is effectively a no-op, and will throw an
    exception for anything other than a seek to the current position.
    """

    def seek(self, position, whence=SEEK_SET):
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise (Exception("Unsupported `whence`"))
        else:
            raise (Exception("Seeking is unsupported"))


class ResponseStream(object):
    """
    This is intended to provide a file-like wrapper around an iterator that
    yields block chunks. Useful for wrapping around streaming `reaponse`
    requests, but not as fast as `CURLStreamFile`
    """

    def __init__(self, request_iterator):
        self._iterator = request_iterator
        self._curr = 0
        self._buff = bytearray()
        self.closed = False

    def _read_until(self, goal_position):
        while len(self._buff) + self._curr < goal_position:
            try:
                self._buff.extend(next(self._iterator))
            except StopIteration:
                break
        if len(self._buff) + self._curr >= goal_position:
            ret_buff = bytes(self._buff[: goal_position - self._curr])
            self._buff = self._buff[goal_position - self._curr :]
            self._curr = goal_position
            return ret_buff

    def tell(self):
        return self._curr

    def read(self, size=None):
        if self.closed:
            raise IOError("ResponseStream closed.")
        goal_position = self._curr + size
        return self._read_until(goal_position)

    @staticmethod
    def writable():
        return False

    @staticmethod
    def fileno():
        return -1

    def readline(self):
        raise Exception("Unimplemented")

    def close(self):
        self.closed = True
        del self._iterator

    def seek(self, position, whence=SEEK_SET):
        if position == self._curr:
            return
        if whence == SEEK_END:
            raise (Exception("Unsupported `whence`"))
        else:
            raise (Exception("Seeking is unsupported"))


class GooseTensorizer:
    """
    Given a file-like object, either for read or write, serialize tensors
    to the file, or deserialize the tensors.
    """

    def __init__(self, file_obj, compress_tensors=False) -> None:
        self._file = file_obj
        self._tensors = 0
        self.total_tensor_bytes = 0
        self.total_compressed_tensor_bytes = 0
        self.compress_tensors = compress_tensors
        self.read_bytes = 0
        self._buffer = bytearray(1024 * 1024)
        if self.compress_tensors:
            import lz4.frame

    @staticmethod
    def _pickle_dump_fix(obj, memo_start_idx=0) -> Tuple[bytes, int]:
        """
        Pickle and object and optimize its string by changing MEMOIZE into PUT,
        removing unused PUT/MEMOIZE, fixing GET opcodes, and remove PROTO,
        FRAME, and STOP opcodes.

        Returns the pickle bytes, and the end memo index.
        """
        p = pickle.dumps(obj, 4)
        oldids = set()
        newids = {}
        opcodes = []

        # Trick to avoid instantiating objects (we use the "is" operator)
        put = "PUT"
        get = "GET"

        ops = list(pickletools.genops(p))
        ops = [(x[0], x[1], x[2], y[2]) for x, y in zip(ops[:-1], ops[1:])] + [
            (ops[-1][0], ops[-1][1], ops[-1][2], len(p))
        ]
        for opcode, arg, pos, end_pos in ops:
            if opcode.name in ("FRAME", "STOP"):
                # Ignore these
                pass
            elif opcode.name == "PROTO":
                # Ignore, but check that it's version 4
                assert arg == 4, "Pickle version should be 4"
            elif "PUT" in opcode.name:
                oldids.add(arg)
                opcodes.append((put, arg))
            elif opcode.name == "MEMOIZE":
                idx = len(oldids)
                oldids.add(idx)
                opcodes.append((put, idx))
            elif "GET" in opcode.name:
                newids[arg] = None
                opcodes.append((get, arg))
            else:
                opcodes.append((pos, end_pos))
        del oldids

        out = []
        memo_put_idx = memo_start_idx
        for op, arg in opcodes:
            if op is put:
                if arg not in newids:
                    continue
                newids[arg] = memo_put_idx
                if memo_put_idx < 256:
                    data = pickle.BINPUT + struct.pack("<B", memo_put_idx)
                else:
                    data = pickle.LONG_BINPUT + struct.pack("<I", memo_put_idx)
                memo_put_idx += 1
            elif op is get:
                memo_get_idx = newids[arg]
                if memo_get_idx < 256:
                    data = pickle.BINGET + struct.pack("<B", memo_get_idx)
                else:
                    data = pickle.LONG_BINGET + struct.pack("<I", memo_get_idx)
            else:
                data = p[op:arg]

            out.append(data)
        return b"".join(out), memo_put_idx

    @property
    def total_bytes_read(self) -> int:
        if self._file.closed:
            return self.total_tensor_bytes
        else:
            return self._file.tell()

    def finalize(self) -> None:
        """
        Finalizes the serialization with a zero-length field, and closes out
        the file.
        """
        self._file.write(struct.pack("<Q", 0))
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

    def write_tensor(self, idx, name, tensor_type, tensor) -> None:
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

        if len(str(tensor.dtype)) >= 256:
            raise ValueError("dtype length should be less than 256")

        # Reserve room for our tensor header size.
        ds_header_begin = self._file.tell()
        self._file.write(struct.pack("<Q", 0))

        # Module index.
        self._file.write(struct.pack("<H", idx))

        # Whether this is a parameter or a buffer
        self._file.write(struct.pack("<B", tensor_type))

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
        self._file.write(self._pickle_dump_fix(tensor.shape)[0])

        # Reserve room for our 64-bit tensor length
        tensor_size_loc = self._file.tell()
        self._file.write(struct.pack("<q", 0))

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
            tensor_compressed = lz4.frame.compress(tensor_buffer.read())
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
        self._file.write(struct.pack("<q", tensor_endpos - tensor_startpos))

        # Write our data structure header size.
        self._file.seek(ds_header_begin)
        ds_header_size = tensor_startpos - ds_header_begin
        self._file.write(struct.pack("<Q", ds_header_size))

        # Move to the end of our serialized tensor to prepare for the next one.
        self._file.seek(tensor_endpos)
        self._tensors += 1
        ds_size = self._file.tell() - ds_header_begin
        ds_bytes = f"{ds_size:,} bytes"

        if tensor_type == TENSOR_PARAM:
            typ = "p"
        else:
            typ = "b"
        if self.compress_tensors:
            comp_report = (
                f" - tensor:[raw: {tensor_raw_sz},"
                + f" compressed: {tensor_compressed_sz},"
                + f" ratio: {compression_ratio:.2f}]"
            )
        else:
            comp_report = ""
        logger.info(
            f"{idx}:{typ}:{name} - {tensor.shape} -> {ds_bytes}{comp_report}"
        )

    @staticmethod
    def _read_shapes(shape_bytes) -> List[int]:
        """
        Read the tensor shapes from pickled format.
        """
        shape_buf = io.BytesIO(shape_bytes)
        shapelist = []
        while True:
            shapeelementtype = shape_buf.read(1)
            if shapeelementtype == pickle.BININT1:
                shapeelement = struct.unpack("<B", shape_buf.read(1))[0]
            elif shapeelementtype == pickle.BININT2:
                shapeelement = struct.unpack("<H", shape_buf.read(2))[0]
            elif shapeelementtype == pickle.BININT:
                shapeelement = struct.unpack("<i", shape_buf.read(4))[0]
            elif shapeelementtype == pickle.EMPTY_TUPLE:
                break
            elif shapeelementtype in (
                pickle.TUPLE1,
                pickle.TUPLE2,
                pickle.TUPLE3,
                pickle.TUPLE,
            ):
                # End of tuple
                break
            elif shapeelementtype == pickle.MARK:
                continue  # ignore mark
            else:
                assert False, "Invalid element type: 0x{:02x}".format(
                    ord(shapeelementtype)
                )
            shapelist.append(shapeelement)
        return shapelist

    def read_tensors(self) -> Iterator[Tuple[int, int, str, numpy.ndarray]]:
        """
        A generator that deserializes tensors and returns the `module_idx`,
        `tensor_type`, parameter/buffer `name`, and the numpy `arr` that
        represents the tensor.
        """
        try:
            while True:
                header_sz = struct.unpack("<Q", self._file.read(8))[0]
                if header_sz == 0:
                    break
                headers = self._file.read(header_sz - 8)
                header_len = len(headers)
                module_idx = struct.unpack("<H", headers[0:2])[0]
                tensor_type = struct.unpack("<B", headers[2:3])[0]
                name_sz = struct.unpack("<H", headers[3:5])[0]
                idx = name_sz + 5
                name_bytes = headers[5:idx]
                name: str = name_bytes.decode("utf-8")
                dtype_len = struct.unpack("<B", headers[idx : idx + 1])[0]
                dtype_end = idx + dtype_len + 1
                dtype = headers[idx + 1 : dtype_end]
                shapelist = self._read_shapes(
                    headers[dtype_end : header_len - 8]
                )
                datalength = struct.unpack("<q", headers[header_len - 8 :])[0]
                if datalength > len(self._buffer):
                    self._buffer = bytearray(datalength)
                self._file.readinto(memoryview(self._buffer)[:datalength])
                arr = numpy.ndarray.__new__(
                    numpy.memmap,
                    shapelist,
                    dtype=dtype,
                    buffer=self._buffer,
                    offset=0,
                )
                # Get rid of the error on load.
                yield module_idx, tensor_type, name, arr
        except EOFError:
            return

    def load_tensors_dict(
        self, device=torch.device("cuda"), dtype: str = "float16"
    ) -> (int, OrderedDict):
        tensor_ct = 0
        d = OrderedDict()
        for idx, typ, name, arr in self.read_tensors():
            gradient = False
            if arr.dtype != "bool":
                gradient = True
                if arr.dtype != dtype:
                    arr = arr.astype(dtype)
            d[name] = torch.nn.Parameter(
                torch.as_tensor(arr, device=device), requires_grad=gradient
            )
            tensor_ct += 1
        self.total_tensor_bytes = self._file.tell()
        self._file.close()
        return tensor_ct, d

    def load_tensors(
        self,
        m: torch.nn.Module,
        device=torch.device("cuda"),
        dtype: [None, str] = None,
    ) -> int:
        """
        Given `m`, a torch.nn.Module, load the associate tensors in this
        GooseTensorizer object into the `torch.nn.Module`. Returns the number
        of tensors loaded into the model.

        Note that this assumes that the `m` is serialized along with this.
        """
        if self._file.closed:
            raise IOError("IO closed, instantiate if you want to load again.")

        modules: typing.OrderedDict[str, torch.nn.Module] = OrderedDict()
        for name, module in m.named_modules():
            modules[name] = module

        tensor_ct = 0
        for idx, typ, name, arr in self.read_tensors():
            gradient = False
            if arr.dtype != "bool":
                gradient = True
                if dtype is not None and arr.dtype != dtype:
                    arr = arr.astype(dtype)
            obj_path, attr = name.rsplit(".", 1)
            module: torch.nn.Module = modules[obj_path]
            param = torch.nn.Parameter(
                torch.as_tensor(arr, device=device), requires_grad=gradient
            )
            if typ is TENSOR_PARAM:
                module.register_parameter(attr, param)
            elif typ is TENSOR_BUFFER:
                module.register_buffer(attr, param)
            tensor_ct += 1
        self.total_tensor_bytes = self._file.tell()
        self._file.close()
        return tensor_ct


def get_tokenizer(
    path_uri: str, args={}
) -> Tuple[Tokenizer, Unitrimmer, List[int]]:
    """
    Given a path prefix, load and return:
      * tokenizer loaded from `{prefix}/tokenizer.json`
      * unitrim loaded from `{prefix}/unitrim.json`

    Given an URL, load:
      * tokenizer loaded from `/tokenizer.json`
      * unitrim loaded from `/unitrim.json`
    """

    tokenizer_uri = f"{path_uri}"
    unitrim_uri = f"{path_uri}/unitrim.json"
    wordtokens_uri = f"{path_uri}/wordtokens.json"

    if path_uri.startswith("https://") or path_uri.startswith("http://"):
        unitrim_loader = lambda: Unitrimmer(
            io.BytesIO(CURLStreamFile(unitrim_uri).read())
        )
        wordtokens_loader = lambda: json.loads(
            CURLStreamFile(wordtokens_uri).read()
        )
    else:
        unitrim_loader = lambda: Unitrimmer(open(unitrim_uri, "r"))
        wordtokens_loader = lambda: json.load(open(wordtokens_uri, "r"))

    with tempfile.TemporaryDirectory() as dir:
        for tokenizer_file in [
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ]:
            tokenizer_path = "/".join([tokenizer_uri, tokenizer_file])
            logger.info(f"Loading {tokenizer_path}")
            request.urlretrieve(tokenizer_path, "/".join([dir, tokenizer_file]))
        try:
            tokenizer = AutoTokenizer.from_pretrained(dir, **args)
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_uri}!")
            raise e

    try:
        logger.info(f"Loading {unitrim_uri}")
        unitrim = unitrim_loader()
    except Exception as e:
        logger.error(f"Failed to load unitrim from {unitrim_uri}!")
        raise e

    try:
        logger.info(f"Loading {wordtokens_uri}")
        wordtokens = wordtokens_loader()
    except Exception as e:
        logger.error(f"Failed to load wordtokens.json from {unitrim_uri}!")
        raise e

    return tokenizer, unitrim, wordtokens


def save_to_state_dict(m, destination, prefix, keep_vars):
    r"""Saves module state to `destination` dictionary, containing a state
    of the module, but not its descendants. This is called on every
    submodule in :meth:`~torch.nn.Module.state_dict`.

    In rare cases, subclasses can achieve class-specific behavior by
    overriding this method with custom logic.

    Args:
        destination (dict): a dict where state will be stored
        prefix (str): the prefix for parameters and buffers used in this
            module
    """
    for name, param in m._parameters.items():
        destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in m._buffers.items():
        if name not in m._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()
    extra_state_key = prefix + "_extra_state"
    if (
        getattr(m.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
        is not torch.nn.Module.get_extra_state
    ):
        destination[extra_state_key] = m.get_extra_state()


def state_dict(
    m: torch.nn.Module, destination=None, prefix="", keep_vars=True
) -> OrderedDict:
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=m._version
    )
    save_to_state_dict(m, destination, prefix, keep_vars)
    # m._save_to_state_dict(destination, prefix, keep_vars)
    for name, module in m._modules.items():
        state_dict(
            module, destination, prefix + name + ".", keep_vars=keep_vars
        )
    for hook in m._state_dict_hooks.values():
        hook_result = hook(m, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def serialize_model(m: torch.nn.Module, out_prefix: str):
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays and serialize them to GooseTensor format. The stripped
    model is also serialized to pytorch format.
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    f = open(f"{out_prefix}.tensors", "wb")
    ts = GooseTensorizer(f)

    idx = 0
    for module_name, module in m.named_modules():
        print(module_name)
        for name, param in module.named_parameters(recurse=False):
            v = param.cpu().detach().numpy()
            ts.write_tensor(idx, module_name + "." + name, TENSOR_PARAM, v)
            setattr(module, name, None)
        for name, buf in module.named_buffers(recurse=False):
            v = buf.cpu().detach().numpy()
            ts.write_tensor(idx, module_name + "." + name, TENSOR_BUFFER, v)
            setattr(module, name, None)
        idx += 1

    ts.finalize()

    # m.train(False)

    # torch.save(y, f"{out_prefix}.state", _use_new_zipfile_serialization=False)
    # torch.save(m, f"{out_prefix}.model", _use_new_zipfile_serialization=False)


def load_model(path_uri: str) -> torch.nn.Module:
    """
    Given a path prefix, load:
      * model object from `{prefix}.model`
      * tensors into the model from `{prefix}.tensors`

    Given an URL, load:
      * model object from `/model`
      * tensors into the model from `/tensors`
    """
    if path_uri.startswith("https://") or path_uri.startswith("http://"):
        config_uri = f"{path_uri}/config.json"
        tensors_uri = f"{path_uri}/tensors"
        tensor_loader = lambda: CURLStreamFile(tensors_uri)
    else:
        tensors_uri = f"{path_uri}/tensors"
        config_uri = "file://" + os.path.join(
            os.path.dirname(path_uri), "config.json"
        )
        tensor_loader = lambda: open(tensors_uri, "rb")

    # begin_model_load = time.time()
    maxrss_b4 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1000 * 1000
    )
    maxrss_b4_gb = f"{maxrss_b4:0.2f}gb CPU RAM used"
    logger.info(f"Loading {tensors_uri}, {maxrss_b4_gb}")
    begin_load = time.time()

    with tempfile.TemporaryDirectory() as dir:
        config_path = os.path.join(dir, "config.json")
        request.urlretrieve(config_uri, config_path)
        config = AutoConfig.from_pretrained(dir)
        config.gradient_checkpointing = True

    tensor_deserializer = GooseTensorizer(tensor_loader())

    # TODO: suppress warnings about missing tensors from `state_dict`
    model = no_init_or_tensor(
        lambda: AutoModelForCausalLM.from_pretrained(
            None, config=config, state_dict=OrderedDict()
        )
    )
    tensor_deserializer.load_tensors(model, dtype="float16")

    tensor_load_s = time.time() - begin_load
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1000 * 1000)
    maxrss_gb = f"{maxrss:0.2f}gb CPU RAM used"
    rate_str = convert_bytes(
        tensor_deserializer.total_bytes_read / tensor_load_s
    )
    tensors_sz = convert_bytes(tensor_deserializer.total_bytes_read)
    logger.info(
        f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
        + f"{tensors_sz} @ {rate_str}/s, {maxrss_gb}"
    )

    return model


def main():
    if len(sys.argv) != 3:
        logger.fatal(f"{sys.argv[0]} [input-directory] [output-prefix]")
        sys.exit(1)

    from lm_node.base import GPTModel

    model_name = os.path.basename(sys.argv[1])
    output_prefix = sys.argv[2]
    config = DotMap()
    config.model_type = "GPT"
    config.model_name = model_name
    config.model_path = sys.argv[1]
    config.is_dev = True
    config.model_alias = None
    config.prefix_path = None
    config.deepspeed_enabled = False
    config.load_serialized = False
    print("MODEL PATH:", config.model_path)

    cudadev = torch.cuda.current_device()
    gb_gpu = int(
        torch.cuda.get_device_properties(0).total_memory / (1000 * 1000 * 1000)
    )
    gb_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1000 * 1000)

    logger.info("GPU: " + torch.cuda.get_device_name(cudadev))
    logger.info("GPU RAM: " + str(gb_gpu) + "gb")
    logger.info("PYTHON USED RAM: " + str(gb_cpu) + "gb")

    model = GPTModel(config)
    serialize_model(model.model, output_prefix)
    if os.path.isdir(output_prefix):
        addl_prefix = output_prefix
    else:
        addl_prefix = os.path.dirname(output_prefix)
    if addl_prefix == "":
        addl_prefix = "."
    model.tokenizer.save_pretrained(addl_prefix)
    model.model.config.to_json_file(addl_prefix + "/config.json")
    open(addl_prefix + "/unitrim.json", mode="w").write(
        model.unitrim.serialize()
    )
    open(addl_prefix + "/wordtokens.json", mode="w").write(
        json.dumps(model.word_tokens)
    )

    logger.info("Validating serialization")
    model = load_model(f"{output_prefix}")
    logger.info(f"force_fp32_attn: {model.config.force_fp32_attn}")


if __name__ == "__main__":
    main()

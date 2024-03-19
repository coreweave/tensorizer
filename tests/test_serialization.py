import contextlib
import ctypes
import enum
import functools
import gc
import hashlib
import itertools
import logging
import os
import re
import secrets
import sys
import tempfile
import time
import typing
import unittest
from typing import Iterator, Mapping, NamedTuple, Optional, Tuple
from unittest.mock import patch

import torch

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # avoids excessive warnings about forking after using a tokenizer
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tensorizer import (
    DecryptionParams,
    EncryptionParams,
    TensorDeserializer,
    TensorSerializer,
    serialization,
    stream_io,
    utils,
)
from tensorizer._crypt import available as encryption_available
from tensorizer.serialization import TensorHash, TensorType

try:
    from test_stream_io import start_redis, teardown_redis
except ImportError:
    from .test_stream_io import start_redis, teardown_redis

model_name = "EleutherAI/gpt-neo-125M"
num_hellos = 400
is_cuda_available = torch.cuda.is_available()
default_device = "cuda" if is_cuda_available else "cpu"
salt = secrets.token_bytes(4)
default_read_endpoint = "object.ord1.coreweave.com"


def _stdout_handler(level=logging.DEBUG) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(levelname)s %(name)s: %(msg)s")
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler


debug_handler = _stdout_handler()


@contextlib.contextmanager
def debug_log():
    serialization.logger.addHandler(debug_handler)
    must_enable: bool = not serialization.logger.isEnabledFor(logging.DEBUG)
    old_level = serialization.logger.level
    if must_enable:
        serialization.logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        if must_enable:
            serialization.logger.setLevel(old_level)
        serialization.logger.removeHandler(debug_handler)


class SerializeMethod(enum.Enum):
    Module = 1
    StateDict = 2


class SerializationResult(NamedTuple):
    filename: str
    orig_sd: dict


def serialize_model(
    model_name: str,
    device: str,
    method: SerializeMethod = SerializeMethod.Module,
    encryption: Optional[EncryptionParams] = None,
) -> SerializationResult:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sd = model.state_dict()
    out_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    try:
        start_time = time.monotonic()
        serializer = TensorSerializer(out_file, encryption=encryption)
        if method is SerializeMethod.Module:
            serializer.write_module(model)
        elif method is SerializeMethod.StateDict:
            serializer.write_state_dict(sd)
        else:
            raise ValueError("Invalid serialization method")
        serializer.close()
        end_time = time.monotonic()
        print(f"Serialization took {end_time - start_time:.3f} seconds")
    except Exception:
        os.unlink(out_file.name)
        raise
    return SerializationResult(out_file.name, sd)


@contextlib.contextmanager
@functools.wraps(serialize_model)
def serialize_model_temp(*args, **kwargs):
    filename = serialize_model(*args, **kwargs).filename
    try:
        yield filename
    finally:
        os.unlink(filename)


# Reducing a tensor to a hash makes it faster to compare against the reference
# model in many repeated tests
class TensorInfo(NamedTuple):
    size: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    hash: bytes

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorInfo":
        shape = tuple(tensor.size())
        storage = tensor.untyped_storage().cpu()
        data = ctypes.cast(
            storage.data_ptr(),
            ctypes.POINTER(ctypes.c_ubyte * storage.nbytes()),
        ).contents
        hash_val = hashlib.blake2b(data, digest_size=16, salt=salt).digest()
        return cls(
            size=tensor.size(), shape=shape, dtype=tensor.dtype, hash=hash_val
        )


@functools.lru_cache(maxsize=None)
def model_digest(
    model_name: str, include_non_persistent_buffers: bool = True
) -> Mapping[str, TensorInfo]:
    orig_model = AutoModelForCausalLM.from_pretrained(model_name)
    orig_sd = orig_model.state_dict()
    # Non-persistent buffers are serialized in tensorizer,
    # but aren't included in a state_dict() in PyTorch.
    if include_non_persistent_buffers:
        orig_sd.update(orig_model.named_buffers())
    return {k: TensorInfo.from_tensor(v) for k, v in orig_sd.items()}


def check_deserialized(
    test_case: unittest.TestCase,
    deserialized: TensorDeserializer,
    model_name: str,
    allow_subset: bool = False,
    include_non_persistent_buffers: bool = True,
):
    orig_sd = model_digest(model_name, include_non_persistent_buffers)

    if not allow_subset:
        test_case.assertEqual(
            orig_sd.keys(),
            deserialized.keys(),
            "List of deserialized keys doesn't match list of original keys",
        )

    for k, v in deserialized.items():
        test_case.assertIn(
            k,
            orig_sd,
            f"Key not from original: {k} not in {orig_sd.keys()}",
        )

        v_info = TensorInfo.from_tensor(v)
        orig_info = orig_sd[k]

        test_case.assertEqual(
            v_info.size,
            orig_info.size,
            f"Sizes don't match for tensor {k}: {v_info.size} !="
            f" {orig_info.size}",
        )

        test_case.assertEqual(
            v_info.shape,
            orig_info.shape,
            f"Shapes don't match for tensor {k}: {v_info.shape} !="
            f" {orig_info.shape}",
        )

        test_case.assertEqual(
            v_info.dtype,
            orig_info.dtype,
            f"dtypes don't match for tensor {k}: {v_info.dtype} !="
            f" {orig_info.dtype}",
        )

        test_case.assertEqual(
            v_info.hash,
            orig_info.hash,
            f"Contents don't match for tensor {k}",
        )

    del orig_sd
    gc.collect()


@contextlib.contextmanager
def enable_tokenizers_parallelism():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        yield
    finally:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def check_inference(
    test_case: unittest.TestCase,
    deserializer: TensorDeserializer,
    model_ref: str,
    device: str,
):
    # This ensures that the model is not initialized.
    config = AutoConfig.from_pretrained(model_ref)
    with utils.no_init_or_tensor():
        model = AutoModelForCausalLM.from_config(config)

    deserializer.load_into_module(model)

    # Tokenize and generate
    with enable_tokenizers_parallelism():
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        eos = tokenizer.eos_token_id
        input_ids = tokenizer.encode(
            " hello" * num_hellos, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=50, do_sample=True, pad_token_id=eos
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        test_case.assertGreater(decoded.count("hello"), num_hellos)


@contextlib.contextmanager
@functools.wraps(tempfile.NamedTemporaryFile)
def temporary_file(*args, **kwargs):
    f = tempfile.NamedTemporaryFile(
        *args, **kwargs, prefix="tensorizer-test", delete=False
    )
    try:
        yield f
    finally:
        os.unlink(f.name)


class TestSerialization(unittest.TestCase):
    def test_serialization(self):
        for device, method in itertools.product(
            ("cuda", "cpu"),
            (SerializeMethod.Module, SerializeMethod.StateDict),
        ):
            if device == "cuda" and not is_cuda_available:
                continue
            include_non_persistent_buffers = method is SerializeMethod.Module
            with self.subTest(
                msg=f"Serializing with device {device} and method {method.name}"
            ):
                gc.collect()
                before_serialization = utils.get_mem_usage()
                print(f"\nBefore serialization: {before_serialization}")
                serialized_model, orig_sd = serialize_model(
                    model_name, device, method
                )
                after_serialization = utils.get_mem_usage()
                print(f"After serialization:  {after_serialization}")
                del orig_sd
                try:
                    deserialized = TensorDeserializer(
                        serialized_model, device="cpu"
                    )
                    check_deserialized(
                        self,
                        deserialized,
                        model_name,
                        include_non_persistent_buffers=(
                            include_non_persistent_buffers
                        ),
                    )
                    deserialized.close()
                    del deserialized
                finally:
                    os.unlink(serialized_model)

    def test_large_unbuffered_tensor(self):
        shape = (36000, 36000)  # 4.828 GiB
        dtype = torch.float32
        num_elements: int = 36000 * 36000
        bytes_required: int = num_elements * 4
        assert bytes_required > 1 << 32
        gc.collect()
        free_mem = utils.CPUMemoryUsage.now().free
        working_space: int = 10 << 20
        if free_mem < bytes_required + working_space:
            self.skipTest(
                reason="Insufficient RAM to test large tensor serialization"
            )
        low_mem: bool = free_mem < bytes_required * 2 + working_space
        tensor = torch.empty(shape, device="cpu", dtype=dtype)
        tensor[0, 0] = 1.0101
        tensor[18000, 18000] = 1.2345
        tensor[-1, -1] = 5.4331
        with temporary_file("wb+", buffering=0) as tensorized_file:
            with tensorized_file:
                serializer = TensorSerializer(tensorized_file)
                serializer.write_state_dict({"tensor": tensor})
                serializer.close()
            del serializer
            if low_mem:
                serialized_digest = TensorInfo.from_tensor(tensor)
                tensor = None
                gc.collect()
            else:
                serialized_digest = None
            with open(
                tensorized_file.name, "rb"
            ) as in_file, TensorDeserializer(
                in_file, device="cpu"
            ) as deserializer:
                deserialized_tensor = deserializer["tensor"]
                if low_mem:
                    deserialized_digest = TensorInfo.from_tensor(
                        deserialized_tensor
                    )
                    self.assertTupleEqual(
                        serialized_digest, deserialized_digest
                    )
                else:
                    self.assertTrue(torch.equal(tensor, deserialized_tensor))
        del deserializer, tensor, deserialized_tensor
        gc.collect()

    def test_bfloat16(self):
        shape = (50, 50)
        tensor = torch.normal(0, 0.5, shape, dtype=torch.bfloat16)
        tensorized_file = tempfile.NamedTemporaryFile("wb+", delete=False)

        try:
            serializer = TensorSerializer(tensorized_file)
            serializer.write_tensor(0, "test_tensor", TensorType.PARAM, tensor)
            serializer.close()

            deserializer = TensorDeserializer(
                tensorized_file.name, device="cpu", lazy_load=True
            )
            deserialized_tensor = [t for t in deserializer.read_tensors()][0][
                -1
            ]
        finally:
            os.unlink(tensorized_file.name)

        self.assertTrue(torch.equal(tensor, deserialized_tensor))

    def test_meta_tensors(self):
        # This test is modeled after self.test_persistent_buffers
        shape = (50, 50)
        materialized_tensor = torch.normal(0, 0.5, shape)
        meta_tensor = torch.empty_like(materialized_tensor, device="meta")
        zero_tensor = torch.zeros_like(materialized_tensor)
        nested_module = torch.nn.Module()
        nested_module.register_parameter(
            "materialized_tensor", torch.nn.Parameter(materialized_tensor)
        )
        nested_module.register_parameter(
            "meta_tensor", torch.nn.Parameter(meta_tensor)
        )
        module = torch.nn.Module()
        module.register_module("nested", nested_module)
        model = torch.nn.Module()
        model.register_module("module", module)

        expected = {
            "module.nested.materialized_tensor": materialized_tensor,
            "module.nested.meta_tensor": zero_tensor,
        }

        def assert_deserialized(d: TensorDeserializer) -> None:
            self.assertGreaterEqual(
                d._file_header.version_number,
                serialization.META_TENSOR_TENSORIZER_VERSION,
            )
            self.assertSetEqual(set(d.keys()), set(expected.keys()))
            device = d._device
            for name, value in expected.items():
                self.assertTrue(
                    torch.equal(d[name], expected[name].to(device=device)),
                    msg=f"Tensor {name!r} is incorrect on deserialization",
                )

        def settings() -> Iterator[dict]:
            devices = ("cpu", "cuda") if is_cuda_available else ("cpu",)
            for device, lazy_load, plaid_mode in itertools.product(
                devices, (True, False), (True, False)
            ):
                if device == "cpu" and plaid_mode:
                    continue
                yield dict(
                    device=device, lazy_load=lazy_load, plaid_mode=plaid_mode
                )

        tensorized_file = tempfile.NamedTemporaryFile("wb+", delete=False)
        try:
            serializer = TensorSerializer(tensorized_file)
            serializer.write_module(model)
            serializer.close()

            for setting in settings():
                with self.subTest(encrypted=False, **setting), open(
                    tensorized_file.name, "rb"
                ) as in_file, TensorDeserializer(
                    in_file, **setting
                ) as deserializer:
                    assert_deserialized(deserializer)
                    self.assertNotIn(
                        serialization._FileFeatureFlags.encrypted,
                        deserializer._file_flags,
                    )

            with self.subTest("Meta tensors with encryption"):
                if not encryption_available:
                    self.skipTest(
                        "libsodium must be installed to test encryption"
                    )
                encryption_params = serialization.EncryptionParams.random()
                decryption_params = serialization.DecryptionParams.from_key(
                    encryption_params.key
                )
                serializer = TensorSerializer(
                    tensorized_file.name, encryption=encryption_params
                )
                serializer.write_module(model)
                serializer.close()

                for setting in settings():
                    with self.subTest(encrypted=True, **setting), open(
                        tensorized_file.name, "rb"
                    ) as in_file, TensorDeserializer(
                        in_file, encryption=decryption_params, **setting
                    ) as deserializer:
                        assert_deserialized(deserializer)
                        self.assertIn(
                            serialization._FileFeatureFlags.encrypted,
                            deserializer._file_flags,
                        )

        finally:
            os.unlink(tensorized_file.name)

    def test_meta_tensor_module(self):
        meta_model = AutoModelForCausalLM.from_pretrained(model_name).to(
            device="meta"
        )
        sd = meta_model.state_dict()
        sd.update(meta_model.named_buffers())
        self.assertDictEqual(
            {name: t.device.type for name, t in sd.items()},
            dict.fromkeys(sd, "meta"),
        )
        serialized_file = tempfile.NamedTemporaryFile("wb+", delete=False)
        try:
            serializer = TensorSerializer(serialized_file)
            serializer.write_module(meta_model)
            serializer.close()
            with TensorDeserializer(
                serialized_file.name, device="cpu"
            ) as deserializer, torch.no_grad():
                self.assertSetEqual(set(sd.keys()), set(deserializer.keys()))
                for k in deserializer.keys():
                    zero = torch.zeros_like(sd[k], device="cpu")
                    if torch.any(zero):
                        # Some torch bug causes zeros_like to yield nonzero
                        # results sometimes (TM) when converting from
                        # a meta tensor
                        zero.zero_()
                    self.assertTrue(torch.equal(zero, deserializer[k]))
        finally:
            os.unlink(serialized_file.name)

    def test_persistent_buffers(self):
        def random_tensor(shape=(50, 50)):
            return torch.normal(0, 0.5, shape)

        parameter = torch.nn.Parameter(random_tensor(), requires_grad=False)
        persistent_buffer = random_tensor()
        non_persistent_buffer = random_tensor()
        nested_module = torch.nn.Module()
        nested_module.register_parameter("parameter", parameter)
        nested_module.register_buffer(
            "persistent_buffer", persistent_buffer, persistent=True
        )
        nested_module.register_buffer(
            "non_persistent_buffer",
            non_persistent_buffer,
            persistent=False,
        )
        module = torch.nn.Module()
        module.register_module("nested", nested_module)
        model = torch.nn.Module()
        model.register_module("module", module)
        model.eval()

        for include in (True, False):
            with self.subTest(
                msg=f"Testing include_non_persistent_buffers={include}"
            ):
                expected: dict = {
                    "module.nested.parameter": parameter,
                    "module.nested.persistent_buffer": persistent_buffer,
                }
                if include:
                    expected["module.nested.non_persistent_buffer"] = (
                        non_persistent_buffer
                    )
                tensorized_file = tempfile.NamedTemporaryFile(
                    "wb+", delete=False
                )
                try:
                    serializer = TensorSerializer(tensorized_file)
                    serializer.write_module(
                        model, include_non_persistent_buffers=include
                    )
                    serializer.close()

                    with TensorDeserializer(
                        tensorized_file.name, device="cpu"
                    ) as deserializer:
                        self.assertSetEqual(
                            set(deserializer.keys()),
                            set(expected.keys()),
                        )
                        for name in deserializer.keys():
                            self.assertTrue(
                                torch.equal(deserializer[name], expected[name]),
                                msg=(
                                    f"Contents of tensor {name!r}"
                                    " are different after deserialization"
                                ),
                            )
                finally:
                    os.unlink(tensorized_file.name)

    def test_structure(self):
        with temporary_file(mode="wb+") as file:
            devices = ("cpu", "cuda") if is_cuda_available else ("cpu",)
            for lazy_load, device, num_readers in itertools.product(
                (False, True), devices, (1, 4)
            ):

                def try_serialize(obj):
                    serializer = TensorSerializer(file.name)
                    serializer.write_state_dict(obj)
                    serializer.close()
                    return TensorDeserializer(
                        file.name,
                        lazy_load=lazy_load,
                        device=device,
                        num_readers=num_readers,
                    )

                tensors: typing.List[torch.Tensor] = [
                    torch.tensor((i,), device=device, dtype=torch.uint8)
                    for i in range(10)
                ]

                with self.subTest("Flat list"), try_serialize(
                    tensors
                ) as deserialized:
                    # Should have consecutive integer keys
                    self.assertSequenceEqual(
                        tuple(deserialized.keys()), range(10)
                    )

                    # Iterating over the mapping values should be equivalent
                    # to iterating over the list values
                    self.assertSequenceEqual(
                        [v[0] for v in deserialized.values()],
                        [v[0] for v in tensors],
                    )

                    # The tree view should be an actual flat list
                    tree: list = deserialized.tree()
                    self.assertListEqual(tree, list(deserialized.values()))

                    # Elements should be accessible by index
                    self.assertListEqual(
                        [deserialized[i] for i in range(10)], tree
                    )

                    # Should be able to check for integer keys
                    self.assertIn(5, deserialized)
                    self.assertNotIn(-1, deserialized)
                    self.assertNotIn(11, deserialized)
                    # And there should be no string keys
                    self.assertNotIn("5", deserialized)
                    self.assertNotIn("-1", deserialized)
                    self.assertNotIn("11", deserialized)

                structure = {str(i): v for i, v in enumerate(tensors)}

                with self.subTest("Flat state_dict"), try_serialize(
                    structure
                ) as deserialized:
                    # Should have string keys
                    structure_keys = tuple(map(str, range(10)))
                    self.assertSequenceEqual(
                        tuple(deserialized.keys()), structure_keys
                    )

                    # Iterating over the mapping values should be equivalent
                    # to iterating over the list values
                    self.assertSequenceEqual(
                        [v[0] for v in deserialized.values()],
                        [v[0] for v in tensors],
                    )

                    # The tree view should be a flat dictionary
                    tree: dict = deserialized.tree()
                    self.assertDictEqual(tree, dict(deserialized))

                    # Elements should be accessible by string key
                    self.assertListEqual(
                        [deserialized[i] for i in structure_keys],
                        [tree[i] for i in structure_keys],
                    )

                    # Should be able to check for string keys
                    self.assertIn("5", deserialized)
                    self.assertNotIn("-1", deserialized)
                    self.assertNotIn("11", deserialized)
                    # And there should be no integer keys
                    self.assertNotIn(5, deserialized)
                    self.assertNotIn(-1, deserialized)
                    self.assertNotIn(11, deserialized)

                structure = [{str(i): v} for i, v in enumerate(tensors)]

                with self.subTest(
                    "List of single-element state_dicts"
                ), try_serialize(structure) as deserialized:
                    # Should have integer keys on the first layer
                    structure_keys = range(10)
                    self.assertSequenceEqual(
                        tuple(deserialized.keys()), structure_keys
                    )

                    # Iterating over the mapping values should yield
                    # nested mappings
                    for i, v in deserialized.items():
                        self.assertIsInstance(i, int)
                        self.assertIsInstance(v, typing.Mapping)
                        # With a single string key each
                        key = str(i)
                        self.assertSequenceEqual(tuple(v.keys()), (key,))
                        # And the appropriate value
                        reference = structure[i]
                        self.assertEqual(v[key][0], reference[key][0])

                    # The tree view should look just like the original structure
                    tree: list = deserialized.tree()
                    self.assertListEqual(tree, list(deserialized.values()))

                    # Elements should be accessible by integer key
                    self.assertListEqual(
                        [deserialized[i] for i in structure_keys],
                        [tree[i] for i in structure_keys],
                    )

                    # Should be able to check for integer keys
                    self.assertIn(5, deserialized)
                    self.assertNotIn(-1, deserialized)
                    self.assertNotIn(11, deserialized)
                    # And there should be no string keys in the first layer
                    self.assertNotIn("5", deserialized)
                    self.assertNotIn("-1", deserialized)
                    self.assertNotIn("11", deserialized)

                structure = {
                    str(i): [tensors[i], tensors[-i]] for i in range(10)
                }

                with self.subTest("Dict of two-element lists"), try_serialize(
                    structure
                ) as deserialized:
                    # Should have string keys on the first layer
                    structure_keys = tuple(map(str, range(10)))
                    self.assertSequenceEqual(
                        tuple(deserialized.keys()), structure_keys
                    )

                    # Iterating over the mapping values should yield
                    # nested mappings with integer keys, not lists
                    for i, v in deserialized.items():
                        self.assertIsInstance(i, str)
                        self.assertIsInstance(v, typing.Mapping)
                        # With two integer keys each
                        self.assertSequenceEqual(tuple(v.keys()), (0, 1))
                        reference = structure[i]
                        for idx in range(2):
                            self.assertEqual(v[idx][0], reference[idx][0])

                    # The tree view should look just like the original structure
                    tree: dict = deserialized.tree()
                    self.assertSequenceEqual(tuple(tree.keys()), structure_keys)
                    for k in structure_keys:
                        # The nested lists are actual lists here
                        tree_sublist = tree[k]
                        self.assertIsInstance(tree_sublist, list)
                        # And their contents should match the originals
                        # and the ordering of the originals
                        original_sublist = structure[k]
                        self.assertListEqual(
                            [v[0] for v in tree_sublist],
                            [v[0] for v in original_sublist],
                        )

                    # Should be able to check for string keys in the first layer
                    self.assertIn("5", deserialized)
                    self.assertNotIn("-1", deserialized)
                    self.assertNotIn("11", deserialized)
                    # And there should be no integer keys
                    self.assertNotIn(5, deserialized)
                    self.assertNotIn(-1, deserialized)
                    self.assertNotIn(11, deserialized)

                structure = [
                    {str(i): tensors[i], str(10 - i - 1): tensors[10 - i - 1]}
                    for i in range(10)
                ]

                with self.subTest("List of two-element dicts"), try_serialize(
                    structure
                ) as deserialized:
                    # Should have integer keys on the first layer
                    structure_keys = range(10)
                    self.assertSequenceEqual(
                        tuple(deserialized.keys()), structure_keys
                    )

                    # Iterating over the mapping values should yield
                    # nested mappings
                    for i, v in deserialized.items():
                        self.assertIsInstance(i, int)
                        self.assertIsInstance(v, typing.Mapping)
                        # With two string keys each
                        keys = (str(i), str(10 - i - 1))
                        self.assertSequenceEqual(tuple(v.keys()), keys)
                        reference = structure[i]
                        for k in keys:
                            self.assertEqual(v[k][0], reference[k][0])

                    # The tree view should look just like the original structure
                    tree: list = deserialized.tree()
                    self.assertIsInstance(tree, list)
                    for k in structure_keys:
                        tree_subdict = tree[k]
                        self.assertIsInstance(tree_subdict, typing.Mapping)
                        # And their contents should match the originals
                        # and the ordering of the originals
                        original_subdict = structure[k]
                        self.assertTupleEqual(
                            tuple(tree_subdict.keys()),
                            tuple(original_subdict.keys()),
                        )
                        self.assertListEqual(
                            [v[0] for v in tree_subdict.values()],
                            [v[0] for v in original_subdict.values()],
                        )

                    # Test subtree access of a nested mapping
                    subtree: typing.Mapping = deserialized.tree((1,))
                    self.assertIsInstance(subtree, typing.Mapping)
                    for (k1, v1), (k2, v2) in zip(
                        subtree.items(), tree[1].items()
                    ):
                        self.assertEqual(k1, k2)
                        self.assertEqual(v1[0], v2[0])
                    # Test subtree access of a specific leaf tensor
                    self.assertEqual(
                        deserialized.tree((0, "9"))[0], tree[0]["9"][0]
                    )

                    # Should be able to check for int keys in the first layer
                    self.assertIn(5, deserialized)
                    self.assertNotIn(-1, deserialized)
                    self.assertNotIn(11, deserialized)
                    # And there should be no string keys
                    self.assertNotIn("5", deserialized)
                    self.assertNotIn("-1", deserialized)
                    self.assertNotIn("11", deserialized)


@unittest.skipUnless(
    encryption_available,
    reason="libsodium must be installed to test encryption",
)
class TestEncryption(unittest.TestCase):
    @staticmethod
    def _serialize(enc: Optional[EncryptionParams], device=default_device):
        return serialize_model_temp(
            model_name,
            device,
            method=SerializeMethod.Module,
            encryption=enc,
        )

    @staticmethod
    def _test_first_key_negative(obj):
        k = next(iter(obj.keys()))
        if obj[k] is not None:
            raise RuntimeError()

    def test_encryption(self):
        fixed_salt = bytes(16)
        low_cpu = EncryptionParams.OpsLimit.MIN
        low_mem = EncryptionParams.MemLimit.MIN
        encryption = EncryptionParams.from_string(
            source="test",
            opslimit=low_cpu,
            memlimit=low_mem,
            salt=fixed_salt,
        )
        decryption = DecryptionParams.from_string(source="test")
        incorrect_decryption = DecryptionParams.from_string(source="tset")
        device = default_device

        with self._serialize(encryption, device) as encrypted_model:
            if device == "cuda":
                modes = (
                    (False, False),
                    (False, True),
                    (True, True),
                )
            else:
                modes = (
                    (False, False),
                    (True, False),
                )
            for lazy_load, plaid_mode in modes:
                # Ensure that it works when given a key
                with self.subTest(
                    msg="Deserializing with a correct key",
                    device=device,
                    lazy_load=lazy_load,
                    plaid_mode=plaid_mode,
                ), TensorDeserializer(
                    encrypted_model,
                    device=device,
                    lazy_load=lazy_load,
                    plaid_mode=plaid_mode,
                    verify_hash=True,
                    encryption=decryption,
                ) as deserialized:
                    check_deserialized(
                        self,
                        deserialized,
                        model_name,
                    )
                    del deserialized
                gc.collect()

            # Ensure that it fails to load when not given a key
            with self.subTest(
                msg="Deserializing with a missing key"
            ), self.assertRaises(
                serialization.CryptographyError
            ), TensorDeserializer(
                encrypted_model,
                device=device,
                lazy_load=True,
                encryption=None,
            ) as deserialized:
                self._test_first_key_negative(deserialized)
                del deserialized
            gc.collect()

            # Ensure that it fails to load when given the wrong key
            with self.subTest(
                msg="Deserializing with an incorrect key"
            ), self.assertRaises(
                serialization.CryptographyError
            ), TensorDeserializer(
                encrypted_model,
                device=device,
                lazy_load=True,
                encryption=incorrect_decryption,
            ) as deserialized:
                self._test_first_key_negative(deserialized)
                del deserialized
            gc.collect()

        with self._serialize(None, device) as unencrypted_model:
            # Ensure that it fails to load an unencrypted model
            # when expecting encryption
            with self.subTest(
                msg="Deserializing an unencrypted model with a key"
            ), self.assertRaises(
                serialization.CryptographyError
            ), TensorDeserializer(
                unencrypted_model,
                device=device,
                lazy_load=True,
                encryption=decryption,
            ) as deserialized:
                self._test_first_key_negative(deserialized)
                del deserialized
            gc.collect()

    def test_from_string(self):
        fixed_salt = bytes(16)
        encryption = EncryptionParams.from_string(
            source="test", salt=fixed_salt
        )
        decryption = DecryptionParams.from_string(source="test")
        incorrect_decryption = DecryptionParams.from_string(source="tset")
        self._test_encryption_params(
            encryption, decryption, incorrect_decryption
        )

    def test_random_encryption_params(self):
        encryption = EncryptionParams.random()
        decryption = DecryptionParams.from_key(encryption.key)
        incorrect_decryption = DecryptionParams.from_key(
            bytes(len(encryption.key))
        )
        self._test_encryption_params(
            encryption, decryption, incorrect_decryption
        )

    def _test_encryption_params(
        self,
        encryption: EncryptionParams,
        decryption: DecryptionParams,
        incorrect_decryption: DecryptionParams,
    ):
        device = default_device

        with self._serialize(encryption, device) as encrypted_model:
            # Ensure that it works when given a key
            with self.subTest(
                msg="Deserializing with a correct key",
                device=device,
                lazy_load=False,
            ), TensorDeserializer(
                encrypted_model,
                device=device,
                lazy_load=False,
                verify_hash=True,
                encryption=decryption,
            ) as deserialized:
                check_deserialized(
                    self,
                    deserialized,
                    model_name,
                )
                del deserialized
            gc.collect()

            # Ensure that it fails to load when not given a key
            with self.subTest(
                msg="Deserializing with a missing key"
            ), self.assertRaises(
                serialization.CryptographyError
            ), TensorDeserializer(
                encrypted_model,
                device=device,
                lazy_load=True,
                encryption=None,
            ) as deserialized:
                self._test_first_key_negative(deserialized)
                del deserialized
            gc.collect()

            # Ensure that it fails to load when given the wrong key
            with self.subTest(
                msg="Deserializing with an incorrect key"
            ), self.assertRaises(
                serialization.CryptographyError
            ), TensorDeserializer(
                encrypted_model,
                device=device,
                lazy_load=True,
                encryption=incorrect_decryption,
            ) as deserialized:
                self._test_first_key_negative(deserialized)
                del deserialized
            gc.collect()


class TestDeserialization(unittest.TestCase):
    _serialized_model_path: str

    @classmethod
    def setUpClass(cls):
        serialized_model_path = serialize_model(model_name, "cpu").filename
        cls._serialized_model_path = serialized_model_path
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._serialized_model_path)

    def open_serialized(self):
        return open(self._serialized_model_path, "rb")

    def test_default_cpu(self):
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(
            self._serialized_model_path, device="cpu"
        )
        after_deserialization = utils.get_mem_usage()
        check_deserialized(self, deserialized, model_name)
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization:  {after_deserialization}")

    @unittest.skipIf(not is_cuda_available, "Requires CUDA")
    def test_default_gpu(self):
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(
            self._serialized_model_path, device="cuda"
        )
        check_deserialized(self, deserialized, model_name)
        after_deserialization = utils.get_mem_usage()
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization:  {after_deserialization}")
        del deserialized
        gc.collect()
        after_del = utils.get_mem_usage()
        print(f"After del: {after_del}")

    def test_lazy_load(self):
        supported_devices = ["cpu"]
        if default_device == "cuda":
            supported_devices.append("cuda")

        for device in supported_devices:
            with self.subTest(
                f"Testing lazy_load=True with device={device}"
            ), self.open_serialized() as in_file, TensorDeserializer(
                self._serialized_model_path,
                device=device,
                lazy_load=True,
            ) as deserialized:
                check_deserialized(self, deserialized, model_name)
                check_inference(self, deserialized, model_name, device)

    @unittest.skipIf(not is_cuda_available, "requires CUDA")
    def test_cuda(self):
        deserialized = TensorDeserializer(
            self._serialized_model_path, device="cuda"
        )

        check_deserialized(self, deserialized, model_name)
        deserialized.close()

    @unittest.skipIf(not is_cuda_available, "requires CUDA")
    def test_cuda_inference(self):
        deserialized = TensorDeserializer(
            self._serialized_model_path, device="cuda"
        )

        check_inference(self, deserialized, model_name, "cuda")
        deserialized.close()

    @unittest.skipIf(not is_cuda_available, "requires CUDA")
    def test_cuda_multiple_readers(self):
        with open(
            self._serialized_model_path, "rb"
        ) as in_file, TensorDeserializer(
            in_file, device="cuda", num_readers=4
        ) as deserialized:
            check_deserialized(self, deserialized, model_name)

    def test_cpu_multiple_readers(self):
        with open(
            self._serialized_model_path, "rb"
        ) as in_file, TensorDeserializer(
            in_file, device="cpu", num_readers=4
        ) as deserialized:
            check_deserialized(self, deserialized, model_name)

    @patch.object(stream_io, "_s3_default_config_paths", ())
    @patch.object(stream_io, "default_s3_read_endpoint", default_read_endpoint)
    def test_s3(self):
        deserialized = TensorDeserializer(
            f"s3://tensorized/{model_name}/model.tensors", device=default_device
        )
        check_deserialized(self, deserialized, model_name)
        check_inference(self, deserialized, model_name, default_device)
        deserialized.close()

    @patch.object(stream_io, "_s3_default_config_paths", ())
    @patch.object(stream_io, "default_s3_read_endpoint", default_read_endpoint)
    def test_s3_multiple_readers(self):
        uri: str = f"s3://tensorized/{model_name}/model.tensors"

        with self.subTest("Deserializing from an S3 URI"), TensorDeserializer(
            uri, device=default_device, num_readers=4
        ) as deserialized:
            check_deserialized(self, deserialized, model_name)
            check_inference(self, deserialized, model_name, default_device)

        with self.subTest(
            "Deserializing from a CURLStreamFile"
        ), stream_io.open_stream(
            uri,
            "rb",
            s3_access_key_id="",
            s3_secret_access_key="",
            s3_endpoint="object.ord1.coreweave.com",
        ) as stream, TensorDeserializer(
            stream, device=default_device, num_readers=4
        ) as deserialized:
            check_deserialized(self, deserialized, model_name)
            check_inference(self, deserialized, model_name, default_device)

    @patch.object(stream_io, "_s3_default_config_paths", ())
    @patch.object(stream_io, "default_s3_read_endpoint", default_read_endpoint)
    def test_s3_fp16(self):
        deserialized = TensorDeserializer(
            f"s3://tensorized/{model_name}/fp16/model.tensors",
            device=default_device,
        )
        self.assertGreater(deserialized.total_tensor_bytes, 0)
        if is_cuda_available and default_device != "cpu":
            # FP16 tensors don't work correctly on CPU in PyTorch
            check_inference(self, deserialized, model_name, default_device)
        deserialized.close()

    @patch.object(stream_io, "_s3_default_config_paths", ())
    @patch.object(stream_io, "default_s3_read_endpoint", default_read_endpoint)
    def test_s3_lazy_load(self):
        deserialized = TensorDeserializer(
            f"s3://tensorized/{model_name}/model.tensors",
            device=default_device,
            lazy_load=True,
        )
        check_deserialized(self, deserialized, model_name)
        check_inference(self, deserialized, model_name, default_device)
        deserialized.close()

    def test_redis(self):
        redis_server, redis_client = start_redis(port=6380)
        try:
            redis_model_path = f"redis://localhost:6380/{model_name}"

            deserialized_s3 = TensorDeserializer(
                f"s3://tensorized/{model_name}/model.tensors",
                device=default_device,
                lazy_load=True,
            )
            deserialized_s3.to_redis(redis_client, model_name)
            deserialized_s3.close()

            with self.subTest(
                msg="Testing redis deserialization in eager mode"
            ):
                deserialized_redis = TensorDeserializer(
                    redis_model_path,
                    device=default_device,
                )
                check_deserialized(self, deserialized_redis, model_name)
                check_inference(
                    self, deserialized_redis, model_name, default_device
                )
                deserialized_redis.close()

            with self.subTest(msg="Testing redis deserialization in lazy mode"):
                deserialized_redis = TensorDeserializer(
                    redis_model_path,
                    device=default_device,
                    lazy_load=True,
                )
                check_deserialized(self, deserialized_redis, model_name)
                check_inference(
                    self, deserialized_redis, model_name, default_device
                )
                deserialized_redis.close()
        finally:
            teardown_redis(redis_server, redis_client)

    def test_filter_func(self):
        # These two filters should produce identical results
        pattern = re.compile(r"transformer\.h\.0.*")

        def custom_check(tensor_name: str) -> bool:
            return tensor_name.startswith("transformer.h.0")

        # Testing no filter_func
        deserialized = TensorDeserializer(
            self._serialized_model_path, device=default_device, filter_func=None
        )
        all_keys = set(deserialized.keys())
        self.assertTrue(
            all_keys,
            "Deserializing the model with no filter_func"
            " loaded an empty set of tensors",
        )
        check_deserialized(self, deserialized, model_name)
        deserialized.close()

        expected_regex_keys = set(filter(pattern.match, all_keys))
        expected_custom_keys = set(filter(custom_check, all_keys))

        self.assertTrue(
            expected_regex_keys
            and expected_regex_keys < all_keys
            and expected_custom_keys
            and expected_custom_keys < all_keys,
            (
                "The filter_func test cannot continue"
                " because a filter_func used in the test"
                " does not appear in the test model,"
                " or matches all tensor names."
                " Update the pattern and/or custom_check"
                " to use more informative filtering criteria."
                "\n\nTensors present in the model: " + " ".join(all_keys)
            ),
        )

        with self.subTest(msg="Testing regex filter_func"):
            deserialized = TensorDeserializer(
                self._serialized_model_path,
                device=default_device,
                filter_func=pattern.match,
            )
            regex_keys = set(deserialized.keys())
            # Test that the deserialized tensors form a proper,
            # non-empty subset of the original list of tensors.
            self.assertEqual(regex_keys, expected_regex_keys)
            check_deserialized(
                self, deserialized, model_name, allow_subset=True
            )
            deserialized.close()

        with self.subTest(msg="Testing custom filter_func"):
            deserialized = TensorDeserializer(
                self._serialized_model_path,
                device=default_device,
                filter_func=custom_check,
            )
            custom_keys = set(deserialized.keys())
            self.assertEqual(custom_keys, expected_custom_keys)
            check_deserialized(
                self, deserialized, model_name, allow_subset=True
            )
            deserialized.close()


def mock_invalid_tensor_hash(*args, **kwargs):
    tensor_hash = TensorHash(*args, **kwargs)
    tensor_hash.hash = bytes(len(tensor_hash.hash))
    return tensor_hash


class TestVerification(unittest.TestCase):
    _serialized_model_path: str

    @classmethod
    def setUpClass(cls):
        serialized_model_path = serialize_model(model_name, "cpu").filename
        cls._serialized_model_path = serialized_model_path
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._serialized_model_path)

    def test_verification(self):
        for num_readers in (1, 4):
            for device in "cuda", "cpu":
                if device == "cuda" and not is_cuda_available:
                    continue
                with self.subTest(msg=f"Verifying hashes with device {device}"):
                    deserialized = TensorDeserializer(
                        self._serialized_model_path,
                        device=device,
                        verify_hash=True,
                        num_readers=num_readers,
                    )
                    check_deserialized(self, deserialized, model_name)
                    deserialized.close()
                    del deserialized

    @patch.object(serialization, "TensorHash", mock_invalid_tensor_hash)
    def test_verification_fail(self):
        for num_readers in (1, 4):
            for device in "cuda", "cpu":
                if device == "cuda" and not is_cuda_available:
                    continue
                with self.subTest(msg=f"Verifying hashes with device {device}"):
                    with self.assertRaises(serialization.HashMismatchError):
                        TensorDeserializer(
                            self._serialized_model_path,
                            device=device,
                            verify_hash=True,
                            num_readers=num_readers,
                        ).close()

    def test_module_verification(self):
        model_to_verify = AutoModelForCausalLM.from_pretrained(model_name)
        for device in "cuda", "cpu":
            if device == "cuda" and not is_cuda_available:
                continue
            with self.subTest(msg=f"Verifying hashes with device {device}"):
                deserialized = TensorDeserializer(
                    self._serialized_model_path, device=device, verify_hash=True
                )
                model_to_verify = model_to_verify.to(device)
                result, tensor_status = deserialized.verify_module(
                    model_to_verify
                )
                deserialized.close()
                del deserialized
                self.assertTrue(result)
                for tensor_name, status in tensor_status:
                    self.assertTrue(status, f"Tensor {tensor_name} failed")

    def test_module_verification_fail(self):
        model_to_verify = AutoModelForCausalLM.from_pretrained(model_name)
        for device in "cuda", "cpu":
            if device == "cuda" and not is_cuda_available:
                continue
            with self.subTest(msg=f"Verifying hashes with device {device}"):
                deserialized = TensorDeserializer(
                    self._serialized_model_path, device=device, verify_hash=True
                )
                model_to_verify = model_to_verify.to(device)
                model_to_verify.transformer.h[0].ln_2 = torch.nn.LayerNorm(
                    768, 768
                )
                result, tensor_status = deserialized.verify_module(
                    model_to_verify
                )
                deserialized.close()
                del deserialized
                self.assertFalse(result, "Did not catch altered layer")
                for tensor_name, status in tensor_status:
                    if tensor_name.startswith("transformer.h.0.ln_2"):
                        self.assertFalse(
                            status,
                            f"Intended mismatch on {tensor_name} was"
                            " not reported",
                        )
                    else:
                        self.assertTrue(
                            status, f"Unexpected mismatch on {tensor_name}"
                        )

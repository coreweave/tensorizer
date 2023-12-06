import contextlib
import ctypes
import enum
import functools
import gc
import hashlib
import itertools
import os
import re
import secrets
import tempfile
import time
import unittest
from typing import Mapping, NamedTuple, Optional
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
    dtype: torch.dtype
    hash: bytes

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorInfo":
        storage = tensor.untyped_storage().cpu()
        data = ctypes.cast(
            storage.data_ptr(),
            ctypes.POINTER(ctypes.c_ubyte * storage.nbytes()),
        ).contents
        hash_val = hashlib.blake2b(data, digest_size=16, salt=salt).digest()
        return cls(size=tensor.size(), dtype=tensor.dtype, hash=hash_val)


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
                    with open(serialized_model, "rb") as in_file:
                        deserialized = TensorDeserializer(in_file, device="cpu")
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

    def test_bfloat16(self):
        shape = (50, 50)
        tensor = torch.normal(0, 0.5, shape, dtype=torch.bfloat16)
        tensorized_file = tempfile.NamedTemporaryFile("wb+", delete=False)

        try:
            serializer = TensorSerializer(tensorized_file)
            serializer.write_tensor(0, "test_tensor", TensorType.PARAM, tensor)
            serializer.close()

            with open(tensorized_file.name, "rb") as in_file:
                deserializer = TensorDeserializer(
                    in_file, device="cpu", lazy_load=True
                )
                deserialized_tensor = [
                    t for t in deserializer.read_tensors(num_tensors=1)
                ][0][-1]
                deserializer.close()
        finally:
            os.unlink(tensorized_file.name)

        self.assertTrue(torch.equal(tensor, deserialized_tensor))

    def test_persistent_buffers(self):
        shape = (50, 50)
        persistent_buffer = torch.normal(0, 0.5, shape)
        non_persistent_buffer = torch.normal(0, 0.5, shape)
        nested_module = torch.nn.Module()
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

        for include in (True, False):
            with self.subTest(
                msg=f"Testing include_non_persistent_buffers={include}"
            ):
                tensorized_file = tempfile.NamedTemporaryFile(
                    "wb+", delete=False
                )
                try:
                    serializer = TensorSerializer(tensorized_file)
                    serializer.write_module(
                        model, include_non_persistent_buffers=include
                    )
                    serializer.close()

                    with open(tensorized_file.name, "rb") as in_file:
                        with TensorDeserializer(
                            in_file, device="cpu", lazy_load=True
                        ) as deserializer:
                            self.assertIn(
                                "module.nested.persistent_buffer",
                                deserializer.keys(),
                            )
                            assertion = (
                                self.assertIn if include else self.assertNotIn
                            )
                            assertion(
                                "module.nested.non_persistent_buffer",
                                deserializer.keys(),
                            )
                finally:
                    os.unlink(tensorized_file.name)


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
                ), open(encrypted_model, "rb") as in_file, TensorDeserializer(
                    in_file,
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
            ), self.assertRaises(serialization.CryptographyError), open(
                encrypted_model, "rb"
            ) as in_file, TensorDeserializer(
                in_file,
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
            ), self.assertRaises(serialization.CryptographyError), open(
                encrypted_model, "rb"
            ) as in_file, TensorDeserializer(
                in_file,
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
            ), self.assertRaises(serialization.CryptographyError), open(
                unencrypted_model, "rb"
            ) as in_file, TensorDeserializer(
                in_file,
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
        plaid_mode = device != "cpu"

        with self._serialize(encryption, device) as encrypted_model:
            # Ensure that it works when given a key
            with self.subTest(
                msg="Deserializing with a correct key",
                device=device,
                lazy_load=False,
                plaid_mode=plaid_mode,
            ), open(encrypted_model, "rb") as in_file, TensorDeserializer(
                in_file,
                device=device,
                lazy_load=False,
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
            ), self.assertRaises(serialization.CryptographyError), open(
                encrypted_model, "rb"
            ) as in_file, TensorDeserializer(
                in_file,
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
            ), self.assertRaises(serialization.CryptographyError), open(
                encrypted_model, "rb"
            ) as in_file, TensorDeserializer(
                in_file,
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

    def test_default_cpu(self):
        in_file = open(self._serialized_model_path, "rb")
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(in_file, device="cpu")
        after_deserialization = utils.get_mem_usage()
        check_deserialized(self, deserialized, model_name)
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization:  {after_deserialization}")

    @unittest.skipIf(not is_cuda_available, "Requires CUDA")
    def test_default_gpu(self):
        in_file = open(self._serialized_model_path, "rb")
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(in_file, device="cuda")
        check_deserialized(self, deserialized, model_name)
        after_deserialization = utils.get_mem_usage()
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization:  {after_deserialization}")
        del in_file
        gc.collect()
        after_del = utils.get_mem_usage()
        print(f"After del: {after_del}")

    def test_lazy_load(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(
            in_file, device=default_device, lazy_load=True
        )

        check_deserialized(self, deserialized, model_name)
        check_inference(self, deserialized, model_name, default_device)
        deserialized.close()

    @unittest.skipIf(not is_cuda_available, "plaid_mode requires CUDA")
    def test_plaid_mode(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(
            in_file, device="cuda", plaid_mode=True
        )

        check_deserialized(self, deserialized, model_name)
        deserialized.close()

    @unittest.skipIf(not is_cuda_available, "plaid_mode requires CUDA")
    def test_plaid_mode_inference(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(
            in_file, device="cuda", plaid_mode=True
        )

        check_inference(self, deserialized, model_name, "cuda")
        deserialized.close()

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
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(
            in_file, device=default_device, filter_func=None
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
                "\n\nTensors present in the model: "
                + " ".join(all_keys)
            ),
        )

        with self.subTest(msg="Testing regex filter_func"):
            in_file = open(self._serialized_model_path, "rb")
            deserialized = TensorDeserializer(
                in_file, device=default_device, filter_func=pattern.match
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
            in_file = open(self._serialized_model_path, "rb")
            deserialized = TensorDeserializer(
                in_file, device=default_device, filter_func=custom_check
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
        for device in "cuda", "cpu":
            if device == "cuda" and not is_cuda_available:
                continue
            with self.subTest(msg=f"Verifying hashes with device {device}"):
                with open(self._serialized_model_path, "rb") as in_file:
                    deserialized = TensorDeserializer(
                        in_file, device=device, verify_hash=True
                    )
                    check_deserialized(self, deserialized, model_name)
                    deserialized.close()
                    del deserialized

    @patch.object(serialization, "TensorHash", mock_invalid_tensor_hash)
    def test_verification_fail(self):
        for device in "cuda", "cpu":
            if device == "cuda" and not is_cuda_available:
                continue
            with self.subTest(msg=f"Verifying hashes with device {device}"):
                with open(self._serialized_model_path, "rb") as in_file:
                    with self.assertRaises(serialization.HashMismatchError):
                        TensorDeserializer(
                            in_file, device=device, verify_hash=True
                        ).close()

    def test_module_verification(self):
        model_to_verify = AutoModelForCausalLM.from_pretrained(model_name)
        for device in "cuda", "cpu":
            if device == "cuda" and not is_cuda_available:
                continue
            with self.subTest(msg=f"Verifying hashes with device {device}"):
                with open(self._serialized_model_path, "rb") as in_file:
                    deserialized = TensorDeserializer(in_file, device=device)
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
                with open(self._serialized_model_path, "rb") as in_file:
                    deserialized = TensorDeserializer(in_file, device=device)
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

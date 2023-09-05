import contextlib
import ctypes
import functools
import gc
import hashlib
import os
import re
import secrets
import tempfile
import time
import unittest
from typing import Mapping, NamedTuple, Tuple
from unittest.mock import patch

import torch

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # avoids excessive warnings about forking after using a tokenizer
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tensorizer import (
    TensorDeserializer,
    TensorSerializer,
    serialization,
    stream_io,
    utils,
)
from tensorizer.serialization import TensorHash, TensorType

model_name = "EleutherAI/gpt-neo-125M"
num_hellos = 400
is_cuda_available = torch.cuda.is_available()
default_device = "cuda" if is_cuda_available else "cpu"
salt = secrets.token_bytes(4)
default_read_endpoint = "object.ord1.coreweave.com"


def serialize_model(model_name: str, device: str) -> Tuple[str, dict]:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sd = model.state_dict()
    out_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    try:
        start_time = time.monotonic()
        serializer = TensorSerializer(out_file)
        serializer.write_module(model)
        serializer.close()
        end_time = time.monotonic()
        print(f"Serialization took {end_time - start_time:.3f} seconds")
    except Exception:
        os.unlink(out_file.name)
        raise
    return out_file.name, sd


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
def model_digest(model_name: str) -> Mapping[str, TensorInfo]:
    orig_model = AutoModelForCausalLM.from_pretrained(model_name)
    orig_sd = orig_model.state_dict()
    # Non-persistent buffers are serialized in tensorizer,
    # but aren't included in a state_dict() in PyTorch.
    orig_sd.update(orig_model.named_buffers())
    return {k: TensorInfo.from_tensor(v) for k, v in orig_sd.items()}


def check_deserialized(
    test_case: unittest.TestCase,
    deserialized: TensorDeserializer,
    model_name: str,
    allow_subset: bool = False,
):
    orig_sd = model_digest(model_name)

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
        for device in "cuda", "cpu":
            if device == "cuda" and not is_cuda_available:
                continue
            with self.subTest(msg=f"Serializing with device {device}"):
                gc.collect()
                before_serialization = utils.get_mem_usage()
                serialized_model, orig_sd = serialize_model(model_name, device)
                after_serialization = utils.get_mem_usage()
                print(f"Before serialization: {before_serialization}")
                print(f"After serialization:  {after_serialization}")
                del orig_sd
                try:
                    with open(serialized_model, "rb") as in_file:
                        deserialized = TensorDeserializer(in_file, device="cpu")
                        check_deserialized(self, deserialized, model_name)
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


class TestDeserialization(unittest.TestCase):
    _serialized_model_path: str

    @classmethod
    def setUpClass(cls):
        serialized_model_path, sd = serialize_model(model_name, "cpu")
        del sd
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

    @unittest.skipIf(not is_cuda_available, "plaid_mode requires CUDA")
    def test_plaid_mode_guards(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(
            in_file, device="cuda", plaid_mode=True
        )
        keys = list(deserialized.keys())
        _ = deserialized[keys[0]]
        _ = deserialized[keys[1]]

        with self.assertRaises(RuntimeError):
            _ = deserialized[keys[0]]

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
    tensor_hash["hash"] = bytes(len(tensor_hash["hash"]))
    return tensor_hash


class TestVerification(unittest.TestCase):
    _serialized_model_path: str

    @classmethod
    def setUpClass(cls):
        serialized_model_path = serialize_model(model_name, "cpu")[0]
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

import contextlib
import gc
import os
import re
import tempfile
import unittest
from typing import Tuple

import torch

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # avoids excessive warnings about forking after using a tokenizer
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tensorizer import TensorDeserializer, TensorSerializer, utils

model_name = "EleutherAI/gpt-neo-125M"
num_hellos = 400
is_cuda_available = torch.cuda.is_available()
default_device = "cuda" if is_cuda_available else "cpu"


def serialize_model(model_name: str, device: str) -> Tuple[str, dict]:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sd = model.state_dict()
    out_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    try:
        serializer = TensorSerializer(out_file)
        serializer.write_module(model)
        serializer.close()
    except Exception:
        os.unlink(out_file)
        raise
    return out_file.name, sd


def check_deserialized(deserialized, model_name: str, allow_subset=False):
    orig_sd = AutoModelForCausalLM.from_pretrained(model_name).state_dict()
    if not allow_subset:
        assert orig_sd.keys() == deserialized.keys()
    for k, v in deserialized.items():
        # fmt: off
        assert k in orig_sd, \
            f"{k} not in {orig_sd.keys()}"

        assert v.size() == orig_sd[k].size(), \
            f"{v.size()} != {orig_sd[k].size()}"

        assert v.dtype == orig_sd[k].dtype, \
            f"{v.dtype} != {orig_sd[k].dtype}"

        assert torch.all(orig_sd[k].to(v.device) == v)
        # fmt: on
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
    deserializer: TensorDeserializer, model_ref: str, device: str
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
        assert decoded.count("hello") > num_hellos


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
                        check_deserialized(deserialized, model_name)
                        deserialized.close()
                        del deserialized
                finally:
                    os.unlink(serialized_model)


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
        check_deserialized(deserialized, model_name)
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization:  {after_deserialization}")

    @unittest.skipIf(not is_cuda_available, "Requires CUDA")
    def test_default_gpu(self):
        in_file = open(self._serialized_model_path, "rb")
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(in_file, device="cuda")
        check_deserialized(deserialized, model_name)
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

        check_deserialized(deserialized, model_name)
        check_inference(deserialized, model_name, default_device)
        deserialized.close()

    @unittest.skipIf(not is_cuda_available, "plaid_mode requires CUDA")
    def test_plaid_mode(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(
            in_file, device="cuda", plaid_mode=True
        )

        check_deserialized(deserialized, model_name)
        deserialized.close()

    @unittest.skipIf(not is_cuda_available, "plaid_mode requires CUDA")
    def test_plaid_mode_inference(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(
            in_file, device="cuda", plaid_mode=True
        )

        check_inference(deserialized, model_name, "cuda")
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

    def test_s3(self):
        deserialized = TensorDeserializer(
            f"s3://tensorized/{model_name}/model.tensors", device=default_device
        )
        check_deserialized(deserialized, model_name)
        check_inference(deserialized, model_name, default_device)
        deserialized.close()

    def test_s3_fp16(self):
        deserialized = TensorDeserializer(
            f"s3://tensorized/{model_name}/fp16/model.tensors",
            device=default_device,
        )
        assert deserialized.total_tensor_bytes > 0
        if is_cuda_available and default_device != "cpu":
            # FP16 tensors don't work correctly on CPU in PyTorch
            check_inference(deserialized, model_name, default_device)
        deserialized.close()

    def test_s3_lazy_load(self):
        deserialized = TensorDeserializer(
            f"s3://tensorized/{model_name}/model.tensors",
            device=default_device,
            lazy_load=True,
        )
        check_deserialized(deserialized, model_name)
        check_inference(deserialized, model_name, default_device)
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
        assert all_keys, (
            "Deserializing the model with no filter_func"
            " loaded an empty set of tensors"
        )
        check_deserialized(deserialized, model_name)
        deserialized.close()

        expected_regex_keys = set(filter(pattern.match, all_keys))
        expected_custom_keys = set(filter(custom_check, all_keys))

        assert (
            expected_regex_keys
            and expected_regex_keys < all_keys
            and expected_custom_keys
            and expected_custom_keys < all_keys
        ), (
            "The filter_func test cannot continue"
            " because a filter_func used in the test"
            " does not appear in the test model,"
            " or matches all tensor names."
            " Update the pattern and/or custom_check"
            " to use more informative filtering criteria."
            "\n\nTensors present in the model: "
            + " ".join(all_keys)
        )

        with self.subTest(msg="Testing regex filter_func"):
            in_file = open(self._serialized_model_path, "rb")
            deserialized = TensorDeserializer(
                in_file, device=default_device, filter_func=pattern.match
            )
            regex_keys = set(deserialized.keys())
            # Test that the deserialized tensors form a proper,
            # non-empty subset of the original list of tensors.
            assert regex_keys == expected_regex_keys
            check_deserialized(deserialized, model_name, allow_subset=True)
            deserialized.close()

        with self.subTest(msg="Testing custom filter_func"):
            in_file = open(self._serialized_model_path, "rb")
            deserialized = TensorDeserializer(
                in_file, device=default_device, filter_func=custom_check
            )
            custom_keys = set(deserialized.keys())
            assert custom_keys == expected_custom_keys
            check_deserialized(deserialized, model_name, allow_subset=True)
            deserialized.close()

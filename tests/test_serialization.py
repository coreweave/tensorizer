import gc
import tempfile
import unittest
from typing import Tuple

import torch

from transformers import AutoModelForCausalLM

from tensorizer.serialization import TensorSerializer, TensorDeserializer
from tensorizer import utils

model_name = "EleutherAI/gpt-neo-125M"


def serialize_model(model_name: str, device: str) -> Tuple[str, dict]:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sd = model.state_dict()
    out_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    serializer = TensorSerializer(out_file)
    serializer.write_state_dict(sd)
    serializer.close()
    return out_file.name, sd


def check_deserialized(deserialized, model_name: str):
    orig_sd = AutoModelForCausalLM.from_pretrained(model_name).state_dict()
    for k, v in deserialized.items():
        assert k in orig_sd
        assert v.size() == orig_sd[k].size()
        assert v.dtype == orig_sd[k].dtype
        assert torch.all(orig_sd[k] == v)


class TestSerialization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        serialized_model_path, sd = serialize_model(model_name, "cpu")
        del sd
        self._serialized_model_path = serialized_model_path
        gc.collect()

    def test_serialization(self):
        serialized_model, orig_sd = serialize_model(model_name, "cpu")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file, preload=False, device="cpu")
        check_deserialized(deserialized, model_name)
        deserialized.close()

    def test_preload(self):
        in_file = open(self._serialized_model_path, "rb")
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(in_file, preload=True, device="cpu")
        after_deserialization = utils.get_mem_usage()
        check_deserialized(deserialized, model_name)
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization: {after_deserialization}")

    def test_mmap(self):
        in_file = open(self._serialized_model_path, "rb")
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(in_file,
                                          device="cpu",
                                          use_mmap=True)
        after_deserialization = utils.get_mem_usage()
        check_deserialized(deserialized, model_name)
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization: {after_deserialization}")

    def test_mmap_gpu(self):
        before_serialization = utils.get_mem_usage()
        serialized_model, orig_sd = serialize_model(model_name, device="cuda")
        after_serialization = utils.get_mem_usage()
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cuda",
                                          use_mmap=True)
        after_deserialization = utils.get_mem_usage()
        check_deserialized(deserialized, orig_sd)
        print(f"Before serialization: {before_serialization}")
        print(f"After serialization: {after_serialization}")
        print(f"After deserialization: {after_deserialization}")
        del serialized_model, orig_sd, in_file
        gc.collect()
        after_del = utils.get_mem_usage()
        print(f"After del: {after_del}")

    def test_mmap_preload(self):
        serialized_model, orig_sd = serialize_model(model_name, device="cpu")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cpu",
                                          use_mmap=True,
                                          preload=True)

        check_deserialized(deserialized, model_name)
        deserialized.close()

    def test_oneshot(self):
        serialized_model, orig_sd = serialize_model(model_name, device="cuda")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cuda",
                                          oneshot=True)

        check_deserialized(deserialized, model_name)
        deserialized.close()

    def test_oneshot_guards(self):
        serialized_model, orig_sd = serialize_model(model_name, device="cuda")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cuda",
                                          oneshot=True)
        keys = list(deserialized.keys())
        _ = deserialized[keys[0]]
        _ = deserialized[keys[1]]

        with self.assertRaises(RuntimeError):
            _ = deserialized[keys[0]]

        deserialized.close()
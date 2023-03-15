import gc
import os
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
    try:
        serializer = TensorSerializer(out_file)
        serializer.write_state_dict(sd)
        serializer.close()
    except Exception:
        os.unlink(out_file)
        raise
    return out_file.name, sd


def check_deserialized(deserialized, model_name: str):
    orig_sd = AutoModelForCausalLM.from_pretrained(model_name).state_dict()
    for k, v in deserialized.items():
        assert k in orig_sd
        assert v.size() == orig_sd[k].size()
        assert v.dtype == orig_sd[k].dtype
        assert torch.all(orig_sd[k].to(v.device) == v)


class TestSerialization(unittest.TestCase):
    def test_serialization(self):
        for device in "cuda", "cpu":
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
                        deserialized = TensorDeserializer(
                            in_file, preload=False, device="cpu"
                        )
                        check_deserialized(deserialized, model_name)
                        deserialized.close()
                        del deserialized
                finally:
                    os.unlink(serialized_model)


class TestDeserialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serialized_model_path, sd = serialize_model(model_name, "cpu")
        del sd
        cls._serialized_model_path = serialized_model_path
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._serialized_model_path)

    def test_preload(self):
        in_file = open(self._serialized_model_path, "rb")
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(in_file, preload=True, device="cpu")
        after_deserialization = utils.get_mem_usage()
        check_deserialized(deserialized, model_name)
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization:  {after_deserialization}")

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
        print(f"After deserialization:  {after_deserialization}")

    def test_mmap_gpu(self):
        in_file = open(self._serialized_model_path, "rb")
        gc.collect()
        before_deserialization = utils.get_mem_usage()
        deserialized = TensorDeserializer(in_file,
                                          device="cuda",
                                          use_mmap=True)
        check_deserialized(deserialized, model_name)
        after_deserialization = utils.get_mem_usage()
        deserialized.close()
        print(f"Before deserialization: {before_deserialization}")
        print(f"After deserialization:  {after_deserialization}")
        del in_file
        gc.collect()
        after_del = utils.get_mem_usage()
        print(f"After del: {after_del}")

    def test_mmap_preload(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cpu",
                                          use_mmap=True,
                                          preload=True)

        check_deserialized(deserialized, model_name)
        deserialized.close()

    def test_oneshot(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cuda",
                                          oneshot=True)

        check_deserialized(deserialized, model_name)
        deserialized.close()

    def test_oneshot_guards(self):
        in_file = open(self._serialized_model_path, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cuda",
                                          oneshot=True)
        keys = list(deserialized.keys())
        _ = deserialized[keys[0]]
        _ = deserialized[keys[1]]

        with self.assertRaises(RuntimeError):
            _ = deserialized[keys[0]]

        deserialized.close()

import tempfile
import unittest
from typing import Tuple

import torch

from transformers import AutoModelForCausalLM

from tensorizer.serialization import TensorSerializer, TensorDeserializer

model_name = "EleutherAI/gpt-neo-125M"


def serialize_model(model_name: str, device: str) -> Tuple[str, dict]:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    sd = model.state_dict()
    out_file = tempfile.NamedTemporaryFile("wb+", delete=False)
    serializer = TensorSerializer(out_file)
    serializer.write_state_dict(sd)
    serializer.close()
    return out_file.name, sd


def check_deserialized(deserialized, orig_sd):
    for k, v in deserialized.items():
        print(k)
        assert k in orig_sd
        assert v.size() == orig_sd[k].size()
        assert v.dtype == orig_sd[k].dtype
        assert torch.all(orig_sd[k] == v)


class TestSerialization(unittest.TestCase):
    def test_serialization(self):
        serialized_model, orig_sd = serialize_model(model_name, "cpu")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file, preload=False, device="cpu")

        check_deserialized(deserialized, orig_sd)
        deserialized.close()

    def test_preload(self):
        serialized_model, orig_sd = serialize_model(model_name, device="cpu")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file, preload=True, device="cpu")

        check_deserialized(deserialized, orig_sd)
        deserialized.close()

    def test_mmap(self):
        serialized_model, orig_sd = serialize_model(model_name, device="cpu")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cpu",
                                          use_mmap=True)

        check_deserialized(deserialized, orig_sd)
        deserialized.close()

    def test_mmap_preload(self):
        serialized_model, orig_sd = serialize_model(model_name, device="cpu")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cpu",
                                          use_mmap=True,
                                          preload=True)

        check_deserialized(deserialized, orig_sd)
        deserialized.close()

    def test_oneshot(self):
        serialized_model, orig_sd = serialize_model(model_name, device="cuda")
        in_file = open(serialized_model, "rb")
        deserialized = TensorDeserializer(in_file,
                                          device="cuda",
                                          oneshot=True)

        check_deserialized(deserialized, orig_sd)
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
import tempfile
import unittest
import torch

from transformers import AutoModelForCausalLM

from tensorizer.serialization import TensorSerializer, TensorDeserializer

model_name = "EleutherAI/gpt-neo-125M"


class TestSerialization(unittest.TestCase):
    def test_serialization(self):
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        sd = model.state_dict()
        out_file = tempfile.NamedTemporaryFile("wb+", delete=False)
        serializer = TensorSerializer(out_file)
        serializer.write_state_dict(sd)
        serializer.close()

        in_file = open(out_file.name, "rb")
        deserialized = TensorDeserializer(in_file, preload=True)

        for k, v in deserialized.items():
            print(k)
            assert k in sd
            assert v.size() == sd[k].size()
            assert v.dtype == sd[k].dtype
            assert torch.all(sd[k] == v)

        deserialized.close()
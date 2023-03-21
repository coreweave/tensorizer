import io
import serializer
import unittest
import torch
from torch import Tensor


class TestSerializer(unittest.TestCase):
    def test_serializer_int(self):
        t = Tensor([1, 2, 3])
        t_serialized = serializer.serialize_tensor(t)
        t_deserialized = serializer.deserialize_tensor(t_serialized)
        self.assertTrue(torch.equal(t, t_deserialized))

    def test_serializer_fp16(self):
        t = Tensor([1.0, 2.0, 3.0]).half()
        t_serialized = serializer.serialize_tensor(t)
        t_deserialized = serializer.deserialize_tensor(t_serialized)
        self.assertTrue(torch.equal(t, t_deserialized))

    def test_serializer_multi_dim(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        t_serialized = serializer.serialize_tensor(t)
        t_deserialized = serializer.deserialize_tensor(t_serialized)
        self.assertTrue(torch.equal(t, t_deserialized))

    def test_serializer_model(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(2, 3))

        model = TestModel()
        model2 = TestModel()
        f = io.BytesIO()
        serializer.serialize_model(model, f)
        f.seek(0)
        serializer.deserialize_model(model2, f)
        self.assertTrue(torch.equal(model.weight, model2.weight))

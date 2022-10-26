import serializer
import unittest
import torch
from torch import Tensor


class TestSerializer(unittest.TestCase):
    def test_serializer_int(self):
        t = Tensor([1, 2, 3])
        t_serialized = serializer.serialize(t)
        t_deserialized = serializer.deserialize(t_serialized)
        self.assertTrue(torch.equal(t, t_deserialized))

    def test_serializer_fp16(self):
        t = Tensor([1.0, 2.0, 3.0]).half()
        t_serialized = serializer.serialize(t)
        t_deserialized = serializer.deserialize(t_serialized)
        self.assertTrue(torch.equal(t, t_deserialized))

    def test_serializer_multi_dim(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        t_serialized = serializer.serialize(t)
        t_deserialized = serializer.deserialize(t_serialized)
        self.assertTrue(torch.equal(t, t_deserialized))

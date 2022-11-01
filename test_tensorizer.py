import tensorizer
import unittest
import torch

class TestTensorizer(unittest.TestCase):
    def test_tensorizer_gptj(self):
        from transformers import GPTJConfig, GPTJForCausalLM
        # instantiate dummy model config
        config = GPTJConfig(
            n_positions=128,
            n_embd=16,
            n_layer=2,
            n_head=2
        )
        model = GPTJForCausalLM(config=config)
        tensorizer.serialize_model(model, 'test/test')
        # deserialize model
        file_stream = open("test/test.tensors", "rb")
        ts = tensorizer.GooseTensorizer(file_stream)
        model2 = GPTJForCausalLM(config=config)
        ts.load_tensors(model2, dtype='float16')
        # compare models
        for name, param in model.named_parameters():
            param2 = model2.state_dict()[name]
            self.assertTrue(torch.allclose(param, param2, atol=1e-3))
    
    def tearDown(self):
        import shutil
        shutil.rmtree('test')

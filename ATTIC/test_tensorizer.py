from tensorizer import tensorizer

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
        tensorizer.serialize_model(model, config, 'test')
        model2 = tensorizer.load_model('test', GPTJForCausalLM, GPTJConfig, None, 'float16')
        # compare models
        for name, param in model.named_parameters():
            param2 = model2.state_dict()[name]
            self.assertTrue(torch.allclose(param, param2, atol=1e-3))
    
    def test_tensorizer_vae(self):
        from diffusers import AutoencoderKL

        # instantiate dummy VAE
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": (64,),
            "latent_channels": 2,
            "norm_num_groups": 2,
            "sample_size": 2,
        }

        model = AutoencoderKL(**config)
        tensorizer.serialize_model(model, config, 'test')
        model2 = tensorizer.load_model('test', AutoencoderKL, None, None, 'float16')
        # compare models
        for name, param in model.named_parameters():
            param2 = model2.state_dict()[name]
            self.assertTrue(torch.allclose(param, param2, atol=1e-3))

    def tearDown(self):
        import shutil
        shutil.rmtree('test')

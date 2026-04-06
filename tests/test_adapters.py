"""Tests for adapter modules: output shapes and length computation."""

import pytest
import torch

from src.models.adapters import ConcatMLP, ConvMLP, build_adapter


class TestConcatMLP:
    def test_output_shape(self):
        adapter = ConcatMLP(encoder_dim=768, concat_k=5, output_dim=896)
        x = torch.randn(2, 100, 768)
        lengths = torch.tensor([100, 80])
        out, out_lengths = adapter(x, lengths)
        assert out.shape == (2, 20, 896)  # 100 // 5 = 20
        assert out_lengths.tolist() == [20, 16]  # 100//5, 80//5

    def test_truncates_to_multiple_of_k(self):
        adapter = ConcatMLP(encoder_dim=768, concat_k=5, output_dim=896)
        x = torch.randn(1, 103, 768)  # 103 not divisible by 5
        lengths = torch.tensor([103])
        out, out_lengths = adapter(x, lengths)
        assert out.shape[1] == 20  # floor(103/5) = 20

    def test_hidden_dims(self):
        adapter = ConcatMLP(encoder_dim=768, concat_k=5, hidden_dims=[1024, 512], output_dim=896)
        x = torch.randn(1, 50, 768)
        lengths = torch.tensor([50])
        out, _ = adapter(x, lengths)
        assert out.shape == (1, 10, 896)

    def test_gradients_flow(self):
        adapter = ConcatMLP(encoder_dim=768, concat_k=5, output_dim=896)
        x = torch.randn(1, 50, 768, requires_grad=True)
        lengths = torch.tensor([50])
        out, _ = adapter(x, lengths)
        out.sum().backward()
        assert x.grad is not None


class TestConvMLP:
    def test_output_shape(self):
        adapter = ConvMLP(
            encoder_dim=768,
            conv_channels=1024,
            kernel_size=5,
            stride=5,
            output_dim=896,
        )
        x = torch.randn(2, 100, 768)
        lengths = torch.tensor([100, 80])
        out, out_lengths = adapter(x, lengths)
        # Conv output: floor((L - kernel) / stride) + 1
        assert out.shape[0] == 2
        assert out.shape[2] == 896
        expected_len_0 = (100 - 5) // 5 + 1  # 20
        expected_len_1 = (80 - 5) // 5 + 1  # 16
        assert out_lengths.tolist() == [expected_len_0, expected_len_1]

    def test_gradients_flow(self):
        adapter = ConvMLP(encoder_dim=768, stride=5, output_dim=896)
        x = torch.randn(1, 50, 768, requires_grad=True)
        lengths = torch.tensor([50])
        out, _ = adapter(x, lengths)
        out.sum().backward()
        assert x.grad is not None


class TestBuildAdapter:
    def test_build_concat_mlp(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "name": "concat_mlp",
                "encoder_dim": 768,
                "concat_k": 5,
                "hidden_dims": [2048],
                "output_dim": 896,
                "dropout": 0.1,
                "activation": "relu",
            }
        )
        adapter = build_adapter(cfg)
        assert isinstance(adapter, ConcatMLP)

    def test_build_conv_mlp(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "name": "conv_mlp",
                "encoder_dim": 768,
                "conv_channels": 1024,
                "kernel_size": 5,
                "stride": 5,
                "hidden_dims": [2048],
                "output_dim": 896,
                "dropout": 0.1,
                "activation": "relu",
            }
        )
        adapter = build_adapter(cfg)
        assert isinstance(adapter, ConvMLP)

    def test_unknown_adapter_raises(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown adapter"):
            build_adapter(cfg)

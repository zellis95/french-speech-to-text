"""Swappable adapter modules: ConcatMLP, ConvMLP, and registry."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAdapter(nn.Module, ABC):
    """Base class for encoder→LLM adapters.

    All adapters map (B, T, encoder_dim) → (B, T_out, output_dim)
    with potential temporal downsampling, and return updated lengths.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, encoder_dim) encoder output
            lengths: (B,) valid frame counts

        Returns:
            adapted: (B, T_out, output_dim) adapted features
            out_lengths: (B,) valid frame counts after downsampling
        """
        ...


class ConcatMLP(BaseAdapter):
    """Concatenate k consecutive frames then project through MLP.

    SLAM-ASR baseline architecture: reshape (B, T, D) → (B, T//k, k*D) then MLP.
    Provides k-fold temporal downsampling (50fps → 10fps at k=5).
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        concat_k: int = 5,
        hidden_dims: list[int] | None = None,
        output_dim: int = 896,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.concat_k = concat_k
        input_dim = encoder_dim * concat_k

        if hidden_dims is None:
            hidden_dims = [2048]

        activation_fn = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    activation_fn(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = hidden_states.shape
        k = self.concat_k

        # Truncate to multiple of k
        T_trunc = (T // k) * k
        hidden_states = hidden_states[:, :T_trunc, :]

        # Reshape: (B, T, D) → (B, T//k, k*D)
        hidden_states = hidden_states.reshape(B, T_trunc // k, k * D)

        out_lengths = lengths // k
        return self.mlp(hidden_states), out_lengths


class ConvMLP(BaseAdapter):
    """Conv1d downsampling followed by MLP projection.

    Learned downsampling variant: Conv1d with stride provides temporal
    reduction, then MLP projects to LLM hidden dimension.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        conv_channels: int = 1024,
        kernel_size: int = 5,
        stride: int = 5,
        hidden_dims: list[int] | None = None,
        output_dim: int = 896,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.stride = stride

        activation_fn = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]

        self.conv = nn.Conv1d(encoder_dim, conv_channels, kernel_size, stride=stride)
        self.conv_activation = activation_fn()

        if hidden_dims is None:
            hidden_dims = [2048]

        layers: list[nn.Module] = []
        prev_dim = conv_channels
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    activation_fn(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Conv1d expects (B, C, T)
        x = hidden_states.transpose(1, 2)  # (B, D, T)
        x = self.conv_activation(self.conv(x))  # (B, conv_channels, T_out)
        x = x.transpose(1, 2)  # (B, T_out, conv_channels)

        # Compute output lengths: floor((L - kernel_size) / stride) + 1
        out_lengths = (lengths - self.conv.kernel_size[0]) // self.stride + 1
        out_lengths = out_lengths.clamp(min=0)

        return self.mlp(x), out_lengths


ADAPTER_REGISTRY = {
    "concat_mlp": ConcatMLP,
    "conv_mlp": ConvMLP,
}


def build_adapter(cfg) -> BaseAdapter:
    """Build an adapter from a Hydra config object.

    Args:
        cfg: OmegaConf adapter config with 'name' field matching a registry key,
             plus adapter-specific params (concat_k, hidden_dims, etc.)
    """
    name = cfg.name
    if name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter: {name!r}. Available: {list(ADAPTER_REGISTRY)}")

    cls = ADAPTER_REGISTRY[name]

    # Build kwargs from config, excluding 'name'
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return cls(**kwargs)

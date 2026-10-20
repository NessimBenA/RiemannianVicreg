from typing import Literal, Optional

import torch
from torch import nn


class SimSiamProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SingleLayerProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class VICRegProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ProductFactorProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        trunk_dim: int,
        num_factors: int,
        factor_dim: int,
        head_hidden: int = 64,
        act: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        self.num_factors = num_factors
        self.factor_dim = factor_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.ReLU(inplace=True) if act == "relu" else nn.GELU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.ReLU(inplace=True) if act == "relu" else nn.GELU(),
        )

        Act = nn.ReLU if act == "relu" else nn.GELU
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, head_hidden),
                Act(),
                nn.Linear(head_hidden, factor_dim, bias=False),
            )
            for _ in range(num_factors)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        z = torch.cat([head(h) for head in self.heads], dim=-1)
        return z


class PredictorHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size: int, hidden_act: str = "gelu") -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if hidden_act == "gelu":
            self.transform_act_fn = nn.GELU()
        else:
            self.transform_act_fn = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MIMHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=patch_size**2 * num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


def build_projection_head(
    projection_type: Literal["SimSiam", "SingleLayer", "VICReg"],
    in_dim: int,
    out_dim: int,
    hidden_dim: Optional[int] = None,
) -> nn.Module:
    if projection_type == "SimSiam":
        if hidden_dim is None:
            hidden_dim = out_dim
        return SimSiamProjectionHead(in_dim, hidden_dim, out_dim)
    elif projection_type == "SingleLayer":
        return SingleLayerProjectionHead(in_dim, out_dim)
    elif projection_type == "VICReg":
        if hidden_dim is None:
            hidden_dim = out_dim
        return VICRegProjectionHead(in_dim, hidden_dim, out_dim)
    raise ValueError(f"Unknown projection type: {projection_type}")

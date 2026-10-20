from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class FusionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[torch.Tensor] = None


class TransformerFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.output_dim = embed_dim

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> FusionOutput:
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        out = self.transformer(x)

        return FusionOutput(
            last_hidden_state=out,
            pooler_output=out[:, 0, :],
        )


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        use_flava: bool = True,
        flava_pretrained: bool = True,
        flava_trainable: bool = True,
        embed_dim: int = 768,
        mlp_hidden_dim: int = 2048,
        mlp_output_dim: int = 768,
        num_modalities: int = 3,
        fusion_type: str = "transformer",
    ) -> None:
        super().__init__()
        self.fusion = TransformerFusion(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
        )
        self.output_dim = embed_dim

    def forward(self, embeddings: torch.Tensor) -> FusionOutput:
        return self.fusion(embeddings)

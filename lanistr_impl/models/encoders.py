from dataclasses import dataclass
from typing import Final, Literal, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig, ViTModel, ViTConfig


@dataclass
class TabularEncoderOutput:
    last_hidden_state: torch.Tensor
    masked_last_hidden_state: Optional[torch.Tensor] = None
    unmasked_last_hidden_state: Optional[torch.Tensor] = None
    masked_loss: Optional[torch.Tensor] = None


class ImageEncoder(nn.Module):
    DEFAULT_MODEL: Final[str] = "WinKawaks/vit-tiny-patch16-224"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: bool = True,
        trainable: bool = True,
        use_mask_token: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            self.encoder = ViTModel.from_pretrained(model_name, use_mask_token=use_mask_token)
        else:
            config = ViTConfig.from_pretrained(model_name)
            config.use_mask_token = use_mask_token
            self.encoder = ViTModel(config)

        if use_mask_token and self.encoder.embeddings.mask_token is None:
            self.encoder.embeddings.mask_token = nn.Parameter(
                torch.zeros(1, 1, self.encoder.config.hidden_size)
            )

        for param in self.encoder.parameters():
            param.requires_grad = trainable

        self._embedding_dim: Optional[int] = None

    @property
    def config(self) -> ViTConfig:
        return self.encoder.config

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._embedding_dim = self.encoder.config.hidden_size
        return self._embedding_dim

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            return_dict=True,
        )
        return outputs.last_hidden_state


class TextEncoder(nn.Module):
    DEFAULT_MODEL: Final[str] = "prajjwal1/bert-mini"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)

        for param in self.encoder.parameters():
            param.requires_grad = trainable

        self._embedding_dim: Optional[int] = None

    @property
    def config(self):
        return self.encoder.config

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._embedding_dim = self.encoder.config.hidden_size
        return self._embedding_dim

    @property
    def vocab_size(self) -> int:
        return self.encoder.config.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state


class TabularEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cat_idxs: list[int],
        cat_dims: list[int],
        cat_emb_dim: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim

        self.embeddings = nn.ModuleList()
        for dim in cat_dims:
            self.embeddings.append(nn.Embedding(dim + 1, cat_emb_dim))

        num_numerical = input_dim - len(cat_idxs)
        self.post_embed_dim = num_numerical + len(cat_idxs) * cat_emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = []
        cat_idx_ptr = 0
        for i in range(self.input_dim):
            if i in self.cat_idxs:
                cat_values = x[:, i].long()
                emb = self.embeddings[cat_idx_ptr](cat_values)
                cols.append(emb)
                cat_idx_ptr += 1
            else:
                cols.append(x[:, i : i + 1])
        return torch.cat(cols, dim=-1)


class TabularEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cat_idxs: list[int],
        cat_dims: list[int],
        output_dim: int = 128,
        cat_emb_dim: int = 3,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        mask_type: Literal["sparsemax", "entmax"] = "sparsemax",
        pretraining_ratio: float = 0.15,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.pretraining_ratio = pretraining_ratio
        self.output_dim = output_dim

        self.embedding = TabularEmbedding(input_dim, cat_idxs, cat_dims, cat_emb_dim)
        post_embed_dim = self.embedding.post_embed_dim

        self.initial_bn = nn.BatchNorm1d(post_embed_dim)
        self.encoder_layers = nn.Sequential(
            nn.Linear(post_embed_dim, n_d),
            nn.BatchNorm1d(n_d),
            nn.ReLU(),
            nn.Linear(n_d, n_d),
            nn.BatchNorm1d(n_d),
            nn.ReLU(),
            nn.Linear(n_d, output_dim),
        )

        self.reconstruction_layer = nn.Linear(output_dim, post_embed_dim)

        for param in self.parameters():
            param.requires_grad = trainable

        self._embedding_dim = output_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self,
        features: torch.Tensor,
        pretraining: bool = False,
    ) -> TabularEncoderOutput:
        embedded = self.embedding(features)
        normalized = self.initial_bn(embedded)

        if pretraining:
            mask = torch.rand(normalized.shape, device=normalized.device) < self.pretraining_ratio
            masked_input = normalized.clone()
            masked_input[mask] = 0.0

            masked_hidden = self.encoder_layers(masked_input)
            unmasked_hidden = self.encoder_layers(normalized)

            reconstructed = self.reconstruction_layer(masked_hidden)
            masked_loss = nn.functional.mse_loss(
                reconstructed[mask], normalized[mask]
            )

            return TabularEncoderOutput(
                last_hidden_state=unmasked_hidden,
                masked_last_hidden_state=masked_hidden,
                unmasked_last_hidden_state=unmasked_hidden,
                masked_loss=masked_loss,
            )

        hidden = self.encoder_layers(normalized)
        return TabularEncoderOutput(last_hidden_state=hidden)

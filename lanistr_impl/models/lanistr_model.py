from dataclasses import dataclass
from typing import Mapping, Optional

import math
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig

from .encoders import ImageEncoder, TextEncoder, TabularEncoder
from .fusion import MultiModalFusion
from .heads import (
    build_projection_head,
    PredictorHead,
    ClassifierHead,
    MLMHead,
    MIMHead,
    ProductFactorProjector,
)
from ..losses import (
    negative_cosine_similarity,
    vicreg_loss,
    riemannian_vicreg_loss_paired_product,
    riemannian_vicreg_loss_paired_product_per_factor,
    MaskedImageModelingLoss,
    MLMLoss,
    LANISTRLossComputer,
)


@dataclass
class LANISTRPretrainOutput:
    loss: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss_mlm: Optional[torch.Tensor] = None
    loss_mim: Optional[torch.Tensor] = None
    loss_mfm: Optional[torch.Tensor] = None
    loss_mmm: Optional[torch.Tensor] = None
    loss_inv: Optional[torch.Tensor] = None
    loss_var: Optional[torch.Tensor] = None
    loss_cov: Optional[torch.Tensor] = None
    boundary_proximity_max: Optional[torch.Tensor] = None
    boundary_proximity_mean: Optional[torch.Tensor] = None
    proj_activity: Optional[torch.Tensor] = None
    tangent_norm_mean: Optional[torch.Tensor] = None
    kappa: Optional[torch.Tensor] = None
    kappa_std: Optional[torch.Tensor] = None
    kappa_min: Optional[torch.Tensor] = None
    kappa_max: Optional[torch.Tensor] = None
    clip_activity: Optional[torch.Tensor] = None


@dataclass
class LANISTRFinetuneOutput:
    loss: torch.Tensor
    logits: torch.Tensor


@dataclass
class LANISTRRepresentationCapture:
    fusion_unmasked: torch.Tensor
    fusion_masked: torch.Tensor
    projector_unmasked: torch.Tensor
    projector_masked: torch.Tensor


class LANISTRPretrainModel(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        tabular_data_info: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.device_holder = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.capture_representations = False
        self.representation_capture: Optional[LANISTRRepresentationCapture] = None

        self.use_image = config.dataset.image
        self.use_text = config.dataset.text
        self.use_tab = config.dataset.tab
        self.ssl_objective = config.training.get("ssl_objective", "simsiam")

        self.image_encoder: Optional[ImageEncoder] = None
        self.text_encoder: Optional[TextEncoder] = None
        self.tabular_encoder: Optional[TabularEncoder] = None

        self.image_proj: Optional[nn.Module] = None
        self.text_proj: Optional[nn.Module] = None
        self.tabular_proj: Optional[nn.Module] = None

        self.mlm_head: Optional[MLMHead] = None
        self.mim_head: Optional[MIMHead] = None

        projection_dim = config.projection.output_dim

        if self.use_image:
            self.image_encoder = ImageEncoder(
                model_name=config.encoders.image.name,
                pretrained=config.encoders.image.pretrained,
                trainable=config.encoders.image.trainable,
                use_mask_token=True,
            )
            self.image_proj = build_projection_head(
                projection_type="SingleLayer",
                in_dim=self.image_encoder.embedding_dim,
                out_dim=projection_dim,
            )
            self.mim_head = MIMHead(
                hidden_size=self.image_encoder.embedding_dim,
                patch_size=self.image_encoder.config.patch_size,
                num_channels=self.image_encoder.config.num_channels,
            )

        if self.use_text:
            self.text_encoder = TextEncoder(
                model_name=config.encoders.text.name,
                pretrained=config.encoders.text.pretrained,
                trainable=config.encoders.text.trainable,
            )
            self.text_proj = build_projection_head(
                projection_type="SingleLayer",
                in_dim=self.text_encoder.embedding_dim,
                out_dim=projection_dim,
            )
            self.mlm_head = MLMHead(
                hidden_size=self.text_encoder.embedding_dim,
                vocab_size=self.text_encoder.vocab_size,
            )

        if self.use_tab:
            self.tabular_encoder = TabularEncoder(
                input_dim=tabular_data_info["input_dim"],
                cat_idxs=tabular_data_info["cat_idxs"],
                cat_dims=tabular_data_info["cat_dims"],
                output_dim=config.encoders.tabular.embedding_dim,
                cat_emb_dim=config.encoders.tabular.cat_emb_dim,
                n_d=config.encoders.tabular.n_d,
                n_a=config.encoders.tabular.n_a,
                pretraining_ratio=config.encoders.tabular.pretraining_ratio,
                trainable=config.encoders.tabular.trainable,
            )
            self.tabular_proj = build_projection_head(
                projection_type="SingleLayer",
                in_dim=self.tabular_encoder.embedding_dim,
                out_dim=projection_dim,
            )

        num_modalities = sum([self.use_image, self.use_text, self.use_tab])
        self.fusion = MultiModalFusion(
            use_flava=config.fusion.use_flava,
            flava_pretrained=config.fusion.flava_pretrained,
            flava_trainable=config.fusion.flava_trainable,
            embed_dim=projection_dim,
            mlp_hidden_dim=config.fusion.mlp_hidden_dim,
            mlp_output_dim=config.fusion.output_dim,
            num_modalities=num_modalities,
            fusion_type=config.fusion.get("type", "transformer"),
        )

        mm_proj_type = "VICReg" if self.ssl_objective in ("vicreg", "rvicreg") else config.projection.type
        self.mm_proj = build_projection_head(
            projection_type=mm_proj_type,
            in_dim=self.fusion.output_dim,
            out_dim=config.projection.output_dim,
            hidden_dim=config.projection.hidden_dim,
        )

        self.mm_predictor = PredictorHead(
            in_dim=config.projection.output_dim,
            hidden_dim=config.simsiam.predictor_hidden_dim,
            out_dim=config.simsiam.predictor_output_dim,
        )

        self.mlm_probability = config.encoders.text.mlm_probability
        self.loss_computer = LANISTRLossComputer(
            lambda_mlm=config.loss_weights.lambda_mlm,
            lambda_mim=config.loss_weights.lambda_mim,
            lambda_mfm=config.loss_weights.lambda_mfm,
            lambda_mmm=config.loss_weights.lambda_mmm,
        )

        if self.ssl_objective == "vicreg":
            vicreg_cfg = config.get("vicreg", {})
            self.vicreg_sim_coeff = vicreg_cfg.get("sim_coeff", 25.0)
            self.vicreg_std_coeff = vicreg_cfg.get("std_coeff", 25.0)
            self.vicreg_cov_coeff = vicreg_cfg.get("cov_coeff", 1.0)

        if self.ssl_objective == "rvicreg":
            riem_cfg = config.get("riemannian", {})
            if not riem_cfg.get("enabled", False):
                raise ValueError("ssl_objective='rvicreg' requires riemannian.enabled=true in config")
            self.rvicreg_sim_coeff = riem_cfg.get("lambda_set", 25.0)
            self.rvicreg_std_coeff = riem_cfg.get("lambda_var", 25.0)
            self.rvicreg_cov_coeff = riem_cfg.get("lambda_cov", 1.0)
            self.global_kappa = riem_cfg.get("kappa", -1.0)
            self.learnable_kappa = riem_cfg.get("learnable_kappa", False)
            self.per_factor_kappa = riem_cfg.get("per_factor_kappa", False)
            self.factor_separated_projector = riem_cfg.get("factor_separated_projector", False)
            self.kappa_max_abs = riem_cfg.get("kappa_max_abs", 1.5)
            self.eps = riem_cfg.get("eps", 0.0001)
            self.clip_delta = riem_cfg.get("clip_delta", 1e-3)
            self.gamma = riem_cfg.get("gamma", 1.0)
            self.num_factors = riem_cfg.get("num_factors", 1)
            self.factor_dim = riem_cfg.get("factor_dim", 256)
            self.factor_head_hidden = riem_cfg.get("factor_head_hidden", 64)
            self.use_isotropy = riem_cfg.get("use_isotropy", False)
            self.use_frechet_mean = riem_cfg.get("use_frechet_mean", False)
            self.frechet_normalize_tangents = riem_cfg.get("frechet_normalize_tangents", False)
            if self.num_factors > 1:
                D = config.projection.output_dim
                F_factors = self.num_factors
                d = self.factor_dim
                if D != F_factors * d:
                    raise ValueError(f"projection.output_dim={D} must equal num_factors*factor_dim={F_factors*d}")
            self.kappa_init_noise = riem_cfg.get("kappa_init_noise", 0.01)
            self.kappa_init_spread = riem_cfg.get("kappa_init_spread", 0.0)
            if self.learnable_kappa and self.num_factors > 1 and self.per_factor_kappa:
                kappa_init = self.global_kappa
                p = min(max(abs(kappa_init) / self.kappa_max_abs, 1e-4), 1 - 1e-4)
                theta0 = math.log(p / (1 - p))
                if self.kappa_init_spread > 0:
                    spread = torch.linspace(-self.kappa_init_spread, self.kappa_init_spread, self.num_factors)
                else:
                    spread = torch.zeros(self.num_factors)
                noise = self.kappa_init_noise * torch.randn(self.num_factors)
                self.kappa_theta = nn.Parameter(
                    torch.full((self.num_factors,), float(theta0)) + spread + noise
                )
            elif self.learnable_kappa:
                self.kappa_param = nn.Parameter(torch.tensor(self.global_kappa))
            if self.factor_separated_projector and self.num_factors > 1:
                self.mm_proj = ProductFactorProjector(
                    in_dim=self.fusion.output_dim,
                    trunk_dim=config.projection.hidden_dim,
                    num_factors=self.num_factors,
                    factor_dim=self.factor_dim,
                    head_hidden=self.factor_head_hidden,
                    act="relu",
                )

        self.mim_loss_fn = MaskedImageModelingLoss(
            patch_size=config.encoders.image.patch_size if self.use_image else 16,
            num_channels=3,
        )
        self.mlm_loss_fn = MLMLoss()

    @property
    def device(self) -> torch.device:
        return self.device_holder.device

    def get_kappa(self) -> torch.Tensor:
        if hasattr(self, 'learnable_kappa') and self.learnable_kappa:
            if hasattr(self, 'kappa_theta'):
                return -self.kappa_max_abs * torch.sigmoid(self.kappa_theta)
            return self.kappa_param
        return torch.tensor(getattr(self, 'global_kappa', -1.0), device=self.device)

    def get_kappas_product(self) -> torch.Tensor:
        if self.learnable_kappa and getattr(self, "per_factor_kappa", False) and hasattr(self, "kappa_theta"):
            return -self.kappa_max_abs * torch.sigmoid(self.kappa_theta)
        if self.learnable_kappa and hasattr(self, "kappa_param"):
            return self.kappa_param.unsqueeze(0).expand(self.num_factors)
        return torch.full((self.num_factors,), float(self.global_kappa), device=self.device)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> LANISTRPretrainOutput:
        device = self.device

        loss_mlm = torch.zeros(1, device=device)
        loss_mim = torch.zeros(1, device=device)
        loss_mfm = torch.zeros(1, device=device)

        embeds: list[torch.Tensor] = []
        masked_embeds: list[torch.Tensor] = []

        if self.use_text and self.text_encoder is not None:
            input_ids = batch["input_ids"].squeeze(1)
            attention_mask = batch["attention_mask"].squeeze(1)
            batch_size = input_ids.shape[0]

            masked_input_ids = input_ids.clone()
            mlm_labels = input_ids.clone()

            rand = torch.rand(input_ids.shape, device=device)
            special_tokens = (input_ids == 101) | (input_ids == 102) | (input_ids == 0)
            mask_arr = (rand < self.mlm_probability) & ~special_tokens

            masked_input_ids[mask_arr] = 103
            mlm_labels[~mask_arr] = -100

            mlm_outputs = self.text_encoder(masked_input_ids, attention_mask)
            mlm_prediction_scores = self.mlm_head(mlm_outputs)
            loss_mlm = self.mlm_loss_fn(
                mlm_prediction_scores,
                mlm_labels,
                self.text_encoder.vocab_size,
            )

            mlm_text_emb = self.text_proj(mlm_outputs[:, 0, :])
            mlm_text_emb = F.normalize(mlm_text_emb, dim=1)
            masked_embeds.append(mlm_text_emb.unsqueeze(1))

            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_emb = self.text_proj(text_outputs[:, 0, :])
            text_emb = F.normalize(text_emb, dim=1)
            embeds.append(text_emb.unsqueeze(1))

        if self.use_image and self.image_encoder is not None:
            pixel_values = batch["pixel_values"]
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.squeeze(1)
            bool_masked_pos = batch["bool_masked_pos"]

            mim_outputs = self.image_encoder(pixel_values, bool_masked_pos)
            sequence_output = mim_outputs[:, 1:]
            batch_size, seq_len, hidden_dim = sequence_output.shape
            height = width = int(math.sqrt(seq_len))
            sequence_output = sequence_output.permute(0, 2, 1).reshape(
                batch_size, hidden_dim, height, width
            )
            reconstructed = self.mim_head(sequence_output)

            loss_mim = self.mim_loss_fn(
                reconstructed,
                pixel_values,
                bool_masked_pos,
                self.image_encoder.config.image_size,
            )

            mim_img_emb = self.image_proj(mim_outputs)
            mim_img_emb = F.normalize(mim_img_emb, dim=-1)
            masked_embeds.append(mim_img_emb)

            img_outputs = self.image_encoder(pixel_values, None)
            img_emb = self.image_proj(img_outputs)
            img_emb = F.normalize(img_emb, dim=-1)
            embeds.append(img_emb)

        if self.use_tab and self.tabular_encoder is not None:
            features = batch["features"]
            tab_output = self.tabular_encoder(features, pretraining=True)
            loss_mfm = tab_output.masked_loss if tab_output.masked_loss is not None else torch.zeros(1, device=device)

            masked_tab_emb = self.tabular_proj(tab_output.masked_last_hidden_state)
            masked_tab_emb = F.normalize(masked_tab_emb, dim=1)
            masked_embeds.append(masked_tab_emb.unsqueeze(1))

            unmasked_tab_emb = self.tabular_proj(tab_output.unmasked_last_hidden_state)
            unmasked_tab_emb = F.normalize(unmasked_tab_emb, dim=1)
            embeds.append(unmasked_tab_emb.unsqueeze(1))

        concat_emb = torch.cat(embeds, dim=1)
        concat_masked_emb = torch.cat(masked_embeds, dim=1)

        mm_out = self.fusion(concat_emb)
        mm_out_masked = self.fusion(concat_masked_emb)

        loss_mmm = torch.zeros(1, device=device)
        loss_inv = torch.zeros(1, device=device)
        loss_var = torch.zeros(1, device=device)
        loss_cov = torch.zeros(1, device=device)
        boundary_proximity_max = None
        boundary_proximity_mean = None
        proj_activity = None
        tangent_norm_mean = None
        kappa_value = None
        kappa_std = None
        kappa_min = None
        kappa_max = None
        clip_activity = None

        if self.ssl_objective == "vicreg":
            mm_hidden = mm_out.last_hidden_state
            mm_hidden_masked = mm_out_masked.last_hidden_state
            if mm_hidden.dim() == 3:
                mm_hidden = mm_hidden[:, 0, :]
                mm_hidden_masked = mm_hidden_masked[:, 0, :]
            z1 = self.mm_proj(mm_hidden)
            z2 = self.mm_proj(mm_hidden_masked)
            loss_mmm, vicreg_stats = vicreg_loss(
                z1,
                z2,
                sim_coeff=self.vicreg_sim_coeff,
                std_coeff=self.vicreg_std_coeff,
                cov_coeff=self.vicreg_cov_coeff,
            )
            loss_inv = vicreg_stats["inv_loss"]
            loss_var = vicreg_stats["var_loss"]
            loss_cov = vicreg_stats["cov_loss"]
            logits = z1
        elif self.ssl_objective == "rvicreg":
            mm_hidden = mm_out.last_hidden_state
            mm_hidden_masked = mm_out_masked.last_hidden_state
            if mm_hidden.dim() == 3:
                mm_hidden = mm_hidden[:, 0, :]
                mm_hidden_masked = mm_hidden_masked[:, 0, :]
            z1 = self.mm_proj(mm_hidden)
            z2 = self.mm_proj(mm_hidden_masked)
            if self.num_factors > 1 and getattr(self, "per_factor_kappa", False):
                kappas = self.get_kappas_product()
                loss_mmm, rvicreg_stats = riemannian_vicreg_loss_paired_product_per_factor(
                    z1,
                    z2,
                    kappas=kappas,
                    num_factors=self.num_factors,
                    factor_dim=self.factor_dim,
                    sim_coeff=self.rvicreg_sim_coeff,
                    std_coeff=self.rvicreg_std_coeff,
                    cov_coeff=self.rvicreg_cov_coeff,
                    eps=self.eps,
                    clip_delta=self.clip_delta,
                    gamma=self.gamma,
                    use_isotropy=self.use_isotropy,
                    use_frechet_mean=self.use_frechet_mean,
                    frechet_normalize_tangents=self.frechet_normalize_tangents,
                )
            else:
                kappa = self.get_kappa()
                loss_mmm, rvicreg_stats = riemannian_vicreg_loss_paired_product(
                    z1,
                    z2,
                    kappa=kappa,
                    num_factors=self.num_factors,
                    factor_dim=self.factor_dim,
                    sim_coeff=self.rvicreg_sim_coeff,
                    std_coeff=self.rvicreg_std_coeff,
                    cov_coeff=self.rvicreg_cov_coeff,
                    eps=self.eps,
                    clip_delta=self.clip_delta,
                    gamma=self.gamma,
                )
            loss_inv = rvicreg_stats["inv_loss"]
            loss_var = rvicreg_stats["var_loss"]
            loss_cov = rvicreg_stats["cov_loss"]
            boundary_proximity_max = rvicreg_stats.get("boundary_proximity_max")
            boundary_proximity_mean = rvicreg_stats.get("boundary_proximity_mean")
            proj_activity = rvicreg_stats.get("proj_activity")
            tangent_norm_mean = rvicreg_stats.get("tangent_norm_mean")
            kappa_value = rvicreg_stats.get("kappa")
            kappa_std = rvicreg_stats.get("kappa_std")
            kappa_min = rvicreg_stats.get("kappa_min")
            kappa_max = rvicreg_stats.get("kappa_max")
            clip_activity = rvicreg_stats.get("clip_activity")
            logits = z1
        else:
            mm_hidden = mm_out.last_hidden_state
            mm_hidden_masked = mm_out_masked.last_hidden_state
            if mm_hidden.dim() == 3:
                mm_hidden = mm_hidden[:, 0, :]
                mm_hidden_masked = mm_hidden_masked[:, 0, :]
            z1 = self.mm_proj(mm_hidden)
            z2 = self.mm_proj(mm_hidden_masked)
            p1 = self.mm_predictor(z1)
            p2 = self.mm_predictor(z2)
            loss_mmm = (
                negative_cosine_similarity(p1, z2) / 2
                + negative_cosine_similarity(p2, z1) / 2
            )
            logits = p1

        total_loss = self.loss_computer.compute_total_loss(
            loss_mlm, loss_mim, loss_mfm, loss_mmm
        )

        if self.capture_representations:
            self.representation_capture = LANISTRRepresentationCapture(
                fusion_unmasked=mm_hidden.detach(),
                fusion_masked=mm_hidden_masked.detach(),
                projector_unmasked=z1.detach(),
                projector_masked=z2.detach(),
            )

        return LANISTRPretrainOutput(
            loss=total_loss,
            logits=logits,
            loss_mlm=loss_mlm,
            loss_mim=loss_mim,
            loss_mfm=loss_mfm,
            loss_mmm=loss_mmm,
            loss_inv=loss_inv,
            loss_var=loss_var,
            loss_cov=loss_cov,
            boundary_proximity_max=boundary_proximity_max,
            boundary_proximity_mean=boundary_proximity_mean,
            proj_activity=proj_activity,
            tangent_norm_mean=tangent_norm_mean,
            kappa=kappa_value,
            kappa_std=kappa_std,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            clip_activity=clip_activity,
        )


class LANISTRFinetuneModel(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        tabular_data_info: dict,
        num_classes: int,
        pretrain_model: Optional[LANISTRPretrainModel] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device_holder = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.use_image = config.dataset.image
        self.use_text = config.dataset.text
        self.use_tab = config.dataset.tab

        projection_dim = config.projection.output_dim

        if pretrain_model is not None:
            self.image_encoder = pretrain_model.image_encoder
            self.text_encoder = pretrain_model.text_encoder
            self.tabular_encoder = pretrain_model.tabular_encoder
            self.image_proj = pretrain_model.image_proj
            self.text_proj = pretrain_model.text_proj
            self.tabular_proj = pretrain_model.tabular_proj
            self.fusion = pretrain_model.fusion
        else:
            self.image_encoder: Optional[ImageEncoder] = None
            self.text_encoder: Optional[TextEncoder] = None
            self.tabular_encoder: Optional[TabularEncoder] = None
            self.image_proj: Optional[nn.Module] = None
            self.text_proj: Optional[nn.Module] = None
            self.tabular_proj: Optional[nn.Module] = None

            if self.use_image:
                self.image_encoder = ImageEncoder(
                    model_name=config.encoders.image.name,
                    pretrained=config.encoders.image.pretrained,
                    trainable=config.encoders.image.trainable,
                    use_mask_token=False,
                )
                self.image_proj = build_projection_head(
                    projection_type="SingleLayer",
                    in_dim=self.image_encoder.embedding_dim,
                    out_dim=projection_dim,
                )

            if self.use_text:
                self.text_encoder = TextEncoder(
                    model_name=config.encoders.text.name,
                    pretrained=config.encoders.text.pretrained,
                    trainable=config.encoders.text.trainable,
                )
                self.text_proj = build_projection_head(
                    projection_type="SingleLayer",
                    in_dim=self.text_encoder.embedding_dim,
                    out_dim=projection_dim,
                )

            if self.use_tab:
                self.tabular_encoder = TabularEncoder(
                    input_dim=tabular_data_info["input_dim"],
                    cat_idxs=tabular_data_info["cat_idxs"],
                    cat_dims=tabular_data_info["cat_dims"],
                    output_dim=config.encoders.tabular.embedding_dim,
                    cat_emb_dim=config.encoders.tabular.cat_emb_dim,
                    n_d=config.encoders.tabular.n_d,
                    n_a=config.encoders.tabular.n_a,
                    trainable=config.encoders.tabular.trainable,
                )
                self.tabular_proj = build_projection_head(
                    projection_type="SingleLayer",
                    in_dim=self.tabular_encoder.embedding_dim,
                    out_dim=projection_dim,
                )

            num_modalities = sum([self.use_image, self.use_text, self.use_tab])
            self.fusion = MultiModalFusion(
                use_flava=config.fusion.use_flava,
                flava_pretrained=config.fusion.flava_pretrained,
                flava_trainable=config.fusion.flava_trainable,
                embed_dim=projection_dim,
                mlp_hidden_dim=config.fusion.mlp_hidden_dim,
                mlp_output_dim=config.fusion.output_dim,
                num_modalities=num_modalities,
                fusion_type=config.fusion.get("type", "transformer"),
            )

        self.classifier = ClassifierHead(
            in_dim=self.fusion.output_dim,
            hidden_dim=config.classifier.hidden_dim,
            num_classes=num_classes,
            dropout=config.classifier.dropout,
        )

    @property
    def device(self) -> torch.device:
        return self.device_holder.device

    def freeze_encoders(self) -> None:
        if self.image_encoder is not None:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if self.tabular_encoder is not None:
            for param in self.tabular_encoder.parameters():
                param.requires_grad = False

    def encode_fusion(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        embeds: list[torch.Tensor] = []

        if self.use_text and self.text_encoder is not None:
            input_ids = batch["input_ids"]
            if input_ids.dim() == 3:
                batch_size, text_num, seq_len = input_ids.shape
                input_ids = input_ids.flatten(0, 1)
                attention_mask = batch["attention_mask"].flatten(0, 1)
            else:
                batch_size = input_ids.shape[0]
                text_num = 1
                attention_mask = batch["attention_mask"]

            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_emb = self.text_proj(text_outputs[:, 0, :])

            if text_num > 1:
                text_emb = text_emb.view(batch_size, text_num, -1).mean(dim=1)

            text_emb = F.normalize(text_emb, dim=1)
            embeds.append(text_emb.unsqueeze(1))

        if self.use_image and self.image_encoder is not None:
            pixel_values = batch["pixel_values"]
            if pixel_values.dim() == 5:
                batch_size, img_num, c, h, w = pixel_values.shape
                pixel_values = pixel_values.flatten(0, 1)
            else:
                batch_size = pixel_values.shape[0]
                img_num = 1

            img_outputs = self.image_encoder(pixel_values, None)
            img_emb = self.image_proj(img_outputs)

            if img_num > 1:
                img_emb = img_emb.view(batch_size, img_num, img_emb.shape[1], -1).mean(dim=1)

            img_emb = F.normalize(img_emb, dim=-1)
            embeds.append(img_emb)

        if self.use_tab and self.tabular_encoder is not None:
            features = batch["features"]
            tab_output = self.tabular_encoder(features, pretraining=False)
            tab_emb = self.tabular_proj(tab_output.last_hidden_state)
            tab_emb = F.normalize(tab_emb, dim=1)
            embeds.append(tab_emb.unsqueeze(1))

        concat_emb = torch.cat(embeds, dim=1)
        mm_out = self.fusion(concat_emb)
        return mm_out.last_hidden_state[:, 0, :]

    def forward(self, batch: Mapping[str, torch.Tensor]) -> LANISTRFinetuneOutput:
        fusion_cls = self.encode_fusion(batch)
        logits = self.classifier(fusion_cls)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)
        return LANISTRFinetuneOutput(loss=loss, logits=logits)


def build_lanistr_model(
    config: DictConfig,
    tabular_data_info: dict,
    num_classes: int,
    pretrain_checkpoint: Optional[str] = None,
) -> nn.Module:
    if config.training.mode == "simsiam":
        return LANISTRPretrainModel(config, tabular_data_info)

    elif config.training.mode == "supervised":
        return LANISTRFinetuneModel(
            config,
            tabular_data_info,
            num_classes,
            pretrain_model=None,
        )

    elif config.training.mode == "simsiam_finetune":
        if pretrain_checkpoint is not None:
            pretrain_model = LANISTRPretrainModel(config, tabular_data_info)
            checkpoint = torch.load(pretrain_checkpoint, map_location="cpu")
            pretrain_model.load_state_dict(checkpoint["model_state_dict"])
            return LANISTRFinetuneModel(
                config,
                tabular_data_info,
                num_classes,
                pretrain_model=pretrain_model,
            )
        return LANISTRPretrainModel(config, tabular_data_info)

    raise ValueError(f"Unknown training mode: {config.training.mode}")

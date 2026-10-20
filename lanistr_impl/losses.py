from typing import Literal, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .geometry import dist_sq, log_0, exp_0, clip_tangent_norm, project_to_ball, log_mu, frechet_mean, mobius_add

ScalarLike = Union[float, Tensor]


def negative_cosine_similarity(
    p: torch.Tensor,
    z: torch.Tensor,
    version: Literal["original", "simplified"] = "simplified",
) -> torch.Tensor:
    if version == "original":
        z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()
    elif version == "simplified":
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    raise ValueError(f"Invalid version: {version}")


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def isotropy_loss_per_factor(
    t1_factors: list[Tensor],
    t2_factors: list[Tensor],
    eps: float = 1e-4,
) -> Tensor:
    F_factors = len(t1_factors)
    iso_loss_sum = torch.tensor(0.0, device=t1_factors[0].device, dtype=t1_factors[0].dtype)
    for f_idx in range(F_factors):
        t1_f = t1_factors[f_idx]
        t2_f = t2_factors[f_idx]
        B, d = t1_f.shape
        t1_centered = t1_f - t1_f.mean(dim=0)
        t2_centered = t2_f - t2_f.mean(dim=0)
        cov_t1 = (t1_centered.T @ t1_centered) / (B - 1)
        cov_t2 = (t2_centered.T @ t2_centered) / (B - 1)
        trace_t1 = cov_t1.trace()
        trace_t2 = cov_t2.trace()
        frob_sq_t1 = (cov_t1 * cov_t1).sum()
        frob_sq_t2 = (cov_t2 * cov_t2).sum()
        mean_lambda_t1 = trace_t1 / d
        mean_lambda_t2 = trace_t2 / d
        iso_t1 = frob_sq_t1 / (d * mean_lambda_t1.pow(2).clamp_min(eps)) - 1.0
        iso_t2 = frob_sq_t2 / (d * mean_lambda_t2.pow(2).clamp_min(eps)) - 1.0
        iso_loss_sum = iso_loss_sum + iso_t1 + iso_t2
    return iso_loss_sum / F_factors


def vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 0.0001,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + eps)
    std_y = torch.sqrt(y.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    batch_size = x.shape[0]
    num_features = x.shape[1]
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

    total_loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss

    stats = {
        "inv_loss": repr_loss.detach(),
        "var_loss": std_loss.detach(),
        "cov_loss": cov_loss.detach(),
    }
    return total_loss, stats


def riemannian_vicreg_loss_paired_product(
    z1: torch.Tensor,
    z2: torch.Tensor,
    kappa: ScalarLike,
    num_factors: int,
    factor_dim: int,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
    clip_delta: float = 1e-3,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert z1.dim() == 2 and z2.dim() == 2
    B, D = z1.shape
    assert z2.shape == (B, D)

    F_factors = num_factors
    d = factor_dim
    assert D == F_factors * d, f"D={D} != F*d={F_factors*d}"

    z1f = z1.reshape(B * F_factors, d)
    z2f = z2.reshape(B * F_factors, d)

    k_t = torch.as_tensor(kappa, dtype=z1.dtype, device=z1.device)
    sqrt_abs_k = k_t.abs().clamp_min(eps).sqrt()
    if (k_t < 0).item():
        r_max = torch.atanh(torch.tensor(1.0 - clip_delta, device=z1.device, dtype=z1.dtype)) / sqrt_abs_k
        clip_act_1 = (z1f.norm(dim=-1) > r_max).float().mean()
        clip_act_2 = (z2f.norm(dim=-1) > r_max).float().mean()
        clip_activity = (clip_act_1 + clip_act_2) / 2.0
    else:
        clip_activity = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

    z1c = clip_tangent_norm(z1f, kappa, delta=clip_delta)
    z2c = clip_tangent_norm(z2f, kappa, delta=clip_delta)

    x1_before = exp_0(z1c, kappa, eps)
    x2_before = exp_0(z2c, kappa, eps)
    x1 = project_to_ball(x1_before, kappa)
    x2 = project_to_ball(x2_before, kappa)

    d2 = dist_sq(x1, x2, kappa, eps)
    inv_loss = d2.mean() / (4.0 * d)

    t1 = log_0(x1, kappa, eps).reshape(B, D)
    t2 = log_0(x2, kappa, eps).reshape(B, D)

    t1 = t1 - t1.mean(dim=0)
    t2 = t2 - t2.mean(dim=0)

    std_t1 = torch.sqrt(t1.var(dim=0) + eps)
    std_t2 = torch.sqrt(t2.var(dim=0) + eps)
    var_loss = (torch.relu(gamma - std_t1).mean() + torch.relu(gamma - std_t2).mean()) / 2.0

    cov_t1 = (t1.T @ t1) / (B - 1)
    cov_t2 = (t2.T @ t2) / (B - 1)
    cov_loss = (
        off_diagonal(cov_t1).pow(2).sum() / D
        + off_diagonal(cov_t2).pow(2).sum() / D
    )

    total = sim_coeff * inv_loss + std_coeff * var_loss + cov_coeff * cov_loss

    x1_norm = x1.norm(dim=-1)
    x2_norm = x2.norm(dim=-1)
    boundary_proximity_mean = (sqrt_abs_k * torch.cat([x1_norm, x2_norm])).mean()
    boundary_proximity_max = (sqrt_abs_k * torch.cat([x1_norm, x2_norm])).max()

    proj_active_1 = (x1.norm(dim=-1) < x1_before.norm(dim=-1) - 1e-8).float().mean()
    proj_active_2 = (x2.norm(dim=-1) < x2_before.norm(dim=-1) - 1e-8).float().mean()
    proj_activity = (proj_active_1 + proj_active_2) / 2.0

    tangent_norm_mean = (t1.norm(dim=-1).mean() + t2.norm(dim=-1).mean()) / 2.0

    stats = {
        "inv_loss": inv_loss.detach(),
        "var_loss": var_loss.detach(),
        "cov_loss": cov_loss.detach(),
        "kappa": k_t.detach(),
        "clip_activity": clip_activity.detach(),
        "boundary_proximity_mean": boundary_proximity_mean.detach(),
        "boundary_proximity_max": boundary_proximity_max.detach(),
        "proj_activity": proj_activity.detach(),
        "tangent_norm_mean": tangent_norm_mean.detach(),
    }
    return total, stats


def riemannian_vicreg_loss_paired_product_per_factor(
    z1: torch.Tensor,
    z2: torch.Tensor,
    kappas: torch.Tensor,
    num_factors: int,
    factor_dim: int,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
    clip_delta: float = 1e-3,
    gamma: float = 1.0,
    use_isotropy: bool = False,
    use_frechet_mean: bool = False,
    frechet_normalize_tangents: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert z1.dim() == 2 and z2.dim() == 2
    B, D = z1.shape
    assert z2.shape == (B, D)

    F_factors = num_factors
    d = factor_dim
    assert D == F_factors * d, f"D={D} != F*d={F_factors*d}"
    assert kappas.shape == (F_factors,), f"kappas shape {kappas.shape} != ({F_factors},)"

    z1f = z1.reshape(B, F_factors, d)
    z2f = z2.reshape(B, F_factors, d)

    kappas_expanded = kappas.view(1, F_factors, 1)
    sqrt_abs_k = kappas_expanded.abs().clamp_min(eps).sqrt()
    atanh_val = torch.atanh(torch.tensor(1.0 - clip_delta, device=z1.device, dtype=z1.dtype))
    r_max = atanh_val / sqrt_abs_k.squeeze(-1)

    z1f_norms = z1f.norm(dim=-1)
    z2f_norms = z2f.norm(dim=-1)
    is_hyperbolic = (kappas < 0).view(1, F_factors)
    clip_mask_1 = (z1f_norms > r_max) & is_hyperbolic
    clip_mask_2 = (z2f_norms > r_max) & is_hyperbolic
    clip_activity = ((clip_mask_1.float().mean() + clip_mask_2.float().mean()) / 2.0)

    x1_list = []
    x2_list = []
    x1_before_list = []
    x2_before_list = []
    for f_idx in range(F_factors):
        kappa_f = kappas[f_idx]
        z1_f = z1f[:, f_idx, :]
        z2_f = z2f[:, f_idx, :]

        z1c_f = clip_tangent_norm(z1_f, kappa_f, delta=clip_delta)
        z2c_f = clip_tangent_norm(z2_f, kappa_f, delta=clip_delta)

        x1_before_f = exp_0(z1c_f, kappa_f, eps)
        x2_before_f = exp_0(z2c_f, kappa_f, eps)
        x1_f = project_to_ball(x1_before_f, kappa_f)
        x2_f = project_to_ball(x2_before_f, kappa_f)

        x1_list.append(x1_f)
        x2_list.append(x2_f)
        x1_before_list.append(x1_before_f)
        x2_before_list.append(x2_before_f)

    x1 = torch.stack(x1_list, dim=1)
    x2 = torch.stack(x2_list, dim=1)
    x1_before = torch.stack(x1_before_list, dim=1)
    x2_before = torch.stack(x2_before_list, dim=1)

    inv_loss_per_factor = []
    for f_idx in range(F_factors):
        kappa_f = kappas[f_idx]
        d2_f = dist_sq(x1[:, f_idx, :], x2[:, f_idx, :], kappa_f, eps)
        inv_loss_per_factor.append(d2_f.mean() / (4.0 * d))
    inv_loss = torch.stack(inv_loss_per_factor).mean()

    t1_list = []
    t2_list = []
    for f_idx in range(F_factors):
        kappa_f = kappas[f_idx]
        x1_f = x1[:, f_idx, :]
        x2_f = x2[:, f_idx, :]
        if use_frechet_mean:
            x_combined = torch.cat([x1_f, x2_f], dim=0)
            mu_f = frechet_mean(x_combined, kappa_f, num_iters=3, eps=eps)
            mu_f = mu_f.detach()
            mu_f_expanded = mu_f.unsqueeze(0).expand(B, -1)
            if frechet_normalize_tangents:
                y1_f = mobius_add(-mu_f_expanded, x1_f, kappa_f, eps)
                y2_f = mobius_add(-mu_f_expanded, x2_f, kappa_f, eps)
                t1_f = log_0(y1_f, kappa_f, eps)
                t2_f = log_0(y2_f, kappa_f, eps)
            else:
                t1_f = log_mu(x1_f, mu_f_expanded, kappa_f, eps)
                t2_f = log_mu(x2_f, mu_f_expanded, kappa_f, eps)
        else:
            t1_f = log_0(x1_f, kappa_f, eps)
            t2_f = log_0(x2_f, kappa_f, eps)
        t1_list.append(t1_f)
        t2_list.append(t2_f)

    t1 = torch.cat(t1_list, dim=-1)
    t2 = torch.cat(t2_list, dim=-1)

    t1_centered = t1 - t1.mean(dim=0)
    t2_centered = t2 - t2.mean(dim=0)

    std_t1 = torch.sqrt(t1_centered.var(dim=0) + eps)
    std_t2 = torch.sqrt(t2_centered.var(dim=0) + eps)
    var_loss = (torch.relu(gamma - std_t1).mean() + torch.relu(gamma - std_t2).mean()) / 2.0

    iso_loss_mean = torch.tensor(0.0, device=z1.device)
    iso_abs_loss_mean = torch.tensor(0.0, device=z1.device)
    trace_mean = torch.tensor(0.0, device=z1.device)

    if use_isotropy:
        cov_loss = isotropy_loss_per_factor(t1_list, t2_list, eps)
        for f_idx in range(F_factors):
            t1_f = t1_list[f_idx]
            t2_f = t2_list[f_idx]
            t1_c = t1_f - t1_f.mean(dim=0)
            t2_c = t2_f - t2_f.mean(dim=0)
            cov1 = (t1_c.T @ t1_c) / (B - 1)
            cov2 = (t2_c.T @ t2_c) / (B - 1)
            tr1, tr2 = cov1.trace(), cov2.trace()
            frob1, frob2 = (cov1 * cov1).sum(), (cov2 * cov2).sum()
            mean_lam1, mean_lam2 = tr1 / d, tr2 / d
            iso1 = frob1 / (d * mean_lam1.pow(2).clamp_min(eps)) - 1.0
            iso2 = frob2 / (d * mean_lam2.pow(2).clamp_min(eps)) - 1.0
            iso_abs1 = frob1 / d - mean_lam1.pow(2)
            iso_abs2 = frob2 / d - mean_lam2.pow(2)
            iso_loss_mean = iso_loss_mean + iso1 + iso2
            iso_abs_loss_mean = iso_abs_loss_mean + iso_abs1 + iso_abs2
            trace_mean = trace_mean + tr1 + tr2
        iso_loss_mean = iso_loss_mean / (2 * F_factors)
        iso_abs_loss_mean = iso_abs_loss_mean / (2 * F_factors)
        trace_mean = trace_mean / (2 * F_factors)
    else:
        cov_t1 = (t1_centered.T @ t1_centered) / (B - 1)
        cov_t2 = (t2_centered.T @ t2_centered) / (B - 1)
        cov_loss = (
            off_diagonal(cov_t1).pow(2).sum() / D
            + off_diagonal(cov_t2).pow(2).sum() / D
        )

    frechet_residual_mean = torch.tensor(0.0, device=z1.device)
    if use_frechet_mean:
        residual_sum = torch.tensor(0.0, device=z1.device)
        for f_idx in range(F_factors):
            t1_f = t1_list[f_idx]
            t2_f = t2_list[f_idx]
            combined_mean = (t1_f.mean(dim=0) + t2_f.mean(dim=0)) / 2.0
            residual_sum = residual_sum + combined_mean.norm()
        frechet_residual_mean = residual_sum / F_factors

    total = sim_coeff * inv_loss + std_coeff * var_loss + cov_coeff * cov_loss

    x1_norm = x1.norm(dim=-1)
    x2_norm = x2.norm(dim=-1)
    boundary_proximity_all = sqrt_abs_k.squeeze(-1) * torch.cat([x1_norm, x2_norm], dim=0)
    boundary_proximity_mean = boundary_proximity_all.mean()
    boundary_proximity_max = boundary_proximity_all.max()

    proj_active_1 = (x1.norm(dim=-1) < x1_before.norm(dim=-1) - 1e-8).float().mean()
    proj_active_2 = (x2.norm(dim=-1) < x2_before.norm(dim=-1) - 1e-8).float().mean()
    proj_activity = (proj_active_1 + proj_active_2) / 2.0

    tangent_norm_mean = (t1.norm(dim=-1).mean() + t2.norm(dim=-1).mean()) / 2.0

    kappa_mean = kappas.mean()
    kappa_std = kappas.std() if F_factors > 1 else torch.tensor(0.0, device=kappas.device)

    stats = {
        "inv_loss": inv_loss.detach(),
        "var_loss": var_loss.detach(),
        "cov_loss": cov_loss.detach(),
        "kappa": kappa_mean.detach(),
        "kappa_std": kappa_std.detach(),
        "kappa_min": kappas.min().detach(),
        "kappa_max": kappas.max().detach(),
        "clip_activity": clip_activity.detach(),
        "boundary_proximity_mean": boundary_proximity_mean.detach(),
        "boundary_proximity_max": boundary_proximity_max.detach(),
        "proj_activity": proj_activity.detach(),
        "tangent_norm_mean": tangent_norm_mean.detach(),
        "iso_loss_mean": iso_loss_mean.detach(),
        "iso_abs_loss_mean": iso_abs_loss_mean.detach(),
        "trace_mean": trace_mean.detach(),
        "frechet_residual_mean": frechet_residual_mean.detach(),
    }
    return total, stats


class MaskedImageModelingLoss(nn.Module):
    def __init__(self, patch_size: int = 16, num_channels: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels

    def forward(
        self,
        reconstructed_pixel_values: torch.Tensor,
        original_pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        size = image_size // self.patch_size
        bool_masked_pos_reshaped = bool_masked_pos.reshape(-1, size, size)
        mask = (
            bool_masked_pos_reshaped.repeat_interleave(self.patch_size, 1)
            .repeat_interleave(self.patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )
        reconstruction_loss = F.l1_loss(
            original_pixel_values, reconstructed_pixel_values, reduction="none"
        )
        masked_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5)
        return masked_loss / self.num_channels


class MLMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        prediction_scores: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        return self.loss_fn(
            prediction_scores.view(-1, vocab_size),
            labels.view(-1),
        )


class LANISTRLossComputer:
    def __init__(
        self,
        lambda_mlm: float = 1.0,
        lambda_mim: float = 1.0,
        lambda_mfm: float = 0.01,
        lambda_mmm: float = 1.0,
    ) -> None:
        self.lambda_mlm = lambda_mlm
        self.lambda_mim = lambda_mim
        self.lambda_mfm = lambda_mfm
        self.lambda_mmm = lambda_mmm

    def compute_total_loss(
        self,
        loss_mlm: torch.Tensor,
        loss_mim: torch.Tensor,
        loss_mfm: torch.Tensor,
        loss_mmm: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.lambda_mlm * loss_mlm
            + self.lambda_mim * loss_mim
            + self.lambda_mfm * loss_mfm
            + self.lambda_mmm * loss_mmm
        )

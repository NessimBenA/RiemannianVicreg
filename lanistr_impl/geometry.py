from typing import Union

import torch
from torch import Tensor

Scalar = Union[float, Tensor]


def _as_tensor(x: Scalar, ref: Tensor) -> Tensor:
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def tan_kappa(r: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    k = _as_tensor(kappa, r)
    abs_k = k.abs()
    use_taylor = abs_k < 1e-2
    taylor = r + (k * r.pow(3)) / 3.0
    sqrt_abs = abs_k.clamp_min(eps).sqrt()
    hyp = torch.tanh(sqrt_abs * r) / sqrt_abs
    sph_arg = (sqrt_abs * r).clamp(max=0.99 * (torch.pi / 2))
    sph = torch.tan(sph_arg) / sqrt_abs
    large = torch.where(k < 0, hyp, sph)
    return torch.where(use_taylor, taylor, large)


def arctan_kappa(r: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    k = _as_tensor(kappa, r)
    abs_k = k.abs()
    use_taylor = abs_k < 1e-2
    taylor = r - (k * r.pow(3)) / 3.0
    sqrt_abs = abs_k.clamp_min(eps).sqrt()
    hyp_arg = (sqrt_abs * r).clamp(min=-0.999999, max=0.999999)
    hyp = torch.atanh(hyp_arg) / sqrt_abs
    sph = torch.atan(sqrt_abs * r) / sqrt_abs
    large = torch.where(k < 0, hyp, sph)
    return torch.where(use_taylor, taylor, large)


def exp_0(u: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    new_norm = tan_kappa(norm_u, kappa, eps)
    return new_norm * (u / norm_u)


def log_0(z: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    norm_z = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    r = arctan_kappa(norm_z, kappa, eps)
    return r * (z / norm_z)


def clip_tangent_norm(
    u: Tensor,
    kappa: Scalar,
    delta: float = 1e-3,
    eps: float = 1e-8,
) -> Tensor:
    k = _as_tensor(kappa, u)
    if torch.all(k >= 0):
        return u
    sqrt_abs = k.abs().clamp_min(eps).sqrt()
    r_max = torch.atanh(torch.tensor(1.0 - delta, device=u.device, dtype=u.dtype)) / sqrt_abs
    r = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = (r_max / r).clamp(max=1.0)
    return u * scale


def project_to_ball(
    x: Tensor,
    kappa: Scalar,
    margin: float = 1e-5,
    eps: float = 1e-8,
) -> Tensor:
    k = _as_tensor(kappa, x)
    if torch.all(k >= 0):
        return x
    sqrt_abs = k.abs().clamp_min(eps).sqrt()
    max_norm = (1.0 - margin) / sqrt_abs
    r = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = (max_norm / r).clamp(max=1.0)
    return x * scale


def dist_sq(z1: Tensor, z2: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    k = _as_tensor(kappa, z1)
    x1_sq = (z1 * z1).sum(dim=-1, keepdim=True)
    x2_sq = (z2 * z2).sum(dim=-1, keepdim=True)
    x1x2 = (z1 * z2).sum(dim=-1, keepdim=True)
    num = -(1 + 2 * k * x1x2 - k * x2_sq) * z1 + (1 + k * x1_sq) * z2
    denom = (1 + 2 * k * x1x2 + k.pow(2) * x1_sq * x2_sq).clamp_min(eps)
    diff = num / denom
    norm_diff = diff.norm(dim=-1)
    d = 2.0 * arctan_kappa(norm_diff, kappa, eps)
    return d.pow(2)


def lambda_x(x: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    k = _as_tensor(kappa, x)
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 + k * x_sq).clamp_min(eps)


def mobius_add(x: Tensor, y: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    k = _as_tensor(kappa, x)
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 - 2.0 * k * xy - k * y_sq) * x + (1.0 + k * x_sq) * y
    denom = (1.0 - 2.0 * k * xy + k.pow(2) * x_sq * y_sq).clamp_min(eps)
    return num / denom


def exp_mu(v: Tensor, mu: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    lam_mu = lambda_x(mu, kappa, eps)
    scaled_v = (lam_mu / 2.0) * v
    exp_scaled = exp_0(scaled_v, kappa, eps)
    return mobius_add(mu, exp_scaled, kappa, eps)


def log_mu(x: Tensor, mu: Tensor, kappa: Scalar, eps: float = 1e-6) -> Tensor:
    diff = mobius_add(-mu, x, kappa, eps)
    log_diff = log_0(diff, kappa, eps)
    lam_mu = lambda_x(mu, kappa, eps)
    return (2.0 / lam_mu) * log_diff


def frechet_mean(points: Tensor, kappa: Scalar, num_iters: int = 3, eps: float = 1e-6) -> Tensor:
    B, d = points.shape
    mu = torch.zeros(d, dtype=points.dtype, device=points.device)
    for _ in range(num_iters):
        tangents = log_mu(points, mu.unsqueeze(0).expand(B, -1), kappa, eps)
        g = tangents.mean(dim=0)
        mu = exp_mu(g.unsqueeze(0), mu.unsqueeze(0), kappa, eps).squeeze(0)
    return mu

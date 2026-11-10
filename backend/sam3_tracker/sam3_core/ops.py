"""Pure PyTorch standalone helpers for SAM 3.1 (zero ComfyUI dependency)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StandardOps:
    Linear = nn.Linear
    Conv2d = nn.Conv2d
    ConvTranspose2d = nn.ConvTranspose2d
    LayerNorm = nn.LayerNorm
    GroupNorm = nn.GroupNorm
    Embedding = nn.Embedding


def cast_to_input(weight: Tensor, input_tensor: Tensor) -> Tensor:
    if weight.device != input_tensor.device or weight.dtype != input_tensor.dtype:
        return weight.to(device=input_tensor.device, dtype=input_tensor.dtype)
    return weight


def common_upscale(samples: Tensor, width: int, height: int, upscale_method: str = "bilinear", crop: str = "disabled") -> Tensor:
    mode = "bicubic" if upscale_method == "bicubic" else "bilinear"
    return F.interpolate(samples, size=(height, width), mode=mode, align_corners=False)


def optimized_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    mask: Tensor | None = None,
    skip_reshape: bool = False,
    low_precision_attention: bool = False,
) -> Tensor:
    if skip_reshape:
        # q, k, v are [B, heads, N, head_dim]
        b, _, n_q, head_dim = q.shape
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).reshape(b, n_q, heads * head_dim)
    else:
        b, n_q, c = q.shape
        head_dim = c // heads
        n_k = k.shape[1]
        q_h = q.view(b, n_q, heads, head_dim).transpose(1, 2)
        k_h = k.view(b, n_k, heads, head_dim).transpose(1, 2)
        v_h = v.view(b, n_k, heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).reshape(b, n_q, c)


def rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    assert dim % 2 == 0
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega.to(dtype=torch.float32))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    b, n, d, _ = out.shape
    out = out.view(b, n, d, 2, 2)
    return out.to(dtype=torch.float32, device=pos.device)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: float, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def apply_rope1(x: Tensor, freqs_cis: Tensor) -> Tensor:
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out = x_out + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def LayerNorm2d_op(operations):
    class LayerNorm2d(operations.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x: Tensor) -> Tensor:
            return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    return LayerNorm2d

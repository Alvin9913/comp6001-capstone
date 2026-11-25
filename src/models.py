"""Models: CM-GAN (from scratch) plus the benchmark baselines.

CM-GAN
    text tokens ----\
                     >-- bidirectional cross-attention --> gated incongruity --> MLP
    visual tokens --/

The gated incongruity unit is the novel part:

    d      = |h_t - h_v|                     element-wise semantic divergence
    g      = sigma(W_g [h_t ; h_v] + b_g)    learned per-dimension gate
    Delta  = d * g                           amplified divergence

The classifier sees [h_t ; h_v ; Delta], so it can use the disagreement signal
directly instead of having to infer it from concatenated features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention with residual + LayerNorm."""

    def __init__(self, dim: int, heads: int, dropout: float, use_cross: bool = True):
        super().__init__()
        self.use_cross = use_cross
        self.txt_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.vis_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_t = nn.LayerNorm(dim)
        self.ln_v = nn.LayerNorm(dim)
        self.ffn_t = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(dim * 2, dim))
        self.ffn_v = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(dim * 2, dim))

    def forward(self, h_t: torch.Tensor, h_v: torch.Tensor, txt_mask: torch.Tensor):
        """h_t (B, L, d), h_v (B, K, d), txt_mask (B, L) with 1 = keep."""
        if not self.use_cross:
            return self.ln_t(h_t + self.ffn_t(h_t)), self.ln_v(h_v + self.ffn_v(h_v))

        key_pad = txt_mask == 0
        t2v, _ = self.txt_attn(query=h_t, key=h_v, value=h_v)                    # text sees vision
        v2t, _ = self.vis_attn(query=h_v, key=h_t, value=h_t, key_padding_mask=key_pad)  # vision sees text
        h_t = self.ln_t(h_t + t2v)
        h_v = self.ln_v(h_v + v2t)
        h_t = h_t + self.ffn_t(h_t)
        h_v = h_v + self.ffn_v(h_v)
        return h_t, h_v


class GatedIncongruity(nn.Module):
    """Delta = |h_t - h_v| * sigmoid(W_g [h_t ; h_v] + b_g)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, h_t: torch.Tensor, h_v: torch.Tensor):
        diff = (h_t - h_v).abs()
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_v], dim=-1)))
        return diff * g, g


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class CMGAN(nn.Module):
    def __init__(
        self,
        txt_dim: int = 768,
        vis_dim: int = 2048,
        dim: int = 256,
        heads: int = 4,
        dropout: float = 0.2,
        use_gate: bool = True,
        use_cross: bool = True,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.use_cross = use_cross
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, dim), nn.LayerNorm(dim))
        self.vis_proj = nn.Sequential(nn.Linear(vis_dim, dim), nn.LayerNorm(dim))
        self.block = CrossAttentionBlock(dim, heads, dropout, use_cross=use_cross)
        self.incong = GatedIncongruity(dim)
        out_dim = dim * 3 if use_gate else dim * 2
        self.head = nn.Sequential(
            nn.Linear(out_dim, dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, txt, txt_mask, vis, return_parts: bool = False):
        h_t = self.txt_proj(txt)
        h_v = self.vis_proj(vis)
        h_t, h_v = self.block(h_t, h_v, txt_mask)
        t_pool = masked_mean(h_t, txt_mask)
        v_pool = h_v.mean(1)
        if self.use_gate:
            delta, gate = self.incong(t_pool, v_pool)
            logits = self.head(torch.cat([t_pool, v_pool, delta], dim=-1)).squeeze(-1)
        else:
            delta, gate = None, None
            logits = self.head(torch.cat([t_pool, v_pool], dim=-1)).squeeze(-1)
        if return_parts:
            return logits, {"h_t": t_pool, "h_v": v_pool, "delta": delta, "gate": gate}
        return logits


class UnimodalNet(nn.Module):
    """Shared MLP head used by the text-only / vision-only baselines."""

    def __init__(self, in_dim: int, dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x, return_parts: bool = False):
        logits = self.net(x).squeeze(-1)
        return (logits, {}) if return_parts else logits


class TextOnly(nn.Module):
    def __init__(self, txt_dim: int = 768, dim: int = 256, dropout: float = 0.2, **_):
        super().__init__()
        self.body = UnimodalNet(txt_dim, dim, dropout)

    def forward(self, txt, txt_mask, vis, return_parts: bool = False):
        return self.body(masked_mean(txt, txt_mask), return_parts=return_parts)


class VisionOnly(nn.Module):
    def __init__(self, vis_dim: int = 2048, dim: int = 256, dropout: float = 0.2, **_):
        super().__init__()
        self.body = UnimodalNet(vis_dim, dim, dropout)

    def forward(self, txt, txt_mask, vis, return_parts: bool = False):
        return self.body(vis.mean(1), return_parts=return_parts)


class EarlyFusion(nn.Module):
    """Concat pooled features, no inter-modal interaction."""

    def __init__(self, txt_dim: int = 768, vis_dim: int = 2048, dim: int = 256,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.body = UnimodalNet(txt_dim + vis_dim, dim, dropout)

    def forward(self, txt, txt_mask, vis, return_parts: bool = False):
        x = torch.cat([masked_mean(txt, txt_mask), vis.mean(1)], dim=-1)
        return self.body(x, return_parts=return_parts)


class LateFusion(nn.Module):
    """Decision-level ensemble: alpha * p_text + (1 - alpha) * p_vision.

    The mixing weight is fitted on the fold's validation part (grid search over
    alpha in [0, 1], minimising BCE), so it stays a genuine benchmark rather
    than an oracle.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, p_text: torch.Tensor, p_vision: torch.Tensor) -> torch.Tensor:
        return self.alpha * p_text + (1.0 - self.alpha) * p_vision


def build_model(name: str, dim: int = 256, heads: int = 4, dropout: float = 0.2,
                txt_dim: int = 768, vis_dim: int = 2048) -> nn.Module:
    name = name.lower()
    if name == "cmgan":
        return CMGAN(txt_dim, vis_dim, dim, heads, dropout, use_gate=True, use_cross=True)
    if name == "cmgan_nogate":
        return CMGAN(txt_dim, vis_dim, dim, heads, dropout, use_gate=False, use_cross=True)
    if name == "cmgan_nocross":
        return CMGAN(txt_dim, vis_dim, dim, heads, dropout, use_gate=True, use_cross=False)
    if name == "text_only":
        return TextOnly(txt_dim, dim, dropout)
    if name == "vision_only":
        return VisionOnly(vis_dim, dim, dropout)
    if name == "early_fusion":
        return EarlyFusion(txt_dim, vis_dim, dim, dropout)
    raise ValueError(f"unknown model: {name}")

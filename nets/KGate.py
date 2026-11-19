# -*- coding: utf-8 -*-
"""K-Gate: spatial context and pixel-wise KAN modulation."""
from __future__ import annotations
import torch
from torch import nn
from nets.kan import KANLinear


def deterministic_spatial_average(x, output_size):
    """Adaptive-average-pool equivalent with deterministic CUDA gradients.

    The five production scales (224/112/56/28/14) are integer multiples of
    the C2Seg pooling grid (14). For that common case, a reshape and mean
    performs the *same* equal-sized bin averages without invoking CUDA's
    nondeterministic adaptive_avg_pool2d_backward kernel.

    For other input sizes, follow AdaptiveAvgPool2d's floor/ceil bin boundaries
    explicitly. This slower fallback keeps non-divisible sizes functional too.
    """
    if x.ndim != 4:
        raise ValueError("Expected a [B,C,H,W] tensor")
    out_h, out_w = (int(output_size[0]), int(output_size[1]))
    b, c, h, w = x.shape
    if out_h <= 0 or out_w <= 0 or out_h > h or out_w > w:
        raise ValueError("Pooling output size must be positive and no larger than the input")
    if (out_h, out_w) == (h, w):
        return x
    if h % out_h == 0 and w % out_w == 0:
        # Blocks are nonoverlapping: backward is deterministic and needs no
        # atomic additions. The reshape is a view for contiguous tensors.
        return x.reshape(b, c, out_h, h // out_h, out_w, w // out_w).mean(dim=(3, 5))
    rows = []
    for i in range(out_h):
        y0 = i * h // out_h
        y1 = ((i + 1) * h + out_h - 1) // out_h
        cols = []
        for j in range(out_w):
            x0 = j * w // out_w
            x1 = ((j + 1) * w + out_w - 1) // out_w
            cols.append(x[..., y0:y1, x0:x1].mean(dim=(-2, -1)))
        rows.append(torch.stack(cols, dim=-1))
    return torch.stack(rows, dim=-2)


def deterministic_bilinear_resize(x, output_size):
    """Bilinear resize (align_corners=False) without CUDA's upsample backward.

    Separable linear interpolation using index_select is differentiable and
    supported in PyTorch's deterministic CUDA mode. No trainable parameters.
    This matters for C2Seg because the K-Gate module upsamples a 14x14
    context lattice to 28/56/112/224, and strict determinism is enabled.
    """
    if x.ndim != 4:
        raise ValueError("Expected a [B,C,H,W] tensor")
    dest_h, dest_w = map(int, output_size)
    h, w = x.shape[-2:]
    if dest_h <= 0 or dest_w <= 0:
        raise ValueError("Resize output dimensions must be positive")
    if (dest_h, dest_w) == (h, w):
        return x

    def coordinates(length, target):
        # The half-pixel mapping is exactly align_corners=False.
        pos = ((torch.arange(target, device=x.device, dtype=torch.float64) + 0.5)
               * (length / target) - 0.5).clamp(0, length - 1)
        idx0 = pos.floor().long()
        idx1 = (idx0 + 1).clamp(max=length - 1)
        weight = (pos - idx0).to(dtype=x.dtype)
        return idx0, idx1, weight

    h0, h1, wh = coordinates(h, dest_h)
    w0, w1, ww = coordinates(w, dest_w)
    # Height first; the intermediate still has only the low-resolution width.
    top = x.index_select(-2, h0)
    bottom = x.index_select(-2, h1)
    blended = top + (bottom - top) * wh.view(1, 1, -1, 1)
    left = blended.index_select(-1, w0)
    right = blended.index_select(-1, w1)
    return left + (right - left) * ww.view(1, 1, 1, -1)


class KGate(nn.Module):
    """Lightweight two-KAN spatial gate with an inexpensive pixel-wise path.

    Input/output are [B, H*W, C]. Expensive spline evaluation is performed
    only on an adaptively pooled <= pool_size x pool_size lattice, with a
    narrow channel bottleneck. A learned pixel-wise affine branch preserves
    local information lost during spatial pooling.
    """
    def __init__(self, in_channels, img_size, hidden_dim=8, pool_size=14, kan_args=None,
                 context_scale=0.1):
        super().__init__()
        if in_channels <= 0 or img_size <= 0 or hidden_dim <= 0 or pool_size <= 0:
            raise ValueError("KGate dimensions must be positive")
        self.in_channels = int(in_channels)
        self.img_size = int(img_size)
        self.hidden_dim = int(hidden_dim)
        self.pooled_size = min(int(img_size), int(pool_size))
        self.reduce = nn.Linear(in_channels, hidden_dim, bias=False)
        # Tiny spline workspaces do not need activation recomputation.
        args = {**dict(kan_args or {}), "activation_checkpoint": False}
        self.kan1 = KANLinear(hidden_dim, hidden_dim, **args)
        self.kan2 = KANLinear(hidden_dim, hidden_dim, **args)
        self.expand = nn.Linear(hidden_dim, in_channels, bias=False)
        self.gamma = nn.Parameter(torch.ones(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale)))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.in_channels or x.shape[1] != self.img_size ** 2:
            raise ValueError("KGate expects [B, img_size*img_size, in_channels]")
        b, pixels, c = x.shape
        spatial = x.transpose(1, 2).reshape(b, c, self.img_size, self.img_size)
        pooled = deterministic_spatial_average(
            spatial, (self.pooled_size, self.pooled_size))
        lowres = pooled.flatten(2).transpose(1, 2)
        context = self.expand(self.kan2(torch.relu(self.kan1(self.reduce(lowres)))))
        context = context.transpose(1, 2).reshape(b, c, self.pooled_size, self.pooled_size)
        if self.pooled_size != self.img_size:
            context = deterministic_bilinear_resize(
                context, (self.img_size, self.img_size))
        context = context.flatten(2).transpose(1, 2)
        # Perform local modulation without constructing a costly dense KAN basis.
        gate = torch.tanh(x * self.gamma + self.beta + self.context_scale * context)
        return x * gate

# -*- coding: utf-8 -*-
"""BCAM: bidirectional cross-modal attention."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from nets.KGate import KGate

class Sinusoidal2DPositionEmbedding(nn.Module):
    def __init__(self, num_channels, height, width):
        super().__init__()
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.register_buffer("pos_encoding", self._build_sincos_pos_encoding())

    def _build_sincos_pos_encoding(self):
        assert self.num_channels % 4 == 0, "Embedding dim must be divisible by 4 for 2D encoding"
        pe = torch.zeros(self.num_channels, self.height, self.width)
        y_pos = torch.arange(0, self.height, dtype=torch.float32).unsqueeze(1).repeat(1, self.width)
        x_pos = torch.arange(0, self.width, dtype=torch.float32).unsqueeze(0).repeat(self.height, 1)
        dim_half = self.num_channels // 2
        div_term = torch.exp(torch.arange(0, dim_half, 2).float() * -(math.log(10000.0) / dim_half))
        pe[0::4, :, :] = torch.sin(y_pos[None, :, :] * div_term[:, None, None])
        pe[1::4, :, :] = torch.cos(y_pos[None, :, :] * div_term[:, None, None])
        pe[2::4, :, :] = torch.sin(x_pos[None, :, :] * div_term[:, None, None])
        pe[3::4, :, :] = torch.cos(x_pos[None, :, :] * div_term[:, None, None])
        return pe.unsqueeze(0)

    def forward(self, x):
        # Keep the original E04 operation unchanged for baseline parity.
        return x + self.pos_encoding


def masked_instance_norm(output, valid_mask, eps=1e-5):
    """InstanceNorm1d without padding contamination (no affine or running stats).

    In particular, this also covers the low-level Conv+InstanceNorm text
    projections of the attention module; masking only Softmax would not be sufficient.
    """
    if valid_mask.ndim != 2 or valid_mask.shape != (output.shape[0], output.shape[-1]):
        raise ValueError("Invalid text token mask shape")
    valid = valid_mask[:, None, :].to(device=output.device, dtype=torch.float32)
    with torch.autocast(device_type=output.device.type, enabled=False):
        y = output.float()
        counts = valid.sum(dim=-1, keepdim=True).clamp_min(1.)
        mean = (y * valid).sum(dim=-1, keepdim=True) / counts
        var = (((y - mean) ** 2) * valid).sum(dim=-1, keepdim=True) / counts
        y = (y - mean) * torch.rsqrt(var + eps)
        y = y * valid
    return y.to(output.dtype)


class BCAModule(nn.Module):
    def __init__(self, img_size, in_channels, num_heads=8, dropout=0.0,
                 kan_args=None, attention_checkpoint=False,
                 mask_tokens=False, head_scale=False, token_mean=False,
                 kgate_hidden_dim=8, kgate_pool_size=14, kgate_context_scale=0.1):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("in_channels must be divisible by num_heads")
        if token_mean and not mask_tokens:
            raise ValueError("Token mean requires valid-token mask")
        self.img_size = int(img_size)
        self.in_channels = int(in_channels)
        self.num_heads = int(num_heads)
        self.attention_checkpoint = bool(attention_checkpoint)
        self.mask_tokens = bool(mask_tokens)
        self.head_scale = bool(head_scale)
        self.token_mean = bool(token_mean)
        self.posE = Sinusoidal2DPositionEmbedding(self.in_channels, self.img_size, self.img_size)
        self.v_key = nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels, 1), nn.InstanceNorm1d(self.in_channels))
        self.v_value = nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels, 1), nn.InstanceNorm1d(self.in_channels))
        self.l_key = nn.Sequential(nn.Conv1d(512, self.in_channels, 1), nn.InstanceNorm1d(self.in_channels))
        self.l_value = nn.Sequential(nn.Conv1d(512, self.in_channels, 1), nn.InstanceNorm1d(self.in_channels))
        a = dict(kan_args or {})
        self.gate_v = KGate(self.in_channels, self.img_size, kgate_hidden_dim,
                            kgate_pool_size, a, kgate_context_scale)
        self.gate_l = KGate(self.in_channels, self.img_size, kgate_hidden_dim,
                            kgate_pool_size, a, kgate_context_scale)
        self.project = nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels, 1),
                                     nn.InstanceNorm1d(self.in_channels),
                                     nn.GELU(), nn.Dropout(dropout))

    def forward(self, img_feat, text_feat, text_mask=None):
        """Bidirectional cross-modal attention and multiplicative residual fusion."""
        tem_v = img_feat.flatten(2)
        b, c, h, w = img_feat.shape
        hw = h * w
        n_t = text_feat.size(1)
        v = self.posE(img_feat).flatten(2)
        l = text_feat.transpose(1, 2)
        use_mask = self.mask_tokens or self.token_mean
        if use_mask:
            if text_mask is None:
                raise ValueError("Masked attention requires text_mask")
            valid = text_mask.reshape(b, n_t).to(device=text_feat.device, dtype=torch.bool)
            counts = valid.sum(dim=-1)
            if not bool(torch.all(counts > 0)):
                raise ValueError("Each sample requires a valid text token")
        else:
            valid, counts = None, None
        v_key = self.v_key(v)
        v_value = self.v_value(v)
        if use_mask:
            l_key = masked_instance_norm(self.l_key[0](l), valid, self.l_key[1].eps)
            l_value = masked_instance_norm(self.l_value[0](l), valid, self.l_value[1].eps)
        else:
            l_key = self.l_key(l)
            l_value = self.l_value(l)
        v_key = v_key.reshape(b, self.num_heads, self.in_channels // self.num_heads, hw).permute(0, 1, 3, 2)
        l_key = l_key.reshape(b, self.num_heads, self.in_channels // self.num_heads, n_t)
        v_value = v_value.reshape(b, self.num_heads, self.in_channels // self.num_heads, hw).permute(0, 1, 3, 2)
        l_value = l_value.reshape(b, self.num_heads, self.in_channels // self.num_heads, n_t).permute(0, 1, 3, 2)
        def legacy_attention(vk, vv, lk, lv):
            sim_map = torch.matmul(vk, lk)
            # The default path uses the original channel scaling and unmasked Softmax.
            scale_dim = (self.in_channels // self.num_heads) if self.head_scale else self.in_channels
            sim_map = (scale_dim ** -.5) * sim_map
            if self.mask_tokens:
                sim_map1 = F.softmax(sim_map.masked_fill(~valid[:, None, None, :], float('-inf')), dim=-1)
            else:
                sim_map1 = F.softmax(sim_map, dim=-1)
            sim_map2 = F.softmax(sim_map.transpose(-1, -2), dim=-1)
            if use_mask:
                sim_map2 = sim_map2 * valid[:, None, :, None].to(dtype=sim_map2.dtype)
            visual_out = torch.matmul(sim_map1, lv)
            language_out = torch.einsum('b h i j, b h j k -> b h j k', sim_map2, vv)
            if self.token_mean:
                language_out = language_out / counts[:, None, None, None].to(language_out.dtype)
            return visual_out, language_out

        if self.attention_checkpoint and self.training and torch.is_grad_enabled():
            v_out, l_out = checkpoint(legacy_attention, v_key, v_value, l_key, l_value,
                                      use_reentrant=False, preserve_rng_state=False)
        else:
            v_out, l_out = legacy_attention(v_key, v_value, l_key, l_value)
        v_out = v_out.permute(0, 2, 1, 3).contiguous().reshape(b, hw, self.in_channels)
        a_v = self.gate_v(v_out).permute(0, 2, 1)
        l_out = l_out.permute(0, 2, 1, 3).contiguous().reshape(b, hw, self.in_channels)
        a_l = self.gate_l(l_out).permute(0, 2, 1)
        return self.project(tem_v + a_v * a_l).view(b, c, h, w)


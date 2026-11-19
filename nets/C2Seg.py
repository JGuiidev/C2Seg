# -*- coding: utf-8 -*-
"""C2Seg: hybrid CNN+KAN encoder, BCAM, K-Gate and U-Net++ decoder."""
from __future__ import annotations
import torch
import torch.nn as nn
from modules.module_clip import load_clip_model
from modules.dilated_conv import ASPP
from nets.BCAM import BCAModule
from nets.kan import KANLinear


class DownBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class KANDownBlock(nn.Module):
    """High-level KAN stage: channel projection + token-wise spline mapping."""
    def __init__(self, in_channels, out_channels, kan_args=None, residual_scale=1.0):
        super().__init__()
        args = dict(kan_args or {})
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.kan = KANLinear(out_channels, out_channels, **args)
        self.residual_scale = float(residual_scale)

    def forward(self, x):
        x = self.proj(x)
        seq = x.flatten(2).transpose(1, 2)
        out = seq + self.residual_scale * self.kan(self.norm(seq))
        return out.transpose(1, 2).reshape_as(x)


class HybridVisualEncoder(nn.Module):
    def __init__(self, n_channels=3, channels=(32, 64, 128, 256, 512), kan_args=None,
                 kan_residual_scale=1.0):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("Hybrid visual encoder requires five scale widths")
        self.stage0 = DownBlock(n_channels, channels[0], channels[0])
        self.stage1 = DownBlock(channels[0], channels[1], channels[1])
        self.stage2 = DownBlock(channels[1], channels[2], channels[2])
        self.stage3 = KANDownBlock(channels[2], channels[3], kan_args, kan_residual_scale)
        self.stage4 = KANDownBlock(channels[3], channels[4], kan_args, kan_residual_scale)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, image):
        features = []
        x = self.stage0(image); features.append(x)
        x = self.stage1(self.pool(x)); features.append(x)
        x = self.stage2(self.pool(x)); features.append(x)
        x = self.stage3(self.pool(x)); features.append(x)
        x = self.stage4(self.pool(x)); features.append(x)
        return features


class UpBlock(nn.Module):
    def __init__(self, skip_channels, up_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_channels, up_channels, 2, 2)
        self.conv1 = nn.Conv2d(skip_channels + up_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip, x):
        up = self.up(x)
        out = torch.cat([skip, up], 1)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class C2Seg(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        channels = [32, 64, 128, 256, 512]
        img_size = [224, 112, 56, 28, 14]
        self.pool = nn.MaxPool2d(2, 2)
        self.vision_encoder_mode = "hybrid_3cnn2kan"
        kan_args = {
            "grid_size": int(config.kan_grid_size),
            "spline_order": int(config.kan_spline_order),
            "grid_range": (-float(config.kan_range), float(config.kan_range)),
            "chunk_size": int(config.kan_chunk_size),
            "activation_checkpoint": bool(config.kan_activation_checkpoint),
        }
        self.encoder = HybridVisualEncoder(n_channels, channels, kan_args, config.kan_residual_scale)

        self.bottleneck = ASPP(channels[4], channels[4], rates=(1, 2, 3), dropout=0.3)
        self.up0_1 = UpBlock(channels[0], channels[1], channels[0])
        self.up1_1 = UpBlock(channels[1], channels[2], channels[1])
        self.up2_1 = UpBlock(channels[2], channels[3], channels[2])
        self.up3_1 = UpBlock(channels[3], channels[4], channels[3])
        self.up0_2 = UpBlock(channels[0] * 2, channels[1], channels[0])
        self.up1_2 = UpBlock(channels[1] * 2, channels[2], channels[1])
        self.up2_2 = UpBlock(channels[2] * 2, channels[3], channels[2])
        self.up0_3 = UpBlock(channels[0] * 3, channels[1], channels[0])
        self.up1_3 = UpBlock(channels[1] * 3, channels[2], channels[1])
        self.up0_4 = UpBlock(channels[0] * 4, channels[1], channels[0])

        self.clip = load_clip_model(config)
        for idx, (size, width) in enumerate(zip(img_size, channels)):
            block = BCAModule(
                img_size=size, in_channels=width, num_heads=config.num_heads,
                kan_args=kan_args, attention_checkpoint=(idx in config.attention_checkpoint_scales),
                mask_tokens=config.attention_mask_tokens,
                head_scale=config.attention_head_scale,
                token_mean=config.attention_token_mean,
                kgate_hidden_dim=config.kgate_hidden_dims[idx],
                kgate_pool_size=config.kgate_pool_size,
                kgate_context_scale=config.kgate_context_scale,
            )
            setattr(self, f"bcam{idx}", block)

        self.out4 = nn.Sequential(nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1), nn.Sigmoid())
        if self.deep_supervision:
            self.out3 = nn.Sequential(nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1), nn.Sigmoid())
            self.out2 = nn.Sequential(nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1), nn.Sigmoid())
            self.out1 = nn.Sequential(nn.Conv2d(channels[0], n_classes, kernel_size=1, stride=1), nn.Sigmoid())

    def _encode_and_fuse_hybrid(self, image, text_hidden, text_mask):
        v0 = self.encoder.stage0(image)
        f0 = self.bcam0(v0, text_hidden, text_mask)
        v1 = self.encoder.stage1(self.pool(f0))
        f1 = self.bcam1(v1, text_hidden, text_mask)
        v2 = self.encoder.stage2(self.pool(f1))
        f2 = self.bcam2(v2, text_hidden, text_mask)
        v3 = self.encoder.stage3(self.pool(f2))
        f3 = self.bcam3(v3, text_hidden, text_mask)
        v4 = self.encoder.stage4(self.pool(f3))
        f4 = self.bcam4(v4, text_hidden, text_mask)
        return f0, f1, f2, f3, f4

    def forward(self, image, text_token, text_mask):
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        _, language_hidden = self.clip.encode_text(text_token, return_hidden=True, mask=text_mask)
        f0, f1, f2, f3, f4 = self._encode_and_fuse_hybrid(image, language_hidden, text_mask)

        up4 = self.bottleneck(f4)
        v0_1 = self.up0_1(f0, f1)
        v1_1 = self.up1_1(f1, f2)
        v2_1 = self.up2_1(f2, f3)
        v3_1 = self.up3_1(f3, up4)
        v0_2 = self.up0_2(torch.cat([f0, v0_1], 1), v1_1)
        v1_2 = self.up1_2(torch.cat([f1, v1_1], 1), v2_1)
        v2_2 = self.up2_2(torch.cat([f2, v2_1], 1), v3_1)
        v0_3 = self.up0_3(torch.cat([f0, v0_1, v0_2], 1), v1_2)
        v1_3 = self.up1_3(torch.cat([f1, v1_1, v1_2], 1), v2_2)
        v0_4 = self.up0_4(torch.cat([f0, v0_1, v0_2, v0_3], 1), v1_3)
        out4 = self.out4(v0_4)
        if self.deep_supervision:
            out3 = self.out3(v0_3)
            out2 = self.out2(v0_2)
            out1 = self.out1(v0_1)
            return out4, out3, out2, out1
        return out4

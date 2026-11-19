"""Atrous spatial pyramid pooling used by the C2Seg decoder."""
from __future__ import annotations
import torch
import torch.nn as nn


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = bool(act)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_c),
        )
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.act else x


class ASPPConv(nn.Module):
    def __init__(self, in_c, out_c, dilation):
        super().__init__()
        self.cbr = CBR(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return self.cbr(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = CBR(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.shape[-2:]
        # Explicit spatial mean has deterministic CUDA backward behavior.
        y = x.mean(dim=(-2, -1), keepdim=True)
        y = self.proj(y)
        return y.expand(-1, -1, *size)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 2, 3), dropout=0.1):
        super().__init__()
        branches = [CBR(in_channels, out_channels, kernel_size=1, padding=0)]
        branches.extend(ASPPConv(in_channels, out_channels, dilation=rate) for rate in rates)
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)
        projected_channels = out_channels * len(self.branches)
        self.project = nn.Sequential(
            CBR(projected_channels, out_channels, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))

"""Segmentation losses used by the default training configuration."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, mode="weighted_bce_dice", foreground_bce_weight=0.5):
        super().__init__()
        if mode != "weighted_bce_dice":
            raise ValueError(f"Unsupported segmentation loss: {mode}")
        self.foreground_bce_weight = float(foreground_bce_weight)
        if not 0 <= self.foreground_bce_weight <= 1:
            raise ValueError("foreground_bce_weight must lie in [0,1]")

    def forward(self, prob, target):
        # Prediction heads already apply sigmoid. Keep BCE arithmetic in FP32 under AMP.
        with torch.autocast(device_type=prob.device.type, enabled=False):
            p = prob.float().clamp(1e-7, 1 - 1e-7)
            t = target.float()
            if p.shape != t.shape:
                raise ValueError(f"Prediction {tuple(p.shape)} != mask {tuple(t.shape)}")
            dims = tuple(range(1, p.ndim))
            loss_map = F.binary_cross_entropy(p, t, reduction="none")
            pos = (t > 0.5).float()
            neg = 1 - pos
            alpha = self.foreground_bce_weight
            bce = (
                alpha * (loss_map * pos).sum() / pos.sum().clamp_min(1)
                + (1 - alpha) * (loss_map * neg).sum() / neg.sum().clamp_min(1)
            )
            weight = 0.5 + 0 * t
            pp, tt = weight * p, weight * t
            intersection = (pp * tt).sum(dim=dims)
            denominator = (pp.square() + tt.square()).sum(dim=dims)
            dice = (1 - (2 * intersection + 1e-5) / (denominator + 1e-5)).mean()
            return 0.5 * bce + 0.5 * dice


def multi_output_loss(outputs, target, criterion, weights):
    predictions = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    if len(predictions) == 1:
        return criterion(predictions[0], target)
    if len(predictions) != len(weights):
        raise ValueError("Deep supervision heads and weights mismatch")
    total_weight = sum(float(weight) for weight in weights)
    if total_weight <= 0:
        raise ValueError("At least one deep-supervision weight must be positive")
    return sum(float(weight) * criterion(output, target)
               for weight, output in zip(weights, predictions)) / total_weight

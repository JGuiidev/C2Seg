"""Per-image segmentation metrics, aggregated independent of DataLoader batch size."""
from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

NAMES = ("Dice", "mIoU", "Accuracy", "Sensitivity", "Specificity")


def batch_metrics(prob, mask, threshold=0.5):
    if prob.shape != mask.shape:
        raise ValueError(f"Incompatible metrics tensors: {prob.shape} vs {mask.shape}")
    p = prob.detach().ge(float(threshold)).flatten(1)
    t = mask.detach().gt(0.5).flatten(1)
    tp = (p & t).sum(dim=1).float()
    tn = (~p & ~t).sum(dim=1).float()
    fp = (p & ~t).sum(dim=1).float()
    fn = (~p & t).sum(dim=1).float()
    den_dice = 2*tp + fp + fn
    den_iou = tp + fp + fn
    sens_den = tp + fn
    spec_den = tn + fp
    return {
        "Dice": torch.where(den_dice > 0, 2*tp / den_dice.clamp_min(1), torch.ones_like(tp)),
        "mIoU": torch.where(den_iou > 0, tp / den_iou.clamp_min(1), torch.ones_like(tp)),
        "Accuracy": (tp+tn) / (tp+tn+fp+fn).clamp_min(1),
        "Sensitivity": torch.where(sens_den > 0, tp / sens_den.clamp_min(1), torch.ones_like(tp)),
        "Specificity": torch.where(spec_den > 0, tn / spec_den.clamp_min(1), torch.ones_like(tp)),
    }


def surface_metrics(pred, target):
    """Symmetric HD95/ASSD in resized-image pixels; one-empty pairs are invalid."""
    p = np.asarray(pred, dtype=bool).squeeze()
    t = np.asarray(target, dtype=bool).squeeze()
    if not p.any() and not t.any():
        return 0.0, 0.0
    if not p.any() or not t.any():
        return float("nan"), float("nan")
    st = generate_binary_structure(2, 1)
    ps = p ^ binary_erosion(p, structure=st, border_value=0)
    ts = t ^ binary_erosion(t, structure=st, border_value=0)
    if not ps.any() or not ts.any():
        return float("nan"), float("nan")
    dt_p = distance_transform_edt(~ps)
    dt_t = distance_transform_edt(~ts)
    pt = dt_t[ps]
    tp = dt_p[ts]
    hd95 = float(np.percentile(np.concatenate([pt, tp]), 95))
    assd = float((pt.mean() + tp.mean()) * 0.5)
    return hd95, assd


class MetricAccumulator:
    """Per-sample averages, and optionally compact per-image diagnostics."""
    def __init__(self, *, surface=False, collect_samples=False):
        self.sums = {key: 0.0 for key in NAMES}
        self.count = 0
        self.surface = bool(surface)
        self.collect_samples = bool(collect_samples)
        self.surface_sums = {"HD95": 0.0, "ASSD": 0.0}
        self.surface_valid = {"HD95": 0, "ASSD": 0}
        self.invalid_surface = 0
        self.sample_records = []

    def update(self, prob, target, threshold=0.5, filenames=None, token_lengths=None):
        values = batch_metrics(prob, target, threshold)
        count = int(prob.shape[0])
        for key, value in values.items():
            self.sums[key] += float(value.sum().item())
        if self.surface or self.collect_samples:
            p = prob.detach().ge(threshold).to("cpu").numpy()
            t = target.detach().gt(0.5).to("cpu").numpy()
            for i, (a, b) in enumerate(zip(p, t)):
                hd, assd = (surface_metrics(a, b) if self.surface else (float("nan"), float("nan")))
                if np.isfinite(hd):
                    self.surface_sums["HD95"] += hd
                    self.surface_valid["HD95"] += 1
                if np.isfinite(assd):
                    self.surface_sums["ASSD"] += assd
                    self.surface_valid["ASSD"] += 1
                if self.surface and (not np.isfinite(hd) or not np.isfinite(assd)):
                    self.invalid_surface += 1
                if self.collect_samples:
                    pred, truth = a.astype(bool), b.astype(bool)
                    pixels = pred.size
                    tp = int((pred & truth).sum())
                    fp = int((pred & ~truth).sum())
                    fn = int((~pred & truth).sum())
                    idx = self.count + i
                    self.sample_records.append({
                        "Filename": str(filenames[i]) if filenames is not None else f"sample_{idx:06d}",
                        **{key: float(values[key][i].item()) for key in NAMES},
                        "HD95": float(hd) if np.isfinite(hd) else None,
                        "ASSD": float(assd) if np.isfinite(assd) else None,
                        "GroundTruthFraction": float(truth.mean()),
                        "PredictionFraction": float(pred.mean()),
                        "FPFraction": fp / pixels,
                        "FNFraction": fn / pixels,
                        "TruePositivePixels": tp,
                        "EmptyPrediction": int(not pred.any()),
                        "EmptyGroundTruth": int(not truth.any()),
                        "TokenLength": int(token_lengths[i]) if token_lengths is not None else None,
                    })
        self.count += count

    def result(self):
        if self.count == 0:
            raise RuntimeError("No samples evaluated")
        result = {k: self.sums[k]/self.count for k in NAMES}
        if self.surface:
            for name in ("HD95", "ASSD"):
                n = self.surface_valid[name]
                result[name] = self.surface_sums[name] / n if n else None
                result[name + "_valid"] = n
            result["Surface_invalid"] = self.invalid_surface
        if self.collect_samples:
            scores = np.asarray([r["Dice"] for r in self.sample_records], dtype=float)
            for name, percentile in (("DiceP10", 10), ("DiceP50", 50), ("DiceP90", 90)):
                result[name] = float(np.percentile(scores, percentile))
            def group(frac):
                if frac == 0:
                    return "empty"
                if frac < 0.01:
                    return "small_lt1pct"
                if frac < 0.05:
                    return "medium_1to5pct"
                return "large_ge5pct"
            groups = {}
            for rec in self.sample_records:
                name = group(rec["GroundTruthFraction"])
                group_stats = groups.setdefault(name, {"Count": 0, "DiceSum": 0., "mIoUSum": 0.})
                group_stats["Count"] += 1
                group_stats["DiceSum"] += rec["Dice"]
                group_stats["mIoUSum"] += rec["mIoU"]
            result["ForegroundGroups"] = {
                k: {"Count": v["Count"], "Dice": v["DiceSum"] / v["Count"],
                    "mIoU": v["mIoUSum"] / v["Count"]}
                for k, v in groups.items()
            }
        result["NumSamples"] = self.count
        return result

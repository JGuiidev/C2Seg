"""Training, validation, evaluation, and reproducibility utilities."""
from __future__ import annotations
import os
import random
import time
import numpy as np
import torch
from .losses import multi_output_loss
from .metrics import MetricAccumulator


def seed_everything(seed, deterministic=True):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def device_for(config):
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is None or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but that CUDA device is unavailable")
    return device


def amp_mode(config, device):
    if device.type != "cuda" or config.amp == "off":
        return False, torch.float32
    if config.amp == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[config.amp]
    return True, dtype


def run_epoch(model, loader, criterion, config, device, optimizer=None, scaler=None, freeze_encoder=False):
    training = optimizer is not None
    model.train(training)
    model.clip.eval()  # frozen CLIP never changes behavior in training mode
    if training and freeze_encoder:
        model.encoder.eval()  # also freeze BN running statistics during warm-start
    stats = MetricAccumulator(surface=False)
    samples = 0
    loss_sum = 0.0
    tic = time.perf_counter()
    if training and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    enabled, dtype = amp_mode(config, device)
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                # Release previous-step gradients BEFORE allocating activations.
                optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            tokens = batch["text_token"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype):
                outputs = model(image, tokens, text_mask)
                loss = multi_output_loss(outputs, target, criterion, config.loss_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite segmentation loss")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            main = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            with torch.no_grad():
                stats.update(main.detach(), target.detach(), config.threshold)
            n = int(image.size(0))
            loss_sum += float(loss.detach().item()) * n
            samples += n
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = stats.result()
    result["Loss"] = loss_sum/max(1, samples)
    result["EpochTimeSec"] = time.perf_counter()-tic
    result["NumBatches"] = len(loader)
    if training and device.type == "cuda":
        result["PeakAllocatedBytes"] = int(torch.cuda.max_memory_allocated(device))
        result["PeakReservedBytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


@torch.inference_mode()
def evaluate(model, loader, config, device, *, return_samples=False):
    model.eval()
    acc = MetricAccumulator(surface=True, collect_samples=return_samples)
    tic = time.perf_counter()
    for batch in loader:
        image = batch["image"].to(device)
        token = batch["text_token"].to(device)
        mask = batch["text_mask"].to(device)
        target = batch["mask"].to(device)
        out = model(image, token, mask)
        main = out[0] if isinstance(out, (tuple, list)) else out
        acc.update(main, target, config.threshold,
                   filenames=batch.get("filename"),
                   token_lengths=(batch["text_mask"].sum(1).tolist() if return_samples else None))
    output = acc.result()
    output["TestTimeSec"] = time.perf_counter()-tic
    return (output, acc.sample_records) if return_samples else output


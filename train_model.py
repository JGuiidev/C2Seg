"""Run CaCL pretraining, segmentation training, and independent test evaluation."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime
import random
import numpy as np
from pathlib import Path
import torch
from args import get_config, parse_args, model_config, run_paths
from datasets.dataset import make_loader
from nets.C2Seg import C2Seg
from tools.log_utils import create_logger, format_epoch_lines
from tools.checkpoint import save_best
from tools.losses import SegmentationLoss
from tools.train_utils import seed_everything, device_for, amp_mode, run_epoch


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def make_model(config):
    model = C2Seg(model_config(config), n_channels=3, n_classes=1,
                   deep_supervision=bool(config.deep_supervision))
    model.clip.requires_grad_(False)
    return model


def make_optimizer(model, config):
    klass = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    if float(config.stage2_encoder_lr_scale) == 1.0:
        # Keep a single optimizer group and the default parameter ordering.
        trainable = (p for p in model.parameters() if p.requires_grad)
        optimizer = klass(trainable, lr=config.lr, weight_decay=config.weight_decay)
    else:
        encoder_ids={id(p) for p in model.encoder.parameters()}
        encoder_group=[p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids]
        other_group=[p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
        optimizer=klass([{'params':encoder_group,'lr':config.lr*config.stage2_encoder_lr_scale},
                         {'params':other_group,'lr':config.lr}],weight_decay=config.weight_decay)
    if config.scheduler == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.t0, T_mult=1, eta_min=config.eta_min)
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cosine_t_max, eta_min=config.eta_min)
    else:
        scheduler = None
    return optimizer, scheduler


def step_scheduler(scheduler, config, epoch):
    """Hold eta_min after the chosen cosine horizon; do not restart implicitly."""
    if scheduler is not None and (config.scheduler != "cosine" or epoch <= config.cosine_t_max):
        scheduler.step()


def run_training(config):
    results_dir, weights_dir = run_paths(config)
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / "best.pth"
    status_file = results_dir.parent / "status.json"
    logger = create_logger(results_dir / "train.log", logger_name=f"C2Seg.{config.dataset}.{config.run_name}", append=False)
    write_json(results_dir / "config.json", vars(config))
    start = time.perf_counter()
    try:
        seed_everything(config.seed, config.deterministic)
        device = device_for(config)
        logger.info("C2Seg | %s | %s | device=%s | seed=%d", config.dataset, config.run_name, device, config.seed)
        train_loader = make_loader(config, "Train")
        val_loader = make_loader(config, "Val")
        logger.info("Data | train=%d | val=%d | batch=%d", len(train_loader.dataset), len(val_loader.dataset), config.batch_size)
        model = make_model(config)
        stage1_info = None
        if config.use_stage1_pretrain:
            # Isolate Stage II stochastic training from Stage I epoch count and
            # whether the pretrained encoder came from a cache or a fresh run.
            stage2_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                          torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            from pretrain_stage1 import run_stage1
            try:
                stage1_info = run_stage1(config, model.encoder, model.clip, logger)
            finally:
                random.setstate(stage2_rng[0])
                np.random.set_state(stage2_rng[1])
                torch.set_rng_state(stage2_rng[2])
                if stage2_rng[3] is not None:
                    torch.cuda.set_rng_state_all(stage2_rng[3])
            write_json(status_file, {
                "status": "running", "stage1": "completed",
                "stage2": "running", "test": "pending",
                "stage1_checkpoint": stage1_info["checkpoint"],
            })
            # The entire Stage II RNG state now matches the pre-Stage-I boundary.
        stage2_start = time.perf_counter()
        model = model.to(device)
        criterion = SegmentationLoss(config.loss, config.foreground_bce_weight)
        optimizer, scheduler = make_optimizer(model, config)
        enabled, dtype = amp_mode(config, device)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler(enabled=enabled and dtype == torch.float16)
        logger.info("Model | vision=3CNN+2KAN | BCAM=Conv K/V | K-Gate=KAN | "
                    "StageI=%s | AMP=%s",
                    bool(config.use_stage1_pretrain),
                    str(dtype) if enabled else "off")
        logger.info("K-Gate | dims=%s | pool=%d | attention checkpoint scales=%s | encoder KAN checkpoint=%s",
                    list(config.kgate_hidden_dims), int(config.kgate_pool_size),
                    list(config.attention_checkpoint_scales), bool(config.kan_activation_checkpoint))
        best = float("-inf")
        best_epoch = 0
        peak_allocated = 0
        peak_reserved = 0
        for epoch in range(1, config.epochs+1):
            lr = optimizer.param_groups[0]["lr"]
            frozen = bool(config.use_stage1_pretrain and epoch <= config.stage2_freeze_epochs)
            for param in model.encoder.parameters():
                param.requires_grad_(not frozen)
            train = run_epoch(model, train_loader, criterion, config, device,
                              optimizer=optimizer, scaler=scaler, freeze_encoder=frozen)
            # Restore all trainable params before saving a compact checkpoint.
            for param in model.encoder.parameters():
                param.requires_grad_(True)
            peak_allocated = max(peak_allocated, int(train.get("PeakAllocatedBytes", 0)))
            peak_reserved = max(peak_reserved, int(train.get("PeakReservedBytes", 0)))
            val = run_epoch(model, val_loader, criterion, config, device)
            train["LR"] = val["LR"] = lr
            improved = val["Dice"] > best or best_epoch == 0
            if improved:
                best = val["Dice"]
                best_epoch = epoch
                save_best(model, checkpoint_path, epoch=epoch, val_dice=best, config=config)
            patience_used = epoch - best_epoch
            for line in format_epoch_lines(
                epoch=epoch, total_epochs=config.epochs, train_metrics=train, valid_metrics=val,
                best_dice=best, best_epoch=best_epoch, improved=improved,
                patience=patience_used, patience_limit=config.patience,
            ):
                logger.info("%s", line)
            step_scheduler(scheduler, config, epoch)
            if patience_used >= config.patience:
                break
        if best_epoch == 0:
            raise RuntimeError("Training did not produce a valid checkpoint")
        stage2_seconds = time.perf_counter()-stage2_start
        summary = {
            "BestEpoch": best_epoch, "BestValidationDice": float(best),
            "Stage2TimeSec": stage2_seconds,
            "TrainingTimeSec": time.perf_counter() - start,
            "Checkpoint": str(checkpoint_path.resolve()),
        }
        if device.type == "cuda":
            summary["TrainPeakAllocatedBytes"] = peak_allocated
            summary["TrainPeakReservedBytes"] = peak_reserved
        if stage1_info is not None:
            summary.update({
                "Stage1Checkpoint": stage1_info["checkpoint"],
                "Stage1Reused": bool(stage1_info["reused"]),
                "Stage1SelectedEpoch": stage1_info["selected_epoch"],
                "Stage1TimeSec": stage1_info["training_seconds"],
            })
        write_json(results_dir / "metrics.json", summary)
        write_json(status_file, {
            "status": "running", "stage1": "completed",
            "stage2": "completed", "test": "pending",
            "best_epoch": best_epoch,
            "stage1_checkpoint": stage1_info["checkpoint"] if stage1_info else None,
            "checkpoint": str(checkpoint_path.resolve()),
        })
        logger.info("Stage II finished | best_epoch=%d | validation Dice=%.5f", best_epoch, best)
        return summary
    except Exception as exc:
        logger.exception("Training failed")
        write_json(status_file, {"status": "failed", "stage": "training", "error": str(exc)})
        raise
    finally:
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def launch_pipeline(config):
    """Start a new run and isolate training from test evaluation."""
    if config.run_name == "auto":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        config.run_name = f"{run_id}/stage2"
    if not config.run_name.endswith("/stage2"):
        raise ValueError("run_name must end with /stage2")
    results_dir, weights_dir = run_paths(config)
    if results_dir.exists() or weights_dir.exists():
        raise FileExistsError(f"Run already exists: {results_dir}")
    results_dir.mkdir(parents=True)
    weights_dir.mkdir(parents=True)
    config_file = results_dir / "config.json"
    status_file = results_dir.parent / "status.json"
    write_json(config_file, vars(config))
    write_json(status_file, {
        "status": "running", "dataset": config.dataset,
        "stage1": "pending", "stage2": "pending", "test": "pending",
    })

    project = Path(__file__).resolve().parent
    commands = (
        [sys.executable, str(project / "train_model.py"),
         "--_train-stage2", "--config", str(config_file)],
        [sys.executable, str(project / "test_model.py"), "--config", str(config_file),
         "--checkpoint", str(weights_dir / "best.pth")],
    )
    start = time.perf_counter()
    print(f"C2Seg | dataset={config.dataset} | device={config.device}", flush=True)
    print(f"Results: {results_dir}", flush=True)
    try:
        for label, command in zip(("Training", "Test evaluation"), commands):
            print(f"\n{label} started", flush=True)
            subprocess.run(command, cwd=project, check=True)
        metrics_file = results_dir / "metrics.json"
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        metrics["TotalPipelineTimeSec"] = time.perf_counter() - start
        write_json(metrics_file, metrics)
        write_json(status_file, {
            "status": "completed", "dataset": config.dataset,
            "stage1": "completed", "stage2": "completed", "test": "completed",
            "best_epoch": metrics["BestEpoch"], "test_dice": metrics["Dice"],
            "checkpoint": metrics["Checkpoint"],
        })
        print(f"\nCompleted | Test Dice={metrics['Dice']:.5f} | mIoU={metrics['mIoU']:.5f}", flush=True)
        print(f"Checkpoint: {metrics['Checkpoint']}", flush=True)
        return metrics
    except Exception as exc:
        old_status = {}
        if status_file.is_file():
            old_status = json.loads(status_file.read_text(encoding="utf-8"))
        write_json(status_file, {**old_status, "status": "failed", "error": str(exc)})
        raise


def main():
    internal_flag = "--_train-stage2"
    if internal_flag in sys.argv:
        sys.argv.remove(internal_flag)
        config, _ = parse_args()
        if config.run_name == "auto":
            raise ValueError("Internal training requires an explicit run_name")
        run_training(config)
    else:
        config, _ = parse_args()
        launch_pipeline(config)


if __name__ == "__main__":
    main()

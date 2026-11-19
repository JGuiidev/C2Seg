"""Evaluate a saved C2Seg checkpoint on the held-out test split."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from args import get_config, parse_args, run_paths
from datasets.dataset import make_loader
from tools.checkpoint import load_best
from tools.log_utils import build_ascii_table, format_seconds
from tools.train_utils import device_for, evaluate, seed_everything


def result_table(config, metrics):
    def value(name, decimals=4):
        item = metrics.get(name)
        return "N/A" if item is None else f"{item:.{decimals}f}"

    rows = [
        ["Dice", value("Dice")],
        ["mIoU", value("mIoU")],
        ["Accuracy", value("Accuracy")],
        ["Sensitivity", value("Sensitivity")],
        ["Specificity", value("Specificity")],
        ["HD95 (pixels)", value("HD95", 3)],
        ["ASSD (pixels)", value("ASSD", 3)],
        ["Training Time", format_seconds(metrics["TrainingTimeSec"]) if metrics.get("TrainingTimeSec") is not None else "N/A"],
        ["Best Epoch", str(metrics["BestEpoch"])],
    ]
    return f"{config.run_name} | Test Results\n" + build_ascii_table(["Metric", "Value"], rows)


def save_samples(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def evaluate_checkpoint(config, checkpoint=None, logger=None, training_time=None, model=None):
    from train_model import make_model, write_json

    results_dir, weights_dir = run_paths(config)
    checkpoint_path = Path(checkpoint) if checkpoint else weights_dir / "best.pth"
    seed_everything(config.seed, config.deterministic)
    device = device_for(config)
    if model is None:
        model = make_model(config).to(device)
    payload = load_best(model, checkpoint_path, config)
    model.to(device).eval()
    test_loader = make_loader(config, "Test")

    if config.save_sample_metrics:
        metrics, sample_records = evaluate(model, test_loader, config, device, return_samples=True)
    else:
        metrics = evaluate(model, test_loader, config, device)
        sample_records = []

    metrics["BestEpoch"] = int(payload["epoch"])
    metrics["BestValidationDice"] = float(payload["val_dice"]) if payload.get("val_dice") is not None else None
    previous_metrics_file = results_dir / "metrics.json"
    previous = {}
    if previous_metrics_file.is_file():
        try:
            previous = json.loads(previous_metrics_file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            previous = {}
    if training_time is None:
        training_time = previous.get("TrainingTimeSec")
    metrics["TrainingTimeSec"] = training_time
    metrics["Checkpoint"] = str(checkpoint_path.resolve())
    metrics["EvaluationProtocol"] = config.evaluation_protocol
    metrics["VisionEncoderMode"] = config.vision_encoder_mode
    metrics["BCAMProjection"] = "conv"
    metrics["K-Gate"] = "KAN"
    metrics["K-GateHiddenDims"] = list(config.kgate_hidden_dims)
    metrics["K-GatePoolSize"] = int(config.kgate_pool_size)
    metrics["Stage1Pretraining"] = bool(config.use_stage1_pretrain)

    # Preserve Stage I/II training metadata in the final evaluation report.
    metrics = {**previous, **metrics}
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "metrics.json", metrics)
    if config.save_sample_metrics:
        save_samples(results_dir / "sample_metrics.csv", sample_records)

    table = result_table(config, metrics)
    if logger:
        logger.info("\n%s", table)
    else:
        print(table)
    return metrics


def standalone():
    config, checkpoint = parse_args(test=True)
    if config.run_name == "auto":
        root = Path(config.results_root).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parent / root
        candidates = sorted((root / config.dataset).glob("*/stage2/config.json"))
        if not candidates:
            raise FileNotFoundError("No saved run was found; supply --config for a specific run")
        config = get_config(str(candidates[-1]), {"device": config.device})
    result_path, _ = run_paths(config)
    config_file = result_path / "config.json"
    if config_file.is_file():
        config = get_config(str(config_file), {"device": config.device})
    return evaluate_checkpoint(config, checkpoint=checkpoint)


if __name__ == "__main__":
    standalone()

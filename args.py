"""Configuration and dataset presets for C2Seg."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

# Select one of the four supported datasets here.
DATASET = "QaTa-COV19"
DATA_ROOT = "/datasets"
CLIP_WEIGHTS_PATH = "/CLIP/ViT-B-32.pt"
DEVICE = "cuda:0"
SEED = 42

DATASET_AUGMENTATIONS = {
    "QaTa-COV19": {"geometry_aug": "none", "use_photometric_aug": False},
    "MosMedData+": {"geometry_aug": "original", "use_photometric_aug": False},
    "CVC-ClinicDB": {"geometry_aug": "none", "use_photometric_aug": False},
    "Kvasir-SEG": {"geometry_aug": "original", "use_photometric_aug": True},
}

# Shared training settings
DEFAULTS = {'dataset': 'MosMedData+',
 'data_root': '/datasets',
 'clip_weights_path': '/CLIP/ViT-B-32.pt',
 'clip_backbone': 'ViT-B/32',
 'frozen_clip': True,
 'results_root': './results',
 'weights_root': './weights',
 'device': 'cuda:0',
 'run_name': 'auto',
 'seed': 42,
 'image_size': 224,
 'text_len': 18,
 'pair_rule': 'auto',
 'preprocess': 'legacy',
 'image_channel_order': 'bgr',
 'image_normalization': 'none',
 'train_mask_interpolation': 'bilinear',
 'evaluation_protocol': 'legacy_mask_bilinear_v1',
 'geometry_aug': 'original',
 'use_geometric_aug': True,
 'use_photometric_aug': False,
 'use_blur_aug': False,
 'batch_size': 32,
 'val_batch_size': 32,
 'test_batch_size': 1,
 'num_workers': 4,
 'test_num_workers': 0,
 'epochs': 500,
 'patience': 50,
 'lr': 0.001,
 'optimizer': 'adam',
 'weight_decay': 0.0,
 'scheduler': 'cosine',
 't0': 10,
 'eta_min': 0.0001,
 'cosine_t_max': 120,
 'amp': 'auto',
 'threshold': 0.5,
 'loss': 'weighted_bce_dice',
 'foreground_bce_weight': 0.5,
 'deep_supervision': True,
 'loss_weights': [0.25, 0.25, 0.25, 0.25],
 'save_sample_metrics': True,
 'deterministic': True,
 'vision_encoder_mode': 'hybrid_3cnn2kan',
 'num_heads': 8,
 'kan_grid_size': 5,
 'kan_spline_order': 3,
 'kan_range': 1.0,
 'kan_chunk_size': 4096,
 'kan_activation_checkpoint': True,
 'kan_residual_scale': 1.0,
 'use_stage1_pretrain': True,
 'stage1_batch_size': 256,
 'stage1_epochs': 120,
 'stage1_train_epochs': 30,
 'stage1_checkpoint_epochs': [20],
 'stage1_checkpoint_epoch': 20,
 'stage1_lr_cnn': 3e-5,
 'stage1_lr_kan': 1e-4,
 'stage1_lr_head': 3e-4,
 'stage1_weight_decay': 1e-4,
 'stage1_warmup_epochs': 5,
 'stage1_eta_min': 1e-6,
 'stage1_tau': 0.07,
 'stage1_alpha': 0.7,
 'stage1_top_k': 4,
 'stage1_rho_max': 0.3,
 'stage1_neighbor_margin': 0.5,
 'stage1_multiscale_global': True,
 'stage1_use_neighbors': True,
 'stage1_use_region': False,
 'stage1_use_separation': True,
 'stage1_region_scales': '23',
 'stage1_global_weight': 0.5,
 'stage1_region_weight': 0.35,
 'stage1_separation_weight': 0.15,
 'stage1_separation_margin': 0.1,
 'stage1_region_tau': 0.1,
 'stage1_photometric_aug': True,
 'stage2_freeze_epochs': 0,
 'stage2_encoder_lr_scale': 1.0,
 'attention_mask_tokens': False,
 'attention_head_scale': False,
 'attention_token_mean': False,
 'attention_checkpoint_scales': [0, 1, 2],
 'kgate_hidden_dims': [8, 8, 8, 16, 32],
 'kgate_pool_size': 14,
 'kgate_context_scale': 0.1}



DEFAULTS.update(
    dataset=DATASET, data_root=DATA_ROOT, clip_weights_path=CLIP_WEIGHTS_PATH,
    device=DEVICE, seed=SEED,
)

# Dataset-specific training presets (validated by the previous experiments).
# Settings not listed here are shared through DEFAULTS.
DATASET_TRAINING_PRESETS = {
    "QaTa-COV19": {
        "stage1_separation_weight": 0.20,
    },
    "MosMedData+": {},
    "CVC-ClinicDB": {},
    "Kvasir-SEG": {
        "stage1_lr_cnn": 1.5e-5,
        "stage1_lr_kan": 5e-5,
        "stage1_lr_head": 1.5e-4,
        # "batch_size": 16,
        "lr": 5e-4,
        "cosine_t_max": 180,
        "eta_min": 5e-5,
        "stage2_encoder_lr_scale": 0.75,
        "stage2_freeze_epochs": 10,
        "text_len": 32,
        "attention_mask_tokens": True,
    },
}

# Common settings that remain fixed for every dataset.
FIXED_TRAINING_SETTINGS = {
    "image_size": 224,
    "attention_token_mean": False,
    "stage1_batch_size": 256,
    "stage1_train_epochs": 30,
    "stage1_checkpoint_epochs": [20],
    "stage1_checkpoint_epoch": 20,
    "threshold": 0.5,
    "loss": "weighted_bce_dice",
    "use_stage1_pretrain": True,
    "stage1_epochs": 120,
    "epochs": 500,
    "patience": 50,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "scheduler": "cosine",
}

def get_config(config_path=None, overrides=None):
    supplied = {}
    if config_path:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config must be a JSON object")
        supplied.update(data)
    if overrides:
        supplied.update(overrides)
    dataset = supplied.get("dataset", DATASET)
    if dataset not in DATASET_AUGMENTATIONS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {tuple(DATASET_AUGMENTATIONS)}")
    config = dict(DEFAULTS)
    config.update(DATASET_AUGMENTATIONS[dataset])
    config.update(DATASET_TRAINING_PRESETS[dataset])
    unknown = set(supplied) - set(config)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config.update(supplied)
    for name, value in FIXED_TRAINING_SETTINGS.items():
        if config[name] != value:
            raise ValueError(f"{name} is fixed to {value!r} in the C2Seg training recipe")
    for name, value in {**DATASET_AUGMENTATIONS[dataset], **DATASET_TRAINING_PRESETS[dataset]}.items():
        if config[name] != value:
            raise ValueError(f"{name} must be {value!r} for {dataset}")
    data_root = Path(config["data_root"]).expanduser()
    if not data_root.is_absolute():
        data_root = (Path(__file__).resolve().parent / data_root).resolve()
    config["data_root"] = str(data_root)
    enums = {
        "image_channel_order": {"rgb", "bgr"},
        "image_normalization": {"clip", "none"},
        "train_mask_interpolation": {"nearest", "bilinear"},
        "evaluation_protocol": {"nearest_v2", "legacy_mask_bilinear_v1"},
        "geometry_aug": {"original", "none", "rotate90", "small_rotate", "flip"},
        "pair_rule": {"auto", "mask_prefix", "same_name"},
        "optimizer": {"adam", "adamw"},
        "scheduler": {"cosine_warm_restarts", "cosine", "none"},
        "loss": {"weighted_bce_dice"},
        "amp": {"auto", "off", "bf16", "fp16"},
        "vision_encoder_mode": {"hybrid_3cnn2kan"},
        "stage1_region_scales": {"23", "3"},
    }
    for key, choices in enums.items():
        if config[key] not in choices:
            raise ValueError(f"Invalid {key}={config[key]!r}; expected {sorted(choices)}")
    dims = config["kgate_hidden_dims"]
    if not isinstance(dims, (tuple,list)) or len(dims) != 5 or any(type(d) is not int or d <= 0 for d in dims):
        raise ValueError("kgate_hidden_dims must have five positive integers")
    if not 1 <= int(config["kgate_pool_size"]) <= config["image_size"]:
        raise ValueError("kgate_pool_size outside valid range")
    if float(config["kgate_context_scale"]) < 0:
        raise ValueError("kgate_context_scale must be nonnegative")
    if config["attention_token_mean"] and not config["attention_mask_tokens"]:
        raise ValueError("Token mean requires attention_mask_tokens=True")
    scales = config["attention_checkpoint_scales"]
    if not isinstance(scales, (tuple,list)) or any(type(v) is not int or v<0 or v>4 for v in scales) or len(set(scales)) != len(scales):
        raise ValueError("attention_checkpoint_scales requires unique indices 0..4")
    if config["evaluation_protocol"] == "legacy_mask_bilinear_v1" and config["preprocess"] != "legacy":
        raise ValueError("Legacy evaluation requires legacy preprocessing")
    if int(config["image_size"]) != 224 or not 2 <= int(config["text_len"]) <= 77:
        raise ValueError("C2Seg expects image_size=224 and text_len in [2,77]")
    if int(config["num_heads"]) < 1 or any(c % int(config["num_heads"]) for c in (32,64,128,256,512)):
        raise ValueError("num_heads must divide all BCAM widths")
    if not isinstance(config["loss_weights"], (tuple,list)) or len(config["loss_weights"])!=4 or sum(config["loss_weights"])<=0 or any(w<0 for w in config["loss_weights"]):
        raise ValueError("loss_weights require four nonnegative positive-sum values")
    positive = ("epochs", "patience", "batch_size", "val_batch_size", "test_batch_size", "t0", "cosine_t_max", "kan_grid_size", "kan_spline_order", "kan_chunk_size", "stage1_epochs", "stage1_batch_size", "stage1_top_k", "stage1_train_epochs", "stage1_checkpoint_epoch")
    if any(int(config[key])<=0 for key in positive) or int(config["num_workers"])<0 or int(config["test_num_workers"])<0:
        raise ValueError("Invalid positive integer training parameter")
    checkpoints=config['stage1_checkpoint_epochs']
    if not isinstance(checkpoints,(tuple,list)) or not checkpoints or any(
        type(v) is not int or v<1 or v>int(config['stage1_train_epochs']) for v in checkpoints
    ) or len(set(checkpoints))!=len(checkpoints):
        raise ValueError('Stage I checkpoint epochs must be unique and within train epochs')
    if int(config['stage1_checkpoint_epoch']) not in checkpoints:
        raise ValueError('Selected Stage I epoch must appear in checkpoint epochs')
    if int(config['stage1_train_epochs'])>int(config['stage1_epochs']):
        raise ValueError('Stage I training must not exceed schedule horizon')
    if int(config['stage1_batch_size']) != 256:
        raise ValueError('Stage I global batch must be 256; no gradient accumulation')
    if not 0 <= int(config['stage1_warmup_epochs']) < int(config['stage1_epochs']):
        raise ValueError('Invalid Stage I warmup')
    if not 0 <= int(config['stage2_freeze_epochs']) < int(config['epochs']):
        raise ValueError('Invalid Stage II freeze epochs')
    if not 0 < float(config['stage2_encoder_lr_scale']) <= 1.0:
        raise ValueError('Invalid Stage II encoder LR scale')
    if not 0 <= config['stage1_alpha'] <= 1 or not 0 <= config['stage1_rho_max'] <= 1:
        raise ValueError('Stage I alpha/rho must be [0,1]')
    if config['stage1_tau'] <= 0 or config['stage1_region_tau']<=0 or config['stage1_neighbor_margin']<0:
        raise ValueError('Invalid Stage I temperatures or neighbour cutoff')
    if config['stage1_separation_margin'] < 0 or config['stage1_eta_min'] < 0:
        raise ValueError('Invalid Stage I margin / eta minimum')
    if min(config['stage1_lr_cnn'],config['stage1_lr_kan'],config['stage1_lr_head'])<=0:
        raise ValueError('Invalid Stage I LR')
    if config['stage1_weight_decay']<0 or min(config['stage1_global_weight'],config['stage1_region_weight'],config['stage1_separation_weight'])<0:
        raise ValueError('Invalid Stage I loss weights')
    active=config['stage1_global_weight']+(config['stage1_region_weight'] if config['stage1_use_region'] else 0)+(config['stage1_separation_weight'] if config['stage1_use_separation'] else 0)
    if active<=0:raise ValueError('Stage I total active loss weight must be positive')
    if config["lr"]<=0 or not 0<=config["eta_min"]<config["lr"]:
        raise ValueError("Expected lr>eta_min>=0")
    if not 0<=config["foreground_bce_weight"]<=1 or not 0<config["threshold"]<1:
        raise ValueError("Invalid foreground BCE weight or threshold")
    return SimpleNamespace(**config)


def parse_args(argv=None, *, test=False):
    parser=argparse.ArgumentParser(description="C2Seg two-stage medical image segmentation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    if test:
        parser.add_argument("--checkpoint", default=None)
    cli=parser.parse_args(argv)
    supplied={k:v for k,v in vars(cli).items() if k not in ("config","checkpoint") and v is not None}
    return get_config(cli.config, supplied), getattr(cli,"checkpoint",None)


def model_config(config):
    return SimpleNamespace(**vars(config))


def run_paths(config):
    project_dir=Path(__file__).resolve().parent
    def resolve(root):
        path=Path(root).expanduser()
        return (path if path.is_absolute() else project_dir/path).resolve()
    return (resolve(config.results_root)/config.dataset/config.run_name,
            resolve(config.weights_root)/config.dataset/config.run_name)

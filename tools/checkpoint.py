"""Compact checkpoints: trainable parameters and BatchNorm running statistics."""
from __future__ import annotations
import os
from pathlib import Path
import torch
FORMAT = "C2SEG_STAGE2_V1"


def compact_state_dict(model):
    trainable = {name for name,p in model.named_parameters() if p.requires_grad}
    mutable_bn = {name for name,_ in model.named_buffers()
                  if not name.startswith("clip.") and name.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    return {name: tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()
            if name in trainable or name in mutable_bn}


def save_best(model, path, *, epoch, val_dice, config):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={"format_version":FORMAT,"epoch":int(epoch),"val_dice":float(val_dice),
             "state_dict":compact_state_dict(model)}
    tmp=path.with_name(path.name+".tmp")
    try:
        torch.save(payload,tmp)
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def load_best(model,path,config=None):
    try:
        payload=torch.load(path,map_location="cpu",weights_only=True)
    except TypeError:
        payload=torch.load(path,map_location="cpu")
    if not isinstance(payload,dict) or payload.get("format_version") != FORMAT:
        raise ValueError("Checkpoint must use the independent C2Seg Stage II format")
    saved=payload["state_dict"]
    expected=compact_state_dict(model)
    if set(saved)!=set(expected):
        raise RuntimeError("Checkpoint has missing or extra trainable parameters")
    current=model.state_dict()
    for name,tensor in saved.items():
        if current[name].shape!=tensor.shape:
            raise RuntimeError(f"Shape mismatch: {name}")
    current.update(saved)
    model.load_state_dict(current, strict=True)
    return payload

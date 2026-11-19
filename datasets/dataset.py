"""C2Seg paired image/mask/text loading adapted to frozen OpenAI CLIP."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def read_text_map(path):
    df = pd.read_excel(path, engine="openpyxl")
    for col in ("Image", "Description"):
        if col not in df.columns:
            raise KeyError(f"Missing Excel column {col!r}: {path}")
    output = {}
    for image, text in zip(df["Image"], df["Description"]):
        if pd.isna(image):
            continue
        key = str(image).strip().lower()
        if key in output:
            raise ValueError(f"Duplicate text image key: {key} in {path}")
        output[key] = "" if pd.isna(text) else str(text).split("\n")[0].strip()
    return output


def match_image(mask_path, image_dir, pair_rule):
    names = [mask_path.name]
    if mask_path.stem.lower().startswith("mask_"):
        names.append(mask_path.name[5:])
    if pair_rule == "mask_prefix":
        names = names[1:]
    elif pair_rule == "same_name":
        names = names[:1]
    for name in names:
        p = image_dir / name
        if p.is_file():
            return p
        candidates = [q for q in image_dir.iterdir() if q.is_file() and q.stem.lower() == Path(name).stem.lower() and q.suffix.lower() in IMAGE_SUFFIXES]
        if len(candidates) == 1:
            return candidates[0]
    raise FileNotFoundError(f"Could not pair mask {mask_path} in {image_dir}; pair_rule={pair_rule}")


class MedicalDataset(Dataset):
    def __init__(self, root, split, config, tokenizer=None):
        self.split = split
        self.config = config
        self.root = Path(root) / config.dataset / f"{split}_Folder"
        image_dir = self.root / "images"
        mask_dir = self.root / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Expected images/ and masks/ in {self.root}")
        texts = read_text_map(self.root / f"{split}_text.xlsx")
        if tokenizer is None:
            import clip
            tokenizer = lambda value: clip.tokenize(value, context_length=config.text_len, truncate=True)
        self.samples = []
        masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not masks:
            raise FileNotFoundError(f"No masks found in {mask_dir}")
        for mask in masks:
            image = match_image(mask, image_dir, config.pair_rule)
            lookup = (image.name, mask.name, image.stem, mask.stem)
            text = next((texts[k.lower()] for k in lookup if k.lower() in texts), None)
            if text is None or not text:
                raise ValueError(f"Missing/empty description for {image.name} in {self.root}")
            token = tokenizer(text)
            if token.ndim == 2:
                token = token.squeeze(0)
            token = token.to(torch.long)
            if token.numel() != config.text_len:
                raise ValueError(f"Tokenizer returned {token.numel()} tokens; expected {config.text_len}")
            self.samples.append((image, mask, token, text))
        # Cache unique token sequences for semantic Stage I. Stage II still receives
        # the same per-sample token tensor; the extra integer ID is diagnostic-only.
        token_to_id = {}
        unique_tokens = []
        indexed_samples = []
        for image, mask, token, text in self.samples:
            key = tuple(int(x) for x in token.tolist())
            if key not in token_to_id:
                token_to_id[key] = len(unique_tokens)
                unique_tokens.append(token)
            indexed_samples.append((image, mask, token, text, token_to_id[key]))
        self.samples = indexed_samples
        self.unique_tokens = torch.stack(unique_tokens, dim=0)
        self.unique_text_count = len(unique_tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, token, text, text_id = self.samples[index]
        size = self.config.image_size
        with Image.open(image_path) as im, Image.open(mask_path) as ma:
            image = im.convert("RGB")
            mask = ma.convert("L")
            if self.split == "Train":
                strategy = self.config.geometry_aug
                if strategy in {"original", "flip"}:
                    if random.random() < 0.5:
                        image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if random.random() < 0.5:
                        image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if strategy == "original":
                    if random.random() < 0.5:
                        k = random.randrange(4)
                        if k:
                            image, mask = image.rotate(k * 90), mask.rotate(k * 90)
                    elif random.random() < 0.5:
                        angle = random.uniform(-20, 20)
                        image = image.rotate(angle, resample=Image.Resampling.BICUBIC)
                        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
                elif strategy == "rotate90":
                    k = random.randrange(4)
                    if k:
                        image, mask = image.rotate(k * 90), mask.rotate(k * 90)
                elif strategy == "small_rotate":
                    angle = random.uniform(-15, 15)
                    image = image.rotate(angle, resample=Image.Resampling.BICUBIC)
                    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
            if self.split == "Train" and self.config.use_photometric_aug:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if self.split == "Train" and self.config.use_blur_aug and random.random() < 0.5:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            mask_method = (self.config.train_mask_interpolation if self.split == "Train"
                           else ("bilinear" if self.config.evaluation_protocol == "legacy_mask_bilinear_v1" else "nearest"))
            mask_interp = Image.Resampling.NEAREST if mask_method == "nearest" else Image.Resampling.BILINEAR
            mask = mask.resize((size, size), mask_interp)
            arr = np.asarray(image, dtype=np.float32).copy() / 255.0
            if self.config.image_normalization == "clip":
                arr = (arr - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
            if self.config.image_channel_order == "bgr":
                arr = arr[:, :, ::-1].copy()
            mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))),
            "mask": torch.from_numpy(mask_arr[None, ...]),
            "text_token": token.clone(),
            "text_mask": token.ne(0),
            "text_id": torch.tensor(text_id, dtype=torch.long),
            "filename": image_path.name,
        }


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(config, split, tokenizer=None):
    dataset = MedicalDataset(config.data_root, split, config, tokenizer=tokenizer)
    train = split == "Train"
    gen = torch.Generator().manual_seed(int(config.seed) + {"Train": 0, "Val": 1, "Test": 2}[split])
    workers = config.test_num_workers if split == "Test" else config.num_workers
    return DataLoader(dataset, batch_size=config.batch_size if train else (config.val_batch_size if split == "Val" else config.test_batch_size),
                      shuffle=train, num_workers=workers, pin_memory=True,
                      drop_last=train and len(dataset) % config.batch_size == 1 and len(dataset) > config.batch_size,
                      worker_init_fn=worker_seed, generator=gen,
                      persistent_workers=workers > 0)

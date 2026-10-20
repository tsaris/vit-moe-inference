import os
import re
import argparse
import random
import json
import time
import math
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import deepspeed
try:
    from deepspeed.moe.utils import is_moe_param as _ds_is_moe_param
except Exception:
    _ds_is_moe_param = None

import geobench

from geo_projector_lib import TinyViTEncoder, S2GeoTiffCSVDataset

import matplotlib
import pickle
from pathlib import Path
from PIL import Image
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class TinyImageNetDataset(torch.utils.data.Dataset):
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, root: str, split: str, target_hw: Optional[Tuple[int, int]] = None,
                 std_floor: float = 1e-6, normalize_mode: str = 'standard'):
        self.root = os.path.abspath(root)
        self.split = str(split)
        self.target_hw = target_hw
        self.std_floor = float(std_floor)
        self.normalize_mode = str(normalize_mode)
        if self.split not in ('train', 'val'):
            raise ValueError(f'TinyImageNetDataset supports split=train/val only, got {self.split}')
        wnids_path = os.path.join(self.root, 'wnids.txt')
        if os.path.isfile(wnids_path):
            with open(wnids_path, 'r', encoding='utf-8') as f:
                self.wnids = [ln.strip() for ln in f if ln.strip()]
        else:
            train_dir = os.path.join(self.root, 'train')
            self.wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        if len(self.wnids) == 0:
            raise RuntimeError(f'No Tiny-ImageNet classes found under {self.root}')
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.files = []
        self.labels = []
        if self.split == 'train':
            train_dir = os.path.join(self.root, 'train')
            for wnid in self.wnids:
                img_dir = os.path.join(train_dir, wnid, 'images')
                if not os.path.isdir(img_dir):
                    continue
                paths = []
                for patt in ('*.JPEG', '*.jpeg', '*.jpg', '*.png'):
                    paths.extend(sorted(str(p) for p in Path(img_dir).glob(patt)))
                cls_idx = int(self.wnid_to_idx[wnid])
                for p in paths:
                    self.files.append(p)
                    self.labels.append(cls_idx)
        else:
            ann_path = os.path.join(self.root, 'val', 'val_annotations.txt')
            img_dir = os.path.join(self.root, 'val', 'images')
            if not os.path.isfile(ann_path):
                raise FileNotFoundError(ann_path)
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(img_dir)
            with open(ann_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    if wnid not in self.wnid_to_idx:
                        continue
                    path = os.path.join(img_dir, fname)
                    if not os.path.isfile(path):
                        continue
                    self.files.append(path)
                    self.labels.append(int(self.wnid_to_idx[wnid]))
        if len(self.files) == 0:
            raise RuntimeError(f'No Tiny-ImageNet samples found for split={self.split} under {self.root}')
        self.num_classes = int(len(self.wnids))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        y = int(self.labels[idx])
        with Image.open(path) as img:
            img = img.convert('RGB')
            x_np = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(np.transpose(x_np, (2, 0, 1)))
        if self.target_hw is not None and tuple(x.shape[-2:]) != tuple(self.target_hw):
            raise RuntimeError(f'[TinyImageNetDataset] Shape mismatch: {path} got {tuple(x.shape[-2:])}, expected {tuple(self.target_hw)}. Interpolation is DISABLED.')
        if self.normalize_mode == 'standard':
            mean = torch.tensor(self.IMAGENET_MEAN, dtype=x.dtype).view(3,1,1)
            std = torch.tensor(self.IMAGENET_STD, dtype=x.dtype).view(3,1,1)
            x = (x - mean) / std
        elif self.normalize_mode == 'perimage_zscore':
            mean = x.mean(dim=(1,2), keepdim=True)
            std = x.std(dim=(1,2), keepdim=True).clamp_min(self.std_floor)
            x = (x - mean) / std
        else:
            raise ValueError(f'Unknown normalize_mode={self.normalize_mode}')
        return x, y


class CIFAR100Dataset(torch.utils.data.Dataset):
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)

    def __init__(self, root: str, split: str, target_hw: Optional[Tuple[int, int]] = None,
                 std_floor: float = 1e-6, normalize_mode: str = 'standard'):
        self.root = os.path.abspath(root)
        self.split = str(split)
        self.target_hw = target_hw
        self.std_floor = float(std_floor)
        self.normalize_mode = str(normalize_mode)
        if self.split not in ('train', 'test'):
            raise ValueError(f'CIFAR100Dataset supports split=train/test only, got {self.split}')
        self.base_dir = self._resolve_base_dir(self.root)
        meta = self._load_pickle(os.path.join(self.base_dir, 'meta'))
        fine_names = meta.get(b'fine_label_names', meta.get('fine_label_names', None))
        if fine_names is None:
            raise RuntimeError('Could not find fine_label_names in CIFAR-100 meta')
        self.class_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in fine_names]
        self.num_classes = int(len(self.class_names))
        data_obj = self._load_pickle(os.path.join(self.base_dir, self.split))
        data = data_obj.get(b'data', data_obj.get('data', None))
        labels = data_obj.get(b'fine_labels', data_obj.get('fine_labels', None))
        if data is None or labels is None:
            raise RuntimeError(f'[CIFAR100Dataset] Missing data/fine_labels in {os.path.join(self.base_dir, self.split)}')
        data = np.asarray(data, dtype=np.uint8)
        labels = np.asarray(labels, dtype=np.int64)
        self.data = data.reshape(-1, 3, 32, 32)
        self.labels = labels.tolist()

    def _resolve_base_dir(self, root: str) -> str:
        direct = root
        nested = os.path.join(root, 'cifar-100-python')
        def ok(p: str) -> bool:
            return all(os.path.isfile(os.path.join(p, nm)) for nm in ('train','test','meta'))
        if ok(direct):
            return direct
        if ok(nested):
            return nested
        raise FileNotFoundError(f'[CIFAR100Dataset] Could not find CIFAR-100 files under root={root}.')

    def _load_pickle(self, path: str):
        with open(path, 'rb') as f:
            try:
                return pickle.load(f, encoding='bytes')
            except TypeError:
                return pickle.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data[idx].astype(np.float32) / 255.0)
        y = int(self.labels[idx])
        if self.target_hw is not None and tuple(x.shape[-2:]) != tuple(self.target_hw):
            raise RuntimeError(f'[CIFAR100Dataset] Shape mismatch idx={idx} got {tuple(x.shape[-2:])}, expected {tuple(self.target_hw)}. Interpolation is DISABLED.')
        if self.normalize_mode == 'standard':
            mean = torch.tensor(self.CIFAR100_MEAN, dtype=x.dtype).view(3,1,1)
            std = torch.tensor(self.CIFAR100_STD, dtype=x.dtype).view(3,1,1)
            x = (x - mean) / std
        elif self.normalize_mode == 'perimage_zscore':
            mean = x.mean(dim=(1,2), keepdim=True)
            std = x.std(dim=(1,2), keepdim=True).clamp_min(self.std_floor)
            x = (x - mean) / std
        else:
            raise ValueError(f'Unknown normalize_mode={self.normalize_mode}')
        return x, y


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ddp_allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


def ddp_sum_floats(device, *vals: float) -> List[float]:
    t = torch.tensor(list(vals), device=device, dtype=torch.float32)
    ddp_allreduce_sum(t)
    return [float(x.item()) for x in t]


def ddp_broadcast_object(obj, src: int = 0):
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return obj
    obj_list = [obj] if torch.distributed.get_rank() == src else [None]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class RGBTrainAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, hflip_prob: float = 0.0, crop_padding: int = 0):
        super().__init__()
        self.base_ds = base_ds
        self.hflip_prob = float(max(0.0, min(1.0, hflip_prob)))
        self.crop_padding = int(max(0, crop_padding))

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        x, y = self.base_ds[idx]

        if self.hflip_prob > 0.0 and torch.rand(1).item() < self.hflip_prob:
            x = torch.flip(x, dims=[2])

        if self.crop_padding > 0:
            pad = int(self.crop_padding)
            x4 = x.unsqueeze(0)
            x4 = F.pad(x4, (pad, pad, pad, pad), mode="reflect")
            H, W = int(x.shape[-2]), int(x.shape[-1])
            max_y = int(x4.shape[-2] - H)
            max_x = int(x4.shape[-1] - W)
            oy = int(torch.randint(0, max_y + 1, (1,)).item())
            ox = int(torch.randint(0, max_x + 1, (1,)).item())
            x = x4[:, :, oy:oy + H, ox:ox + W].squeeze(0)

        return x, y


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def one_hot_targets(y: torch.Tensor, num_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return F.one_hot(y.to(device=device, non_blocking=True).long(), num_classes=int(num_classes)).to(dtype=dtype)


def mixup_cutmix_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
    switch_prob: float,
):
    mix_prob = float(max(0.0, min(1.0, mix_prob)))
    mixup_alpha = float(max(0.0, mixup_alpha))
    cutmix_alpha = float(max(0.0, cutmix_alpha))
    switch_prob = float(max(0.0, min(1.0, switch_prob)))

    y1 = one_hot_targets(y, num_classes=num_classes, device=x.device, dtype=x.dtype)

    if mix_prob <= 0.0 or ((mixup_alpha <= 0.0) and (cutmix_alpha <= 0.0)):
        return x, y1, False, 1.0, "none"

    if torch.rand(1, device=x.device).item() >= mix_prob:
        return x, y1, False, 1.0, "none"

    perm = torch.randperm(x.size(0), device=x.device)
    x2 = x[perm]
    y2 = y1[perm]

    use_cutmix = (cutmix_alpha > 0.0) and ((mixup_alpha <= 0.0) or (torch.rand(1, device=x.device).item() < switch_prob))

    if use_cutmix:
        beta = torch.distributions.Beta(cutmix_alpha, cutmix_alpha)
        lam = float(beta.sample(()).item())
        B, C, H, W = x.shape
        cut_rat = math.sqrt(max(0.0, 1.0 - lam))
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = int(torch.randint(0, W, (1,), device=x.device).item())
        cy = int(torch.randint(0, H, (1,), device=x.device).item())

        x1 = max(cx - cut_w // 2, 0)
        x2b = min(cx + cut_w // 2, W)
        y1b = max(cy - cut_h // 2, 0)
        y2b = min(cy + cut_h // 2, H)

        x_mixed = x.clone()
        x_mixed[:, :, y1b:y2b, x1:x2b] = x2[:, :, y1b:y2b, x1:x2b]

        box_area = max(0, x2b - x1) * max(0, y2b - y1b)
        lam_adj = 1.0 - (float(box_area) / float(max(1, H * W)))
        y_mixed = (lam_adj * y1) + ((1.0 - lam_adj) * y2)
        return x_mixed, y_mixed, True, float(lam_adj), "cutmix"

    beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
    lam = float(beta.sample(()).item())
    x_mixed = (lam * x) + ((1.0 - lam) * x2)
    y_mixed = (lam * y1) + ((1.0 - lam) * y2)
    return x_mixed, y_mixed, True, float(lam), "mixup"



def apply_geo_safe_spatial_aug(
    x: torch.Tensor,
    hflip_prob: float,
    vflip_prob: float,
    rot90_prob: float,
) -> torch.Tensor:
    
    if x.ndim != 4:
        return x

    B = int(x.shape[0])
    out = x

    if float(hflip_prob) > 0.0:
        mask = (torch.rand(B, device=out.device) < float(hflip_prob))
        if bool(mask.any()):
            out = out.clone()
            out[mask] = torch.flip(out[mask], dims=[3])

    if float(vflip_prob) > 0.0:
        mask = (torch.rand(B, device=out.device) < float(vflip_prob))
        if bool(mask.any()):
            out = out.clone()
            out[mask] = torch.flip(out[mask], dims=[2])

    if float(rot90_prob) > 0.0:
        mask = (torch.rand(B, device=out.device) < float(rot90_prob))
        if bool(mask.any()):
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            out = out.clone()
            ks = torch.randint(low=1, high=4, size=(int(idx.numel()),), device=out.device)
            for j, bi in enumerate(idx.tolist()):
                out[bi] = torch.rot90(out[bi], k=int(ks[j].item()), dims=[1, 2])

    return out


def resolve_rgb_recipe(args) -> Dict[str, float]:
    recipe = {
        "hflip": 0.0,
        "crop_pad": 0.0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "mix_prob": 0.0,
        "mix_switch_prob": 0.5,
        "head_dropout": 0.0,
        "label_smoothing": float(args.label_smoothing),
        "encoder_dropout": float(args.dropout),
    }

    if args.dataset in ("cifar100", "tinyimagenet"):
        recipe.update({
            "hflip": 0.5,
            "crop_pad": 4.0,
            "mixup_alpha": 0.8,
            "cutmix_alpha": 1.0,
            "mix_prob": 1.0,
            "mix_switch_prob": 0.5,
            "head_dropout": 0.2,
            "label_smoothing": 0.1 if float(args.label_smoothing) == 0.0 else float(args.label_smoothing),
            "encoder_dropout": 0.10 if float(args.dropout) == 0.05 else float(args.dropout),
        })

    if float(args.rgb_aug_hflip) >= 0.0:
        recipe["hflip"] = float(args.rgb_aug_hflip)
    if int(args.rgb_aug_crop_pad) >= 0:
        recipe["crop_pad"] = float(args.rgb_aug_crop_pad)
    if float(args.mixup_alpha) >= 0.0:
        recipe["mixup_alpha"] = float(args.mixup_alpha)
    if float(args.cutmix_alpha) >= 0.0:
        recipe["cutmix_alpha"] = float(args.cutmix_alpha)
    if float(args.mix_prob) >= 0.0:
        recipe["mix_prob"] = float(args.mix_prob)
    if float(args.mix_switch_prob) >= 0.0:
        recipe["mix_switch_prob"] = float(args.mix_switch_prob)
    if float(args.head_dropout) >= 0.0:
        recipe["head_dropout"] = float(args.head_dropout)

    return recipe

def resolve_geo_recipe(args) -> Dict[str, float]:
    recipe = {
        "hflip": 0.0,
        "vflip": 0.0,
        "rot90": 0.0,
    }

    if args.dataset in ("so2sat", "forestnet"):
        recipe.update({
            "hflip": 0.5,
            "vflip": 0.5,
            "rot90": 0.5,
        })

    if float(args.geo_aug_hflip) >= 0.0:
        recipe["hflip"] = float(args.geo_aug_hflip)
    if float(args.geo_aug_vflip) >= 0.0:
        recipe["vflip"] = float(args.geo_aug_vflip)
    if float(args.geo_aug_rot90) >= 0.0:
        recipe["rot90"] = float(args.geo_aug_rot90)

    return recipe


def extract_single_labels(ds) -> List[int]:
    labels_attr = getattr(ds, "labels", None)
    if labels_attr is not None:
        try:
            return [int(v) for v in labels_attr]
        except Exception:
            pass

    base = getattr(ds, "base", None)
    if base is not None:
        labels_attr = getattr(base, "labels", None)
        if labels_attr is not None:
            try:
                return [int(v) for v in labels_attr]
            except Exception:
                pass
        out = []
        for i in range(len(base)):
            s = base[i]
            lab = getattr(s, "label", None)
            out.append(int(lab))
        return out

    out = []
    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append(int(item[1]))
        else:
            raise RuntimeError("Could not extract labels for stratified split.")
    return out


def build_cosine_lr_scheduler_state(base_lrs: List[float], warmup_steps: int, total_steps: int, min_lr_scale: float):
    return {
        "base_lrs": [float(v) for v in base_lrs],
        "warmup_steps": int(max(0, warmup_steps)),
        "total_steps": int(max(1, total_steps)),
        "min_lr_scale": float(max(0.0, min(1.0, min_lr_scale))),
    }


def cosine_lr_scale(step_idx: int, sched_state: Dict[str, Any]) -> float:
    warmup_steps = int(sched_state["warmup_steps"])
    total_steps = int(sched_state["total_steps"])
    min_lr_scale = float(sched_state["min_lr_scale"])

    if total_steps <= 1:
        return 1.0

    s = int(max(0, step_idx))
    if warmup_steps > 0 and s < warmup_steps:
        return float(s + 1) / float(max(1, warmup_steps))

    if total_steps <= warmup_steps:
        return 1.0

    t = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
    t = min(max(t, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(min_lr_scale + (1.0 - min_lr_scale) * cosine)


def apply_lr_scale(optimizer, sched_state: Dict[str, Any], step_idx: int):
    scale = cosine_lr_scale(step_idx=step_idx, sched_state=sched_state)
    base_lrs = sched_state["base_lrs"]
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = float(base_lr) * scale
    return scale


@torch.no_grad()
def eval_split_limited(engine, dataloader, device, label_smoothing: float, max_batches: int = 0) -> Tuple[float, float]:
    engine.eval()
    loss_sum_local = 0.0
    steps_local = 0.0
    correct_local = 0.0
    samples_local = 0.0

    for bi, (x, y) in enumerate(dataloader):
        if int(max_batches) > 0 and bi >= int(max_batches):
            break

        x = x.to(device, non_blocking=True)
        if engine.fp16_enabled():
            x = x.half()
        elif engine.bfloat16_enabled():
            x = x.bfloat16()

        logits, _ = engine.module(x, labels=None)
        y_i = y.to(device, non_blocking=True).long()
        loss = F.cross_entropy(logits, y_i, label_smoothing=float(label_smoothing))
        preds = logits.argmax(dim=1)

        correct_local += float((preds == y_i).sum().item())
        samples_local += float(y_i.numel())
        loss_sum_local += float(loss.item())
        steps_local += 1.0

    loss_sum, steps = ddp_sum_floats(device, loss_sum_local, steps_local)
    mean_loss = loss_sum / max(steps, 1.0)
    correct, samples = ddp_sum_floats(device, correct_local, samples_local)
    metric = correct / max(samples, 1.0)
    return mean_loss, metric


class GeoBench6Band(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        rank: int,
        split: str,
        dataset_kind: str,
        std_floor: float = 1e-6,
        target_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.base = base_dataset
        self.rank = int(rank)
        self.split = str(split)
        self.dataset_kind = str(dataset_kind)
        self.std_floor = float(std_floor)
        self.target_hw = target_hw

        if self.dataset_kind not in ("so2sat", "forestnet"):
            raise ValueError(f"Unsupported dataset_kind={self.dataset_kind}")

        self.num_classes = self._infer_num_classes()

    def __len__(self):
        return len(self.base)

    def _fail(self, msg: str):
        raise RuntimeError(f"[Rank {self.rank}][{self.split}] {msg}")

    def _band_text(self, band) -> str:
        vals = []
        for attr in ("name", "band_name", "id", "info"):
            v = getattr(band, attr, None)
            if v is not None:
                vals.append(str(v))

        if len(vals) == 0:
            try:
                s = str(band)
                if s is not None and str(s).strip() != "":
                    vals.append(str(s))
            except Exception:
                pass

        if len(vals) == 0:
            try:
                r = repr(band)
                if r is not None and str(r).strip() != "":
                    vals.append(str(r))
            except Exception:
                pass

        return " | ".join(vals).upper()

    def _extract_band_number(self, txt: str) -> Optional[int]:
        """
        Robustly extract the optical band number from strings like:
          'Band 02 - Blue ...'
          'Band(info=Band 11 - SWIR ...)'
        Returns:
          int band number if found, else None
        """
        m = re.search(r"\bBAND\s*0*([0-9]{1,2})(?:\s*-|\b)", txt)
        if m is not None:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _band_semantic_key(self, band) -> str:
        txt = self._band_text(band)
        bnum = self._extract_band_number(txt)

        if self.dataset_kind == "so2sat":
            if bnum == 2 and "BLUE" in txt:
                return "BLUE"
            if bnum == 3 and "GREEN" in txt:
                return "GREEN"
            if bnum == 4 and "RED" in txt and "RED EDGE" not in txt:
                return "RED"
            if bnum == 8 and "NIR" in txt:
                return "NIR"
            if bnum == 11 and "SWIR" in txt:
                return "SWIR1"
            if bnum == 12 and "SWIR" in txt:
                return "SWIR2"
            return ""

        if self.dataset_kind == "forestnet":
            if bnum == 2 and "BLUE" in txt:
                return "BLUE"
            if bnum == 3 and "GREEN" in txt:
                return "GREEN"
            if bnum == 4 and "RED" in txt and "RED EDGE" not in txt:
                return "RED"
            if bnum == 5 and "NIR" in txt:
                return "NIR"
            if bnum == 6 and "SWIR1" in txt:
                return "SWIR1"
            if bnum == 7 and "SWIR2" in txt:
                return "SWIR2"
            return ""

        return ""

    def _select_6_bands(self, bands):
        wanted = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]

        key_to_band = {}
        keys_seen = []
        raw = []

        for b in bands:
            txt = self._band_text(b)
            raw.append(txt)
            k = self._band_semantic_key(b)
            keys_seen.append(k)
            if k and k not in key_to_band:
                key_to_band[k] = b

        missing = [k for k in wanted if k not in key_to_band]
        if missing:
            self._fail(
                f"Missing required 6-band overlap {missing}. "
                f"Available parsed keys={keys_seen}. Raw bands={raw}"
            )

        return [key_to_band[k] for k in wanted]

    def _infer_num_classes(self) -> int:
        
        labels_attr = getattr(self.base, "labels", None)
        if labels_attr is not None:
            try:
                max_single = max(int(v) for v in labels_attr)
                if max_single >= 0:
                    return int(max_single + 1)
            except Exception:
                pass

        max_single = -1
        for i in range(len(self.base)):
            s = self.base[i]
            lab = getattr(s, "label", None)
            try:
                iv = int(lab)
                if iv > max_single:
                    max_single = iv
            except Exception:
                pass

        if max_single < 0:
            self._fail("Could not infer num_classes from labels.")
        return int(max_single + 1)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        bands_use = self._select_6_bands(sample.bands)

        x_np = np.stack([np.asarray(b.data, dtype=np.float32) for b in bands_use], axis=0)
        x = torch.from_numpy(x_np)

        if self.target_hw is not None and tuple(x.shape[-2:]) != tuple(self.target_hw):
            self._fail(
                f"Shape mismatch idx={idx}: got {tuple(x.shape[-2:])} expected {tuple(self.target_hw)}"
            )

        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            mn = torch.nanmin(x).item()
            mx = torch.nanmax(x).item()
            self._fail(f"Non-finite RAW x idx={idx} count={bad} min={mn} max={mx}")

        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(self.std_floor)
        x = (x - mean) / std

        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            mn = torch.nanmin(x).item()
            mx = torch.nanmax(x).item()
            self._fail(f"Non-finite POST-NORM x idx={idx} count={bad} min={mn} max={mx}")

        lab = getattr(sample, "label", None)
        y = int(lab)
        return x, y


class EncoderOnlyMAE(nn.Module):
    def __init__(self, encoder: TinyViTEncoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x: torch.Tensor):
        return self.encoder.encode(x, mod_ids=None)


class FTModel(nn.Module):
    def __init__(self, mae: EncoderOnlyMAE, embed_dim: int, num_classes: int, ft_mode: str, moe_aux_weight: float, head_dropout: float = 0.0):
        super().__init__()
        self.mae = mae
        self.head_drop = nn.Dropout(float(head_dropout))
        self.head = nn.Linear(int(embed_dim), int(num_classes))
        self.ft_mode = str(ft_mode)
        self.moe_aux_weight = float(moe_aux_weight)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        if self.ft_mode == "linear_probe":
            with torch.no_grad():
                tokens = self.mae.encode(x)
        else:
            tokens = self.mae.encode(x)

        feats = tokens.mean(dim=1)
        feats = self.head_drop(feats)
        logits = self.head(feats)

        if labels is None:
            return logits, {}

        aux = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        if self.moe_aux_weight != 0.0:
            try:
                aux = self.mae.encoder.aux_balance_loss()
                aux = torch.as_tensor(aux, device=x.device, dtype=torch.float32)
            except Exception:
                aux = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        if torch.is_tensor(labels) and labels.dtype.is_floating_point:
            target = labels.to(device=x.device, non_blocking=True, dtype=logits.dtype)
            cls_loss = soft_target_cross_entropy(logits, target)
        else:
            labels_i = labels.to(x.device, non_blocking=True).long()
            cls_loss = F.cross_entropy(logits, labels_i, label_smoothing=float(label_smoothing))

        loss = cls_loss + (self.moe_aux_weight * aux)
        return loss, {
            "cls_loss": float(cls_loss.item()),
            "moe_aux": float(aux.item()) if torch.is_tensor(aux) else float(aux),
            "moe_aux_w": float(self.moe_aux_weight),
            "total_loss": float(loss.item()),
        }



def find_moe_params(model: nn.Module):
    moe_params = []

    if _ds_is_moe_param is not None:
        for p in model.parameters():
            if _ds_is_moe_param(p):
                moe_params.append(p)

    if len(moe_params) == 0:
        seen = set()
        mae = getattr(model, "mae", None)
        enc = getattr(mae, "encoder", None) if mae is not None else None
        blocks = getattr(enc, "blocks", None) if enc is not None else None
        if blocks is not None:
            for blk in blocks:
                moe_mod = getattr(blk, "moe", None)
                if moe_mod is None:
                    continue
                for p in moe_mod.parameters():
                    pid = id(p)
                    if pid not in seen:
                        moe_params.append(p)
                        seen.add(pid)

    return moe_params, (len(moe_params) > 0)


def format_expert_util(counts: torch.Tensor) -> str:
    total = float(counts.sum().item())
    if total <= 0:
        return "util: (no tokens)"
    pct = (counts.float() / total) * 100.0
    counts_i = [int(x) for x in counts.detach().cpu().tolist()]
    pct_f = [float(x) for x in pct.detach().cpu().tolist()]
    parts = [f"e{i}:{pct_f[i]:.1f}%({counts_i[i]})" for i in range(len(counts_i))]
    return "util: " + " ".join(parts)


def epoch_util_summary(counts: torch.Tensor):
    eps = 1e-12
    total = float(counts.sum().item())
    if total <= 0:
        return {
            "tokens": 0.0, "active": 0, "top_idx": -1, "top_pct": 0.0,
            "max_pct": 0.0, "min_pct": 0.0, "entropy": 0.0, "gini": 0.0, "cv": 0.0
        }

    p = counts.float() / counts.sum().clamp_min(eps)
    active = int((counts > 0).sum().item())
    top_idx = int(torch.argmax(counts).item())
    top_pct = float((p[top_idx] * 100.0).item())
    max_pct = float((p.max() * 100.0).item())
    min_pct = float((p[p > 0].min() * 100.0).item()) if active > 0 else 0.0

    H = float((-(p * torch.log(p.clamp_min(eps))).sum()).item())
    H_norm = H / max(math.log(p.numel()), eps)

    hhi = float((p * p).sum().item())
    gini_like = 1.0 - hhi

    mean = float(p.mean().item())
    std = float(p.std(unbiased=False).item())
    cv = std / max(mean, eps)

    return {
        "tokens": total,
        "active": active,
        "top_idx": top_idx,
        "top_pct": top_pct,
        "max_pct": max_pct,
        "min_pct": min_pct,
        "entropy": H_norm,
        "gini": gini_like,
        "cv": cv,
    }


def epoch_topk_str_from_counts(counts: torch.Tensor, k: int = 8) -> str:
    total = float(counts.sum().item())
    if total <= 0:
        return ""
    p = (counts.float() / counts.sum().clamp_min(1e-12)) * 100.0
    vals, idx = torch.topk(p, k=min(k, p.numel()))
    pairs = [f"{int(i.item())}:{float(v.item()):.1f}" for v, i in zip(vals, idx)]
    return ",".join(pairs)


@torch.no_grad()
def eval_split(engine, dataloader, device, label_smoothing: float) -> Tuple[float, float]:
    engine.eval()
    loss_sum_local = 0.0
    steps_local = 0.0
    correct_local = 0.0
    samples_local = 0.0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)

        if engine.fp16_enabled():
            x = x.half()
        elif engine.bfloat16_enabled():
            x = x.bfloat16()

        logits, _ = engine.module(x, labels=None)

        y_i = y.to(device, non_blocking=True).long()
        loss = F.cross_entropy(logits, y_i, label_smoothing=float(label_smoothing))
        preds = logits.argmax(dim=1)
        correct = (preds == y_i).sum().item()
        correct_local += float(correct)
        samples_local += float(y_i.numel())

        loss_sum_local += float(loss.item())
        steps_local += 1.0

    loss_sum, steps = ddp_sum_floats(device, loss_sum_local, steps_local)
    mean_loss = loss_sum / max(steps, 1.0)

    correct, samples = ddp_sum_floats(device, correct_local, samples_local)
    metric = correct / max(samples, 1.0)
    return mean_loss, metric


def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def _counts_to_pct(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64, copy=False)
    s = float(np.sum(counts))
    if s <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return (counts / s) * 100.0


def plot_expert_occupancy_bars(out_path: str, pct: np.ndarray, title: str):
    E = int(pct.shape[0])
    x = np.arange(E, dtype=np.int32)
    plt.figure()
    plt.bar(x, pct)
    plt.xlabel("expert id")
    plt.ylabel("token share (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_training_plots(
    out_dir: str,
    history: Dict[str, List[float]],
    metric_name: str,
):
    _safe_makedirs(out_dir)

    epochs = np.arange(len(history["train_loss"]), dtype=np.int32)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_mix_loss")
    if "train_clean_loss" in history and not np.isnan(np.asarray(history["train_clean_loss"], dtype=np.float32)).all():
        plt.plot(epochs, history["train_clean_loss"], label="train_clean_loss")
    if not np.isnan(np.asarray(history["val_loss"], dtype=np.float32)).all():
        plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["test_loss"], label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, np.asarray(history["train_metric"]) * 100.0, label=f"train_mix_{metric_name}")
    if "train_clean_metric" in history and not np.isnan(np.asarray(history["train_clean_metric"], dtype=np.float32)).all():
        plt.plot(epochs, np.asarray(history["train_clean_metric"]) * 100.0, label=f"train_clean_{metric_name}")
    if not np.isnan(np.asarray(history["val_metric"], dtype=np.float32)).all():
        plt.plot(epochs, np.asarray(history["val_metric"]) * 100.0, label=f"val_{metric_name}")
    plt.plot(epochs, np.asarray(history["test_metric"]) * 100.0, label=f"test_{metric_name}")
    plt.xlabel("epoch")
    plt.ylabel(f"{metric_name} (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves_acc.png"), dpi=200)
    plt.close()




def _split_indices_stratified(labels: List[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    val_frac = float(max(0.0, min(0.9, val_frac)))
    n_total = len(labels)
    if val_frac <= 0.0:
        return list(range(n_total)), []

    rng = np.random.RandomState(int(seed))
    per_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        yi = int(y)
        if yi not in per_class:
            per_class[yi] = []
        per_class[yi].append(int(i))

    train_idx: List[int] = []
    val_idx: List[int] = []

    for y in sorted(per_class.keys()):
        idxs = list(per_class[y])
        rng.shuffle(idxs)
        n_cls = len(idxs)
        n_val = int(round(n_cls * val_frac))
        if n_val <= 0 and n_cls > 1:
            n_val = 1
        if n_val >= n_cls and n_cls > 1:
            n_val = n_cls - 1

        val_part = idxs[:n_val]
        train_part = idxs[n_val:]

        if len(train_part) == 0 and len(val_part) > 1:
            train_part.append(val_part.pop())
        if len(val_part) == 0 and len(train_part) > 1:
            val_part.append(train_part.pop())

        train_idx.extend(train_part)
        val_idx.extend(val_part)

    train_idx = sorted(int(i) for i in train_idx)
    val_idx = sorted(int(i) for i in val_idx)
    return train_idx, val_idx

def _compute_num_classes_from_csvs_rank0(csv_paths: List[str]) -> int:
    max_lab = -1
    for cp in csv_paths:
        ds = S2GeoTiffCSVDataset(
            csv_path=cp,
            bands=[2, 3, 4, 8, 12, 13],
            normalize="perimage_zscore",
            target_hw=None,
        )
        for lab in ds.labels:
            if int(lab) > max_lab:
                max_lab = int(lab)
    if max_lab < 0:
        raise RuntimeError("Could not infer num_classes from CSV labels.")
    return int(max_lab + 1)


def _split_indices_by_hash(n_total: int, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    val_frac = float(max(0.0, min(0.9, val_frac)))
    if val_frac <= 0.0:
        return list(range(n_total)), []

    train_idx: List[int] = []
    val_idx: List[int] = []

    def h64(x: int) -> int:
        z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return z & 0xFFFFFFFFFFFFFFFF

    denom = float(2**64 - 1)
    for i in range(int(n_total)):
        u = h64(int(i) ^ (int(seed) * 1000003)) / denom
        if u < val_frac:
            val_idx.append(i)
        else:
            train_idx.append(i)

    if len(val_idx) == 0 and n_total > 1:
        val_idx.append(train_idx.pop())
    if len(train_idx) == 0 and n_total > 1:
        train_idx.append(val_idx.pop())

    return train_idx, val_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FT (DeepSpeed): supports GeoBench So2Sat + ForestNet + customcsv supervised CSV splits + Tiny-ImageNet + CIFAR-100."
    )
    parser.add_argument(
        "--ft-mode",
        type=str,
        default="scratch_ft",
        choices=["linear_probe", "full_ft", "scratch_ft"],
    )

    parser.add_argument("--dataset", type=str, default="so2sat", choices=["so2sat", "forestnet", "customcsv", "tinyimagenet", "cifar100"])
    parser.add_argument("--tiny-imagenet-root", type=str, default="")
    parser.add_argument("--cifar100-root", type=str, default="")

    parser.add_argument("--csv-train", type=str, default="")
    parser.add_argument("--csv-val", type=str, default="")
    parser.add_argument("--csv-test", type=str, default="")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--val-frac", type=float, default=0.1)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.001)

    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--lr-moe", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-scale", type=float, default=0.05)
    parser.add_argument("--clean-train-eval-batches", type=int, default=25)

    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--head-dropout", type=float, default=-1.0)
    parser.add_argument("--rgb-aug-hflip", type=float, default=-1.0)
    parser.add_argument("--rgb-aug-crop-pad", type=int, default=-1)
    parser.add_argument("--mixup-alpha", type=float, default=-1.0)
    parser.add_argument("--cutmix-alpha", type=float, default=-1.0)
    parser.add_argument("--mix-prob", type=float, default=-1.0)
    parser.add_argument("--mix-switch-prob", type=float, default=-1.0)
    parser.add_argument("--geo-aug-hflip", type=float, default=-1.0)
    parser.add_argument("--geo-aug-vflip", type=float, default=-1.0)
    parser.add_argument("--geo-aug-rot90", type=float, default=-1.0)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)

    parser.add_argument("--img-h", type=int, default=32)
    parser.add_argument("--img-w", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--enc-heads", type=int, default=8)

    parser.add_argument("--ckpt-dir", type=str, default="ckpts_pretrain")
    parser.add_argument("--tag", type=str, default="latest")

    parser.add_argument("--ft-ckpt-dir", type=str, default="ckpts_finetune")

    args = deepspeed.add_config_arguments(parser).parse_args()
    conf = json.load(open(args.deepspeed_config, "r"))
    print(conf)

    moe_conf = conf.get("moe", {})
    moe_num_experts = int(moe_conf.get("num_experts", 0))
    moe_ep_size     = int(moe_conf.get("ep_size", 1))
    moe_topk        = int(moe_conf.get("top_k", 1))
    moe_capacity    = float(moe_conf.get("capacity_factor", 1.25))
    moe_noisy_gate  = moe_conf.get("noisy_gate_policy", "None")
    moe_aux_weight  = float(moe_conf.get("moe_aux_weight", 0.0))

    set_seed(args.seed)

    os.environ["MASTER_ADDR"] = str(os.environ.get("HOSTNAME", "127.0.0.1"))
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"]  = os.environ["SLURM_NTASKS"]
    os.environ["RANK"]        = os.environ["SLURM_PROCID"]

    world_size  = int(os.environ["SLURM_NTASKS"])
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank  = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    deepspeed.init_distributed()

    dist_rank = torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0
    is_rank0 = (dist_rank == 0)

    ep_size = max(1, moe_ep_size)
    dp_world_size = world_size
    dp_rank = global_rank

    if is_rank0:
        print(f"[FT] dist_rank={dist_rank} local_rank={local_rank} world={world_size} ep_size={ep_size}", flush=True)
        print(f"[FT] Mode: {args.ft_mode}", flush=True)
        print(f"[FT] Dataset: {args.dataset}", flush=True)
        print(
            f"[FT] MoE cfg: enabled={bool(moe_conf.get('enabled', False))} "
            f"num_experts={moe_num_experts} top_k={moe_topk} ep_size={ep_size} "
            f"capacity_factor={moe_capacity} noisy_gate={moe_noisy_gate} moe_aux_weight={moe_aux_weight}",
            flush=True,
        )

    TARGET_HW = (int(args.img_h), int(args.img_w))
    if TARGET_HW[0] % int(args.patch) != 0 or TARGET_HW[1] % int(args.patch) != 0:
        raise RuntimeError(f"[FT] img_hw={TARGET_HW} must be divisible by patch={args.patch}")

    rgb_recipe = resolve_rgb_recipe(args)
    geo_recipe = resolve_geo_recipe(args)

    if is_rank0:
        print(
            f"[FT] RGB recipe: hflip={rgb_recipe['hflip']} crop_pad={int(rgb_recipe['crop_pad'])} "
            f"mixup_alpha={rgb_recipe['mixup_alpha']} cutmix_alpha={rgb_recipe['cutmix_alpha']} "
            f"mix_prob={rgb_recipe['mix_prob']} mix_switch_prob={rgb_recipe['mix_switch_prob']} "
            f"label_smoothing={rgb_recipe['label_smoothing']} encoder_dropout={rgb_recipe['encoder_dropout']} "
            f"head_dropout={rgb_recipe['head_dropout']}",
            flush=True,
        )

    if args.dataset in ("so2sat", "forestnet"):
        if args.dataset == "so2sat":
            dataset_name = "m-so2sat"
        else:
            dataset_name = "m-forestnet"

        task = next(t for t in geobench.task_iterator("classification_v1.0") if t.dataset_name == dataset_name)
        base_train = task.get_dataset(split="train")
        base_test  = task.get_dataset(split="test")

        ds_train_full = GeoBench6Band(
            base_train,
            rank=global_rank,
            split=f"{dataset_name}/train",
            dataset_kind=args.dataset,
            target_hw=TARGET_HW,
        )
        ds_test = GeoBench6Band(
            base_test,
            rank=global_rank,
            split=f"{dataset_name}/test",
            dataset_kind=args.dataset,
            target_hw=TARGET_HW,
        )

        num_classes = int(ds_train_full.num_classes)
        metric_name = "acc"
        in_ch = 6

        if is_rank0:
            x0, y0 = ds_train_full[0]
            print(f"[FT] Example x shape={tuple(x0.shape)} label={y0} num_classes={num_classes}", flush=True)

        n_total = len(ds_train_full)
        val_frac = float(args.val_frac)
        val_frac = max(0.0, min(0.9, val_frac))

        if val_frac > 0.0:
            if is_rank0:
                train_idx, val_idx = _split_indices_stratified(labels=extract_single_labels(ds_train_full), val_frac=val_frac, seed=int(args.seed) + 12345)
            else:
                train_idx, val_idx = None, None

            train_idx = ddp_broadcast_object(train_idx, src=0)
            val_idx = ddp_broadcast_object(val_idx, src=0)

            ds_train = Subset(ds_train_full, train_idx)
            ds_val = Subset(ds_train_full, val_idx)

            if is_rank0:
                print(f"[FT] Train/Val split (stratified): n_total={n_total} n_train={len(ds_train)} n_val={len(ds_val)} (val_frac={val_frac})", flush=True)
        else:
            ds_train = ds_train_full
            ds_val = None
            if is_rank0:
                print(f"[FT] Train/Val split disabled (val_frac={val_frac}).", flush=True)

    elif args.dataset == "tinyimagenet":
        if args.tiny_imagenet_root.strip() == "":
            raise ValueError("[FT][tinyimagenet] You must provide --tiny-imagenet-root.")

        ds_train_full = TinyImageNetDataset(
            root=args.tiny_imagenet_root,
            split="train",
            target_hw=TARGET_HW,
            normalize_mode="standard",
        )
        ds_test = TinyImageNetDataset(
            root=args.tiny_imagenet_root,
            split="val",
            target_hw=TARGET_HW,
            normalize_mode="standard",
        )

        num_classes = int(ds_train_full.num_classes)
        metric_name = "acc"
        in_ch = 3

        if is_rank0:
            x0, y0 = ds_train_full[0]
            print(f"[FT][tinyimagenet] ROOT={args.tiny_imagenet_root}", flush=True)
            print(f"[FT][tinyimagenet] Example x shape={tuple(x0.shape)} label={int(y0)} num_classes={num_classes}", flush=True)
            print(f"[FT][tinyimagenet] Using val/ as TEST because Tiny-ImageNet test labels are not published.", flush=True)
            print(f"[FT][tinyimagenet] RGB normalization: ImageNet mean/std.", flush=True)

        n_total = len(ds_train_full)
        val_frac = float(args.val_frac)
        val_frac = max(0.0, min(0.9, val_frac))

        if val_frac > 0.0:
            if is_rank0:
                train_idx, val_idx = _split_indices_stratified(labels=ds_train_full.labels, val_frac=val_frac, seed=int(args.seed) + 12345)
            else:
                train_idx, val_idx = None, None

            train_idx = ddp_broadcast_object(train_idx, src=0)
            val_idx = ddp_broadcast_object(val_idx, src=0)

            ds_train = Subset(ds_train_full, train_idx)
            ds_val = Subset(ds_train_full, val_idx)

            if is_rank0:
                print(f"[FT][tinyimagenet] Train/Val split (stratified): n_total={n_total} n_train={len(ds_train)} n_val={len(ds_val)} (val_frac={val_frac})", flush=True)
        else:
            ds_train = ds_train_full
            ds_val = None
            if is_rank0:
                print(f"[FT][tinyimagenet] Train/Val split disabled (val_frac={val_frac}).", flush=True)

    elif args.dataset == "cifar100":
        if args.cifar100_root.strip() == "":
            raise ValueError("[FT][cifar100] You must provide --cifar100-root.")

        ds_train_full = CIFAR100Dataset(
            root=args.cifar100_root,
            split="train",
            target_hw=TARGET_HW,
            normalize_mode="standard",
        )
        ds_test = CIFAR100Dataset(
            root=args.cifar100_root,
            split="test",
            target_hw=TARGET_HW,
            normalize_mode="standard",
        )

        num_classes = int(ds_train_full.num_classes)
        metric_name = "acc"
        in_ch = 3

        if is_rank0:
            x0, y0 = ds_train_full[0]
            print(f"[FT][cifar100] ROOT={args.cifar100_root}", flush=True)
            print(f"[FT][cifar100] Example x shape={tuple(x0.shape)} label={int(y0)} num_classes={num_classes}", flush=True)
            print(f"[FT][cifar100] Using official test split as TEST.", flush=True)
            print(f"[FT][cifar100] RGB normalization: CIFAR-100 mean/std.", flush=True)

        n_total = len(ds_train_full)
        val_frac = float(args.val_frac)
        val_frac = max(0.0, min(0.9, val_frac))

        if val_frac > 0.0:
            if is_rank0:
                train_idx, val_idx = _split_indices_stratified(labels=ds_train_full.labels, val_frac=val_frac, seed=int(args.seed) + 12345)
            else:
                train_idx, val_idx = None, None

            train_idx = ddp_broadcast_object(train_idx, src=0)
            val_idx = ddp_broadcast_object(val_idx, src=0)

            ds_train = Subset(ds_train_full, train_idx)
            ds_val = Subset(ds_train_full, val_idx)

            if is_rank0:
                print(f"[FT][cifar100] Train/Val split (stratified): n_total={n_total} n_train={len(ds_train)} n_val={len(ds_val)} (val_frac={val_frac})", flush=True)
        else:
            ds_train = ds_train_full
            ds_val = None
            if is_rank0:
                print(f"[FT][cifar100] Train/Val split disabled (val_frac={val_frac}).", flush=True)

    else:
        if (args.csv_train.strip() == "") or (args.csv_val.strip() == "") or (args.csv_test.strip() == ""):
            raise ValueError("[FT][customcsv] You must provide --csv-train --csv-val --csv-test (fixed splits).")

        bands_custom = [2, 3, 4, 8, 12, 13]

        ds_train = S2GeoTiffCSVDataset(
            csv_path=args.csv_train,
            bands=bands_custom,
            normalize="perimage_zscore",
            target_hw=TARGET_HW,
        )
        ds_val = S2GeoTiffCSVDataset(
            csv_path=args.csv_val,
            bands=bands_custom,
            normalize="perimage_zscore",
            target_hw=TARGET_HW,
        )
        ds_test = S2GeoTiffCSVDataset(
            csv_path=args.csv_test,
            bands=bands_custom,
            normalize="perimage_zscore",
            target_hw=TARGET_HW,
        )

        metric_name = "acc"
        in_ch = 6

        if is_rank0:
            num_classes0 = _compute_num_classes_from_csvs_rank0([args.csv_train, args.csv_val, args.csv_test])
        else:
            num_classes0 = None
        num_classes = int(ddp_broadcast_object(num_classes0, src=0))

        if is_rank0:
            x0, y0 = ds_train[0]
            print(f"[FT][customcsv] TRAIN={args.csv_train} VAL={args.csv_val} TEST={args.csv_test}", flush=True)
            print(f"[FT][customcsv] N_train={len(ds_train)} N_val={len(ds_val)} N_test={len(ds_test)} num_classes={num_classes}", flush=True)
            print(f"[FT][customcsv] Example x shape={tuple(x0.shape)} label={int(y0)} bands_rasterio_1idx={bands_custom}", flush=True)

    ds_train_clean_eval = ds_train

    if args.dataset in ("tinyimagenet", "cifar100"):
        ds_train = RGBTrainAugmentDataset(
            ds_train,
            hflip_prob=rgb_recipe["hflip"],
            crop_padding=int(rgb_recipe["crop_pad"]),
        )

    encoder = TinyViTEncoder(
        in_ch=in_ch,
        embed_dim=args.embed_dim,
        depth=args.enc_depth,
        num_heads=args.enc_heads,
        patch=args.patch,
        img_hw=(args.img_h, args.img_w),
        mlp_ratio=args.mlp_ratio,
        dropout=rgb_recipe["encoder_dropout"],
        moe_num_experts=max(0, moe_num_experts),
        ep_world_size=ep_size,
        top_k=moe_topk,
        capacity_factor=moe_capacity,
        noisy_gate_policy=moe_noisy_gate,
        num_modalities=0,
    ).to(device)

    mae = EncoderOnlyMAE(encoder=encoder).to(device)
    model = FTModel(
        mae=mae,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        ft_mode=args.ft_mode,
        moe_aux_weight=moe_aux_weight,
        head_dropout=rgb_recipe["head_dropout"],
    ).to(device)

    head_params = list(model.head.parameters())
    moe_params_all, has_moe = find_moe_params(model)

    head_param_ids = {id(p) for p in head_params}
    moe_param_ids  = {id(p) for p in moe_params_all}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids and id(p) not in moe_param_ids]

    groups = [{"name": "head", "params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay}]

    if backbone_params:
        if args.ft_mode == "linear_probe":
            lr_backbone = 0.0
            wd_backbone = 0.0
        else:
            lr_backbone = args.lr_backbone
            wd_backbone = args.weight_decay
        groups.append({"name": "backbone_dense", "params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone})

    if has_moe and moe_params_all:
        lr_moe = 0.0 if args.ft_mode == "linear_probe" else args.lr_moe
        groups.append({"name": f"ep_size_{ep_size}", "params": moe_params_all, "lr": lr_moe, "weight_decay": args.weight_decay, "moe": True})

    if is_rank0:
        total_params, trainable_params = count_parameters(model)
        print(f"[FT] Parameters: total={total_params:,} trainable={trainable_params:,}", flush=True)

    engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=groups)

    if args.ft_mode in ("linear_probe", "full_ft"):
        load_path, _ = engine.load_checkpoint(
            args.ckpt_dir,
            tag=args.tag,
            load_module_strict=False,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
        if is_rank0:
            print(f"[FT] DeepSpeed load_checkpoint OK from: {load_path} (tag={args.tag})", flush=True)

    run_id = os.environ.get("SLURM_JOB_ID", None)
    if run_id is None or str(run_id).strip() == "":
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
    run_id = str(run_id)
    args.ft_ckpt_dir = os.path.join(args.ft_ckpt_dir, run_id)

    plots_dir = os.path.join(args.ft_ckpt_dir, f"plots_{args.dataset}")
    if is_rank0:
        os.makedirs(args.ft_ckpt_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    torch.distributed.barrier()

    train_batch_size = int(conf["train_micro_batch_size_per_gpu"])
    if args.steps and int(args.steps) > 0:
        total_sched_steps = int(args.steps)
    else:
        steps_per_epoch_sched = int(math.ceil(float(len(ds_train)) / float(max(1, train_batch_size * dp_world_size))))
        total_sched_steps = int(max(1, steps_per_epoch_sched * int(args.epochs)))
    warmup_steps = int(round(float(args.warmup_epochs) * float(total_sched_steps) / float(max(1, int(args.epochs)))))
    lr_sched_state = build_cosine_lr_scheduler_state(
        base_lrs=[float(pg.get("lr", 0.0)) for pg in optimizer.param_groups],
        warmup_steps=warmup_steps,
        total_steps=total_sched_steps,
        min_lr_scale=float(args.min_lr_scale),
    )
    apply_lr_scale(optimizer, lr_sched_state, step_idx=0)

    if is_rank0:
        print(
            f"[FT] LR scheduler: cosine per-step | total_steps={total_sched_steps} warmup_steps={warmup_steps} "
            f"min_lr_scale={float(args.min_lr_scale):.4f}",
            flush=True,
        )

    best_val_metric = -1.0
    best_epoch = -1
    bad_epochs = 0
    global_step = 0
    saved_step1 = False

    history = {
        "train_loss": [],
        "train_metric": [],
        "train_clean_loss": [],
        "train_clean_metric": [],
        "val_loss": [],
        "val_metric": [],
        "test_loss": [],
        "test_metric": [],
    }

    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        engine.train()

        sampler_train = DistributedSampler(ds_train, num_replicas=dp_world_size, rank=dp_rank, shuffle=True, drop_last=False)
        sampler_train.set_epoch(epoch)

        dl_train = DataLoader(
            ds_train,
            batch_size=conf["train_micro_batch_size_per_gpu"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=sampler_train,
        )

        dl_val = None
        if ds_val is not None:
            sampler_val = DistributedSampler(ds_val, num_replicas=dp_world_size, rank=dp_rank, shuffle=False, drop_last=False)
            dl_val = DataLoader(
                ds_val,
                batch_size=conf["train_micro_batch_size_per_gpu"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                sampler=sampler_val,
            )

        sampler_test = DistributedSampler(ds_test, num_replicas=dp_world_size, rank=dp_rank, shuffle=False, drop_last=False)
        dl_test = DataLoader(
            ds_test,
            batch_size=conf["train_micro_batch_size_per_gpu"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=sampler_test,
        )

        dl_train_clean = None
        if int(args.clean_train_eval_batches) > 0:
            sampler_train_clean = DistributedSampler(ds_train_clean_eval, num_replicas=dp_world_size, rank=dp_rank, shuffle=False, drop_last=False)
            dl_train_clean = DataLoader(
                ds_train_clean_eval,
                batch_size=conf["train_micro_batch_size_per_gpu"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                sampler=sampler_train_clean,
            )

        train_loss_sum_local = 0.0
        train_steps_local = 0.0
        train_correct_local = 0.0
        train_samples_local = 0.0

        L = int(args.enc_depth)
        epoch_counts_layers_global_accum: Optional[torch.Tensor] = None
        epoch_counts_sum_global_accum: Optional[torch.Tensor] = None

        for step, (x, y) in enumerate(dl_train):
            if args.steps and global_step >= args.steps:
                break

            x = x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True).long()

            if engine.fp16_enabled():
                x = x.half()
            elif engine.bfloat16_enabled():
                x = x.bfloat16()

            if args.dataset in ("so2sat", "forestnet"):
                x = apply_geo_safe_spatial_aug(
                    x=x,
                    hflip_prob=geo_recipe["hflip"],
                    vflip_prob=geo_recipe["vflip"],
                    rot90_prob=geo_recipe["rot90"],
                )

            x, y_train, mix_applied, mix_lambda, mix_mode = mixup_cutmix_batch(
                x=x,
                y=y_dev,
                num_classes=num_classes,
                mixup_alpha=rgb_recipe["mixup_alpha"],
                cutmix_alpha=rgb_recipe["cutmix_alpha"],
                mix_prob=rgb_recipe["mix_prob"],
                switch_prob=rgb_recipe["mix_switch_prob"],
            )

            loss, dbg = engine(x, y_train, label_smoothing=rgb_recipe["label_smoothing"])
            engine.backward(loss)
            engine.step()
            lr_scale = apply_lr_scale(optimizer, lr_sched_state, step_idx=global_step + 1)

            if (not saved_step1) and (global_step == 0):
                torch.distributed.barrier()
                tag_step1 = f"epoch_{epoch:03d}_gstep_{(global_step + 1):06d}"
                engine.save_checkpoint(args.ft_ckpt_dir, tag=tag_step1, save_latest=True)
                torch.distributed.barrier()
                if is_rank0:
                    print(f"[FT] Saved checkpoint to {args.ft_ckpt_dir} tag={tag_step1}", flush=True)
                saved_step1 = True

            with torch.no_grad():
                logits, _ = engine.module(x, labels=None)
                preds = logits.argmax(dim=1)
                train_correct_local += float((preds == y_dev).sum().item())
                train_samples_local += float(y_dev.numel())

            train_loss_sum_local += float(loss.item())
            train_steps_local += 1.0

            util_str = ""
            if has_moe and moe_num_experts > 0:
                enc = engine.module.mae.encoder
                layers_counts = torch.zeros(L, moe_num_experts, device=device, dtype=torch.float32)
                any_counts = False
                for li, blk in enumerate(enc.blocks):
                    ec = getattr(blk, "last_exp_counts", None)
                    if ec is None:
                        continue
                    ec_f = ec.to(device=device, dtype=torch.float32)
                    if ec_f.numel() == moe_num_experts:
                        layers_counts[li, :] = ec_f
                        any_counts = True

                if any_counts:
                    ddp_allreduce_sum(layers_counts)
                    step_sum = layers_counts.sum(dim=0)
                    util_str = format_expert_util(step_sum)

                    if epoch_counts_layers_global_accum is None:
                        epoch_counts_layers_global_accum = layers_counts.clone()
                    else:
                        epoch_counts_layers_global_accum += layers_counts

                    if epoch_counts_sum_global_accum is None:
                        epoch_counts_sum_global_accum = step_sum.clone()
                    else:
                        epoch_counts_sum_global_accum += step_sum

            if is_rank0 and util_str:
                aux_s = ""
                if isinstance(dbg, dict) and ("moe_aux" in dbg) and ("moe_aux_w" in dbg):
                    aux_s = f" aux={dbg['moe_aux']:.4f} aux_w={dbg['moe_aux_w']:.3f}"
                print(
                    f"[FT] epoch={epoch} step={step} gstep={global_step} loss={loss.item():.4f}{aux_s} lr_scale={lr_scale:.4f} {util_str}",
                    flush=True,
                )

            global_step += 1

        train_loss_sum, train_steps = ddp_sum_floats(device, train_loss_sum_local, train_steps_local)
        train_loss = train_loss_sum / max(train_steps, 1.0)

        train_correct, train_samples = ddp_sum_floats(device, train_correct_local, train_samples_local)
        train_metric = train_correct / max(train_samples, 1.0)

        train_clean_loss, train_clean_metric = (float("nan"), float("nan"))
        if dl_train_clean is not None:
            train_clean_loss, train_clean_metric = eval_split_limited(
                engine,
                dl_train_clean,
                device,
                rgb_recipe["label_smoothing"] if args.dataset in ("tinyimagenet", "cifar100") else args.label_smoothing,
                max_batches=int(args.clean_train_eval_batches),
            )

        torch.distributed.barrier()

        val_loss, val_metric = (float("nan"), float("nan"))
        if dl_val is not None:
            val_loss, val_metric = eval_split(engine, dl_val, device, args.label_smoothing)

        test_loss, test_metric = eval_split(engine, dl_test, device, args.label_smoothing)

        epoch_time = time.time() - epoch_t0

        if has_moe and moe_num_experts > 0:
            if epoch_counts_sum_global_accum is None:
                epoch_counts_sum_global_accum = torch.zeros(moe_num_experts, device=device, dtype=torch.float32)

            summ = epoch_util_summary(epoch_counts_sum_global_accum)
            topk = epoch_topk_str_from_counts(epoch_counts_sum_global_accum, k=8)

            if is_rank0:
                print(
                    "FT_EPOCH_UTIL_SUM "
                    f"epoch={epoch} tokens={summ['tokens']:.0f} active={summ['active']} "
                    f"top={summ['top_idx']} top_pct={summ['top_pct']:.2f} "
                    f"max_pct={summ['max_pct']:.2f} min_pct={summ['min_pct']:.2f} "
                    f"entropy={summ['entropy']:.3f} gini={summ['gini']:.3f} cv={summ['cv']:.3f}",
                    flush=True,
                )
                print(f"FT_EPOCH_UTIL_TOPK epoch={epoch} top8={topk}", flush=True)

            if is_rank0 and (epoch_counts_layers_global_accum is not None):
                layers_np = epoch_counts_layers_global_accum.detach().cpu().numpy().astype(np.float64, copy=False)
                sum_np = layers_np.sum(axis=0)

                for li in range(layers_np.shape[0]):
                    pct = _counts_to_pct(layers_np[li])
                    plot_expert_occupancy_bars(
                        out_path=os.path.join(plots_dir, f"ft_expert_occupancy_gate{li:02d}.png"),
                        pct=pct,
                        title=f"FT expert occupancy (%) | gate(layer)={li} | latest(epoch={epoch})",
                    )

                pct_sum = _counts_to_pct(sum_np)
                plot_expert_occupancy_bars(
                    out_path=os.path.join(plots_dir, "ft_expert_occupancy_SUM.png"),
                    pct=pct_sum,
                    title=f"FT expert occupancy (%) | SUM(all gates) | latest(epoch={epoch})",
                )

        score = val_metric if (dl_val is not None) else test_metric

        if is_rank0:
            improved = (score - best_val_metric) > float(args.min_delta)
            if improved:
                best_val_metric = score
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            train_clean_s = ""
            if not math.isnan(train_clean_metric):
                train_clean_s = f" | TrainClean loss {train_clean_loss:.4f} {metric_name} {train_clean_metric*100:.2f}%"

            print(
                f"[FT] Epoch {epoch:03d} | time {epoch_time:.2f}s | "
                f"TrainMix loss {train_loss:.4f} {metric_name} {train_metric*100:.2f}%"
                f"{train_clean_s} | "
                f"Val   loss {val_loss:.4f} {metric_name} {val_metric*100:.2f}% | "
                f"Test  loss {test_loss:.4f} {metric_name} {test_metric*100:.2f}% | "
                f"best(score) {best_val_metric*100:.2f}% (epoch {best_epoch}) bad {bad_epochs}/{args.patience}",
                flush=True,
            )

        torch.distributed.barrier()

        tag = f"epoch_{epoch:03d}_gstep_{global_step:06d}"
        engine.save_checkpoint(args.ft_ckpt_dir, tag=tag, save_latest=True)
        torch.distributed.barrier()
        if is_rank0:
            print(f"[FT] Saved checkpoint to {args.ft_ckpt_dir} tag={tag}", flush=True)

        if is_rank0:
            history["train_loss"].append(float(train_loss))
            history["train_metric"].append(float(train_metric))
            history["train_clean_loss"].append(float(train_clean_loss))
            history["train_clean_metric"].append(float(train_clean_metric))
            history["val_loss"].append(float(val_loss))
            history["val_metric"].append(float(val_metric))
            history["test_loss"].append(float(test_loss))
            history["test_metric"].append(float(test_metric))
            save_training_plots(out_dir=plots_dir, history=history, metric_name=metric_name)

        local_stop = 1 if (is_rank0 and bad_epochs >= int(args.patience)) else 0
        stop_t = torch.tensor([local_stop], device=device, dtype=torch.int32)
        ddp_allreduce_sum(stop_t)

        if int(stop_t.item()) >= 1:
            if is_rank0:
                print(f"[FT] Early stopping at epoch {epoch}. Best score={best_val_metric*100:.2f}% at epoch {best_epoch}.", flush=True)
            break

    torch.distributed.barrier()
    if is_rank0:
        print(f"[FT] Done. (Per-epoch checkpoint saving ENABLED in: {args.ft_ckpt_dir})", flush=True)
        print(f"[FT] Plots saved under: {plots_dir}", flush=True)

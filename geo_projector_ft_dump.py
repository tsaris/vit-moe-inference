import os
import re
import argparse
import random
import json
from typing import Optional, Tuple, List, Dict

import pickle
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import deepspeed
import geobench

from geo_projector_lib import TinyViTEncoder, S2GeoTiffCSVDataset


try:
    from deepspeed.moe.layer import MoE as DS_MoE
    _HAS_DS_MOE = True
except Exception:
    DS_MoE = None
    _HAS_DS_MOE = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dist_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def ddp_broadcast_object(obj, src: int = 0):
    if not dist_ready():
        return obj
    obj_list = [obj] if torch.distributed.get_rank() == src else [None]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def ddp_gather_object_to_rank0(obj):
    if not dist_ready():
        return [obj]
    world = torch.distributed.get_world_size()
    out = [None for _ in range(world)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(obj, out, dst=0)
    return out




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

def parse_epoch_gstep_from_tag(tag: str) -> Tuple[str, str]:
    m_e = re.search(r"epoch[_-]?(\d+)", str(tag))
    m_g = re.search(r"gstep[_-]?(\d+)", str(tag))
    epoch_s = m_e.group(1) if m_e else "NA"
    gstep_s = m_g.group(1) if m_g else "NA"
    return epoch_s, gstep_s


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


class TinyImageNetDataset(torch.utils.data.Dataset):

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __init__(
        self,
        root: str,
        split: str,
        target_hw: Optional[Tuple[int, int]] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = str(split)
        self.target_hw = target_hw

        if self.split not in ("train", "val"):
            raise ValueError(f"TinyImageNetDataset supports split=train/val only, got {self.split}")

        wnids_path = os.path.join(self.root, "wnids.txt")
        if os.path.isfile(wnids_path):
            with open(wnids_path, "r", encoding="utf-8") as f:
                self.wnids = [ln.strip() for ln in f if ln.strip() != ""]
        else:
            train_dir = os.path.join(self.root, "train")
            self.wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

        if len(self.wnids) == 0:
            raise RuntimeError(f"No Tiny-ImageNet classes found under {self.root}")

        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        self.files: List[str] = []
        self.labels: List[int] = []

        if self.split == "train":
            train_dir = os.path.join(self.root, "train")
            for wnid in self.wnids:
                img_dir = os.path.join(train_dir, wnid, "images")
                if not os.path.isdir(img_dir):
                    continue
                exts = ("*.JPEG", "*.jpeg", "*.jpg", "*.png")
                paths = []
                for patt in exts:
                    paths.extend(sorted(str(p) for p in Path(img_dir).glob(patt)))
                cls_idx = int(self.wnid_to_idx[wnid])
                for p in paths:
                    self.files.append(p)
                    self.labels.append(cls_idx)
        else:
            ann_path = os.path.join(self.root, "val", "val_annotations.txt")
            img_dir = os.path.join(self.root, "val", "images")
            if not os.path.isfile(ann_path):
                raise FileNotFoundError(ann_path)
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(img_dir)

            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
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
            raise RuntimeError(f"No Tiny-ImageNet samples found for split={self.split} under {self.root}")

        self.num_classes = int(len(self.wnids))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        y = int(self.labels[idx])

        with Image.open(path) as img:
            img = img.convert("RGB")
            x_np = np.asarray(img, dtype=np.float32) / 255.0

        if x_np.ndim != 3 or x_np.shape[2] != 3:
            raise RuntimeError(f"[TinyImageNetDataset] Expected HxWx3 RGB image, got {x_np.shape} for {path}")

        x = torch.from_numpy(np.transpose(x_np, (2, 0, 1)))

        if self.target_hw is not None and tuple(x.shape[-2:]) != tuple(self.target_hw):
            raise RuntimeError(
                f"[TinyImageNetDataset] Shape mismatch: {path} got {tuple(x.shape[-2:])}, expected {tuple(self.target_hw)}. "
                f"Interpolation is DISABLED."
            )

        x = (x - self.IMAGENET_MEAN) / self.IMAGENET_STD

        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            mn = torch.nanmin(x).item()
            mx = torch.nanmax(x).item()
            raise RuntimeError(
                f"[TinyImageNetDataset] Non-finite tensor idx={idx} bad={bad} min={mn} max={mx} file={path}"
            )

        return x, y

class CIFAR100Dataset(torch.utils.data.Dataset):

    CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408], dtype=torch.float32).view(3, 1, 1)
    CIFAR100_STD = torch.tensor([0.2675, 0.2565, 0.2761], dtype=torch.float32).view(3, 1, 1)

    def __init__(
        self,
        root: str,
        split: str,
        target_hw: Optional[Tuple[int, int]] = None,
    ):
        self.root = os.path.abspath(root)
        self.split = str(split)
        self.target_hw = target_hw

        if self.split not in ("train", "test"):
            raise ValueError(f"CIFAR100Dataset supports split=train/test only, got {self.split}")

        self.base_dir = self._resolve_base_dir(self.root)
        self.meta = self._load_pickle(os.path.join(self.base_dir, "meta"))

        fine_names = self.meta.get(b"fine_label_names", self.meta.get("fine_label_names", None))
        if fine_names is None:
            raise RuntimeError(f"[CIFAR100Dataset] Could not find fine_label_names in {os.path.join(self.base_dir, 'meta')}")

        self.class_names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in fine_names
        ]
        self.num_classes = int(len(self.class_names))

        data_obj = self._load_pickle(os.path.join(self.base_dir, self.split))
        data = data_obj.get(b"data", data_obj.get("data", None))
        labels = data_obj.get(b"fine_labels", data_obj.get("fine_labels", None))
        if data is None or labels is None:
            raise RuntimeError(
                f"[CIFAR100Dataset] Missing data/fine_labels in {os.path.join(self.base_dir, self.split)}"
            )

        data = np.asarray(data, dtype=np.uint8)
        labels = np.asarray(labels, dtype=np.int64)

        if data.ndim != 2 or data.shape[1] != 3072:
            raise RuntimeError(
                f"[CIFAR100Dataset] Expected flat data [N,3072], got shape={data.shape} in split={self.split}"
            )

        self.data = data.reshape(-1, 3, 32, 32)
        self.labels = labels.tolist()

        if self.data.shape[0] != len(self.labels):
            raise RuntimeError(
                f"[CIFAR100Dataset] Data/label mismatch: N_data={self.data.shape[0]} N_labels={len(self.labels)}"
            )

    def _resolve_base_dir(self, root: str) -> str:
        direct = root
        nested = os.path.join(root, "cifar-100-python")

        def ok(p: str) -> bool:
            return all(os.path.isfile(os.path.join(p, nm)) for nm in ("train", "test", "meta"))

        if ok(direct):
            return direct
        if ok(nested):
            return nested
        raise FileNotFoundError(
            f"[CIFAR100Dataset] Could not find CIFAR-100 files under root={root}. "
            f"Expected either root/train,test,meta or root/cifar-100-python/train,test,meta."
        )

    def _load_pickle(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            try:
                return pickle.load(f, encoding="bytes")
            except TypeError:
                return pickle.load(f)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x_np = self.data[idx].astype(np.float32) / 255.0
        y = int(self.labels[idx])

        x = torch.from_numpy(x_np)

        if self.target_hw is not None and tuple(x.shape[-2:]) != tuple(self.target_hw):
            raise RuntimeError(
                f"[CIFAR100Dataset] Shape mismatch idx={idx} got {tuple(x.shape[-2:])}, expected {tuple(self.target_hw)}. "
                f"Interpolation is DISABLED."
            )

        x = (x - self.CIFAR100_MEAN) / self.CIFAR100_STD

        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            mn = torch.nanmin(x).item()
            mx = torch.nanmax(x).item()
            raise RuntimeError(
                f"[CIFAR100Dataset] Non-finite tensor idx={idx} bad={bad} min={mn} max={mx}"
            )

        return x, y

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

        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(self.std_floor)
        x = (x - mean) / std

        lab = getattr(sample, "label", None)
        y = int(lab)
        return x, y


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.is_subset = isinstance(ds, Subset)
        self.sub_indices = ds.indices if self.is_subset else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i: int):
        if self.is_subset:
            base_idx = int(self.sub_indices[i])
        else:
            base_idx = int(i)
        x, y = self.ds[i]
        return x, y, base_idx


class EncoderOnlyMAE(nn.Module):
    def __init__(self, encoder: TinyViTEncoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x: torch.Tensor):
        return self.encoder.encode(x, mod_ids=None)


class FTModel(nn.Module):
    def __init__(self, mae: EncoderOnlyMAE, embed_dim: int, num_classes: int, ft_mode: str = "full_ft"):
        super().__init__()
        self.mae = mae
        self.head = nn.Linear(int(embed_dim), int(num_classes))
        self.ft_mode = str(ft_mode)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        if self.ft_mode == "linear_probe":
            with torch.no_grad():
                tokens = self.mae.encode(x)
        else:
            tokens = self.mae.encode(x)

        feats = tokens.mean(dim=1)
        logits = self.head(feats)

        if labels is None:
            return logits, {}

        labels = labels.to(x.device, non_blocking=True).long()
        loss = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
        return loss, {"cls_loss": float(loss.item())}


def list_moe_modules_in_order(module: nn.Module) -> List[DS_MoE]:
    if not _HAS_DS_MOE:
        return []
    out = []
    for m in module.modules():
        if isinstance(m, DS_MoE):
            out.append(m)
    return out


def patch_one_moe_gate_force_expert(moe_mod: DS_MoE, expert_id: int) -> List[Tuple[nn.Module, object]]:
    patched: List[Tuple[nn.Module, object]] = []
    eid = int(expert_id)

    gate = getattr(moe_mod, "gate", None)
    if gate is None:
        dsmoe = getattr(moe_mod, "deepspeed_moe", None)
        gate = getattr(dsmoe, "gate", None) if dsmoe is not None else None

    if gate is None or (not hasattr(gate, "forward")):
        return []

    orig_fwd = gate.forward

    def forced_forward(x, *args, __gate=gate, __eid=eid, **kwargs):
        tokens = int(x.shape[0])

        wg = getattr(__gate, "wg", None)
        if wg is None:
            raise RuntimeError("Forced routing: gate.wg missing; cannot infer num_experts.")
        num_experts = int(getattr(wg, "out_features", 0))
        if num_experts <= 0:
            raise RuntimeError(f"Forced routing: invalid num_experts={num_experts} from gate.wg.out_features")

        if __eid < 0 or __eid >= num_experts:
            raise RuntimeError(f"expert_id={__eid} out of range for num_experts={num_experts}")

        capacity = tokens + 1
        device = x.device

        new_dispatch = torch.zeros(tokens, num_experts, capacity, device=device, dtype=torch.bool)
        new_combine = torch.zeros(tokens, num_experts, capacity, device=device, dtype=torch.float32)

        t_idx = torch.arange(tokens, device=device)
        slot = t_idx
        new_dispatch[t_idx, __eid, slot] = True
        new_combine[t_idx, __eid, slot] = 1.0

        exp_counts = torch.zeros(num_experts, device=device, dtype=torch.int32)
        exp_counts[__eid] = tokens

        l_aux = torch.tensor(0.0, device=device, dtype=torch.float32)
        return (l_aux, new_combine, new_dispatch, exp_counts)

    gate.forward = forced_forward
    patched.append((gate, orig_fwd))
    return patched


def patch_one_moe_gate_force_token_topk(
    moe_mod: DS_MoE,
    token_topk_idx_flat: torch.Tensor,
    token_topk_w_flat: torch.Tensor,
    num_experts: int,
) -> List[Tuple[nn.Module, object]]:
    patched: List[Tuple[nn.Module, object]] = []

    gate = getattr(moe_mod, "gate", None)
    if gate is None:
        dsmoe = getattr(moe_mod, "deepspeed_moe", None)
        gate = getattr(dsmoe, "gate", None) if dsmoe is not None else None

    if gate is None or (not hasattr(gate, "forward")):
        return []

    orig_fwd = gate.forward

    idx_ref = token_topk_idx_flat.detach().clone()
    w_ref = token_topk_w_flat.detach().clone()

    def forced_forward(x, *args, __idx=idx_ref, __w=w_ref, __E=int(num_experts), **kwargs):
        tokens = int(x.shape[0])
        if __idx.shape[0] != tokens:
            raise RuntimeError(f"Forced token-topk routing token mismatch: idx tokens={__idx.shape[0]} vs x tokens={tokens}")
        if __w.shape != __idx.shape:
            raise RuntimeError(f"Forced token-topk routing weight shape mismatch: idx={tuple(__idx.shape)} w={tuple(__w.shape)}")

        K = int(__idx.shape[1])
        device = x.device
        capacity = tokens + 1

        new_dispatch = torch.zeros(tokens, __E, capacity, device=device, dtype=torch.bool)
        new_combine = torch.zeros(tokens, __E, capacity, device=device, dtype=torch.float32)

        t_idx = torch.arange(tokens, device=device, dtype=torch.long)
        slot = t_idx

        idx_local = __idx.to(device=device, dtype=torch.long)
        w_local = __w.to(device=device, dtype=torch.float32)

        for j in range(K):
            ej = idx_local[:, j]
            wj = w_local[:, j]
            new_dispatch[t_idx, ej, slot] = True
            new_combine[t_idx, ej, slot] = wj

        exp_counts = torch.zeros(__E, device=device, dtype=torch.int32)
        for e in range(__E):
            exp_counts[e] = int((idx_local == e).sum().item())

        l_aux = torch.tensor(0.0, device=device, dtype=torch.float32)
        return (l_aux, new_combine, new_dispatch, exp_counts)

    gate.forward = forced_forward
    patched.append((gate, orig_fwd))
    return patched


def unpatch_all_gates(patched: List[Tuple[nn.Module, object]]):
    for gate, orig_fwd in patched:
        gate.forward = orig_fwd


class GateCaptureAllLayers:
    def __init__(self):
        self.enabled = False
        self.batch_size: Optional[int] = None
        self.probs_layers: Optional[List[torch.Tensor]] = None
        self.top1e_layers: Optional[List[torch.Tensor]] = None
        self.top1p_layers: Optional[List[torch.Tensor]] = None

    def reset_for_batch(self, batch_size: int, num_layers: int):
        self.enabled = True
        self.batch_size = int(batch_size)
        self.probs_layers = [None for _ in range(num_layers)]
        self.top1e_layers = [None for _ in range(num_layers)]
        self.top1p_layers = [None for _ in range(num_layers)]


def patch_all_moe_gates_capture_full_probs(module: nn.Module, cap: GateCaptureAllLayers) -> List[Tuple[nn.Module, object]]:
    moes = list_moe_modules_in_order(module)
    if len(moes) == 0:
        return []

    patched: List[Tuple[nn.Module, object]] = []

    for li, moe_mod in enumerate(moes):
        gate = getattr(moe_mod, "gate", None)
        if gate is None:
            dsmoe = getattr(moe_mod, "deepspeed_moe", None)
            gate = getattr(dsmoe, "gate", None) if dsmoe is not None else None
        if gate is None or (not hasattr(gate, "forward")):
            continue

        wg = getattr(gate, "wg", None)
        if wg is None or (not isinstance(wg, nn.Linear)):
            raise RuntimeError("Could not access gate.wg Linear layer to compute full router probs.")

        orig_fwd = gate.forward

        def wrapped_forward(x, *args, __orig=orig_fwd, __cap=cap, __wg=wg, __li=li, **kwargs):
            out = __orig(x, *args, **kwargs)

            if __cap.enabled:
                if __cap.batch_size is None:
                    raise RuntimeError("GateCaptureAllLayers.batch_size not set before forward.")

                logits_full = F.linear(x, __wg.weight, __wg.bias)
                probs_full = torch.softmax(logits_full.float(), dim=1)
                top_prob, top_e = torch.max(probs_full, dim=1)

                __cap.probs_layers[__li] = probs_full.detach()
                __cap.top1e_layers[__li] = top_e.detach()
                __cap.top1p_layers[__li] = top_prob.detach()

            return out

        gate.forward = wrapped_forward
        patched.append((gate, orig_fwd))

    return patched


def make_loader(ds, world_size: int, rank: int, batch_size: int, shuffle: bool):
    ds_wi = DatasetWithIndex(ds)
    sampler = DistributedSampler(ds_wi, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    return DataLoader(
        ds_wi,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
    )


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


@torch.no_grad()
def dump_split(
    engine,
    dataloader,
    device,
    num_experts: int,
    dump_path_rank0: str,
    num_classes: int,
):
    engine.eval()
    E = int(num_experts)
    if E <= 0:
        raise ValueError("dump_split requires num_experts > 0")

    moes = list_moe_modules_in_order(engine.module)
    if len(moes) == 0:
        raise RuntimeError("No DS_MoE modules found in model. Is MoE enabled in this architecture?")
    L = len(moes)
    last_layer_idx = L - 1

    cap = GateCaptureAllLayers()

    idx_local: List[int] = []
    y_local: List[int] = []

    logits_default_batches: List[np.ndarray] = []
    logits_allE_layers_batches: List[np.ndarray] = []

    logits_token_topk_weighted_layers_batches: List[np.ndarray] = []
    logits_token_topk_mean_layers_batches: List[np.ndarray] = []
    logits_token_topk_pick1_layers_batches: List[np.ndarray] = []
    logits_token_topk_oracle_layers_batches: List[np.ndarray] = []

    gate_probs_token_layers_batches: List[np.ndarray] = []
    gate_top1e_layers_batches: List[np.ndarray] = []
    gate_top1p_layers_batches: List[np.ndarray] = []

    is_rank0 = (not dist_ready()) or torch.distributed.get_rank() == 0
    printed_debug = False

    for x, y, idx in dataloader:
        x = x.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True).long()
        y_dev = y.to(device, non_blocking=True).long()

        if engine.fp16_enabled():
            x = x.half()
        elif engine.bfloat16_enabled():
            x = x.bfloat16()

        B = int(idx.numel())

        cap.reset_for_batch(B, num_layers=L)
        patched_cap = patch_all_moe_gates_capture_full_probs(engine.module, cap)
        logits_default, _ = engine.module(x, labels=None)
        unpatch_all_gates(patched_cap)

        assert cap.probs_layers is not None and cap.top1e_layers is not None and cap.top1p_layers is not None
        for li in range(L):
            if cap.probs_layers[li] is None:
                raise RuntimeError(f"Did not capture gate probs for MoE layer li={li} (out of L={L}).")

        tok_e0 = cap.top1e_layers[0]
        assert tok_e0 is not None
        tokens_total = int(tok_e0.numel())
        if tokens_total % B != 0:
            raise RuntimeError(f"Captured tokens_total={tokens_total} not divisible by batch_size={B}")
        T = tokens_total // B

        probs_b_l_t_e = []
        top1e_b_l_t = []
        top1p_b_l_t = []
        for li in range(L):
            p = cap.probs_layers[li]
            e = cap.top1e_layers[li]
            q = cap.top1p_layers[li]
            assert p is not None and e is not None and q is not None
            if p.shape[0] != tokens_total or p.shape[1] != E:
                raise RuntimeError(f"Layer {li} gate_probs shape mismatch: got {tuple(p.shape)} expected ({tokens_total},{E})")
            probs_b_l_t_e.append(p.view(B, T, E))
            top1e_b_l_t.append(e.view(B, T))
            top1p_b_l_t.append(q.view(B, T))

        gate_probs_token_layers = torch.stack(probs_b_l_t_e, dim=1)
        gate_top1e_layers = torch.stack(top1e_b_l_t, dim=1)
        gate_top1p_layers = torch.stack(top1p_b_l_t, dim=1)

        logits_default_np = logits_default.detach().cpu().to(torch.float16).numpy()

        if is_rank0 and (not printed_debug):
            print(f"[DUMP][DEBUG] First batch: B={B} L={L} T={T} E={E}", flush=True)
            printed_debug = True

        logits_layers = []
        for li in range(L):
            moe_li = moes[li]
            logits_stack_e = []
            for e in range(E):
                patched_force = patch_one_moe_gate_force_expert(moe_li, expert_id=e)
                logits_e, _ = engine.module(x, labels=None)
                unpatch_all_gates(patched_force)
                logits_stack_e.append(logits_e.float())
            logits_be_c = torch.stack(logits_stack_e, dim=1)
            logits_layers.append(logits_be_c)

        logits_b_l_e_c = torch.stack(logits_layers, dim=1)
        logits_allE_layers_np = logits_b_l_e_c.detach().cpu().to(torch.float16).numpy()

        logits_token_topk_weighted_layers = []
        logits_token_topk_mean_layers = []
        logits_token_topk_pick1_layers = []
        logits_token_topk_oracle_layers = []

        for li in range(L):
            layer_probs = gate_probs_token_layers[:, li, :, :].float()   # [B,T,E]
            topk_vals_all, topk_idx_all = torch.sort(layer_probs, dim=2, descending=True)

            logits_allE_layer_batch = logits_b_l_e_c[:, li, :, :].float()  # [B,E,C]
            probs_be = torch.softmax(logits_allE_layer_batch, dim=2)
            conf_be = probs_be.max(dim=2).values
            pred_be = logits_allE_layer_batch.argmax(dim=2)
            oracle_score_be = (pred_be == y_dev[:, None]).float()

            weighted_k_list = []
            mean_k_list = []
            pick1_k_list = []
            oracle_k_list = []

            for k in range(1, E + 1):
                idx_k = topk_idx_all[:, :, :k]
                val_k = topk_vals_all[:, :, :k]

                if k == 1:
                    w_weighted = torch.ones_like(val_k, dtype=torch.float32)
                else:
                    denom = val_k.sum(dim=2, keepdim=True).clamp_min(1e-12)
                    w_weighted = (val_k / denom).float()

                w_mean = torch.full_like(val_k, fill_value=(1.0 / float(k)), dtype=torch.float32)

                conf_topk = conf_be[:, None, :].expand(B, T, E).gather(dim=2, index=idx_k)
                best_j_conf = conf_topk.argmax(dim=2, keepdim=True)
                idx_pick1 = idx_k.gather(dim=2, index=best_j_conf)
                w_pick1 = torch.ones_like(idx_pick1, dtype=torch.float32)

                oracle_topk = oracle_score_be[:, None, :].expand(B, T, E).gather(dim=2, index=idx_k)
                best_j_oracle = oracle_topk.argmax(dim=2, keepdim=True)
                idx_oracle = idx_k.gather(dim=2, index=best_j_oracle)
                w_oracle = torch.ones_like(idx_oracle, dtype=torch.float32)

                patched_token = patch_one_moe_gate_force_token_topk(
                    moes[li],
                    token_topk_idx_flat=idx_k.reshape(B * T, k).to(device=device, dtype=torch.long),
                    token_topk_w_flat=w_weighted.reshape(B * T, k).to(device=device, dtype=torch.float32),
                    num_experts=E,
                )
                logits_k, _ = engine.module(x, labels=None)
                unpatch_all_gates(patched_token)
                weighted_k_list.append(logits_k.float())

                patched_token = patch_one_moe_gate_force_token_topk(
                    moes[li],
                    token_topk_idx_flat=idx_k.reshape(B * T, k).to(device=device, dtype=torch.long),
                    token_topk_w_flat=w_mean.reshape(B * T, k).to(device=device, dtype=torch.float32),
                    num_experts=E,
                )
                logits_k, _ = engine.module(x, labels=None)
                unpatch_all_gates(patched_token)
                mean_k_list.append(logits_k.float())

                patched_token = patch_one_moe_gate_force_token_topk(
                    moes[li],
                    token_topk_idx_flat=idx_pick1.reshape(B * T, 1).to(device=device, dtype=torch.long),
                    token_topk_w_flat=w_pick1.reshape(B * T, 1).to(device=device, dtype=torch.float32),
                    num_experts=E,
                )
                logits_k, _ = engine.module(x, labels=None)
                unpatch_all_gates(patched_token)
                pick1_k_list.append(logits_k.float())

                patched_token = patch_one_moe_gate_force_token_topk(
                    moes[li],
                    token_topk_idx_flat=idx_oracle.reshape(B * T, 1).to(device=device, dtype=torch.long),
                    token_topk_w_flat=w_oracle.reshape(B * T, 1).to(device=device, dtype=torch.float32),
                    num_experts=E,
                )
                logits_k, _ = engine.module(x, labels=None)
                unpatch_all_gates(patched_token)
                oracle_k_list.append(logits_k.float())

            logits_token_topk_weighted_layers.append(torch.stack(weighted_k_list, dim=1))
            logits_token_topk_mean_layers.append(torch.stack(mean_k_list, dim=1))
            logits_token_topk_pick1_layers.append(torch.stack(pick1_k_list, dim=1))
            logits_token_topk_oracle_layers.append(torch.stack(oracle_k_list, dim=1))

        logits_token_topk_weighted_layers = torch.stack(logits_token_topk_weighted_layers, dim=1)  # [B,L,E,C]
        logits_token_topk_mean_layers = torch.stack(logits_token_topk_mean_layers, dim=1)
        logits_token_topk_pick1_layers = torch.stack(logits_token_topk_pick1_layers, dim=1)
        logits_token_topk_oracle_layers = torch.stack(logits_token_topk_oracle_layers, dim=1)

        idx_local.extend([int(v) for v in idx.detach().cpu().tolist()])
        y_local.extend([int(v) for v in y_dev.detach().cpu().tolist()])

        logits_default_batches.append(logits_default_np)
        logits_allE_layers_batches.append(logits_allE_layers_np)

        logits_token_topk_weighted_layers_batches.append(
            logits_token_topk_weighted_layers.detach().cpu().to(torch.float16).numpy()
        )
        logits_token_topk_mean_layers_batches.append(
            logits_token_topk_mean_layers.detach().cpu().to(torch.float16).numpy()
        )
        logits_token_topk_pick1_layers_batches.append(
            logits_token_topk_pick1_layers.detach().cpu().to(torch.float16).numpy()
        )
        logits_token_topk_oracle_layers_batches.append(
            logits_token_topk_oracle_layers.detach().cpu().to(torch.float16).numpy()
        )

        gate_probs_token_layers_batches.append(gate_probs_token_layers.detach().cpu().to(torch.float16).numpy())
        gate_top1e_layers_batches.append(gate_top1e_layers.detach().cpu().to(torch.int16).numpy())
        gate_top1p_layers_batches.append(gate_top1p_layers.detach().cpu().to(torch.float16).numpy())

    local_obj = {
        "idx": idx_local,
        "y": y_local,
        "logits_default_batches": logits_default_batches,
        "logits_allE_layers_batches": logits_allE_layers_batches,

        "logits_token_topk_weighted_layers_batches": logits_token_topk_weighted_layers_batches,
        "logits_token_topk_mean_layers_batches": logits_token_topk_mean_layers_batches,
        "logits_token_topk_pick1_layers_batches": logits_token_topk_pick1_layers_batches,
        "logits_token_topk_oracle_layers_batches": logits_token_topk_oracle_layers_batches,

        "gate_probs_token_layers_batches": gate_probs_token_layers_batches,
        "gate_top1e_layers_batches": gate_top1e_layers_batches,
        "gate_top1p_layers_batches": gate_top1p_layers_batches,
        "E": int(E),
        "L": int(L),
        "num_classes": int(num_classes),
    }
    gathered = ddp_gather_object_to_rank0(local_obj)

    if (not dist_ready()) or torch.distributed.get_rank() == 0:
        all_idx: List[int] = []
        all_y: List[int] = []

        all_logits_default: List[np.ndarray] = []
        all_logits_allE_layers: List[np.ndarray] = []

        all_logits_token_topk_weighted_layers: List[np.ndarray] = []
        all_logits_token_topk_mean_layers: List[np.ndarray] = []
        all_logits_token_topk_pick1_layers: List[np.ndarray] = []
        all_logits_token_topk_oracle_layers: List[np.ndarray] = []

        all_gate_probs_token_layers: List[np.ndarray] = []
        all_gate_top1e_layers: List[np.ndarray] = []
        all_gate_top1p_layers: List[np.ndarray] = []

        for g in gathered:
            if g is None:
                continue
            all_idx.extend([int(v) for v in g["idx"]])
            all_y.extend([int(v) for v in g["y"]])

            all_logits_default.extend(g["logits_default_batches"])
            all_logits_allE_layers.extend(g["logits_allE_layers_batches"])

            all_logits_token_topk_weighted_layers.extend(g["logits_token_topk_weighted_layers_batches"])
            all_logits_token_topk_mean_layers.extend(g["logits_token_topk_mean_layers_batches"])
            all_logits_token_topk_pick1_layers.extend(g["logits_token_topk_pick1_layers_batches"])
            all_logits_token_topk_oracle_layers.extend(g["logits_token_topk_oracle_layers_batches"])

            all_gate_probs_token_layers.extend(g["gate_probs_token_layers_batches"])
            all_gate_top1e_layers.extend(g["gate_top1e_layers_batches"])
            all_gate_top1p_layers.extend(g["gate_top1p_layers_batches"])

        idx_arr = np.asarray(all_idx, dtype=np.int64)
        y_arr = np.asarray(all_y, dtype=np.int64)

        logits_default = np.concatenate(all_logits_default, axis=0)
        logits_allE_layers = np.concatenate(all_logits_allE_layers, axis=0)

        logits_token_topk_weighted_layers = np.concatenate(all_logits_token_topk_weighted_layers, axis=0)
        logits_token_topk_mean_layers = np.concatenate(all_logits_token_topk_mean_layers, axis=0)
        logits_token_topk_pick1_layers = np.concatenate(all_logits_token_topk_pick1_layers, axis=0)
        logits_token_topk_oracle_layers = np.concatenate(all_logits_token_topk_oracle_layers, axis=0)

        gate_probs_token_layers = np.concatenate(all_gate_probs_token_layers, axis=0)
        gate_top1_expert_token_layers = np.concatenate(all_gate_top1e_layers, axis=0)
        gate_top1_prob_token_layers = np.concatenate(all_gate_top1p_layers, axis=0)

        N = logits_default.shape[0]
        order = np.argsort(idx_arr)

        idx_arr = idx_arr[order]
        y_arr = y_arr[order]
        logits_default = logits_default[order]
        logits_allE_layers = logits_allE_layers[order]

        logits_token_topk_weighted_layers = logits_token_topk_weighted_layers[order]
        logits_token_topk_mean_layers = logits_token_topk_mean_layers[order]
        logits_token_topk_pick1_layers = logits_token_topk_pick1_layers[order]
        logits_token_topk_oracle_layers = logits_token_topk_oracle_layers[order]

        gate_probs_token_layers = gate_probs_token_layers[order]
        gate_top1_expert_token_layers = gate_top1_expert_token_layers[order]
        gate_top1_prob_token_layers = gate_top1_prob_token_layers[order]

        N2, L2, T, E2 = gate_probs_token_layers.shape
        if N2 != N or L2 != L or E2 != E:
            raise RuntimeError(f"gate_probs_token_layers shape mismatch: {gate_probs_token_layers.shape} vs expected (N,L,T,E)=({N},{L},*,{E})")

        if logits_token_topk_weighted_layers.shape != (N, L, E, num_classes):
            raise RuntimeError(
                f"logits_token_topk_weighted_layers shape mismatch: got {logits_token_topk_weighted_layers.shape}, "
                f"expected {(N, L, E, num_classes)}"
            )

        token_count_top1_layers = np.zeros((N, L, E), dtype=np.int32)
        for li in range(L):
            for e in range(E):
                token_count_top1_layers[:, li, e] = (gate_top1_expert_token_layers[:, li, :] == e).sum(axis=1)

        gate_probs_sample_layers = gate_probs_token_layers.astype(np.float32).mean(axis=2)
        logits_allE_last = logits_allE_layers[:, last_layer_idx, :, :]

        logits_token_topk_weighted_last = logits_token_topk_weighted_layers[:, last_layer_idx, :, :]
        logits_token_topk_mean_last = logits_token_topk_mean_layers[:, last_layer_idx, :, :]
        logits_token_topk_pick1_last = logits_token_topk_pick1_layers[:, last_layer_idx, :, :]
        logits_token_topk_oracle_last = logits_token_topk_oracle_layers[:, last_layer_idx, :, :]

        os.makedirs(os.path.dirname(dump_path_rank0), exist_ok=True)
        np.savez_compressed(
            dump_path_rank0,
            idx=idx_arr,
            y=y_arr,
            logits_default=logits_default.astype(np.float16, copy=False),
            logits_allE_layers=logits_allE_layers.astype(np.float16, copy=False),
            logits_allE_last=logits_allE_last.astype(np.float16, copy=False),

            logits_token_topk_weighted_layers=logits_token_topk_weighted_layers.astype(np.float16, copy=False),
            logits_token_topk_mean_layers=logits_token_topk_mean_layers.astype(np.float16, copy=False),
            logits_token_topk_pick1_layers=logits_token_topk_pick1_layers.astype(np.float16, copy=False),
            logits_token_topk_oracle_layers=logits_token_topk_oracle_layers.astype(np.float16, copy=False),

            logits_token_topk_last=logits_token_topk_weighted_last.astype(np.float16, copy=False),
            logits_token_topk_weighted_last=logits_token_topk_weighted_last.astype(np.float16, copy=False),
            logits_token_topk_mean_last=logits_token_topk_mean_last.astype(np.float16, copy=False),
            logits_token_topk_pick1_last=logits_token_topk_pick1_last.astype(np.float16, copy=False),
            logits_token_topk_oracle_last=logits_token_topk_oracle_last.astype(np.float16, copy=False),

            gate_probs_token_layers=gate_probs_token_layers.astype(np.float16, copy=False),
            gate_top1_expert_token_layers=gate_top1_expert_token_layers.astype(np.int16, copy=False),
            gate_top1_prob_token_layers=gate_top1_prob_token_layers.astype(np.float16, copy=False),
            token_count_top1_layers=token_count_top1_layers.astype(np.int32, copy=False),
            gate_probs_sample_layers=gate_probs_sample_layers.astype(np.float32, copy=False),
        )
        print(
            f"[DUMP] Wrote: {dump_path_rank0}\n"
            f"       idx [{idx_arr.shape[0]}], y {y_arr.shape}, logits_default {logits_default.shape}, "
            f"logits_allE_layers {logits_allE_layers.shape}, "
            f"logits_token_topk_weighted_layers {logits_token_topk_weighted_layers.shape}, "
            f"logits_token_topk_mean_layers {logits_token_topk_mean_layers.shape}, "
            f"logits_token_topk_pick1_layers {logits_token_topk_pick1_layers.shape}, "
            f"logits_token_topk_oracle_layers {logits_token_topk_oracle_layers.shape}, "
            f"gate_probs_token_layers {gate_probs_token_layers.shape}",
            flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump selector inputs for VAL/TEST from an FT checkpoint.")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--ft-ckpt-dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)

    parser.add_argument("--dataset", type=str, default="so2sat", choices=["so2sat", "forestnet", "customcsv", "tinyimagenet", "cifar100"])
    parser.add_argument("--tiny-imagenet-root", type=str, default="")
    parser.add_argument("--cifar100-root", type=str, default="")

    parser.add_argument("--csv-train", type=str, default="")
    parser.add_argument("--csv-val", type=str, default="")
    parser.add_argument("--csv-test", type=str, default="")

    parser.add_argument("--img-h", type=int, default=32)
    parser.add_argument("--img-w", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--eval-bsz", type=int, default=0)
    parser.add_argument("--dump-selector", action="store_true", help="Write selector .npz dumps for VAL/TEST.")

    args = deepspeed.add_config_arguments(parser).parse_args()
    conf = json.load(open(args.deepspeed_config, "r"))

    if args.dataset in ("so2sat", "forestnet"):
        if float(args.val_frac) <= 0.0:
            raise ValueError("--val-frac must be > 0 (VAL always enabled here for GeoBench).")

    if not args.dump_selector:
        raise ValueError("Please pass --dump-selector.")
    if not _HAS_DS_MOE:
        raise RuntimeError("DeepSpeed MoE import failed (deepspeed.moe.layer.MoE). Needed for routing capture/forcing.")

    set_seed(args.seed)

    os.environ["MASTER_ADDR"] = str(os.environ.get("HOSTNAME", "127.0.0.1"))
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS")
    os.environ["RANK"] = os.environ["SLURM_PROCID"]

    world_size = int(os.environ["SLURM_NTASKS"])
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    deepspeed.init_distributed()

    dist_rank = torch.distributed.get_rank() if dist_ready() else 0
    is_rank0 = (dist_rank == 0)

    moe_conf = conf.get("moe", {})
    moe_num_experts = int(moe_conf.get("num_experts", 0))
    moe_ep_size = int(moe_conf.get("ep_size", 1))
    moe_topk = int(moe_conf.get("top_k", 1))
    moe_capacity = float(moe_conf.get("capacity_factor", 1.25))
    moe_noisy_gate = moe_conf.get("noisy_gate_policy", "None")
    ep_size = max(1, moe_ep_size)

    if is_rank0:
        print(f"[DUMP] world={world_size} dist_rank={dist_rank} local_rank={local_rank} ep_size={ep_size}", flush=True)
        print(f"[DUMP] ckpt: dir={args.ft_ckpt_dir} tag={args.tag}", flush=True)
        print(f"[DUMP] moe: E={moe_num_experts} top_k={moe_topk} cap={moe_capacity} noisy_gate={moe_noisy_gate}", flush=True)
        print(f"[DUMP] dataset={args.dataset}", flush=True)

    TARGET_HW = (int(args.img_h), int(args.img_w))

    if args.dataset in ("so2sat", "forestnet"):
        dataset_name = "m-so2sat" if args.dataset == "so2sat" else "m-forestnet"

        task = next(t for t in geobench.task_iterator("classification_v1.0") if t.dataset_name == dataset_name)
        base_train = task.get_dataset(split="train")
        base_test = task.get_dataset(split="test")

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
        in_ch = 6

        n_total = len(ds_train_full)
        val_frac = max(0.0, min(0.9, float(args.val_frac)))

        labels_full = [int(ds_train_full[i][1]) for i in range(n_total)] if is_rank0 else None

        if is_rank0:
            train_idx, val_idx = _split_indices_stratified(
                labels=labels_full,
                val_frac=val_frac,
                seed=int(args.seed) + 12345,
            )
        else:
            train_idx = None
            val_idx = None

        train_idx = ddp_broadcast_object(train_idx, src=0)
        val_idx = ddp_broadcast_object(val_idx, src=0)

        ds_val = Subset(ds_train_full, val_idx)

        if is_rank0:
            print(
                f"[DUMP] Val split (stratified): "
                f"n_total={n_total} n_val={len(ds_val)} "
                f"(val_frac={val_frac})",
                flush=True,
            )

    elif args.dataset == "tinyimagenet":
        if args.tiny_imagenet_root.strip() == "":
            raise ValueError("[DUMP][tinyimagenet] You must provide --tiny-imagenet-root.")

        ds_train_full = TinyImageNetDataset(
            root=args.tiny_imagenet_root,
            split="train",
            target_hw=TARGET_HW,
        )
        ds_test = TinyImageNetDataset(
            root=args.tiny_imagenet_root,
            split="val",
            target_hw=TARGET_HW,
        )
        num_classes = int(ds_train_full.num_classes)
        in_ch = 3

        n_total = len(ds_train_full)
        val_frac = max(0.0, min(0.9, float(args.val_frac)))

        if is_rank0:
            train_idx, val_idx = _split_indices_stratified(
                labels=ds_train_full.labels,
                val_frac=val_frac,
                seed=int(args.seed) + 12345,
            )
        else:
            train_idx = None
            val_idx = None

        train_idx = ddp_broadcast_object(train_idx, src=0)
        val_idx = ddp_broadcast_object(val_idx, src=0)

        ds_val = Subset(ds_train_full, val_idx)

        if is_rank0:
            x0, y0 = ds_val[0]
            print(f"[DUMP][tinyimagenet] ROOT={args.tiny_imagenet_root}", flush=True)
            print(f"[DUMP][tinyimagenet] N_val={len(ds_val)} N_test={len(ds_test)} num_classes={num_classes}", flush=True)
            print(f"[DUMP][tinyimagenet] Example x shape={tuple(x0.shape)} label={int(y0)}", flush=True)
            print(f"[DUMP][tinyimagenet] Val split (stratified): n_total={n_total} n_val={len(ds_val)} (val_frac={val_frac})", flush=True)
            print(f"[DUMP][tinyimagenet] Using val/ as TEST because Tiny-ImageNet test labels are not published.", flush=True)

    elif args.dataset == "cifar100":
        if args.cifar100_root.strip() == "":
            raise ValueError("[DUMP][cifar100] You must provide --cifar100-root.")

        ds_train_full = CIFAR100Dataset(
            root=args.cifar100_root,
            split="train",
            target_hw=TARGET_HW,
        )
        ds_test = CIFAR100Dataset(
            root=args.cifar100_root,
            split="test",
            target_hw=TARGET_HW,
        )
        num_classes = int(ds_train_full.num_classes)
        in_ch = 3

        n_total = len(ds_train_full)
        val_frac = max(0.0, min(0.9, float(args.val_frac)))

        if is_rank0:
            train_idx, val_idx = _split_indices_stratified(
                labels=ds_train_full.labels,
                val_frac=val_frac,
                seed=int(args.seed) + 12345,
            )
        else:
            train_idx = None
            val_idx = None

        train_idx = ddp_broadcast_object(train_idx, src=0)
        val_idx = ddp_broadcast_object(val_idx, src=0)

        ds_val = Subset(ds_train_full, val_idx)

        if is_rank0:
            x0, y0 = ds_val[0]
            print(f"[DUMP][cifar100] ROOT={args.cifar100_root}", flush=True)
            print(f"[DUMP][cifar100] N_val={len(ds_val)} N_test={len(ds_test)} num_classes={num_classes}", flush=True)
            print(f"[DUMP][cifar100] Example x shape={tuple(x0.shape)} label={int(y0)}", flush=True)
            print(f"[DUMP][cifar100] Val split (stratified): n_total={n_total} n_val={len(ds_val)} (val_frac={val_frac})", flush=True)
            print(f"[DUMP][cifar100] Using official test split as TEST.", flush=True)

    else:
        if (args.csv_val.strip() == "") or (args.csv_test.strip() == ""):
            raise ValueError("[DUMP][customcsv] You must provide --csv-val --csv-test.")

        bands_custom = [2, 3, 4, 8, 12, 13]

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
        in_ch = 6

        if is_rank0:
            num_classes0 = _compute_num_classes_from_csvs_rank0([args.csv_val, args.csv_test])
        else:
            num_classes0 = None
        num_classes = int(ddp_broadcast_object(num_classes0, src=0))

        if is_rank0:
            x0, y0 = ds_val[0]
            print(f"[DUMP][customcsv] VAL={args.csv_val} TEST={args.csv_test}", flush=True)
            print(f"[DUMP][customcsv] N_val={len(ds_val)} N_test={len(ds_test)} num_classes={num_classes}", flush=True)
            print(f"[DUMP][customcsv] Example x shape={tuple(x0.shape)} label={int(y0)} bands_rasterio_1idx={bands_custom}", flush=True)

    bsz = int(args.eval_bsz) if int(args.eval_bsz) > 0 else int(conf["train_micro_batch_size_per_gpu"])
    dl_val = make_loader(ds_val, world_size, global_rank, bsz, shuffle=False)
    dl_test = make_loader(ds_test, world_size, global_rank, bsz, shuffle=False)

    encoder = TinyViTEncoder(
        in_ch=in_ch,
        embed_dim=args.embed_dim,
        depth=args.enc_depth,
        num_heads=args.enc_heads,
        patch=args.patch,
        img_hw=(args.img_h, args.img_w),
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        moe_num_experts=max(0, moe_num_experts),
        ep_world_size=ep_size,
        top_k=moe_topk,
        capacity_factor=moe_capacity,
        noisy_gate_policy=moe_noisy_gate,
        num_modalities=0,
    ).to(device)

    model = FTModel(
        mae=EncoderOnlyMAE(encoder=encoder),
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        ft_mode="full_ft",
    ).to(device)

    params = list(model.parameters())
    engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=params)

    load_path, _ = engine.load_checkpoint(
        args.ft_ckpt_dir,
        tag=args.tag,
        load_module_strict=True,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )

    if is_rank0:
        print(f"[DUMP] load_checkpoint OK from: {load_path}", flush=True)

    torch.distributed.barrier()

    ep_s, gs_s = parse_epoch_gstep_from_tag(args.tag)
    dump_dir = os.path.join(args.ft_ckpt_dir, f"selector_dumps_tokenfull_{args.dataset}", f"epoch{ep_s}_gstep{gs_s}_tag{args.tag}")
    os.makedirs(dump_dir, exist_ok=True)

    dump_val_path = os.path.join(dump_dir, f"dump_val_epoch{ep_s}_gstep{gs_s}_tag{args.tag}_seed{args.seed}.npz")
    dump_test_path = os.path.join(dump_dir, f"dump_test_epoch{ep_s}_gstep{gs_s}_tag{args.tag}_seed{args.seed}.npz")

    dump_split(engine, dl_val, device, moe_num_experts, dump_val_path, num_classes=num_classes)
    dump_split(engine, dl_test, device, moe_num_experts, dump_test_path, num_classes=num_classes)

    torch.distributed.barrier()
    if is_rank0:
        print(f"[DUMP] VAL  : {dump_val_path}", flush=True)
        print(f"[DUMP] TEST : {dump_test_path}", flush=True)

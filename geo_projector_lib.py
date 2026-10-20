import math
import os
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio

try:
    import deepspeed
    from deepspeed.moe.layer import MoE as DS_MoE
    _HAS_DS = True
except Exception:
    DS_MoE = None
    _HAS_DS = False


def _dist_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rglob_many(root: str, exts: Tuple[str, ...] = ("tif", "tiff")) -> List[str]:
    """Recursively list all files under `root` matching extensions."""
    rootp = Path(root)
    out: List[str] = []
    for ext in exts:
        out.extend([str(p) for p in rootp.rglob(f"*.{ext}")])
    return sorted(out)


class S2GeoTiffDataset(Dataset):

    def __init__(
        self,
        files: List[str],
        bands: List[int],
        normalize: str = "perimage_zscore",
        stats: Optional[Dict[str, List[float]]] = None,
        target_hw: Optional[Tuple[int, int]] = None,
        std_floor: float = 1e-6,
    ):
        if len(files) == 0:
            raise ValueError("Empty dataset: no files provided.")
        if len(bands) != 6:
            raise ValueError(f"Expected exactly 6 S2 bands, got {len(bands)}")

        self.files = list(files)
        self.bands = list(bands)
        self.normalize = str(normalize)
        self.stats = stats
        self.target_hw = target_hw
        self.std_floor = float(std_floor)

    def __len__(self) -> int:
        return len(self.files)

    def _apply_norm(self, x: np.ndarray) -> np.ndarray:
        norm = self.normalize

        if norm == "reflectance":
            x = x / 10000.0

        elif norm == "minmax":
            mn, mx = float(x.min()), float(x.max())
            x = (x - mn) / (mx - mn + 1e-6)

        elif norm == "zscore":
            if x.max() > 1.0:
                x = x / 10000.0
            if self.stats is None:
                raise ValueError("normalize=zscore but stats=None")
            mean = np.asarray(self.stats["mean"], dtype=np.float32)[:, None, None]
            std = np.asarray(self.stats["std"], dtype=np.float32)[:, None, None]
            x = (x - mean) / (std + 1e-6)

        elif norm == "perimage_zscore":
            mean = x.mean(axis=(1, 2), keepdims=True)
            std = x.std(axis=(1, 2), keepdims=True)
            std = np.maximum(std, self.std_floor)
            x = (x - mean) / std

        elif norm in ("none", "None", "", None):
            pass

        else:
            raise ValueError(f"Unknown normalize mode '{norm}'")

        return x

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]

        with rasterio.open(path) as src:
            arr = src.read()  # [C,H,W]

        sel = [b - 1 for b in self.bands]
        x = arr[sel].astype(np.float32)  # [6,H,W]

        if self.target_hw is not None:
            Ht, Wt = self.target_hw
            if x.shape[1] != Ht or x.shape[2] != Wt:
                raise RuntimeError(
                    f"[S2GeoTiffDataset] Shape mismatch: {path} "
                    f"got {x.shape[1]}x{x.shape[2]}, expected {Ht}x{Wt}. "
                    f"Interpolation is DISABLED."
                )

        x = self._apply_norm(x)
        t = torch.from_numpy(x)

        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).sum().item()
            mn = torch.nanmin(t).item()
            mx = torch.nanmax(t).item()
            raise RuntimeError(
                f"[S2GeoTiffDataset] Non-finite tensor idx={idx} bad={bad} "
                f"min={mn} max={mx} file={path}"
            )

        return t


class S2GeoTiffCSVDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path: str,
        bands: List[int],
        normalize: str = "perimage_zscore",
        target_hw: Optional[Tuple[int, int]] = None,
        std_floor: float = 1e-6,
        resize_if_needed: bool = True,
        resize_mode: str = "bilinear",
        resize_align_corners: bool = False,
    ):
        if len(bands) != 6:
            raise ValueError(f"Expected exactly 6 S2 bands, got {len(bands)}")

        if resize_mode not in ("bilinear", "nearest"):
            raise ValueError(f"resize_mode must be 'bilinear' or 'nearest', got {resize_mode}")

        self.csv_path = csv_path
        self.csv_dir = os.path.dirname(os.path.abspath(csv_path))
        self.bands = bands
        self.normalize = normalize
        self.target_hw = target_hw
        self.std_floor = std_floor
        self.resize_if_needed = resize_if_needed
        self.resize_mode = resize_mode
        self.resize_align_corners = resize_align_corners

        self.files: List[str] = []
        self.labels: List[int] = []

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)

        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            required_cols = {"path", "label_id"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"{csv_path} missing required columns {required_cols}. "
                    f"Found: {reader.fieldnames}"
                )

            for row in reader:
                if row is None:
                    continue

                path_s = str(row["path"]).strip()
                lab_s = str(row["label_id"]).strip()

                if path_s == "":
                    continue

                try:
                    label = int(lab_s)
                except ValueError:
                    raise ValueError(f"Invalid label_id '{lab_s}' in {csv_path}")

                if not os.path.isabs(path_s):
                    path_s = os.path.join(self.csv_dir, path_s)

                self.files.append(path_s)
                self.labels.append(label)

        if len(self.files) == 0:
            raise RuntimeError(f"No valid samples found in {csv_path}")

    def __len__(self):
        return len(self.files)

    def _normalize_np(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "perimage_zscore":
            mean = x.mean(axis=(1, 2), keepdims=True)
            std = x.std(axis=(1, 2), keepdims=True)
            std = np.maximum(std, self.std_floor)
            return (x - mean) / std
        return x

    def _resize_tensor(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        Ht, Wt = target_hw
        if x.shape[-2:] == (Ht, Wt):
            return x

        x4 = x.unsqueeze(0)
        if self.resize_mode == "nearest":
            y4 = F.interpolate(x4, size=(Ht, Wt), mode="nearest")
        else:
            y4 = F.interpolate(
                x4,
                size=(Ht, Wt),
                mode="bilinear",
                align_corners=self.resize_align_corners,
            )
        return y4.squeeze(0)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        label = self.labels[idx]

        with rasterio.open(path) as src:
            arr = src.read()  # [bands,H,W]

        sel = [b - 1 for b in self.bands]
        x = arr[sel].astype(np.float32)  # [6,H,W]

        x = self._normalize_np(x)
        x = torch.from_numpy(x)

        if self.target_hw is not None and x.shape[-2:] != self.target_hw:
            if not self.resize_if_needed:
                raise RuntimeError(
                    f"{path} shape {tuple(x.shape[-2:])} != expected {self.target_hw}"
                )
            x = self._resize_tensor(x, self.target_hw)

        return x, label


class TinyImageNetDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str,
        target_hw: Optional[Tuple[int, int]] = None,
        std_floor: float = 1e-6,
    ):
        self.root = os.path.abspath(root)
        self.split = str(split)
        self.target_hw = target_hw
        self.std_floor = float(std_floor)

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

    def __len__(self) -> int:
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

        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(self.std_floor)
        x = (x - mean) / std

        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            mn = torch.nanmin(x).item()
            mx = torch.nanmax(x).item()
            raise RuntimeError(
                f"[TinyImageNetDataset] Non-finite tensor idx={idx} bad={bad} min={mn} max={mx} file={path}"
            )

        return x, y


class CIFAR100Dataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str,
        target_hw: Optional[Tuple[int, int]] = None,
        std_floor: float = 1e-6,
    ):
        self.root = os.path.abspath(root)
        self.split = str(split)
        self.target_hw = target_hw
        self.std_floor = float(std_floor)

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

        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(self.std_floor)
        x = (x - mean) / std

        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            mn = torch.nanmin(x).item()
            mx = torch.nanmax(x).item()
            raise RuntimeError(
                f"[CIFAR100Dataset] Non-finite tensor idx={idx} bad={bad} min={mn} max={mx}"
            )

        return x, y


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos."
    y, x = torch.meshgrid(
        torch.linspace(0, 1, grid_h, device=device, dtype=dtype),
        torch.linspace(0, 1, grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    y = y.reshape(-1)
    x = x.reshape(-1)

    dim_quarter = embed_dim // 4
    omega = torch.arange(dim_quarter, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / dim_quarter))

    out_x = torch.einsum("n,d->nd", x, omega)
    out_y = torch.einsum("n,d->nd", y, omega)

    emb = torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)],
        dim=1,
    )
    return emb.unsqueeze(0)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        moe_num_experts: int = 0,
        ep_world_size: int = 1,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        noisy_gate_policy: str = "None",
    ):
        super().__init__()
        self.dim = int(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=float(drop),
        )
        self.norm2 = nn.LayerNorm(dim)

        self.total_experts = int(moe_num_experts)
        self.use_moe_ffn = bool(_HAS_DS and self.total_experts > 0)

        self.last_exp_counts: Optional[torch.Tensor] = None

        if self.use_moe_ffn:
            assert DS_MoE is not None

            hidden = int(dim * float(mlp_ratio))

            class ExpertMLP(nn.Module):
                def __init__(self, in_dim: int, hidden_dim: int, drop_p: float):
                    super().__init__()
                    self.fc1 = nn.Linear(int(in_dim), int(hidden_dim))
                    self.act = nn.GELU()
                    self.drop1 = nn.Dropout(float(drop_p))
                    self.fc2 = nn.Linear(int(hidden_dim), int(in_dim))
                    self.drop2 = nn.Dropout(float(drop_p))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.fc1(x)
                    x = self.act(x)
                    x = self.drop1(x)
                    x = self.fc2(x)
                    x = self.drop2(x)
                    return x

            self.moe = DS_MoE(
                hidden_size=int(dim),
                expert=ExpertMLP(int(dim), hidden, float(drop)),
                num_experts=self.total_experts,
                ep_size=int(max(1, ep_world_size)),
                k=int(max(1, top_k)),
                capacity_factor=float(capacity_factor),
                noisy_gate_policy=None if str(noisy_gate_policy).lower() == "none" else str(noisy_gate_policy),
            )
            self.mlp = None
        else:
            self.moe = None

            hidden = int(dim * float(mlp_ratio))
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(float(drop)),
                nn.Linear(hidden, dim),
                nn.Dropout(float(drop)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]

        h2 = self.norm2(x)

        if self.moe is None:
            self.last_exp_counts = None
            x = x + self.mlp(h2)
            return x

        B, N, D = h2.shape
        h2_flat = h2.reshape(B * N, D)

        out_flat, _, _ = self.moe(h2_flat)

        self.last_exp_counts = None
        try:
            ds = self.moe.deepspeed_moe
            ec = getattr(ds, "exp_counts", None)
            if ec is not None:
                self.last_exp_counts = ec.detach()
        except Exception:
            self.last_exp_counts = None

        out = out_flat.view(B, N, D)
        x = x + out
        return x

    def aux_balance_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.moe is None:
            return torch.tensor(0.0, device=device)

        try:
            l_aux = getattr(self.moe.deepspeed_moe, "l_aux", None)
            if l_aux is not None:
                return torch.as_tensor(l_aux, device=device, dtype=torch.float32)
        except Exception:
            pass
        return torch.tensor(0.0, device=device)


class TinyViTEncoder(nn.Module):

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch: int,
        img_hw: Tuple[int, int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        moe_num_experts: int = 0,
        ep_world_size: int = 1,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        noisy_gate_policy: str = "None",
        num_modalities: int = 0,
    ):
        super().__init__()
        H, W = img_hw
        patch = int(patch)
        assert H % patch == 0 and W % patch == 0, f"img_hw {img_hw} must be divisible by patch={patch}"

        self.patch = patch
        self.embed_dim = int(embed_dim)

        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch, bias=True)

        self.mod_embed = nn.Embedding(num_modalities, embed_dim) if int(num_modalities) > 0 else None

        self.total_experts = int(moe_num_experts)
        self.top_k = int(max(1, top_k))
        self.ep_world_size = int(max(1, ep_world_size))
        self.use_moe = bool(_HAS_DS and self.total_experts > 0)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=float(mlp_ratio),
                    drop=float(dropout),
                    moe_num_experts=self.total_experts,
                    ep_world_size=self.ep_world_size,
                    top_k=self.top_k,
                    capacity_factor=float(capacity_factor),
                    noisy_gate_policy=str(noisy_gate_policy),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.last_exp_counts: Optional[torch.Tensor] = None

    def embed_input(
        self,
        x: torch.Tensor,
        mod_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        x = self.patch_embed(x)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)

        pos = build_2d_sincos_pos_embed(self.embed_dim, Hp, Wp, x.device, x.dtype)
        x = x + pos

        if self.mod_embed is not None and mod_ids is not None:
            x = x + self.mod_embed(mod_ids).unsqueeze(1)

        return x, Hp, Wp

    def run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        counts_accum = None
        for blk in self.blocks:
            x = blk(x)
            ec = getattr(blk, "last_exp_counts", None)
            if ec is not None:
                ec_f = ec.to(device=x.device, dtype=torch.float32)
                if counts_accum is None:
                    counts_accum = ec_f.clone()
                else:
                    if counts_accum.shape == ec_f.shape:
                        counts_accum += ec_f

        self.last_exp_counts = counts_accum.detach() if counts_accum is not None else None
        x = self.norm(x)
        return x

    def forward_tokens(self, x: torch.Tensor, mod_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, _, _ = self.embed_input(x, mod_ids=mod_ids)
        return self.run_blocks(x)

    def encode(self, x: torch.Tensor, mod_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_tokens(x, mod_ids=mod_ids)

    def aux_balance_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if not self.use_moe:
            return torch.tensor(0.0, device=device)

        aux_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
        for blk in self.blocks:
            if hasattr(blk, "aux_balance_loss"):
                aux_sum = aux_sum + blk.aux_balance_loss()
        return aux_sum


class TinyMAE(nn.Module):

    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        patch: int = 4,
        img_hw: Tuple[int, int] = (128, 128),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        moe_num_experts: int = 0,
        ep_world_size: int = 1,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        noisy_gate_policy: str = "None",
        moe_aux_weight: float = 0.0,
        num_modalities: int = 0,
    ):
        super().__init__()
        H, W = img_hw
        patch = int(patch)
        assert H % patch == 0 and W % patch == 0, f"img_hw {img_hw} must be divisible by patch={patch}"

        self.patch = patch
        self.img_hw = tuple(img_hw)
        self.in_ch = int(in_ch)

        self.patch_pixels = self.in_ch * patch * patch
        self.moe_aux_weight = float(moe_aux_weight)

        self.encoder = TinyViTEncoder(
            in_ch=self.in_ch,
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            patch=self.patch,
            img_hw=self.img_hw,
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            moe_num_experts=int(moe_num_experts),
            ep_world_size=int(ep_world_size),
            top_k=int(top_k),
            capacity_factor=float(capacity_factor),
            noisy_gate_policy=str(noisy_gate_policy),
            num_modalities=int(num_modalities),
        )

        self.decoder_dim = int(decoder_dim)
        self.dec_embed = nn.Linear(int(embed_dim), self.decoder_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.dec_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.decoder_dim,
                    int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    drop=float(dropout),
                    moe_num_experts=0,
                )
                for _ in range(int(decoder_depth))
            ]
        )
        self.dec_norm = nn.LayerNorm(self.decoder_dim)
        self.pred_head = nn.Linear(self.decoder_dim, self.patch_pixels)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch
        patches = imgs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * p * p)
        return patches

    def random_masking(self, x_tokens: torch.Tensor, mask_ratio: float):
        B, N, D = x_tokens.shape
        len_keep = int(N * (1.0 - float(mask_ratio)))

        noise = torch.rand(B, N, device=x_tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x_tokens, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones(B, N, device=x_tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_vis, mask, ids_restore

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75, mod_ids: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch, W // self.patch

        target_patches = self.patchify(x)

        token_seq, _, _ = self.encoder.embed_input(x, mod_ids=mod_ids)
        x_vis, mask, ids_restore = self.random_masking(token_seq, mask_ratio)
        enc_vis = self.encoder.run_blocks(x_vis)

        dec_tokens = self.dec_embed(enc_vis)
        B_dec, N_vis, D_dec = dec_tokens.shape
        N_tot = mask.shape[1]

        mask_tokens = self.mask_token.expand(B_dec, N_tot - N_vis, D_dec)
        dec_seq = torch.cat([dec_tokens, mask_tokens], dim=1)
        dec_seq = torch.gather(dec_seq, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))

        dec_pos = build_2d_sincos_pos_embed(self.decoder_dim, Hp, Wp, dec_seq.device, dec_seq.dtype)
        dec_seq = dec_seq + dec_pos

        for blk in self.dec_blocks:
            dec_seq = blk(dec_seq)
        dec_seq = self.dec_norm(dec_seq)

        pred = self.pred_head(dec_seq)

        loss_per_patch = ((pred - target_patches) ** 2).mean(dim=-1)
        loss_mae = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)

        aux = self.encoder.aux_balance_loss()
        loss = loss_mae + self.moe_aux_weight * aux

        return loss, {"mask_ratio": float(mask.mean().item()), "aux": float(aux.item())}

    def encode(self, x: torch.Tensor, mod_ids: Optional[torch.Tensor] = None):
        return self.encoder.encode(x, mod_ids=mod_ids)

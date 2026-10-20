import os
import argparse
import random
import time
import json
import math
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from geo_projector_lib import (
    rglob_many,
    S2GeoTiffDataset,
    TinyMAE,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_ep_group_for_rank(global_rank: int, ep_size: int):
    ep_rank = global_rank % ep_size
    base = global_rank - ep_rank
    ep_group_ranks = list(range(base, base + ep_size))
    ep_group = torch.distributed.new_group(ranks=ep_group_ranks) if ep_size > 1 else None
    return ep_rank, ep_group_ranks, ep_group


class PretrainModel(torch.nn.Module):
    """
    Pretraining wrapper around TinyMAE.

    Input:
      x: [B,6,H,W]
    Output:
      loss, debug_dict
    """
    def __init__(self, mae: TinyMAE, mask_ratio: float):
        super().__init__()
        self.mae = mae
        self.mask_ratio = float(mask_ratio)

    def forward(self, x: torch.Tensor):
        return self.mae(x, mask_ratio=self.mask_ratio, mod_ids=None)


def ddp_allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


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


def resolve_pt_geo_recipe(args):
    recipe = {
        "hflip": 0.5,
        "vflip": 0.5,
        "rot90": 0.5,
    }
    if float(args.geo_aug_hflip) >= 0.0:
        recipe["hflip"] = float(args.geo_aug_hflip)
    if float(args.geo_aug_vflip) >= 0.0:
        recipe["vflip"] = float(args.geo_aug_vflip)
    if float(args.geo_aug_rot90) >= 0.0:
        recipe["rot90"] = float(args.geo_aug_rot90)
    return recipe


def compute_lr_scale(step: int, total_steps: int, warmup_steps: int, min_lr_scale: float) -> float:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    min_lr_scale = float(max(0.0, min(1.0, min_lr_scale)))

    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    if total_steps <= warmup_steps:
        return 1.0

    t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    t = max(0.0, min(1.0, t))
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine


def set_optimizer_lrs(optimizer, base_lrs: List[float], lr_scale: float):
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = float(base_lr) * float(lr_scale)


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


def plot_epoch_throughput_mem_and_loss(
    out_dir: str,
    epochs: List[int],
    throughput_sps: List[float],
    mem_reserved_gb: List[float],
    train_loss: List[float],
    lr_scale: List[float],
):
    _safe_makedirs(out_dir)
    x = np.asarray(epochs, dtype=np.int32)

    plt.figure()
    plt.plot(x, np.asarray(throughput_sps, dtype=np.float64))
    plt.xlabel("epoch")
    plt.ylabel("throughput (samples/s)")
    plt.title("Pretrain throughput per epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pt_throughput_per_epoch.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, np.asarray(mem_reserved_gb, dtype=np.float64))
    plt.xlabel("epoch")
    plt.ylabel("max reserved memory (GB)")
    plt.title("Pretrain GPU max reserved memory per epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pt_mem_reserved_per_epoch.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, np.asarray(train_loss, dtype=np.float64))
    plt.xlabel("epoch")
    plt.ylabel("train loss (DP avg)")
    plt.title("Pretrain training loss per epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pt_loss_per_epoch.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, np.asarray(lr_scale, dtype=np.float64))
    plt.xlabel("epoch")
    plt.ylabel("lr scale")
    plt.title("Pretrain LR scale per epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pt_lr_scale_per_epoch.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S2-only MAE Pretrain (DeepSpeed). No interpolation. in_ch=6."
    )
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    parser.add_argument("--img-h", type=int, default=128)
    parser.add_argument("--img-w", type=int, default=128)
    parser.add_argument("--patch", type=int, default=4)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--enc-heads", type=int, default=8)

    parser.add_argument("--dec-dim", type=int, default=128)
    parser.add_argument("--dec-depth", type=int, default=2)

    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-scale", type=float, default=0.05)

    parser.add_argument("--geo-aug-hflip", type=float, default=0.5)
    parser.add_argument("--geo-aug-vflip", type=float, default=0.5)
    parser.add_argument("--geo-aug-rot90", type=float, default=0.5)

    parser.add_argument("--pt-dataset-name", type=str, default="s2")

    parser.add_argument(
        "--s2-root",
        type=str,
        nargs="+",
        required=True,
        default="s2",
    )

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
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")

    ep_size = max(1, moe_ep_size)
    _ep_rank, _ep_group_ranks, _ep_group = build_ep_group_for_rank(global_rank, ep_size)

    TARGET_HW = (int(args.img_h), int(args.img_w))
    PATCH = int(args.patch)
    if TARGET_HW[0] % PATCH != 0 or TARGET_HW[1] % PATCH != 0:
        raise RuntimeError(f"[PT] img_hw={TARGET_HW} must be divisible by patch={PATCH}")

    files_s2 = []
    for root in args.s2_root:
        files_s2.extend(rglob_many(root, exts=("tif", "tiff")))
    files_s2 = sorted(files_s2)

    if args.subset and args.subset > 0:
        files_s2 = files_s2[: min(len(files_s2), int(args.subset))]

    N = len(files_s2)

    # Shared 6-band overlap:
    # [B2, B3, B4, B8, B11, B12] -> rasterio 1-indexed in your custom S2 files
    bands_PT = [2, 3, 4, 8, 12, 13]

    ds_PT = S2GeoTiffDataset(
        files=files_s2,
        bands=bands_PT,
        normalize="perimage_zscore",
        stats=None,
        target_hw=TARGET_HW,
    )

    geo_recipe = resolve_pt_geo_recipe(args)

    if is_rank0:
        print(f"[PT] Using S2-only N={N} roots={args.s2_root}", flush=True)
        print(f"[PT] world_size={world_size} ep_size={ep_size}", flush=True)
        print(
            f"[PT] Model: in_ch=6 patch={PATCH} img_hw={TARGET_HW} "
            f"embed_dim={args.embed_dim} enc_depth={args.enc_depth} enc_heads={args.enc_heads} "
            f"dec_dim={args.dec_dim} dec_depth={args.dec_depth}",
            flush=True,
        )
        print(f"[PT] Norm=perimage_zscore | S2 bands (6) rasterio-1idx={bands_PT}", flush=True)
        print(
            f"[PT] MoE: num_experts={moe_num_experts} top_k={moe_topk} capacity_factor={moe_capacity} "
            f"noisy_gate={moe_noisy_gate} aux_w={moe_aux_weight}",
            flush=True,
        )
        print(
            f"[PT] Geo aug: hflip={geo_recipe['hflip']} vflip={geo_recipe['vflip']} rot90={geo_recipe['rot90']} | "
            f"warmup_epochs={args.warmup_epochs} min_lr_scale={args.min_lr_scale}",
            flush=True,
        )
        x0 = ds_PT[0]
        print(
            f"[PT] Example input shape={tuple(x0.shape)} expected=(6,{TARGET_HW[0]},{TARGET_HW[1]})",
            flush=True,
        )

    sampler = DistributedSampler(ds_PT, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=False)

    g = torch.Generator()
    g.manual_seed(args.seed)

    def _wif(worker_id: int):
        seed = args.seed + 1000 * global_rank + worker_id
        np.random.seed(seed)
        random.seed(seed)

    dl_PT = DataLoader(
        ds_PT,
        batch_size=conf["train_micro_batch_size_per_gpu"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=_wif,
        generator=g,
        sampler=sampler,
    )

    mae = TinyMAE(
        in_ch=6,
        embed_dim=args.embed_dim,
        depth=args.enc_depth,
        num_heads=args.enc_heads,
        decoder_dim=args.dec_dim,
        decoder_depth=args.dec_depth,
        patch=PATCH,
        img_hw=TARGET_HW,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        moe_num_experts=max(0, moe_num_experts),
        ep_world_size=ep_size,
        top_k=moe_topk,
        capacity_factor=moe_capacity,
        noisy_gate_policy=moe_noisy_gate,
        moe_aux_weight=moe_aux_weight,
        num_modalities=0,
    ).to(device)

    model = PretrainModel(mae=mae, mask_ratio=args.mask_ratio).to(device)

    if is_rank0:
        total_params, trainable_params = count_parameters(model)
        print(f"[PT] Parameters (pre-DS): total={total_params:,} trainable={trainable_params:,}", flush=True)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters)

    base_lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]

    total_steps_target = int(args.steps) if int(args.steps) > 0 else int(args.epochs) * int(len(dl_PT))
    warmup_steps = int(args.warmup_epochs) * int(len(dl_PT))

    global_step = 0
    ckpt_dir = "ckpts_pretrain"
    plots_dir = os.path.join(ckpt_dir, f"plots_{args.pt_dataset_name}")
    if is_rank0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    torch.distributed.barrier()

    epoch_ids_rank0: List[int] = []
    throughput_sps_rank0: List[float] = []
    mem_reserved_gb_rank0: List[float] = []
    train_loss_rank0: List[float] = []
    lr_scale_rank0: List[float] = []

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        engine.train()

        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        epoch_start = time.time()
        running_loss_local = 0.0
        running_steps_local = 0.0
        lr_scale_last = float("nan")

        L = int(args.enc_depth)
        epoch_counts_layers_global_accum: Optional[torch.Tensor] = None
        epoch_counts_sum_global_accum: Optional[torch.Tensor] = None

        for step, x in enumerate(dl_PT):
            if args.steps and global_step >= args.steps:
                break

            x = x.to(device, non_blocking=True)
            x = apply_geo_safe_spatial_aug(
                x,
                hflip_prob=geo_recipe["hflip"],
                vflip_prob=geo_recipe["vflip"],
                rot90_prob=geo_recipe["rot90"],
            )

            if engine.fp16_enabled():
                x = x.half()
            elif engine.bfloat16_enabled():
                x = x.bfloat16()

            lr_scale = compute_lr_scale(
                step=global_step,
                total_steps=total_steps_target,
                warmup_steps=warmup_steps,
                min_lr_scale=float(args.min_lr_scale),
            )
            set_optimizer_lrs(optimizer, base_lrs, lr_scale)
            lr_scale_last = float(lr_scale)

            loss, dbg = engine(x)
            engine.backward(loss)
            engine.step()

            running_loss_local += float(loss.item())
            running_steps_local += 1.0

            util_str = ""
            if moe_num_experts > 0:
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
                    step_counts_sum = layers_counts.sum(dim=0)
                    util_str = format_expert_util(step_counts_sum)

                    if epoch_counts_layers_global_accum is None:
                        epoch_counts_layers_global_accum = layers_counts.clone()
                    else:
                        epoch_counts_layers_global_accum += layers_counts

                    if epoch_counts_sum_global_accum is None:
                        epoch_counts_sum_global_accum = step_counts_sum.clone()
                    else:
                        epoch_counts_sum_global_accum += step_counts_sum

            if is_rank0 and util_str:
                print(
                    f"[PT] epoch={epoch} step={step} gstep={global_step} "
                    f"loss={loss.item():.4f} mask={dbg['mask_ratio']:.3f} aux={dbg['aux']:.4f} "
                    f"lr_scale={lr_scale:.5f} {util_str}",
                    flush=True,
                )

            global_step += 1

        if world_size > 1:
            vec = torch.tensor([running_loss_local, running_steps_local], device=device, dtype=torch.float32)
            ddp_allreduce_sum(vec)
            total_loss = float(vec[0].item())
            total_steps = float(vec[1].item())
        else:
            total_loss = running_loss_local
            total_steps = running_steps_local

        epoch_time = time.time() - epoch_start

        try:
            max_alloc_gb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
            max_reserved_gb = float(torch.cuda.max_memory_reserved() / (1024.0 ** 3))
        except Exception:
            max_alloc_gb = float("nan")
            max_reserved_gb = float("nan")

        if is_rank0 and total_steps > 0:
            mean_loss = total_loss / max(total_steps, 1.0)
            global_batch = world_size * conf["train_micro_batch_size_per_gpu"]
            samples_this_epoch = global_batch * (running_steps_local if running_steps_local > 0 else 0.0)
            throughput = (samples_this_epoch / epoch_time) if epoch_time > 0 and samples_this_epoch > 0 else float("nan")

            print(
                f"[PT][EPOCH {epoch}] mean_loss_DP_avg={mean_loss:.4f} over {int(total_steps)} aggregated steps | "
                f"epoch_time={epoch_time:.2f}s | throughput≈{throughput:.2f} samples/s | "
                f"lr_scale_last={lr_scale_last:.5f} | "
                f"mem_max_reserved_gb={max_reserved_gb:.3f} mem_max_alloc_gb={max_alloc_gb:.3f}",
                flush=True,
            )

            epoch_ids_rank0.append(int(epoch))
            throughput_sps_rank0.append(float(throughput))
            mem_reserved_gb_rank0.append(float(max_reserved_gb))
            train_loss_rank0.append(float(mean_loss))
            lr_scale_rank0.append(float(lr_scale_last))

        if moe_num_experts > 0:
            if epoch_counts_sum_global_accum is None:
                epoch_counts_sum_global_accum = torch.zeros(moe_num_experts, device=device, dtype=torch.float32)

            summ = epoch_util_summary(epoch_counts_sum_global_accum)
            topk = epoch_topk_str_from_counts(epoch_counts_sum_global_accum, k=8)

            if is_rank0:
                print(
                    "PT_EPOCH_UTIL_SUM "
                    f"epoch={epoch} tokens={summ['tokens']:.0f} active={summ['active']} "
                    f"top={summ['top_idx']} top_pct={summ['top_pct']:.2f} "
                    f"max_pct={summ['max_pct']:.2f} min_pct={summ['min_pct']:.2f} "
                    f"entropy={summ['entropy']:.3f} gini={summ['gini']:.3f} cv={summ['cv']:.3f}",
                    flush=True,
                )
                print(f"PT_EPOCH_UTIL_TOPK epoch={epoch} top8={topk}", flush=True)

            if is_rank0:
                if epoch_counts_layers_global_accum is not None:
                    layers_np = epoch_counts_layers_global_accum.detach().cpu().numpy().astype(np.float64, copy=False)
                    sum_np = layers_np.sum(axis=0)

                    for li in range(layers_np.shape[0]):
                        pct = _counts_to_pct(layers_np[li])
                        plot_expert_occupancy_bars(
                            out_path=os.path.join(plots_dir, f"pt_expert_occupancy_gate{li:02d}.png"),
                            pct=pct,
                            title=f"PT expert occupancy (%) | gate(layer)={li} | latest(epoch={epoch})",
                        )

                    pct_sum = _counts_to_pct(sum_np)
                    plot_expert_occupancy_bars(
                        out_path=os.path.join(plots_dir, "pt_expert_occupancy_SUM.png"),
                        pct=pct_sum,
                        title=f"PT expert occupancy (%) | SUM(all gates) | latest(epoch={epoch})",
                    )

                plot_epoch_throughput_mem_and_loss(
                    out_dir=plots_dir,
                    epochs=epoch_ids_rank0,
                    throughput_sps=throughput_sps_rank0,
                    mem_reserved_gb=mem_reserved_gb_rank0,
                    train_loss=train_loss_rank0,
                    lr_scale=lr_scale_rank0,
                )

        torch.distributed.barrier()

        tag = f"pt_j{job_id}_E{moe_num_experts}_ep{ep_size}_ws{world_size}_epoch{epoch}_gstep{global_step}"
        engine.save_checkpoint(ckpt_dir, tag=tag, save_latest=True)

        if is_rank0:
            print(f"[PT] DeepSpeed checkpoint saved: {ckpt_dir} tag={tag}", flush=True)

        torch.distributed.barrier()

        if args.steps and global_step >= args.steps:
            break

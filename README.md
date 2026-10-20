# Evaluating Deployable Inference-Time Error Prediction in Vision MoEs

Code accompanying the paper **“Evaluating Deployable Inference-Time Error Prediction in Vision MoEs.”**


## Overview

This repository provides the model and experimental pipeline used to pretrain, fine-tune, and evaluate the Vision Mixture-of-Experts (V-MoE) models considered in the paper.

The release includes the V-MoE architecture, DeepSpeed-based sparse expert layers, RGB and multispectral data pipelines, masked-autoencoder pretraining, downstream classification fine-tuning, checkpoint loading, and inference-time extraction of router-native and non-selected-expert predictions. The experiments use a common compact V-MoE backbone across datasets so that comparisons reflect inference-time expert behavior rather than dataset-specific architecture tuning. Together with the architecture, hyperparameter, dataset, and evaluation specifications in the accompanying paper, this repository documents the experimental setup used throughout the study.

## Model Architecture

The evaluation model is a compact ViT-style sparse Mixture-of-Experts encoder constructed from standard Transformer components and the standard DeepSpeed MoE implementation. The same architecture is trained from scratch for the RGB experiments and used within a masked-autoencoder pretraining and fine-tuning pipeline for the remote-sensing experiments.

The default backbone configuration is:

| Component | Configuration |
|---|---|
| Patch embedding | Convolutional |
| Patch size | 4 × 4 |
| Embedding dimension | 256 |
| Transformer depth | 6 blocks |
| Attention heads | 8 |
| MLP ratio | 4.0 |
| Positional encoding | 2D sinusoidal |
| Normalization | Pre-norm Transformer blocks + final LayerNorm |
| Expert layer | DeepSpeed MoE replacing the Transformer feed-forward sublayer |

The sparse MoE design follows the general Vision MoE formulation, while routing and expert execution use the standard DeepSpeed MoE layer. See the accompanying paper, particularly Section 6.2 and Appendix D, for the complete experimental specification.

## Repository Structure

```text
.
├── geo_projector_lib.py          # model, MoE block, dataset, and preprocessing definitions
├── geo_projector_pretrain.py     # masked-autoencoder pretraining on six-band Sentinel-2 imagery
├── geo_projector_finetune.py     # fine-tuning of the Vision MoE classifier
├── geo_projector_ft_dump.py      # extraction of router-native, router, and expert-forced outputs for validation and test data
├── config/
│   └── ds_config.json            # DeepSpeed and MoE configuration
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md
```

## Execution Environment

The training and inference scripts are designed for distributed execution under Slurm, with one process per GPU. The scripts expect the Slurm environment variables `SLURM_NTASKS`, `SLURM_PROCID`, and `SLURM_LOCALID` to be available.

The commands below are intended to run within an existing Slurm allocation. For example, assuming eight visible GPU processes per node and seven CPU cores per process:

```bash
CMD="python <SCRIPT_AND_ARGUMENTS>"

srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="$((SLURM_NNODES * 8))" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    bash -c "${CMD}"
```

The default DeepSpeed configuration uses ZeRO Stage 2 and bfloat16 training.

## Datasets

The paper evaluates RGB natural-image and multispectral remote-sensing classification settings.

- **CIFAR-100** — RGB, 100 classes, trained from scratch.
- **Tiny-ImageNet-200** — RGB, 200 classes, trained from scratch.
- **So2Sat** — multispectral remote-sensing classification using the GeoBench interface.
- **ForestNet** — remote-sensing classification using the GeoBench interface.
- **Sentinel-2 pretraining data** — six-band imagery using bands `[B2, B3, B4, B8, B11, B12]`.

Dataset paths are supplied through the corresponding command-line arguments.

## Pretraining

Remote-sensing experiments use masked-autoencoder pretraining on six-band Sentinel-2 imagery. The default mask ratio is 0.75.

```bash
python geo_projector_pretrain.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --img-h 128 \
    --img-w 128 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4.0 \
    --dropout 0.1 \
    --mask-ratio 0.75 \
    --seed 0 \
    --epochs 100 \
    --s2-root <PATH_TO_SENTINEL2_DATA>
```

## Fine-tuning and Training

Remote-sensing downstream experiments are initialized from a pretrained checkpoint. CIFAR-100 and Tiny-ImageNet are trained from scratch using the same encoder-style architecture.

### So2Sat

```bash
python geo_projector_finetune.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --lr-head 5e-4 \
    --lr-backbone 2e-4 \
    --lr-moe 1e-4 \
    --weight-decay 1e-3 \
    --label-smoothing 0.05 \
    --dropout 0.1 \
    --img-h 32 \
    --img-w 32 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --epochs 100 \
    --patience 10 \
    --min-delta 0.001 \
    --ft-mode full_ft \
    --val-frac 0.1 \
    --ckpt-dir ckpts_pretrain \
    --tag <PRETRAINED_CHECKPOINT_TAG> \
    --ft-ckpt-dir ckpts_finetune_so2sat_pt \
    --dataset so2sat \
    --geo-aug-hflip 0.5 \
    --geo-aug-vflip 0.5 \
    --geo-aug-rot90 0.5
```

### ForestNet

```bash
python geo_projector_finetune.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --lr-head 5e-4 \
    --lr-backbone 2e-4 \
    --lr-moe 1e-4 \
    --weight-decay 1e-3 \
    --label-smoothing 0.05 \
    --dropout 0.1 \
    --img-h 332 \
    --img-w 332 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --epochs 100 \
    --patience 10 \
    --min-delta 0.001 \
    --ft-mode full_ft \
    --val-frac 0.1 \
    --ckpt-dir ckpts_pretrain \
    --tag <PRETRAINED_CHECKPOINT_TAG> \
    --ft-ckpt-dir ckpts_finetune_forestnet_pt \
    --dataset forestnet \
    --geo-aug-hflip 0.5 \
    --geo-aug-vflip 0.5 \
    --geo-aug-rot90 0.5
```

### CIFAR-100

```bash
python geo_projector_finetune.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --lr-head 5e-4 \
    --lr-backbone 2e-4 \
    --lr-moe 1e-4 \
    --weight-decay 1e-3 \
    --label-smoothing 0.1 \
    --dropout 0.1 \
    --img-h 32 \
    --img-w 32 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --epochs 100 \
    --patience 10 \
    --min-delta 0.001 \
    --ft-mode scratch_ft \
    --val-frac 0.1 \
    --ft-ckpt-dir ckpts_finetune_cifar \
    --dataset cifar100 \
    --cifar100-root <PATH_TO_CIFAR100>
```

### Tiny-ImageNet-200

```bash
python geo_projector_finetune.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --lr-head 5e-4 \
    --lr-backbone 2e-4 \
    --lr-moe 1e-4 \
    --weight-decay 1e-3 \
    --label-smoothing 0.1 \
    --dropout 0.1 \
    --img-h 64 \
    --img-w 64 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --epochs 100 \
    --patience 10 \
    --min-delta 0.001 \
    --ft-mode scratch_ft \
    --val-frac 0.1 \
    --ft-ckpt-dir ckpts_finetune_tiny \
    --dataset tinyimagenet \
    --tiny-imagenet-root <PATH_TO_TINY_IMAGENET>
```

## Generating Inference Artifacts

`geo_projector_ft_dump.py` loads a selected fine-tuned checkpoint and generates validation and test artifacts containing the quantities used by the paper's inference-time evaluation protocol. These artifacts expose the router-native and non-selected-expert outputs needed to study additional expert inference while keeping the trained model weights fixed.

### So2Sat

```bash
python geo_projector_ft_dump.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --seed 42 \
    --dropout 0.0 \
    --img-h 32 \
    --img-w 32 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --val-frac 0.1 \
    --ft-ckpt-dir ckpts_finetune_so2sat_pt \
    --tag <BEST_VALIDATION_CHECKPOINT> \
    --dump-selector \
    --dataset so2sat
```

### ForestNet

```bash
python geo_projector_ft_dump.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --seed 42 \
    --dropout 0.0 \
    --img-h 332 \
    --img-w 332 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --val-frac 0.1 \
    --ft-ckpt-dir ckpts_finetune_forestnet_pt \
    --tag <BEST_VALIDATION_CHECKPOINT> \
    --dump-selector \
    --dataset forestnet
```

### CIFAR-100

```bash
python geo_projector_ft_dump.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --seed 42 \
    --dropout 0.0 \
    --img-h 32 \
    --img-w 32 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --val-frac 0.1 \
    --ft-ckpt-dir ckpts_finetune_cifar \
    --tag <BEST_VALIDATION_CHECKPOINT> \
    --dump-selector \
    --dataset cifar100 \
    --cifar100-root <PATH_TO_CIFAR100>
```

### Tiny-ImageNet-200

```bash
python geo_projector_ft_dump.py \
    --deepspeed \
    --deepspeed_config config/ds_config.json \
    --seed 42 \
    --dropout 0.0 \
    --img-h 64 \
    --img-w 64 \
    --patch 4 \
    --embed-dim 256 \
    --enc-depth 6 \
    --enc-heads 8 \
    --mlp-ratio 4 \
    --val-frac 0.1 \
    --ft-ckpt-dir ckpts_finetune_tiny \
    --tag <BEST_VALIDATION_CHECKPOINT> \
    --dump-selector \
    --dataset tinyimagenet \
    --tiny-imagenet-root <PATH_TO_TINY_IMAGENET>
```

## Experimental Configuration

The paper varies the number of experts while keeping the backbone architecture fixed. To reproduce a particular expert-count configuration, edit the `moe.num_experts` field in `config/ds_config.json`.

For example:

```json
{
  "moe": {
    "enabled": true,
    "ep_size": 1,
    "num_experts": 10,
    "top_k": 1,
    "capacity_factor": 1.25,
    "eval_capacity_factor": 200,
    "noisy_gate_policy": "None",
    "moe_aux_weight": 0.02
  }
}
```

The expert-count sweep reported in the paper uses:

```text
E in {2, 5, 10, 15, 25, 50}
```

Model selection and checkpoint selection should be performed using validation data, following the validation-to-test protocol described in the paper.

## Citation

If you use this code in your work, please cite the accompanying paper.

```bibtex
@inproceedings{vision_moe_error_prediction_2026,
  title     = {Evaluating Deployable Inference-Time Error Prediction in Vision MoEs},
  author    = {Aristeidis Tsaris, JangHyeon Lee, Abhishek Potnis, Philipe Ambrozio Dias, Waqwoya Abebe, Dan Lu, Feiyi Wang, Dalton Lunga},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

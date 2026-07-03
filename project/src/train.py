"""
train.py
========

Training loop end-to-end cho MultimodalFusionModel.

Skeleton — fill in khi tới Phase 5. Đã wire sẵn:
- Multi-task loss với weighted CE cho Head 1 & 2
- AdamW với lr phân biệt (backbone vs các layer mới)
- 3-stage fine-tuning schedule
- CosineAnnealingLR + gradient clipping + early stopping (TODO)

CLI usage (sẽ implement):
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import MultimodalFusionModel
from .utils import get_device, set_seed


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    lambda_detection: float = 1.0
    lambda_echogenicity: float = 0.5
    lambda_reclassify: float = 0.3
    lr_backbone: float = 1e-5
    lr_new_layers: float = 1e-4
    stage1_epochs: int = 5   # freeze backbone
    stage2_epochs: int = 15  # unfreeze layer4
    stage3_epochs: int = 10  # unfreeze all
    seed: int = 42


def build_optimizer(model: MultimodalFusionModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.cnn_branch.backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": model.cnn_branch.projection.parameters(), "lr": cfg.lr_new_layers},
            {"params": model.mlp_branch.parameters(), "lr": cfg.lr_new_layers},
            {"params": model.fusion.parameters(), "lr": cfg.lr_new_layers},
            {"params": model.heads.parameters(), "lr": cfg.lr_new_layers},
        ],
        weight_decay=cfg.weight_decay,
    )


def compute_loss(
    outputs: dict,
    batch: dict,
    cfg: TrainConfig,
    weight_plaque: torch.Tensor,
    weight_echo: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    ce_plaque = nn.functional.cross_entropy(
        outputs["plaque_logits"], batch["plaque"], weight=weight_plaque
    )
    ce_echo = nn.functional.cross_entropy(
        outputs["echo_logits"], batch["echo"], weight=weight_echo
    )
    bce_reclassify = nn.functional.binary_cross_entropy_with_logits(
        outputs["reclassify_logit"], batch["reclassify"]
    )
    total = (
        cfg.lambda_detection * ce_plaque
        + cfg.lambda_echogenicity * ce_echo
        + cfg.lambda_reclassify * bce_reclassify
    )
    return total, {
        "loss_total": float(total.detach()),
        "loss_plaque": float(ce_plaque.detach()),
        "loss_echo": float(ce_echo.detach()),
        "loss_reclassify": float(bce_reclassify.detach()),
    }


def train_one_epoch(
    model: MultimodalFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
    weight_plaque: torch.Tensor,
    weight_echo: torch.Tensor,
) -> dict:
    model.train()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch["images"], batch["n_images"], batch["tabular"])
        loss, parts = compute_loss(outputs, batch, cfg, weight_plaque, weight_echo)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        losses.append(parts)
    keys = losses[0].keys()
    return {k: float(np.mean([d[k] for d in losses])) for k in keys}


@torch.no_grad()
def validate(
    model: MultimodalFusionModel,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    weight_plaque: torch.Tensor,
    weight_echo: torch.Tensor,
) -> dict:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch["images"], batch["n_images"], batch["tabular"])
        _, parts = compute_loss(outputs, batch, cfg, weight_plaque, weight_echo)
        losses.append(parts)
    keys = losses[0].keys()
    return {k: float(np.mean([d[k] for d in losses])) for k in keys}


def main():
    # TODO Phase 5:
    #   1. Load config YAML
    #   2. Build train/val/test loaders (xem dataset.py)
    #   3. Compute class weights từ y_train (sklearn.utils.class_weight)
    #   4. 3-stage training schedule:
    #        - Stage 1 (epoch 1..5):   freeze backbone
    #        - Stage 2 (epoch 6..20):  unfreeze layer4
    #        - Stage 3 (epoch 21..30): unfreeze all
    #   5. Cosine LR schedule + early stopping on val AUC
    #   6. Save best checkpoint vào results/checkpoints/
    raise NotImplementedError("Training loop — implement ở Phase 5")


if __name__ == "__main__":
    set_seed(42)
    main()

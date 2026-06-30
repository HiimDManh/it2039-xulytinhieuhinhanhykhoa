"""
cnn_branch.py
=============

Branch 1 — Image Branch.
ResNet-50 (ImageNet pretrained) + GAP + projection 2048 → 128.

Projection layer là điểm mới so với design v1: tránh image embedding
2048-dim "nuốt" clinical embedding 64-dim trong concat fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CNNBranch(nn.Module):
    """
    Input  : [N_img_total, 3, H, W] + n_images per patient
    Output : image_embedding [B, embed_dim] (mean-pool theo bệnh nhân)

    Khi `use_projection=False` (ablation w/o projection): bỏ qua Linear
    2048→embed_dim, trả về 2048-dim raw GAP feature. `output_dim` reflect
    đầu ra thực — FusionLayer/CrossAttention dựa vào đó để cấu hình kích cỡ.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        use_projection: bool = True,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()  # output 2048-dim GAP feature
        self.backbone = backbone
        self.use_projection = use_projection

        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(2048, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.output_dim = embed_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = 2048

        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Fine-tuning helpers — gọi từ train.py theo 3-stage schedule
    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_layer4(self) -> None:
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor, n_images: torch.Tensor) -> torch.Tensor:
        """
        images   : [sum_N, 3, H, W]
        n_images : [B] — số ảnh của từng bệnh nhân trong batch
        returns  : [B, embed_dim]
        """
        feats = self.backbone(images)          # [sum_N, 2048]
        feats = self.projection(feats)         # [sum_N, embed_dim]

        # Mean-pool theo bệnh nhân
        out = torch.zeros(len(n_images), feats.size(1), device=feats.device)
        offset = 0
        for i, n in enumerate(n_images.tolist()):
            out[i] = feats[offset : offset + n].mean(dim=0)
            offset += n
        return out

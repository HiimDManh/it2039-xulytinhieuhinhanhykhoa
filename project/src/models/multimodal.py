"""
multimodal.py
=============

Full Multimodal Fusion Model = CNNBranch + MLPBranch + Fusion + OutputHeads.

Hỗ trợ:
- `fusion_type`: 'concat' (mặc định) hoặc 'cross_attn' (RQ4)
- `cnn_use_projection`: True (mặc định) hoặc False (ablation w/o projection)
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .cnn_branch import CNNBranch
from .fusion import CrossAttentionFusion, FusionLayer, OutputHeads
from .mlp_branch import MLPBranch


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 128,
        clinical_embed_dim: int = 64,
        tabular_in: int = 9,
        n_echo_classes: int = 4,
        pretrained: bool = True,
        cnn_use_projection: bool = True,
        fusion_type: Literal["concat", "cross_attn"] = "concat",
        fusion_out_dim: int = 64,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        self.cnn_branch = CNNBranch(
            embed_dim=image_embed_dim,
            pretrained=pretrained,
            use_projection=cnn_use_projection,
        )
        # CNN output dim reflects projection choice
        image_dim = self.cnn_branch.output_dim

        self.mlp_branch = MLPBranch(
            in_features=tabular_in,
            embed_dim=clinical_embed_dim,
        )

        if fusion_type == "concat":
            self.fusion = FusionLayer(
                image_dim=image_dim,
                clinical_dim=clinical_embed_dim,
                out_dim=fusion_out_dim,
            )
        elif fusion_type == "cross_attn":
            self.fusion = CrossAttentionFusion(
                image_dim=image_dim,
                clinical_dim=clinical_embed_dim,
                d_model=image_embed_dim,
                out_dim=fusion_out_dim,
            )
        else:
            raise ValueError(f"Unknown fusion_type={fusion_type!r}")

        self.heads = OutputHeads(in_features=fusion_out_dim, n_echo_classes=n_echo_classes)

    def forward(
        self,
        images: torch.Tensor,
        n_images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> dict:
        image_emb = self.cnn_branch(images, n_images)
        clinical_emb = self.mlp_branch(tabular)
        fused = self.fusion(image_emb, clinical_emb)
        return self.heads(fused)

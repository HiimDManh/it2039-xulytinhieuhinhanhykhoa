"""
fusion.py
=========

Fusion layer + multi-task output heads.

- Concatenation fusion (mặc định): [image_128 ⊕ clinical_64] = 192
  → FC(192→128) → FC(128→64)
- Cross-Attention fusion (Phase 7): chưa implement — placeholder
- 3 output heads:
    Head 1 — Plaque Detection      (2 class)
    Head 2 — Echogenicity          (4 class: None/Low/Inter/High)
    Head 3 — Reclassify ESC/EAS    (binary, sigmoid)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """Concat fusion → 64-dim joint embedding."""

    def __init__(
        self,
        image_dim: int = 128,
        clinical_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim + clinical_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, image_emb: torch.Tensor, clinical_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image_emb, clinical_emb], dim=1))


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion (RQ4).

    Image làm Query, clinical làm Key/Value. Cả hai được chiếu về cùng
    d_model qua linear projection. Coi mỗi embedding là sequence length=1
    để feed thẳng vào nn.MultiheadAttention.

    Output: joint embedding shape [B, out_dim] — cùng signature với
    FusionLayer concat để swap qua lại được trong MultimodalFusionModel.
    """

    def __init__(
        self,
        image_dim: int = 128,
        clinical_dim: int = 64,
        d_model: int = 128,
        num_heads: int = 4,
        out_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.img_proj = nn.Linear(image_dim, d_model)
        self.cli_proj = nn.Linear(clinical_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, image_emb: torch.Tensor, clinical_emb: torch.Tensor) -> torch.Tensor:
        q = self.img_proj(image_emb).unsqueeze(1)          # [B, 1, d_model]
        kv = self.cli_proj(clinical_emb).unsqueeze(1)      # [B, 1, d_model]
        attn_out, _ = self.attn(q, kv, kv)
        x = self.norm1(q + attn_out)                       # residual + LN
        x = self.norm2(x + self.ff(x))
        return self.out(x.squeeze(1))                      # [B, out_dim]


class OutputHeads(nn.Module):
    """3 heads ăn chung joint embedding 64-dim."""

    def __init__(self, in_features: int = 64, n_echo_classes: int = 4):
        super().__init__()
        self.head_plaque = nn.Linear(in_features, 2)
        self.head_echo = nn.Linear(in_features, n_echo_classes)
        self.head_reclassify = nn.Linear(in_features, 1)

    def forward(self, fused: torch.Tensor) -> dict:
        return {
            "plaque_logits": self.head_plaque(fused),
            "echo_logits": self.head_echo(fused),
            "reclassify_logit": self.head_reclassify(fused).squeeze(-1),
        }

"""
mlp_branch.py
=============

Branch 2 — Tabular Branch.
9 features → 64 → 128 → 64 (clinical embedding).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPBranch(nn.Module):
    """Input [B, 9] → Output [B, 64]."""

    def __init__(self, in_features: int = 9, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

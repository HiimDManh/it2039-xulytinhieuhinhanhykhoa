"""
evaluate.py
===========

Metrics & subgroup analysis cho 3 task + nhóm Discordance ESC/EAS 2025.

Skeleton — fill in Phase 6.

Metrics chính (theo design doc):
- Plaque Detection : AUC-ROC, F1, Sensitivity, Specificity
- Echogenicity     : F1-macro, accuracy per class
- Reclassification : AUC-ROC, F1, Cohen's Kappa
- Discordance      : Sensitivity, NPV trên 33 cases (subgroup)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .models import MultimodalFusionModel


@dataclass
class EvalResult:
    plaque_auc: float
    plaque_f1: float
    plaque_sensitivity: float
    plaque_specificity: float
    echo_f1_macro: float
    reclassify_auc: float
    reclassify_kappa: float
    # Subgroup — Phase 6 novelty
    discordant_sensitivity: float
    discordant_npv: float
    n_discordant: int


@torch.no_grad()
def collect_predictions(
    model: MultimodalFusionModel,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Chạy model qua loader, trả về DataFrame để phân tích."""
    model.eval()
    rows = []
    for batch in loader:
        images = batch["images"].to(device)
        n_images = batch["n_images"].to(device)
        tabular = batch["tabular"].to(device)
        out = model(images, n_images, tabular)

        plaque_prob = torch.softmax(out["plaque_logits"], dim=1)[:, 1].cpu().numpy()
        echo_pred = out["echo_logits"].argmax(dim=1).cpu().numpy()
        reclass_prob = torch.sigmoid(out["reclassify_logit"]).cpu().numpy()

        for i, pid in enumerate(batch["patient_ids"]):
            rows.append({
                "patient_id": pid,
                "plaque_true": int(batch["plaque"][i]),
                "plaque_prob": float(plaque_prob[i]),
                "echo_true": int(batch["echo"][i]),
                "echo_pred": int(echo_pred[i]),
                "reclassify_true": float(batch["reclassify"][i]),
                "reclassify_prob": float(reclass_prob[i]),
                "discordant": bool(batch["discordant"][i]),
            })
    return pd.DataFrame(rows)


def compute_metrics(preds: pd.DataFrame, plaque_threshold: float = 0.5) -> EvalResult:
    """Tính toàn bộ metrics + subgroup từ predictions DataFrame."""
    y_pl = preds["plaque_true"].values
    p_pl = preds["plaque_prob"].values
    yhat_pl = (p_pl >= plaque_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_pl, yhat_pl, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # Discordance subgroup
    disc = preds[preds["discordant"]]
    if len(disc) == 0:
        disc_sens, disc_npv = float("nan"), float("nan")
    else:
        disc_yhat = (disc["plaque_prob"].values >= plaque_threshold).astype(int)
        disc_y = disc["plaque_true"].values
        dtn, dfp, dfn, dtp = confusion_matrix(disc_y, disc_yhat, labels=[0, 1]).ravel()
        disc_sens = dtp / (dtp + dfn) if (dtp + dfn) else 0.0
        disc_npv = dtn / (dtn + dfn) if (dtn + dfn) else 0.0

    return EvalResult(
        plaque_auc=roc_auc_score(y_pl, p_pl),
        plaque_f1=f1_score(y_pl, yhat_pl),
        plaque_sensitivity=sens,
        plaque_specificity=spec,
        echo_f1_macro=f1_score(preds["echo_true"], preds["echo_pred"], average="macro"),
        reclassify_auc=roc_auc_score(preds["reclassify_true"], preds["reclassify_prob"]),
        reclassify_kappa=cohen_kappa_score(
            preds["reclassify_true"].astype(int),
            (preds["reclassify_prob"].values >= 0.5).astype(int),
        ),
        discordant_sensitivity=float(disc_sens),
        discordant_npv=float(disc_npv),
        n_discordant=int(len(disc)),
    )


def evaluate_esceas_rule(df: pd.DataFrame) -> dict:
    """
    Baseline lâm sàng ESC/EAS Rule (xem clinical_rules.esceas_2025_rule_predict).
    Phase 3 sẽ gọi và đưa vào bảng so sánh ở Phase 6.
    """
    from .clinical_rules import esceas_2025_rule_predict

    yhat = esceas_2025_rule_predict(df)
    y = df["Plaque_present"].values
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y, yhat),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }

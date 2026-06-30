"""
utils.py
========

Helpers dùng chung: seeding, device detection, path resolution, logger.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CSV_PATH: Path = DATA_DIR / "carotid_clinical_dataset_300cases.csv"
IMAGES_DIR: Path = DATA_DIR / "CAROTID_IMAGES"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"
FIGURES_DIR: Path = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Fix RNG seed cho random / numpy / torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Trả về CUDA nếu có, ngược lại CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------
def patient_image_paths(patient_id: str, plaque_present: int) -> list[Path]:
    """
    Trả về list path ảnh cho 1 bệnh nhân.

    - Control (plaque_present=0): 1 ảnh `<ID>_IMT.png`
    - Case    (plaque_present=1): 5 ảnh `<ID>_IMT.png` + 4 cross-section
      (`<ID>_CCA_L1.png`, `_L2`, `_R1`, `_R2`).

    Hàm KHÔNG kiểm tra file tồn tại — caller tự assert nếu cần.
    """
    if plaque_present == 0:
        return [IMAGES_DIR / f"{patient_id}_IMT.png"]
    return [
        IMAGES_DIR / f"{patient_id}_IMT.png",
        IMAGES_DIR / f"{patient_id}_CCA_L1.png",
        IMAGES_DIR / f"{patient_id}_CCA_L2.png",
        IMAGES_DIR / f"{patient_id}_CCA_R1.png",
        IMAGES_DIR / f"{patient_id}_CCA_R2.png",
    ]

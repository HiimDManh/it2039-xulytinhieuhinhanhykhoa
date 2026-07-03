"""
dataset.py
==========

CarotidDataset: trả về (images_tensor, tabular_tensor, labels_dict)
cho mỗi bệnh nhân.

Chiến lược ảnh (theo pipeline_tong_the.md / multimodal_fusion_architecture.md):
- Control (n=205): 1 ảnh IMT.
- Case    (n=95):  5 ảnh (1 IMT + 4 cross-section CCA_{L,R}{1,2}).
- Aggregation 5 ảnh: stack -> [N_img, C, H, W], xử lý ở model side
  (mean-pool embedding sau ResNet-50, ổn định hơn max-pool cho n nhỏ).

Tabular: 9 features đã chốt trong design doc (KHÔNG đưa
`Baseline_Risk_Score` — derived → leakage).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .clinical_rules import ECHOGENICITY_CLASSES, annotate_dataframe
from .utils import IMAGES_DIR, patient_image_paths


TABULAR_FEATURES: list[str] = [
    "Age",
    "Sex",  # encoded Male=1, Female=0
    "Lp(a)_mg_dL",
    "ApoB_mg_dL",
    "LDL_C_mg_dL",
    "Triglyceride_mg_dL",
    "Total_Cholesterol_mg_dL",
    "Non_HDL_mg_dL",
    "IMT_mm",
]


def encode_sex(series: pd.Series) -> pd.Series:
    return series.map({"Male": 1, "Female": 0}).astype(np.float32)


def encode_echogenicity(series: pd.Series) -> np.ndarray:
    """Map 4-class echogenicity sang index 0..3 theo ECHOGENICITY_CLASSES."""
    idx = {c: i for i, c in enumerate(ECHOGENICITY_CLASSES)}
    return series.map(idx).to_numpy(dtype=np.int64)


class CarotidDataset(Dataset):
    """
    Dataset đa phương thức cho bài toán xơ vữa động mạch cảnh.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã qua `annotate_dataframe()` (có cột `is_discordant`,
        `needs_reclassify`, `Plaque_echogenicity` đã fill 'None').
    tabular_scaler : sklearn StandardScaler đã fit trên train set.
        Pass None nếu chưa fit (chỉ dùng cho debug).
    image_transform : Callable[[PIL.Image.Image], torch.Tensor]
        Transforms khác nhau cho train / val|test (xem `build_transforms`).
    images_dir : Path (mặc định project/data/CAROTID_IMAGES/)

    Returns __getitem__
    -------------------
    dict with keys:
        - 'images'    : Tensor [N_img, 3, H, W] (1 hoặc 5 ảnh)
        - 'n_images'  : int
        - 'tabular'   : Tensor [9] (đã scale)
        - 'plaque'    : Tensor [] long (0/1)              -> Head 1
        - 'echo'      : Tensor [] long (0..3)             -> Head 2
        - 'reclassify': Tensor [] float (0/1)             -> Head 3
        - 'discordant': Tensor [] bool   (cho subgroup analysis Phase 6)
        - 'patient_id': str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tabular_scaler=None,
        image_transform: Callable | None = None,
        images_dir: Path = IMAGES_DIR,
    ):
        # Đảm bảo các cột derived có mặt (idempotent — annotate 2 lần OK)
        df = annotate_dataframe(df)
        self.df = df.reset_index(drop=True)
        self.scaler = tabular_scaler
        self.transform = image_transform
        self.images_dir = Path(images_dir)

        # Prepare tabular matrix
        self._X_raw = self._prepare_tabular(self.df)

    @staticmethod
    def _prepare_tabular(df: pd.DataFrame) -> np.ndarray:
        X = df[TABULAR_FEATURES].copy()
        X["Sex"] = encode_sex(X["Sex"])
        return X.to_numpy(dtype=np.float32)

    def fit_scaler(self, scaler) -> None:
        """Fit scaler trên train set rồi gán lại — gọi từ train pipeline."""
        scaler.fit(self._X_raw)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.df)

    def _load_images(self, patient_id: str, plaque_present: int) -> torch.Tensor:
        paths = patient_image_paths(patient_id, plaque_present)
        tensors = []
        for p in paths:
            img = Image.open(self.images_dir / p.name).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            tensors.append(img)
        return torch.stack(tensors, dim=0)  # [N_img, 3, H, W]

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        pid = row["Patient_ID"]
        plaque = int(row["Plaque_present"])

        images = self._load_images(pid, plaque)

        x_tab = self._X_raw[idx]
        if self.scaler is not None:
            x_tab = self.scaler.transform(x_tab.reshape(1, -1)).astype(np.float32)[0]
        tab_tensor = torch.from_numpy(x_tab)

        echo_idx = int(encode_echogenicity(pd.Series([row["Plaque_echogenicity"]]))[0])

        return {
            "images": images,
            "n_images": images.shape[0],
            "tabular": tab_tensor,
            "plaque": torch.tensor(plaque, dtype=torch.long),
            "echo": torch.tensor(echo_idx, dtype=torch.long),
            "reclassify": torch.tensor(float(row["needs_reclassify"]), dtype=torch.float32),
            "discordant": torch.tensor(bool(row["is_discordant"]), dtype=torch.bool),
            "patient_id": pid,
        }


# ---------------------------------------------------------------------------
# Collate — vì N_img khác nhau giữa control (1) và case (5), không thể
# stack trực tiếp. Ta padding hoặc dùng list. Pipeline doc đề xuất
# mean-pool ở model side, nên ở đây ta concat tất cả ảnh theo batch và
# lưu segment lengths để model split lại.
# ---------------------------------------------------------------------------
def carotid_collate(batch: list[dict]) -> dict:
    images = torch.cat([b["images"] for b in batch], dim=0)         # [sum_N, 3, H, W]
    n_images = torch.tensor([b["n_images"] for b in batch], dtype=torch.long)
    tabular = torch.stack([b["tabular"] for b in batch], dim=0)
    plaque = torch.stack([b["plaque"] for b in batch], dim=0)
    echo = torch.stack([b["echo"] for b in batch], dim=0)
    reclassify = torch.stack([b["reclassify"] for b in batch], dim=0)
    discordant = torch.stack([b["discordant"] for b in batch], dim=0)
    pids = [b["patient_id"] for b in batch]
    return {
        "images": images,
        "n_images": n_images,  # dùng để split + mean-pool ở CNN branch
        "tabular": tabular,
        "plaque": plaque,
        "echo": echo,
        "reclassify": reclassify,
        "discordant": discordant,
        "patient_ids": pids,
    }


# ---------------------------------------------------------------------------
# Transforms (lazy import torchvision — không bắt buộc khi chỉ debug logic)
# ---------------------------------------------------------------------------
def build_transforms(train: bool, image_size: int = 224):
    """Train: heavy augmentation. Val/Test: chỉ resize + normalize."""
    from torchvision import transforms as T

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.GaussianBlur(kernel_size=3),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        norm,
    ])


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 theo `Plaque_present`, patient-level.
    Dùng cho prototype (Phase 2 Phương án A).
    """
    from sklearn.model_selection import train_test_split

    y = df["Plaque_present"].values
    df_tv, df_test = train_test_split(
        df, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    y_tv = df_tv["Plaque_present"].values
    df_train, df_val = train_test_split(
        df_tv, test_size=rel_val, stratify=y_tv, random_state=seed
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def stratified_kfold_indices(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42
):
    """
    Yield (train_idx, val_idx) cho 5-fold stratified theo `Plaque_present`.
    Dùng cho báo cáo cuối (Phase 2 Phương án B).
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df["Plaque_present"].values
    yield from skf.split(np.zeros(len(df)), y)

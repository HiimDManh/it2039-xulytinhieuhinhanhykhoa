"""
clinical_rules.py
=================

Implementation of clinical rules from the 2025 Focused Update of the
2019 ESC/EAS Guidelines for the management of dyslipidaemias.

Reference:
    Mach F. et al. 2025 Focused Update of the 2019 ESC/EAS Guidelines
    for the management of dyslipidaemias. European Heart Journal
    2025;46(42):4359-4378. https://doi.org/10.1093/eurheartj/ehaf190

Module này tập trung mọi ngưỡng lâm sàng vào một nơi duy nhất, được
dùng chung bởi EDA, baseline, training, và evaluation. KHÔNG hardcode
ngưỡng ở chỗ khác trong codebase.

Author: Đề tài cuối kỳ - Multimodal Fusion Carotid Atherosclerosis
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


# ============================================================================
# 1. NGƯỠNG CHÍNH THỨC TỪ ESC/EAS 2025
# ============================================================================

# LDL-C goals theo Table 4 (mg/dL)
# Vượt ngưỡng = cần can thiệp bằng thuốc hạ lipid
LDL_C_GOAL_MG_DL: dict[str, float] = {
    "Low":       116.0,  # 3.0 mmol/L  - SCORE2 < 2%
    "Moderate":  100.0,  # 2.6 mmol/L  - SCORE2 2 - <10%
    "High":       70.0,  # 1.8 mmol/L  - SCORE2 10 - <20%, hoặc FH, DM ≥10y, mod CKD
    "Very High":  55.0,  # 1.4 mmol/L  - ASCVD, severe CKD, SCORE2 ≥20%, etc.
}

# Lp(a) tiers theo Figure 3 + Box 1 (mg/dL)
LPA_NORMAL_MAX_MG_DL: float = 30.0        # < 30: bình thường
LPA_SLIGHT_MAX_MG_DL: float = 50.0        # 30 - <50: tăng nhẹ
LPA_ELEVATED_THRESHOLD_MG_DL: float = 50.0  # > 50 (105 nmol/L): risk modifier chính thức
LPA_MARKEDLY_ELEVATED_MG_DL: float = 180.0  # > 180 (430 nmol/L): tự động High risk

# Conversion factor: Lp(a) mg/dL → nmol/L ≈ x2.1 (xấp xỉ, varies by isoform)
LPA_MG_TO_NMOL: float = 2.1

# Các nhóm risk category hợp lệ
VALID_RISK_CATEGORIES = ("Low", "Moderate", "High", "Very High")

# 4 class hợp lệ của Plaque_echogenicity (None = control, không có plaque)
# Thứ tự CỐ ĐỊNH — dùng để encode label nhất quán giữa các fold/CV split
ECHOGENICITY_CLASSES = ("None", "Low", "Intermediate", "High")


# ============================================================================
# 2. CORE FUNCTIONS
# ============================================================================

def get_ldl_goal(risk_category: str) -> float:
    """
    Trả về LDL-C goal (mg/dL) cho một risk category, theo Table 4.

    Parameters
    ----------
    risk_category : str
        Một trong {"Low", "Moderate", "High", "Very High"}.

    Returns
    -------
    float
        LDL-C goal trong mg/dL.

    Raises
    ------
    KeyError
        Nếu risk_category không hợp lệ.

    Examples
    --------
    >>> get_ldl_goal("Low")
    116.0
    >>> get_ldl_goal("Very High")
    55.0
    """
    if risk_category not in LDL_C_GOAL_MG_DL:
        raise KeyError(
            f"Invalid risk_category {risk_category!r}. "
            f"Must be one of {VALID_RISK_CATEGORIES}."
        )
    return LDL_C_GOAL_MG_DL[risk_category]


def is_at_ldl_goal(ldl_c_mg_dl: float, risk_category: str) -> bool:
    """
    Kiểm tra bệnh nhân có đạt LDL-C goal cho category của họ không.

    "Đạt goal" = LDL-C strictly less than goal threshold (ngưỡng treatment).

    Examples
    --------
    >>> is_at_ldl_goal(95.0, "Low")    # 95 < 116
    True
    >>> is_at_ldl_goal(110.0, "Moderate")  # 110 > 100
    False
    """
    return ldl_c_mg_dl < get_ldl_goal(risk_category)


def lpa_tier(lpa_mg_dl: float) -> Literal["normal", "slight", "elevated", "markedly_elevated"]:
    """
    Phân tier Lp(a) theo Figure 3 của ESC/EAS 2025.

    Tiers
    -----
    - normal              : < 30 mg/dL (< 62 nmol/L)
    - slight              : 30 - <50 mg/dL
    - elevated            : 50 - <180 mg/dL (≥105 nmol/L) — risk modifier chính thức
    - markedly_elevated   : ≥ 180 mg/dL (≥430 nmol/L) — tự động xếp High risk

    Examples
    --------
    >>> lpa_tier(15.0)
    'normal'
    >>> lpa_tier(60.0)
    'elevated'
    """
    if lpa_mg_dl < LPA_NORMAL_MAX_MG_DL:
        return "normal"
    if lpa_mg_dl < LPA_SLIGHT_MAX_MG_DL:
        return "slight"
    if lpa_mg_dl < LPA_MARKEDLY_ELEVATED_MG_DL:
        return "elevated"
    return "markedly_elevated"


def is_lpa_elevated(lpa_mg_dl: float) -> bool:
    """
    Kiểm tra Lp(a) có ở mức "elevated" theo Box 1 không (>50 mg/dL).

    Đây là risk modifier chính thức trong ESC/EAS 2025.
    """
    return lpa_mg_dl > LPA_ELEVATED_THRESHOLD_MG_DL


def mg_dl_to_nmol_l(lpa_mg_dl: float) -> float:
    """Conversion Lp(a) mg/dL → nmol/L (xấp xỉ)."""
    return lpa_mg_dl * LPA_MG_TO_NMOL


def normalize_echogenicity(series: pd.Series) -> pd.Series:
    """
    Chuẩn hóa cột Plaque_echogenicity.

    Trong CSV gốc, bệnh nhân control (Plaque_present=0) có giá trị
    NaN ở cột này (vì không có mảng bám để đánh giá echogenicity).
    Đây KHÔNG phải missing data — nó là một nhãn lâm sàng hợp lệ
    ("không có plaque"), nên phải chuẩn hóa thành string "None" để
    dùng làm 1 trong 4 class label cho Head 2 (Echogenicity).

    KHÔNG dùng .astype('category').cat.codes để encode trực tiếp:
    pandas gán -1 cho NaN, là class index không hợp lệ cho
    nn.CrossEntropyLoss (4-class) → crash hoặc tính loss sai âm thầm.
    Luôn gọi fillna('None') TRƯỚC khi encode bằng bất kỳ phương pháp nào.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(["Low", None, "High"])
    >>> list(normalize_echogenicity(s))
    ['Low', 'None', 'High']
    """
    return series.fillna("None")


# ============================================================================
# 3. RISK MODIFIERS & RECLASSIFICATION
# ============================================================================

@dataclass
class RiskModifierStatus:
    """Trạng thái risk modifier của một bệnh nhân theo Box 1."""
    high_lpa: bool             # Lp(a) > 50 mg/dL
    has_plaque: bool           # Carotid plaque present
    any_modifier: bool         # Có ít nhất một modifier

    @property
    def n_modifiers(self) -> int:
        return int(self.high_lpa) + int(self.has_plaque)


def get_risk_modifiers(
    lpa_mg_dl: float,
    plaque_present: int | bool,
) -> RiskModifierStatus:
    """
    Tính trạng thái các risk modifier (subset liên quan đến đề tài).

    Box 1 của ESC/EAS 2025 liệt kê nhiều risk modifier; ở đây chỉ
    implement 2 cái dataset có:
        - Elevated Lp(a) > 50 mg/dL  (biomarker)
        - Carotid plaque present     (imaging finding, Class IIa)
    """
    high_lpa = is_lpa_elevated(lpa_mg_dl)
    has_plaque = bool(plaque_present)
    return RiskModifierStatus(
        high_lpa=high_lpa,
        has_plaque=has_plaque,
        any_modifier=(high_lpa or has_plaque),
    )


def needs_reclassification(
    ldl_c_mg_dl: float,
    risk_category: str,
    lpa_mg_dl: float,
    plaque_present: int | bool,
) -> bool:
    """
    Bệnh nhân có cần reclassify lên risk category cao hơn không?

    Theo ESC/EAS 2025:
        Nếu bệnh nhân ở Low/Moderate risk hiện tại và có risk modifier,
        thì cần cân nhắc lên category cao hơn. Plaque trên carotid US
        theo Table 3 tự động đưa vào Very High Risk.

    Logic đơn giản hóa: any modifier → cần reclassify (trừ khi đã ở Very High).
    """
    if risk_category == "Very High":
        return False  # đã ở mức cao nhất
    return get_risk_modifiers(lpa_mg_dl, plaque_present).any_modifier


# ============================================================================
# 4. DISCORDANCE — định nghĩa CHÍNH THỨC
# ============================================================================

def is_discordant(
    ldl_c_mg_dl: float,
    risk_category: str,
    lpa_mg_dl: float,
    plaque_present: int | bool,
) -> bool:
    """
    Discordance theo ESC/EAS 2025.

    Discordant = bệnh nhân ĐẠT LDL-C goal cho category hiện tại
                 (nhìn lipid panel có vẻ ổn)
                 NHƯNG có ≥1 risk modifier (Lp(a) > 50 HOẶC plaque)
                 → cần reclassify lên risk cao hơn

    Đây là nhóm bị bỏ sót nếu chỉ dùng lipid panel — chính là động lực
    của đề tài Multimodal Fusion.

    Reference: Table 3, Table 4, Box 1 của ESC/EAS 2025.

    Examples
    --------
    >>> # LDL 90 mg/dL (đạt goal Low <116), Lp(a) 70 (>50) → discordant
    >>> is_discordant(90.0, "Low", 70.0, 0)
    True
    >>> # LDL 90, Lp(a) bình thường, không plaque → truly low risk
    >>> is_discordant(90.0, "Low", 20.0, 0)
    False
    >>> # LDL 130 (trên goal) → không phải discordant, đã cần điều trị
    >>> is_discordant(130.0, "Low", 70.0, 1)
    False
    """
    at_goal = is_at_ldl_goal(ldl_c_mg_dl, risk_category)
    if not at_goal:
        return False  # đã trên goal, không phải discordant — cần điều trị thẳng
    modifiers = get_risk_modifiers(lpa_mg_dl, plaque_present)
    return modifiers.any_modifier


def classify_discordance_subtype(
    ldl_c_mg_dl: float,
    risk_category: str,
    lpa_mg_dl: float,
    plaque_present: int | bool,
) -> Literal[
    "above_ldl_goal",      # đã vượt goal — cần điều trị, không phải discordant
    "truly_low_risk",      # đạt goal + không có modifier
    "discordant_lpa_only", # đạt goal + chỉ Lp(a) cao
    "discordant_plaque_only", # đạt goal + chỉ có plaque
    "discordant_both",     # đạt goal + cả hai (mạnh nhất)
]:
    """Phân loại chi tiết nhóm bệnh nhân — dùng cho EDA và subgroup analysis."""
    if not is_at_ldl_goal(ldl_c_mg_dl, risk_category):
        return "above_ldl_goal"

    mods = get_risk_modifiers(lpa_mg_dl, plaque_present)
    if not mods.any_modifier:
        return "truly_low_risk"
    if mods.high_lpa and mods.has_plaque:
        return "discordant_both"
    if mods.high_lpa:
        return "discordant_lpa_only"
    return "discordant_plaque_only"


# ============================================================================
# 5. DATAFRAME-LEVEL HELPERS
# ============================================================================

def annotate_dataframe(
    df: pd.DataFrame,
    *,
    ldl_col: str = "LDL_C_mg_dL",
    lpa_col: str = "Lp(a)_mg_dL",
    risk_cat_col: str = "Baseline_Risk_Category",
    plaque_col: str = "Plaque_present",
    plaque_echo_col: str = "Plaque_echogenicity",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Thêm các cột phân tích lâm sàng vào DataFrame.

    Các cột được tạo:
        - ldl_goal_mg_dl       : LDL goal cho category của row
        - at_ldl_goal          : bool, đạt goal hay không
        - lpa_tier             : 'normal' / 'slight' / 'elevated' / 'markedly_elevated'
        - lpa_elevated         : bool, > 50 mg/dL
        - has_risk_modifier    : bool, có ≥1 modifier (Lp(a) cao hoặc plaque)
        - needs_reclassify     : bool, cần lên risk category cao hơn
        - is_discordant        : bool, theo định nghĩa ESC/EAS 2025
        - discordance_subtype  : nhãn chi tiết

    Cột `plaque_echo_col` (mặc định "Plaque_echogenicity") cũng được
    chuẩn hóa tại chỗ: NaN (control) -> "None", để dùng an toàn làm
    4-class label (None/Low/Intermediate/High) ở downstream (Head 2).

    Examples
    --------
    >>> df = pd.read_csv("carotid_clinical_dataset_300cases.csv")
    >>> df = annotate_dataframe(df)
    >>> df['is_discordant'].sum()
    33
    >>> df['Plaque_echogenicity'].isna().sum()
    0
    """
    out = df if inplace else df.copy()

    # Validate cột bắt buộc
    required = {ldl_col, lpa_col, risk_cat_col, plaque_col}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"DataFrame thiếu các cột: {missing}")

    # Validate giá trị risk category
    invalid_cats = set(out[risk_cat_col].unique()) - set(VALID_RISK_CATEGORIES)
    if invalid_cats:
        raise ValueError(
            f"Risk category không hợp lệ: {invalid_cats}. "
            f"Phải nằm trong {VALID_RISK_CATEGORIES}."
        )

    # Chuẩn hóa Plaque_echogenicity: NaN (control) -> "None"
    if plaque_echo_col in out.columns:
        out[plaque_echo_col] = normalize_echogenicity(out[plaque_echo_col])

    # Tính các derived columns
    out["ldl_goal_mg_dl"] = out[risk_cat_col].map(LDL_C_GOAL_MG_DL)
    out["at_ldl_goal"]    = out[ldl_col] < out["ldl_goal_mg_dl"]
    out["lpa_tier"]       = out[lpa_col].apply(lpa_tier)
    out["lpa_elevated"]   = out[lpa_col] > LPA_ELEVATED_THRESHOLD_MG_DL
    out["has_plaque"]     = out[plaque_col].astype(bool)
    out["has_risk_modifier"] = out["lpa_elevated"] | out["has_plaque"]
    out["needs_reclassify"]  = (
        (out[risk_cat_col] != "Very High") & out["has_risk_modifier"]
    )
    out["is_discordant"]  = out["at_ldl_goal"] & out["has_risk_modifier"]

    out["discordance_subtype"] = out.apply(
        lambda r: classify_discordance_subtype(
            r[ldl_col], r[risk_cat_col], r[lpa_col], r[plaque_col]
        ),
        axis=1,
    )
    return out


def discordance_summary(df_annotated: pd.DataFrame) -> pd.DataFrame:
    """
    Tóm tắt phân nhóm theo định nghĩa ESC/EAS 2025.

    Phải gọi annotate_dataframe() trước.
    """
    if "discordance_subtype" not in df_annotated.columns:
        raise ValueError(
            "DataFrame chưa có cột 'discordance_subtype'. "
            "Gọi annotate_dataframe() trước."
        )

    order = [
        "above_ldl_goal",
        "discordant_lpa_only",
        "discordant_plaque_only",
        "discordant_both",
        "truly_low_risk",
    ]
    counts = df_annotated["discordance_subtype"].value_counts().reindex(order, fill_value=0)
    total = len(df_annotated)
    summary = pd.DataFrame({
        "n":       counts.values,
        "percent": (counts.values / total * 100).round(2),
    }, index=counts.index)
    return summary


# ============================================================================
# 6. BASELINE LÂM SÀNG (RULE-BASED)
# ============================================================================

def esceas_2025_rule_predict(
    df: pd.DataFrame,
    *,
    ldl_col: str = "LDL_C_mg_dL",
    lpa_col: str = "Lp(a)_mg_dL",
    risk_cat_col: str = "Baseline_Risk_Category",
) -> np.ndarray:
    """
    Baseline lâm sàng: dự đoán "high risk / cần để ý" dựa trên rule
    của ESC/EAS 2025, chỉ dùng thông tin có ở screening (KHÔNG dùng
    plaque vì plaque là cái ta đang muốn predict).

    Rule:
        Predict positive (high risk) nếu:
            - LDL-C vượt goal cho category, HOẶC
            - Lp(a) > 50 mg/dL (elevated, risk modifier)

    Đây là benchmark "không có deep learning" — cho thấy phương pháp
    truyền thống đoán chính xác bao nhiêu, và bao nhiêu discordant
    case bị miss.

    Returns
    -------
    np.ndarray of int (0/1) với shape (n_samples,)
    """
    ldl_goals  = df[risk_cat_col].map(LDL_C_GOAL_MG_DL).values
    above_goal = df[ldl_col].values >= ldl_goals
    high_lpa   = df[lpa_col].values > LPA_ELEVATED_THRESHOLD_MG_DL
    return (above_goal | high_lpa).astype(int)


# ============================================================================
# 7. SELF-TEST (chạy `python clinical_rules.py` để kiểm tra)
# ============================================================================

def _self_test() -> None:
    """Kiểm tra nhanh logic — chạy như script."""
    print("=" * 60)
    print("clinical_rules.py — Self test")
    print("=" * 60)

    # Test 1: LDL goals
    assert get_ldl_goal("Low") == 116.0
    assert get_ldl_goal("Very High") == 55.0
    print("✓ LDL goal lookup OK")

    # Test 2: At goal
    assert is_at_ldl_goal(100.0, "Low") is True   # 100 < 116
    assert is_at_ldl_goal(120.0, "Low") is False  # 120 >= 116
    print("✓ is_at_ldl_goal OK")

    # Test 3: Lp(a) tiers
    assert lpa_tier(10.0) == "normal"
    assert lpa_tier(40.0) == "slight"
    assert lpa_tier(80.0) == "elevated"
    assert lpa_tier(200.0) == "markedly_elevated"
    print("✓ lpa_tier OK")

    # Test 4: Discordance
    # Đạt goal + Lp(a) cao → discordant
    assert is_discordant(90.0, "Low", 70.0, 0) is True
    # Đạt goal + plaque → discordant
    assert is_discordant(90.0, "Low", 20.0, 1) is True
    # Đạt goal + không gì → truly low
    assert is_discordant(90.0, "Low", 20.0, 0) is False
    # Trên goal → không phải discordant (cần điều trị thẳng)
    assert is_discordant(130.0, "Low", 70.0, 1) is False
    print("✓ is_discordant OK")

    # Test 4b: Echogenicity normalization (NaN -> "None")
    raw = pd.Series(["Low", np.nan, "High", "Intermediate", np.nan])
    normalized = normalize_echogenicity(raw)
    assert normalized.isna().sum() == 0
    assert list(normalized) == ["Low", "None", "High", "Intermediate", "None"]
    assert set(normalized.unique()).issubset(set(ECHOGENICITY_CLASSES))
    print("✓ normalize_echogenicity OK (NaN -> 'None')")

    # Test 5: Apply lên dataset thực (path tương đối từ project/src/)
    from pathlib import Path
    csv_path = Path(__file__).resolve().parent.parent / "data" / "carotid_clinical_dataset_300cases.csv"
    try:
        df = pd.read_csv(csv_path)
        df_ann = annotate_dataframe(df)
        n_discordant = int(df_ann["is_discordant"].sum())
        print(f"\nDataset analysis (n=300):")
        print(discordance_summary(df_ann))
        print(f"\nTổng discordant cases: {n_discordant}")
        assert n_discordant == 33, f"Expected 33 discordant, got {n_discordant}"
        print("✓ Dataset annotation OK — 33 discordant cases (đúng kỳ vọng)")

        # Echogenicity sau annotate phải hết NaN và chỉ chứa 4 class hợp lệ
        n_nan_echo = int(df_ann["Plaque_echogenicity"].isna().sum())
        assert n_nan_echo == 0, f"Còn {n_nan_echo} NaN trong Plaque_echogenicity"
        invalid_echo = set(df_ann["Plaque_echogenicity"].unique()) - set(ECHOGENICITY_CLASSES)
        assert not invalid_echo, f"Echogenicity không hợp lệ: {invalid_echo}"
        n_none = int((df_ann["Plaque_echogenicity"] == "None").sum())
        assert n_none == 205, f"Expected 205 'None' (control), got {n_none}"
        print(f"✓ Plaque_echogenicity normalized OK — 0 NaN, {n_none} 'None' (control)")
    except FileNotFoundError:
        print("(Skip dataset test — file CSV không có)")

    print("\n" + "=" * 60)
    print("All tests PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
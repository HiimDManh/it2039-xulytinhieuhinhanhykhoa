# Pipeline Tổng Thể — Multimodal Fusion Chẩn Đoán Xơ Vữa Động Mạch Cảnh

> Tài liệu này mô tả 7 phase thực hiện đề tài cuối kỳ, theo thứ tự ưu tiên từ bắt buộc đến nâng cao.
>
> **Phiên bản 2.0** — Cập nhật theo *2025 Focused Update of the 2019 ESC/EAS Guidelines for the management of dyslipidaemias* (eur heart j, ehaf190) và dữ liệu thực tế trong CSV.

---

## Tổng Quan

| Phase | Tên | Mức ưu tiên |
|---|---|---|
| 1 | EDA & Hiểu dữ liệu | 🟢 Bắt buộc |
| 2 | Data pipeline & Dataset class | 🟢 Bắt buộc |
| 3 | Baselines đơn phương thức | 🟣 Bắt buộc |
| 4 | Xây dựng Multimodal Fusion Model | 🟣 Bắt buộc |
| 5 | Training & Fusion layer | 🟡 Quan trọng |
| 6 | Đánh giá & Phân tích | 🟡 Quan trọng |
| 7 | Nâng cao & Báo cáo | ⚪ Nâng cao |

---

## Bối Cảnh Lâm Sàng (theo ESC/EAS 2025)

Trước khi đi vào pipeline, ghi nhận các ngưỡng chuẩn từ guideline (ehaf190) được dùng xuyên suốt:

### LDL-C goal theo nhóm nguy cơ (Table 4)

| Risk Category | LDL-C goal (cần can thiệp khi vượt) |
|---|---|
| Low risk (SCORE2 <2%) | < 116 mg/dL (3.0 mmol/L) |
| Moderate risk (SCORE2 2–<10%) | < 100 mg/dL (2.6 mmol/L) |
| High risk | < 70 mg/dL (1.8 mmol/L) |
| Very high risk | < 55 mg/dL (1.4 mmol/L) |

### Lp(a) — Ngưỡng chính thức (Box 1, Figure 3)

| Mức | mg/dL | nmol/L | Ý nghĩa lâm sàng |
|---|---|---|---|
| Bình thường | < 30 | < 62 | Không tăng nguy cơ đáng kể |
| Tăng nhẹ | 30–50 | 62–105 | Tăng nguy cơ nhẹ |
| **Elevated** | **> 50** | **> 105** | **Risk modifier chính thức** |
| Markedly elevated | > 180 | > 430 | Tự động xếp High risk |

### Risk Modifiers (Box 1)
Hai marker có liên quan trực tiếp đến đề tài:
- **Elevated Lp(a) > 50 mg/dL** — biomarker
- **Mảng bám động mạch cảnh/đùi** (Class IIa) — imaging finding

### Định nghĩa Discordance theo guideline

> Bệnh nhân **đạt LDL-C goal cho nhóm nguy cơ hiện tại** (nhìn lipid panel "ổn") NHƯNG có ≥1 risk modifier (Lp(a) > 50 mg/dL HOẶC mảng bám động mạch cảnh) → cần **reclassify** lên nhóm nguy cơ cao hơn.

Cụ thể với mảng bám có ý nghĩa (significant plaque) trên siêu âm: **Table 3 xếp tự động vào Very High Risk** — đây chính là cơ sở lâm sàng để chứng minh giá trị của Fusion model.

---

## Phase 1 — EDA & Hiểu Dữ Liệu

**Mục tiêu:** Nắm rõ dataset trước khi chạy bất kỳ model nào. Con số từ phase này sẽ được trích dẫn xuyên suốt báo cáo.

### 1.1 Số liệu thực tế của dataset

| Thuộc tính | Giá trị |
|---|---|
| Tổng số bệnh nhân | 300 |
| Plaque_present = 0 (Control) | **205 (68.3%)** |
| Plaque_present = 1 (Case) | **95 (31.7%)** |
| Class imbalance ratio | Control : Case ≈ 2.16 : 1 |
| Risk Category (Low) | 293 |
| Risk Category (Moderate) | 7 |
| Tuổi | 30–80 (mean 54.4) |
| Sex | 166 Male / 134 Female |
| Lp(a) range | 5.4 – 131.6 mg/dL |
| IMT range | 0.45 – 0.88 mm |

> **Lưu ý quan trọng:** Dataset KHÔNG khớp với mô tả ban đầu "100 control / 200 case". Số thực tế là **205/95**, với Control là majority class. Toàn bộ chiến lược chia split và xử lý imbalance cần điều chỉnh theo con số thực này.

### 1.2 Phân tầng Lp(a) theo ESC/EAS 2025 (Figure 3)

| Tier | mg/dL | n | % | Có plaque | Tỷ lệ plaque |
|---|---|---|---|---|---|
| Bình thường | <30 | 169 | 56.3% | 48 | 28.4% |
| Tăng nhẹ | 30–50 | 86 | 28.7% | 30 | 34.9% |
| Elevated | >50 | 45 | 15.0% | 17 | **37.8%** |
| Markedly elevated | >180 | 0 | 0.0% | — | — |

→ Có gradient sinh học tăng dần theo Lp(a) tier, đúng với cơ chế trong guideline.

### 1.3 Phân tích Discordance theo ESC/EAS 2025 ⭐

```python
def get_ldl_goal(risk_category):
    """LDL-C goal theo Table 4 ESC/EAS 2025"""
    return {
        'Low':       116,  # 3.0 mmol/L
        'Moderate':  100,  # 2.6 mmol/L
        'High':       70,  # 1.8 mmol/L
        'Very High':  55,  # 1.4 mmol/L
    }[risk_category]

# Phân nhóm theo guideline
df['ldl_goal']      = df['Baseline_Risk_Category'].map(get_ldl_goal)
df['at_ldl_goal']   = df['LDL_C_mg_dL'] < df['ldl_goal']
df['high_lpa']      = df['Lp(a)_mg_dL'] > 50          # Box 1 ESC/EAS 2025
df['has_modifier']  = df['high_lpa'] | (df['Plaque_present'] == 1)

# DISCORDANT = nhìn lipid panel ổn nhưng có risk modifier
df['discordant']    = df['at_ldl_goal'] & df['has_modifier']
```

**Kết quả phân nhóm trên 300 bệnh nhân:**

| Nhóm | n | Ý nghĩa lâm sàng |
|---|---|---|
| Trên ngưỡng LDL goal | 210 | Đã cần intervention |
| Đạt LDL goal nhưng có risk modifier (**DISCORDANT**) | **33** | **Nhóm phân tích chính — bị bỏ sót nếu chỉ dùng lipid panel** |
| → chỉ Lp(a) > 50 | 7 | |
| → chỉ có plaque | 22 | |
| → cả hai | 4 | |
| Truly low risk (đạt goal + không modifier) | 57 | |

→ **33 discordant cases** đủ để phân tích có ý nghĩa (vs 1 case theo định nghĩa sai trước đây).

### 1.4 Các việc cần làm trong EDA

- Phân tích phân bố các biến lipid: Lp(a), ApoB, LDL-C, Triglyceride, Total Cholesterol, Non-HDL
- Boxplot/histogram của từng biến lipid theo Plaque_present
- Visualize correlation matrix giữa các biomarkers
- Kiểm tra liên kết sinh học: Lp(a) cao ↔ Plaque_echogenicity = Low (vulnerable plaque)
- Phân bố echogenicity: Low=28, Intermediate=40, High=27 (cân bằng 3 lớp)
- Cross-tab Lp(a) tier × Plaque (đã có ở 1.2)
- Định nghĩa và đếm Discordance cases theo công thức ở 1.3
- **KHÔNG dùng định nghĩa cũ "LDL < 100 AND Lp(a) > 50 AND plaque=1"** — chỉ cho 1 case, vi phạm chuẩn guideline

**Output:** Notebook `01_eda.ipynb` với đầy đủ biểu đồ và nhận xét, trích dẫn ESC/EAS 2025.

---

## Phase 2 — Data Pipeline & Dataset Class

**Mục tiêu:** Xây dựng pipeline dữ liệu hoàn chỉnh, sẵn sàng cho training.

### 2.1 Chiến lược chia dataset (cập nhật theo số thực 205/95)

**Phương án A — Stratified 70/15/15 split** (đơn giản, dùng cho prototype):

```
Tổng: 300 bệnh nhân (205 control + 95 case)
├── Train:       210 (70%) — 143 control + 67 case
├── Validation:   45 (15%) —  31 control + 14 case
└── Test:         45 (15%) —  31 control + 14 case
```

⚠️ Với chỉ ~14 case trong val/test, AUC có thể không ổn định giữa các run.

**Phương án B — Stratified 5-fold Cross-Validation** (khuyến nghị cho báo cáo cuối kỳ):

```
5 folds × 60 patients/fold
Mỗi fold: 41 control + 19 case (giữ tỷ lệ 2.16:1)
Báo cáo: mean ± std của AUC, F1 trên 5 fold
```

> Phương án B cho ước lượng performance ổn định hơn với dataset nhỏ, và đủ tổng số discordant unseen (~33) để phân tích subgroup ở Phase 6.

**Lưu ý:** Cả hai phương án chia theo **patient-level**, không chia theo ảnh — tránh data leakage (một bệnh nhân có thể có 5 ảnh).

### 2.2 CarotidDataset class

**Nhánh ảnh (CNN):**
- Control (Plaque_present = 0, n=205): 1 ảnh IMT → sử dụng trực tiếp
- Case (Plaque_present = 1, n=95): 5 ảnh (1 IMT + 4 cross-section) → **mean embedding** hoặc **max-pooling** trên 5 image embeddings
- Augmentation chỉ áp dụng cho training set

**Nhánh tabular (MLP):** 9 input features
- `Age`, `Sex`, `Lp(a)_mg_dL`, `ApoB_mg_dL`, `LDL_C_mg_dL`, `Triglyceride_mg_dL`, `Total_Cholesterol_mg_dL`, `Non_HDL_mg_dL`, `IMT_mm`
- `Sex`: encode Male=1, Female=0
- **Không đưa `Baseline_Risk_Score`** vào input (derived feature → data leakage)
- `StandardScaler`: fit trên train set, transform val/test set

### 2.3 Image transforms

```python
# Training
train_transforms = [
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=(-10, 10)),
    RandomAffine(degrees=0, translate=(0.05, 0.05)),
    GaussianBlur(kernel_size=3, p=0.3),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

# Validation / Test (không augment)
val_transforms = [
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
```

**Output:** File `src/dataset.py` với `CarotidDataset` class và `DataLoader` cho train/val/test.

---

## Phase 3 — Baselines Đơn Phương Thức

**Mục tiêu:** Có điểm so sánh trước khi xây dựng Fusion model. Đây là con số bắt buộc để Phase 6 chứng minh Fusion tốt hơn.

### Baseline 1: CNN-only

- ResNet-50 pretrained ImageNet
- Chỉ dùng nhánh ảnh, không có tabular data
- Output: Binary classification (Plaque_present)
- Weighted loss vì Control là majority (2.16:1)
- Tham khảo: Gao et al. (2026), He et al. (2024)

### Baseline 2: MLP-only

- Tabular model với 9 features
- Có thể thử Random Forest / XGBoost để so sánh thêm
- Tham khảo: Vu et al. (2025) — LASSO + SHAP feature selection

### Baseline 3 (mới — chuẩn lâm sàng): Rule-based theo ESC/EAS 2025

> Đây là baseline lâm sàng quan trọng — cho thấy phương pháp truyền thống miss bao nhiêu case discordant.

```python
def esceas_rule(row):
    """Reclassify theo guideline:
       cao risk nếu LDL trên goal HOẶC có risk modifier"""
    above_goal = row['LDL_C_mg_dL'] >= get_ldl_goal(row['Baseline_Risk_Category'])
    has_modifier = (row['Lp(a)_mg_dL'] > 50)  # Plaque chưa biết tại screening
    return int(above_goal or has_modifier)
```

So sánh: Rule này dự đoán Plaque_present chính xác bao nhiêu? Đây là benchmark "human baseline".

**Metrics cần ghi lại:**

| Metric | Lý do |
|---|---|
| AUC-ROC | Chuẩn chính theo 4 bài báo |
| F1-score | Xử lý imbalance |
| Sensitivity (Recall) | Quan trọng trong lâm sàng |
| Specificity | Tránh over-diagnosis |
| **Sensitivity trên nhóm discordant** | Metric mới — đo trực tiếp việc giải quyết discordance |

**Output:** Notebook `02_baseline.ipynb` + bảng kết quả lưu vào `results/`.

---

## Phase 4 — Xây Dựng Multimodal Fusion Model

**Mục tiêu:** Lắp ghép hai nhánh thành một pipeline end-to-end.

### 4.1 Branch 1 — CNN (Image Branch)

```
Input: US Images [B, C, H, W]
    ↓
ResNet-50 (pretrained ImageNet)
    ↓
Global Average Pooling                  → [B, 2048]
    ↓
Projection Linear(2048 → 128)           ← MỚI: để cân bằng với clinical branch
    ↓
BatchNorm1d + ReLU + Dropout(0.3)
    ↓
Image Embedding [B, 128]
```

> **Thay đổi quan trọng:** Thay vì truyền thẳng 2048-dim (output GAP của ResNet-50) sang fusion, thêm **projection layer xuống 128-dim**. Lý do: clinical embedding chỉ 64-dim → concat 2048+64 sẽ khiến nhánh ảnh "nuốt" tín hiệu lâm sàng. Sau projection, fusion concat thành [B, 128+64=192] cân bằng hơn.

**Fine-tuning 3 giai đoạn:**

| Giai đoạn | Hành động | Epochs |
|---|---|---|
| 1 | Freeze toàn bộ ResNet-50, chỉ train projection + fusion + heads | 5 |
| 2 | Unfreeze layer4 của ResNet-50, tiếp tục train | 15 |
| 3 | Unfreeze toàn bộ, learning rate rất nhỏ | 10 |

### 4.2 Branch 2 — MLP (Tabular Branch)

```
Input: [B, 9]
    ↓
Linear(9 → 64) + BatchNorm1d + ReLU
    ↓
Linear(64 → 128) + BatchNorm1d + ReLU + Dropout(0.2)
    ↓
Linear(128 → 64) + BatchNorm1d + ReLU
    ↓
Clinical Embedding [B, 64]
```

### 4.3 Fusion Layer

**Phương pháp 1 — Concatenation (bắt đầu với cái này):**

```python
fused = torch.cat([image_embedding, clinical_embedding], dim=1)
# Shape: [B, 128 + 64] = [B, 192]   ← cập nhật sau khi thêm projection
```

```
[B, 192]
    ↓
FC(192 → 128) + BatchNorm + ReLU + Dropout(0.5)
    ↓
FC(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
```

**Phương pháp 2 — Cross-Attention (nâng cao, Phase 7):**

```python
# image_embedding làm Query, clinical_embedding làm Key/Value
attention_weight = softmax(Q @ K.T / sqrt(d_k))
fused = attention_weight @ V + image_embedding  # residual
```

### 4.4 Output Heads (Multi-task)

| Head | Task | Architecture | Loss |
|---|---|---|---|
| Head 1 | Plaque Detection | FC(64 → 2) + Softmax | Weighted Binary CrossEntropy |
| Head 2 | Echogenicity Classification | FC(64 → 4) + Softmax | Weighted CrossEntropy |
| Head 3 | Reclassification (ESC/EAS) | FC(64 → 1) + Sigmoid | Binary CrossEntropy |

> **Lưu ý Head 2:** Phải có 4 lớp (None / Low / Intermediate / High), không phải 3, vì 205 control có nhãn echogenicity = "None".
>
> **Head 3 — gợi ý mới:** Thay Risk Score regression bằng nhãn "cần reclassify lên risk cao hơn theo guideline" (binary). Đây là output có ý nghĩa lâm sàng trực tiếp, gắn với ESC/EAS 2025.

**Output:** Files `src/models/cnn_branch.py`, `mlp_branch.py`, `fusion.py`, `multimodal.py`.

---

## Phase 5 — Training & Fusion Layer

**Mục tiêu:** Huấn luyện toàn bộ pipeline Fusion end-to-end.

### 5.1 Multi-task Loss

```python
total_loss = 1.0 * L_detection + 0.5 * L_echogenicity + 0.3 * L_reclassify
```

### 5.2 Xử lý Class Imbalance (CẬP NHẬT — Control là majority)

```python
# Plaque detection: Control 205 vs Case 95 → weight Case cao hơn
weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
# weights ≈ [0.73, 1.58]   (vs trước đây tưởng [1.5, 0.75])
criterion = CrossEntropyLoss(weight=torch.tensor(weights))

# Echogenicity: None=205, Low=28, Inter=40, High=27 → imbalance NẶNG hơn nhiều
echo_weights = compute_class_weight('balanced', classes=[0,1,2,3], y=y_echo_train)
```

### 5.3 Optimizer (learning rate phân biệt)

```python
optimizer = AdamW([
    {'params': cnn_branch.parameters(),       'lr': 1e-5},  # Nhỏ hơn vì pretrained
    {'params': cnn_projection.parameters(),   'lr': 1e-4},  # Layer mới, lr cao hơn
    {'params': mlp_branch.parameters(),       'lr': 1e-4},
    {'params': fusion_layer.parameters(),     'lr': 1e-4},
    {'params': output_heads.parameters(),     'lr': 1e-4}
])
```

### 5.4 Hyperparameters

```python
config = {
    'batch_size': 16,
    'epochs': 50,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'scheduler': 'CosineAnnealingLR',
    'warmup_epochs': 5,
    'gradient_clip': 1.0
}
```

**Output:** File `src/train.py` + checkpoints lưu vào `results/checkpoints/`.

---

## Phase 6 — Đánh Giá & Phân Tích

**Mục tiêu:** Chứng minh Fusion tốt hơn single-modality và giải quyết bài toán Discordance theo định nghĩa chuẩn ESC/EAS 2025.

### 6.1 Baseline so sánh

| Model | Mô tả |
|---|---|
| ESC/EAS Rule | Quy tắc lâm sàng từ guideline (benchmark "không có DL") |
| CNN-only | ResNet-50 chỉ trên ảnh |
| MLP-only | Tabular model (RF / XGBoost / MLP) |
| Fusion (Concat) | Kiến trúc đề xuất — baseline |
| Fusion (Attention) | Kiến trúc đề xuất — nâng cao |

### 6.2 Ablation Study

| Experiment | Mô tả | Mục đích |
|---|---|---|
| w/o Lp(a) | Bỏ Lp(a) khỏi tabular input | Chứng minh vai trò Lp(a) theo guideline |
| w/o ApoB | Bỏ ApoB khỏi tabular input | Chứng minh vai trò ApoB |
| w/o IMT_mm | Bỏ IMT tabular, chỉ dùng ảnh | So sánh IMT từ ảnh vs tabular |
| w/o projection (CNN) | Bỏ projection layer 2048→128 | Chứng minh giá trị của projection |
| Image branch only | CNN không có fusion | Baseline CNN |
| Tabular branch only | MLP không có fusion | Baseline MLP |
| Full model | Tất cả features + fusion | Kết quả tốt nhất |

### 6.3 Phân Tích Discordance theo ESC/EAS 2025 ⭐⭐⭐

**Đây là đóng góp quan trọng nhất của đề tài (Novelty), và là phần khác biệt rõ nhất so với các paper hiện có.**

```python
def identify_discordant(df):
    """Định nghĩa Discordance CHÍNH THỨC theo ESC/EAS 2025
       (ehaf190, Table 3+4, Box 1)
       
       Bệnh nhân 'đạt LDL goal cho category hiện tại' nhưng có
       risk modifier (Lp(a) > 50 hoặc plaque) — cần reclassify."""
    
    ldl_goal_map = {'Low': 116, 'Moderate': 100, 'High': 70, 'Very High': 55}
    df['ldl_goal']    = df['Baseline_Risk_Category'].map(ldl_goal_map)
    df['at_goal']     = df['LDL_C_mg_dL'] < df['ldl_goal']
    df['high_lpa']    = df['Lp(a)_mg_dL'] > 50
    df['has_plaque']  = df['Plaque_present'] == 1
    df['discordant']  = df['at_goal'] & (df['high_lpa'] | df['has_plaque'])
    return df

# Áp dụng trên unseen test fold predictions (gộp toàn bộ 5-fold CV)
# → có ~33 discordant cases trong tập unseen
```

**So sánh trên nhóm discordant:**

| Model | AUC overall | AUC trên discordant | Sensitivity trên discordant |
|---|---|---|---|
| ESC/EAS Rule | — | — | — |
| CNN-only | | | |
| MLP-only | | | |
| Fusion | **kỳ vọng cao nhất** | | |

**Câu hỏi trả lời:**
- Có bao nhiêu discordant cases bị **MLP-only miss** mà Fusion bắt được? (vì lipid panel "ổn")
- Có bao nhiêu discordant cases bị **CNN-only miss** mà Fusion bắt được? (vì plaque nhỏ/khó nhìn)
- → Đây là giá trị lâm sàng cụ thể của Fusion.

**Metrics đánh giá:**

| Task | Metric |
|---|---|
| Plaque Detection | AUC-ROC, F1-score, Sensitivity, Specificity |
| Echogenicity | F1-score (macro), Accuracy per class |
| Reclassification | AUC-ROC, F1-score, Cohen's Kappa |
| Discordance subgroup | Sensitivity, NPV trên 33 discordant cases |

**Output:** Notebook `03_fusion.ipynb` + file `src/evaluate.py` + figures lưu vào `results/figures/`.

---

## Phase 7 — Nâng Cao & Báo Cáo

**Mục tiêu:** Hoàn thiện model và viết báo cáo trả lời 4 Research Questions.

### 7.1 Nâng cao kỹ thuật

- Thử **Cross-Attention Fusion** và so sánh với Concat (trả lời RQ4)
- **Grad-CAM visualization**: visualize vùng mảng bám CNN đang chú ý
- Thử tăng image size lên 384×384 nếu GPU cho phép (224 → 384, ResNet-50 vẫn ổn)
- Thử CLAHE preprocessing cho US images

### 7.2 Research Questions cần trả lời trong báo cáo

| RQ | Câu hỏi | Trả lời bằng |
|---|---|---|
| RQ1 | Fusion có AUC cao hơn CNN-only và MLP-only không? | Bảng so sánh baseline |
| RQ2 | Thêm Lp(a)/ApoB có cải thiện echogenicity accuracy không? | Ablation study |
| RQ3 | Fusion giải quyết Discordance (theo ESC/EAS 2025) tốt hơn single-modality không? | Discordance subgroup analysis trên 33 cases |
| RQ4 | Concat hay Cross-Attention cho kết quả tốt hơn (n=300)? | Fusion strategy comparison |

### 7.3 Lưu ý khi viết báo cáo

- Trích dẫn chính: **Mach F. et al., 2025 Focused Update of the 2019 ESC/EAS Guidelines for the management of dyslipidaemias, Eur Heart J 2025;46:4359–4378** — đây là cơ sở khoa học của bài toán
- Dataset là **synthetic** — không claim clinical validity trực tiếp, nhưng tuân theo logic sinh học hợp lý (Lp(a) tier ↔ plaque rate)
- 300 cases là nhỏ — nhấn mạnh regularization, augmentation, và 5-fold CV
- Kết quả Fusion > Single-modality là expected và phải có ablation để chứng minh contribution của từng nhánh
- Phần discordance phải nói rõ định nghĩa theo guideline, không tự đặt ngưỡng

---

## Cấu Trúc Thư Mục Đề Xuất

```
project/
├── data/
│   ├── carotid_clinical_dataset_300cases.csv
│   ├── ehaf190.pdf                        ← Guideline gốc làm reference
│   └── CAROTID_IMAGES/
│       ├── control/        (205 subjects × 1 image)
│       └── cases/          (95 subjects × 5 images)
│
├── src/
│   ├── dataset.py
│   ├── models/
│   │   ├── cnn_branch.py
│   │   ├── mlp_branch.py
│   │   ├── fusion.py
│   │   └── multimodal.py
│   ├── train.py
│   ├── evaluate.py
│   ├── clinical_rules.py   ← MỚI: implement ESC/EAS rules (LDL goal, Lp(a) tier, discordance)
│   └── utils.py
│
├── configs/
│   └── default.yaml
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_fusion.ipynb
│
└── results/
    ├── checkpoints/
    └── figures/
```

---

## Kết Nối Với Bài Báo Tham Khảo

| Thành phần | Bài báo nguồn |
|---|---|
| **Định nghĩa bài toán (Lp(a), LDL goal, Discordance)** | **Mach et al. (Eur Heart J 2025) — ESC/EAS Focused Update** |
| ResNet-50 backbone | Gao et al. (2026), He et al. (2024) |
| Echogenicity classification | Ren et al. (2026) |
| IMT_mm là clinical predictor | Vu et al. (2025) — top-1 SHAP feature |
| Tabular feature selection | Vu et al. (2025) — LASSO + SHAP |
| Fusion với clinical data | Gao et al. (2026) — future work → đề tài này thực hiện |
| Data augmentation cho US | He et al. (2024) — flip, rotation |

---

*Pipeline v2.0 — Được cập nhật để phù hợp với dữ liệu thực tế (205 control / 95 case) và chuẩn lâm sàng ESC/EAS 2025 (ehaf190).*
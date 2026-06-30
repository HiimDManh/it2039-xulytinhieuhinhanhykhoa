# Kiến Trúc Multimodal Fusion cho Chẩn Đoán Xơ Vữa Động Mạch Cảnh

> **Tài liệu thiết kế kỹ thuật** — Dùng làm context cho Claude trong suốt quá trình phát triển đề tài cuối kỳ.
>
> **Phiên bản 2.0** — Cập nhật theo *2025 Focused Update of the 2019 ESC/EAS Guidelines for the management of dyslipidaemias* (Mach F. et al., Eur Heart J 2025;46:4359–4378) và dữ liệu thực tế trong dataset.

---

## 1. Tổng Quan Đề Tài

### 1.1 Bài Toán Cần Giải Quyết

Xơ vữa động mạch cảnh (carotid atherosclerosis) là nguyên nhân của ~25% ca đột quỵ thiếu máu não. Theo ESC/EAS 2025 Focused Update, có hai hướng tiếp cận chính trong đánh giá nguy cơ:

- **Hướng 1 — Lipid Screening (tabular):** Đo LDL-C, ApoB, Lp(a). Phân tầng nguy cơ theo SCORE2/SCORE2-OP rồi đặt LDL-C goal tương ứng (Table 4 guideline). **Giới hạn:** có nhóm bệnh nhân đạt LDL-C goal nhưng vẫn có Lp(a) tăng cao (>50 mg/dL) hoặc mảng bám trên siêu âm — đây là cơ sở của "Discordance" theo guideline.

- **Hướng 2 — Imaging (ultrasound):** Siêu âm động mạch cảnh để đo IMT và đánh giá echogenicity mảng bám. Guideline 2025 ghi nhận significant plaque trên carotid US **tự động xếp bệnh nhân vào Very High Risk** (Table 3). **Giới hạn:** phụ thuộc kỹ năng bác sĩ, chưa tích hợp thông tin lipid burden.

**Mục tiêu của đề tài:** Xây dựng mô hình Multimodal Fusion kết hợp cả hai nguồn dữ liệu, giải quyết bài toán Discordance theo đúng định nghĩa ESC/EAS 2025.

### 1.2 Cơ Sở Lâm Sàng — Ngưỡng Chính Thức ESC/EAS 2025

**LDL-C goal theo nhóm nguy cơ (Table 4):**

| Risk Category | LDL-C goal |
|---|---|
| Low risk (SCORE2 <2%) | < 116 mg/dL (3.0 mmol/L) |
| Moderate risk (SCORE2 2–<10%) | < 100 mg/dL (2.6 mmol/L) |
| High risk | < 70 mg/dL (1.8 mmol/L) |
| Very high risk | < 55 mg/dL (1.4 mmol/L) |

**Lp(a) — phân tầng theo Figure 3 + Box 1:**

| Mức | mg/dL | nmol/L | Ý nghĩa |
|---|---|---|---|
| Bình thường | < 30 | < 62 | — |
| Tăng nhẹ | 30–50 | 62–105 | Cân nhắc |
| **Elevated (chính thức)** | **> 50** | **> 105** | **Risk modifier — Box 1** |
| Markedly elevated | > 180 | > 430 | Tự động High risk |

**Risk modifiers liên quan đến đề tài (Box 1):**
- Elevated Lp(a) > 50 mg/dL
- Mảng bám động mạch cảnh/đùi (Class IIa)

### 1.3 Định Nghĩa Discordance Chính Thức

> **Discordance (theo ESC/EAS 2025):** Bệnh nhân **đạt LDL-C goal cho nhóm nguy cơ hiện tại** (nhìn lipid panel "ổn") NHƯNG có ≥1 risk modifier — cần **reclassify** lên nhóm nguy cơ cao hơn.

Áp dụng vào dataset của đề tài: **33 bệnh nhân discordant** trong tổng 300 (xem chi tiết Section 2.4).

### 1.4 Đóng Góp Khoa Học (Novelty)

Dựa trên phân tích 4 bài báo liên quan (Ren et al. 2026, Vu et al. 2025, Gao et al. 2026, He et al. 2024), không có nghiên cứu nào kết hợp:
- Ảnh siêu âm mảng bám + biomarkers lâm sàng (đặc biệt Lp(a), ApoB) trong cùng pipeline end-to-end
- Đặc biệt: chưa có nghiên cứu nào nhắm thẳng vào nhóm "discordant" theo định nghĩa guideline mới
- Gao et al. (2026) là bài gần nhất — họ đề xuất hướng này trong "future work" nhưng chưa thực hiện

**Contribution cốt lõi:** Chứng minh rằng Fusion model:
1. Cải thiện AUC tổng thể so với single-modality models
2. Đặc biệt cải thiện sensitivity trên nhóm bệnh nhân **discordant theo ESC/EAS 2025** — nhóm mà phương pháp truyền thống chỉ dựa vào lipid panel sẽ bỏ sót

---

## 2. Dataset

### 2.1 Tổng Quan Dataset

| Thuộc tính | Giá trị |
|---|---|
| Tên file | `carotid_clinical_dataset_300cases.csv` / `.xlsx` |
| Số bệnh nhân | 300 |
| Số cột | 15 |
| Loại dữ liệu | Synthetic (biologically-constrained) |
| Mục tiêu thiết kế | Huấn luyện và validation mô hình Multimodal Deep Learning |

### 2.2 Cấu Trúc Dữ Liệu Bảng (Tabular)

#### Nhân khẩu học
| Cột | Kiểu | Mô tả |
|---|---|---|
| `Patient_ID` | String | Định danh bệnh nhân |
| `Age` | Integer | Tuổi (30–80, mean 54.4) |
| `Sex` | String | Male (166) / Female (134) |

#### Hồ Sơ Lipid (mg/dL)
| Cột | Kiểu | Vai trò trong mô hình |
|---|---|---|
| `Lp(a)_mg_dL` | Float | **Risk modifier chính theo ESC/EAS 2025** — ngưỡng 50 mg/dL |
| `ApoB_mg_dL` | Float | Marker lipid atherogenic |
| `LDL_C_mg_dL` | Float | So với LDL goal của nhóm nguy cơ |
| `Triglyceride_mg_dL` | Float | Lipid phụ |
| `Total_Cholesterol_mg_dL` | Float | Cholesterol tổng |
| `Non_HDL_mg_dL` | Float | Lipid non-protective |

#### Marker Lâm Sàng
| Cột | Kiểu | Mô tả |
|---|---|---|
| `IMT_mm` | Float | Intima-Media Thickness (mm), range 0.45–0.88 |

#### Nhãn và Phân Tầng Nguy Cơ
| Cột | Kiểu | Mô tả |
|---|---|---|
| `Plaque_present` | Integer (0/1) | **Target chính** — 0: Control, 1: Có mảng bám |
| `Plaque_echogenicity` | String | None / Low / Intermediate / High (4 lớp, vì 205 control = "None") |
| `Baseline_Risk_Score` | Float | Điểm nguy cơ (range 0.000–2.370) — **derived feature, KHÔNG dùng làm input** |
| `Baseline_Risk_Category` | String | Trong dataset: chỉ có Low (293) và Moderate (7) |
| `Associated_Images` | String | Đường dẫn ảnh siêu âm |

### 2.3 Phân Bố Dataset Thực Tế

> **⚠️ Cập nhật quan trọng:** Số liệu thực tế trong CSV KHÁC với mô tả ban đầu.

- **Control (Plaque_present = 0):** **205** bệnh nhân — mỗi người có 1 ảnh IMT
- **Case (Plaque_present = 1):** **95** bệnh nhân — mỗi người có 1 ảnh IMT + 4 ảnh cross-section

Imbalance ratio: **Control : Case = 2.16 : 1** (Control là majority — ngược với mô tả ban đầu).

**Phân bố Plaque_echogenicity:**

| Echogenicity | n | Ý nghĩa |
|---|---|---|
| None (control) | 205 | Không có mảng bám |
| Low (echolucent) | 28 | Vulnerable plaque, giàu lipid |
| Intermediate | 40 | Mảng bám trung bình |
| High (echogenic) | 27 | Stable plaque, calcified |

→ **Head Echogenicity phải là 4-class**, không phải 3-class như thiết kế cũ.

### 2.4 Phân Tích Discordance theo ESC/EAS 2025

Áp dụng định nghĩa chính thức của guideline vào 300 bệnh nhân:

```python
ldl_goal_map = {'Low': 116, 'Moderate': 100, 'High': 70, 'Very High': 55}
df['ldl_goal']   = df['Baseline_Risk_Category'].map(ldl_goal_map)
df['at_goal']    = df['LDL_C_mg_dL'] < df['ldl_goal']
df['high_lpa']   = df['Lp(a)_mg_dL'] > 50
df['has_plaque'] = df['Plaque_present'] == 1
df['discordant'] = df['at_goal'] & (df['high_lpa'] | df['has_plaque'])
```

**Kết quả:**

| Nhóm | n | % |
|---|---|---|
| Trên ngưỡng LDL goal (cần intervention) | 210 | 70.0% |
| Đạt LDL goal | 90 | 30.0% |
| ├── **Discordant** (có risk modifier) | **33** | **11.0%** |
| │   ├── chỉ Lp(a) > 50 | 7 | |
| │   ├── chỉ có plaque | 22 | |
| │   └── có cả hai | 4 | |
| └── Truly low risk | 57 | 19.0% |

**Lp(a) tier × Plaque cross-tab:**

| Lp(a) tier | n | Có plaque | % có plaque |
|---|---|---|---|
| < 30 mg/dL | 169 | 48 | 28.4% |
| 30–50 mg/dL | 86 | 30 | 34.9% |
| > 50 mg/dL | 45 | 17 | **37.8%** |

→ Gradient sinh học hợp lý: Lp(a) cao → khả năng có plaque tăng. Phù hợp với Figure 3 của guideline.

### 2.5 Liên Kết Sinh Học (Biologically-Constrained Logic)

Dataset được thiết kế với ràng buộc sinh học:

```
Lp(a) cao  →  Plaque echogenicity = Low (echolucent)
                  ↓
              Mảng bám giàu lipid, mỏng fibrous cap
                  ↓
              Nguy cơ vỡ → đột quỵ cao hơn
```

**Ý nghĩa cho mô hình:** Branch tabular (MLP) học correlation Lp(a) → echogenicity. Branch imaging (CNN) nhìn thấy echogenicity trực tiếp từ ảnh. Fusion layer kết hợp hai nguồn này.

### 2.6 Chiến Lược Chia Dataset (cập nhật theo 205/95)

**Phương án A — Stratified 70/15/15:**

```
Tổng: 300 bệnh nhân
├── Train:      210 (70%) — 143 control + 67 case
├── Validation:  45 (15%) —  31 control + 14 case
└── Test:        45 (15%) —  31 control + 14 case
```

**Phương án B — Stratified 5-fold CV** (khuyến nghị):

```
5 folds × 60 patients/fold
Mỗi fold: ~41 control + ~19 case (giữ tỷ lệ 2.16:1)
Báo cáo mean ± std của AUC/F1 trên 5 fold
Gộp test predictions → ~33 discordant unseen cho Phase 6
```

> Chia theo bệnh nhân (patient-level split) để tránh data leakage.

---

## 3. Kiến Trúc Mô Hình

### 3.1 Sơ Đồ Tổng Thể (cập nhật)

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION MODEL                  │
├──────────────────────┬──────────────────────────────────────┤
│   BRANCH 1 — CNN     │        BRANCH 2 — MLP                │
│   (Image Branch)     │        (Tabular Branch)              │
│                      │                                      │
│  Input: US Images    │  Input: Clinical Tabular Data        │
│  [B, C, H, W]        │  [B, 9]                              │
│         ↓            │         ↓                            │
│   Preprocessing      │   Feature Normalization              │
│   (resize 224,norm)  │   (StandardScaler)                   │
│         ↓            │         ↓                            │
│   ResNet-50          │   FC Layer (9 → 64)                  │
│   (pretrained        │   BatchNorm + ReLU                   │
│    ImageNet)         │         ↓                            │
│         ↓            │   FC Layer (64 → 128)                │
│   GAP → [B, 2048]    │   BatchNorm + ReLU + Dropout(0.2)    │
│         ↓            │         ↓                            │
│   Projection         │   FC Layer (128 → 64)                │
│   (2048 → 128) ★MỚI  │   BatchNorm + ReLU                   │
│   BN + ReLU + Drop   │         ↓                            │
│         ↓            │   Clinical Embedding                 │
│   Image Embedding    │   [B, 64]                            │
│   [B, 128]           │                                      │
└──────────┬───────────┴───────────────┬──────────────────────┘
           │                           │
           └───────────┬───────────────┘
                       ↓
           ┌───────────────────────┐
           │     FUSION LAYER      │
           │  Concatenation:       │
           │  [B,128] ⊕ [B,64]    │
           │       = [B, 192]      │
           │           ↓           │
           │  FC (192 → 128)       │
           │  BatchNorm + ReLU     │
           │  Dropout(0.5)         │
           │           ↓           │
           │  FC (128 → 64)        │
           │  BatchNorm + ReLU     │
           │  Dropout(0.3)         │
           └───────────┬───────────┘
                       ↓
           ┌───────────────────────┐
           │     OUTPUT HEADS      │
           ├───────────────────────┤
           │  Head 1: Plaque       │
           │  Detection            │
           │  FC(64 → 2) Softmax   │
           ├───────────────────────┤
           │  Head 2: Echogenicity │
           │  FC(64 → 4) Softmax   │
           │  → None/Low/Int/High  │
           ├───────────────────────┤
           │  Head 3: Reclassify   │
           │  (theo ESC/EAS 2025)  │
           │  FC(64 → 1) Sigmoid   │
           │  → Cần lên risk cao?  │
           └───────────────────────┘
```

### 3.2 Branch 1 — CNN (Image Branch)

**Backbone:** ResNet-50 pretrained trên ImageNet

**Lý do chọn ResNet-50:**
- Được dùng làm chuẩn trong Gao et al. (2026) và He et al. (2024) với kết quả tốt nhất
- Đủ sâu để học subvisual texture features của mảng bám
- Lightweight — inference < 15ms/image
- Tránh overfitting tốt hơn các kiến trúc phức tạp hơn (ViT, ConvNeXt) khi dataset chỉ 300 cases

**★ Thêm Projection Layer (thay đổi so với v1):**

Trước đây image embedding 2048-dim concat với clinical 64-dim → tỷ lệ 32:1, fusion bị dominated bởi nhánh ảnh.

Giải pháp: Sau GAP, thêm `Linear(2048 → 128) + BN + ReLU + Dropout(0.3)`. Concat thành [B, 192] cân bằng hơn (128:64 = 2:1).

**Preprocessing ảnh:**
```python
transforms = [
    Resize((224, 224)),          # hoặc (384, 384) tùy GPU memory
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.2), # Mô phỏng gain variation
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])  # ImageNet stats
]
```

**Xử lý ảnh cho bệnh nhân có plaque (95 cases):**
- Mỗi bệnh nhân có 5 ảnh (1 IMT + 4 cross-section)
- Chiến lược: lấy **mean embedding** trên 5 image embeddings (ổn định hơn max-pooling cho dataset nhỏ)
- Hoặc: **max-pooling** (worst-case aggregation)

**Xử lý ảnh cho control (205 cases):**
- Chỉ có 1 ảnh IMT → sử dụng trực tiếp

**Fine-tuning strategy:**

| Giai đoạn | Hành động | Epochs |
|---|---|---|
| 1 | Freeze ResNet-50, train projection + fusion + heads | 5 |
| 2 | Unfreeze layer4 + projection, tiếp tục train | 15 |
| 3 | Unfreeze toàn bộ, lr rất nhỏ | 10 |

### 3.3 Branch 2 — MLP (Tabular Branch)

**Input features (9 features):**
```python
tabular_features = [
    'Age',
    'Sex',              # Encode: Male=1, Female=0
    'Lp(a)_mg_dL',      # Quan trọng nhất — ngưỡng 50 theo ESC/EAS 2025
    'ApoB_mg_dL',
    'LDL_C_mg_dL',
    'Triglyceride_mg_dL',
    'Total_Cholesterol_mg_dL',
    'Non_HDL_mg_dL',
    'IMT_mm'
]
```

> **Không đưa `Baseline_Risk_Score`** vào input — derived feature từ các biến khác → data leakage và làm mờ contribution của từng biomarker.

**Kiến trúc MLP:**
```
Input [B, 9]
    ↓
Linear(9 → 64) + BatchNorm1d + ReLU
    ↓
Linear(64 → 128) + BatchNorm1d + ReLU + Dropout(0.2)
    ↓
Linear(128 → 64) + BatchNorm1d + ReLU
    ↓
Clinical Embedding [B, 64]
```

**Normalization:**
- Fit StandardScaler trên train set, áp dụng cho val/test set
- Không dùng MinMaxScaler vì nhạy cảm với outliers trong dữ liệu lâm sàng

### 3.4 Fusion Layer

**Phương pháp 1 — Concatenation (baseline, đơn giản nhất):**
```python
fused = torch.cat([image_embedding, clinical_embedding], dim=1)
# Shape: [B, 128 + 64] = [B, 192]  ← cập nhật sau khi thêm projection
```

**Phương pháp 2 — Cross-Attention Fusion (nâng cao):**
```python
# image_embedding làm Query, clinical_embedding làm Key/Value
# (sau khi đã projection về cùng 128-dim)
attention_weight = softmax(Q @ K^T / sqrt(d_k))
fused = attention_weight @ V + image_embedding  # residual
```

> **Khuyến nghị:** Bắt đầu với Concatenation. Sau khi có baseline results, thử Cross-Attention và so sánh.

### 3.5 Output Heads — Multi-task

| Head | Task | Số class | Architecture | Loss |
|---|---|---|---|---|
| Head 1 | Plaque Detection | 2 | FC(64 → 2) + Softmax | Weighted BCE |
| Head 2 | Echogenicity | **4** (None/Low/Inter/High) | FC(64 → 4) + Softmax | Weighted CE |
| Head 3 | Reclassification | 2 (cần lên risk?) | FC(64 → 1) + Sigmoid | BCE |

> **Lưu ý đặc biệt — Head 3 mới:** Thay vì regression "Risk Score" (số liệu derived), dùng nhãn binary "có cần reclassify lên nhóm nguy cơ cao hơn theo ESC/EAS 2025 không" — gắn trực tiếp với hành động lâm sàng từ guideline.

---

## 4. Hàm Loss

### 4.1 Multi-Task Loss

```python
total_loss = λ1 * L_detection + λ2 * L_echogenicity + λ3 * L_reclassify

# Mặc định: λ1=1.0, λ2=0.5, λ3=0.3
```

### 4.2 Xử Lý Class Imbalance (CẬP NHẬT)

```python
# Plaque: Control 205 vs Case 95
weights_plaque = compute_class_weight('balanced',
                                       classes=[0,1],
                                       y=y_train)
# ≈ [0.73, 1.58] — Case nặng hơn

# Echogenicity (imbalance nặng hơn): None=205, Low=28, Inter=40, High=27
weights_echo = compute_class_weight('balanced',
                                     classes=[0,1,2,3],
                                     y=y_echo_train)
# Low/Inter/High có weight cao hơn ~2-3 lần None
```

### 4.3 Loss cho Từng Output Head

| Head | Loss Function | Lý do |
|---|---|---|
| Head 1 (Detection) | Weighted Binary CrossEntropy | Imbalance 2.16:1 |
| Head 2 (Echogenicity) | Weighted CrossEntropy (4-class) | Imbalance nặng hơn |
| Head 3 (Reclassification) | Binary CrossEntropy | Tỷ lệ ~49/51 trong train |

---

## 5. Chiến Lược Huấn Luyện

### 5.1 Hyperparameters

```python
config = {
    'batch_size': 16,          # Nhỏ do dataset nhỏ (300 cases)
    'learning_rate': 1e-4,     # AdamW optimizer
    'weight_decay': 1e-4,      # L2 regularization
    'epochs': 50,
    'early_stopping_patience': 10,
    'scheduler': 'CosineAnnealingLR',
    'warmup_epochs': 5,
    'gradient_clip': 1.0
}
```

### 5.2 Optimizer (lr phân biệt)

```python
optimizer = AdamW([
    {'params': cnn_branch.parameters(),     'lr': 1e-5},  # Pretrained
    {'params': cnn_projection.parameters(), 'lr': 1e-4},  # Layer mới
    {'params': mlp_branch.parameters(),     'lr': 1e-4},
    {'params': fusion_layer.parameters(),   'lr': 1e-4},
    {'params': output_heads.parameters(),   'lr': 1e-4}
])
```

### 5.3 Data Augmentation cho Ảnh Siêu Âm

```python
# Chỉ áp dụng trong training
train_augment = [
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=(-10, 10)),
    RandomAffine(degrees=0, translate=(0.05, 0.05)),
    GaussianBlur(kernel_size=3, p=0.3),  # Mô phỏng US noise
    ColorJitter(brightness=0.2, contrast=0.2)
]
```

---

## 6. Metrics Đánh Giá

### 6.1 Metrics Chính

| Task | Metric | Lý do |
|---|---|---|
| Plaque Detection | AUC-ROC, F1-score, Sensitivity, Specificity | Chuẩn 4 bài báo |
| Echogenicity | F1-score (macro), Accuracy per class | Theo Ren et al. (2026) |
| Reclassification | AUC-ROC, F1, Cohen's Kappa | Gắn với ESC/EAS 2025 |
| **Discordance subgroup** | **Sensitivity, NPV trên 33 cases** | **Metric novelty** |

### 6.2 Baseline So Sánh

| Model | Mô tả | Paper tham khảo |
|---|---|---|
| **ESC/EAS Rule** | Rule lâm sàng từ guideline | **Mach et al. 2025** |
| CNN-only | ResNet-50 chỉ trên ảnh | Gao et al. (2026) |
| MLP-only | Tabular model (RF/XGBoost/MLP) | Vu et al. (2025) |
| Fusion (Concat) | Kiến trúc đề xuất (baseline) | Đề tài này |
| Fusion (Attention) | Kiến trúc đề xuất (nâng cao) | Đề tài này |

### 6.3 Ablation Study

| Experiment | Mô tả |
|---|---|
| w/o Lp(a) | Bỏ Lp(a) khỏi tabular input |
| w/o ApoB | Bỏ ApoB khỏi tabular input |
| w/o IMT_mm | Bỏ IMT tabular, chỉ dùng ảnh |
| **w/o CNN projection** | Bỏ Linear 2048→128, dùng 2048-dim trực tiếp |
| Image branch only | CNN không có fusion |
| Tabular branch only | MLP không có fusion |
| Full model | Tất cả features + fusion |

### 6.4 Phân Tích Discordance theo ESC/EAS 2025 ⭐⭐⭐

**Đây là đóng góp quan trọng nhất của đề tài.**

```python
def identify_discordant_esceas2025(df):
    """Định nghĩa CHÍNH THỨC theo ESC/EAS 2025 Focused Update
       (Mach et al., Eur Heart J 2025, ehaf190)
       
       Discordant = bệnh nhân đạt LDL goal cho category hiện tại
                    nhưng có risk modifier → cần reclassify
                    
       Reference: Table 3 + Table 4 + Box 1
    """
    ldl_goal = {'Low': 116, 'Moderate': 100, 'High': 70, 'Very High': 55}
    
    df['ldl_goal']    = df['Baseline_Risk_Category'].map(ldl_goal)
    df['at_goal']     = df['LDL_C_mg_dL'] < df['ldl_goal']
    df['high_lpa']    = df['Lp(a)_mg_dL'] > 50    # Box 1
    df['has_plaque']  = df['Plaque_present'] == 1  # Risk modifier
    
    df['discordant']  = df['at_goal'] & (df['high_lpa'] | df['has_plaque'])
    return df

# Apply trên unseen 5-fold CV predictions → ~33 discordant cases
```

**So sánh trên nhóm discordant:**

| Model | AUC overall | Sens trên discordant | NPV trên discordant |
|---|---|---|---|
| ESC/EAS Rule | — | — | — |
| CNN-only | | | |
| MLP-only | | | |
| Fusion | **kỳ vọng cao nhất** | | |

**Câu hỏi cụ thể:**
- MLP-only miss bao nhiêu discordant cases? (vì lipid panel đạt goal)
- CNN-only miss bao nhiêu discordant cases? (vì plaque nhỏ, khó nhìn)
- Fusion bắt được bao nhiêu trong số đó?

→ Đây là giá trị lâm sàng cụ thể, định lượng được, gắn trực tiếp với guideline.

---

## 7. Cấu Trúc Code Đề Xuất

```
project/
├── data/
│   ├── carotid_clinical_dataset_300cases.csv
│   ├── ehaf190.pdf                  ← Guideline gốc
│   └── CAROTID_IMAGES/
│       ├── control/      (205 subjects × 1 image)
│       └── cases/        (95 subjects × 5 images)
│
├── src/
│   ├── dataset.py        # CarotidDataset class
│   ├── models/
│   │   ├── cnn_branch.py     # ResNet-50 + projection
│   │   ├── mlp_branch.py     # Tabular MLP
│   │   ├── fusion.py         # Fusion layer + output heads
│   │   └── multimodal.py     # Full model assembly
│   ├── train.py
│   ├── evaluate.py
│   ├── clinical_rules.py     # ESC/EAS 2025 rules (LDL goal, Lp(a), discordance)
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

## 8. Kết Nối Với Bài Báo Tham Khảo

| Thành phần | Bài báo nguồn |
|---|---|
| **Định nghĩa Discordance, ngưỡng Lp(a), LDL goal** | **Mach F. et al. (Eur Heart J 2025;46:4359–4378), ehaf190** |
| ResNet-50 backbone | Gao et al. (2026), He et al. (2024) |
| Echogenicity classification | Ren et al. (2026) |
| IMT_mm là clinical predictor | Vu et al. (2025) — top-1 SHAP |
| Tabular feature selection | Vu et al. (2025) — LASSO + SHAP |
| Fusion với clinical data | Gao et al. (2026) — future work |
| Data augmentation cho US | He et al. (2024) — flip, rotation |

---

## 9. Câu Hỏi Nghiên Cứu (Research Questions)

**RQ1:** Mô hình Fusion có AUC cao hơn CNN-only và MLP-only trong bài toán phát hiện mảng bám không?

**RQ2:** Khi thêm Lp(a) và ApoB vào CNN-based model, accuracy của echogenicity classification có tăng không?

**RQ3:** Mô hình Fusion có giải quyết được bài toán Discordance **(theo định nghĩa chính thức ESC/EAS 2025)** — nhận diện đúng bệnh nhân đạt LDL goal nhưng có risk modifier — tốt hơn single-modality models và ESC/EAS rule-based baseline không?

**RQ4:** Trong các chiến lược fusion (concatenation, cross-attention), chiến lược nào cho kết quả tốt hơn với dataset kích thước 300 cases?

---

## 10. Lưu Ý Quan Trọng

### 10.1 Dataset Limitations
- Dataset là **synthetic** — kết quả không thể claim clinical validity trực tiếp
- 300 cases là dataset nhỏ — cần regularization mạnh, data augmentation, và 5-fold CV
- Imbalance: **205 control vs 95 case** (Control là majority) — cần weighted loss
- Risk Category chỉ có Low và Moderate trong dataset — phù hợp với cohort primary prevention

### 10.2 Implementation Priorities
1. **Ưu tiên đầu tiên:** Chạy được pipeline end-to-end (kể cả đơn giản)
2. **Ưu tiên thứ hai:** Có baseline comparison (ESC/EAS Rule + CNN-only + MLP-only + Fusion)
3. **Ưu tiên thứ ba:** Ablation study (đặc biệt w/o Lp(a), w/o CNN projection)
4. **Ưu tiên thứ tư:** Discordance subgroup analysis trên 33 cases
5. **Nâng cao:** Cross-attention fusion, Grad-CAM

### 10.3 Khi Hỏi Claude Trong Quá Trình Làm
- Cung cấp file `.md` này + `ehaf190.pdf` làm context
- Ghi rõ đang ở bước nào (EDA, training, evaluation)
- Paste error message + relevant code snippet
- Hỏi cụ thể: "Tôi đang viết `CarotidDataset.__getitem__`, làm sao xử lý bệnh nhân có 5 ảnh?"

---

*Tài liệu v2.0 — Cập nhật theo Mach F. et al., 2025 Focused Update of the 2019 ESC/EAS Guidelines for the management of dyslipidaemias, European Heart Journal 2025;46:4359–4378 (ehaf190), và dữ liệu thực tế trong `carotid_clinical_dataset_300cases.csv` (205 control / 95 case).*

*Bài báo tham khảo phụ: Ren et al. (Bioengineering 2026), Vu et al. (JMIR Cardio 2025), Gao et al. (Frontiers in Medicine 2026), He et al. (Frontiers in AI 2024).*
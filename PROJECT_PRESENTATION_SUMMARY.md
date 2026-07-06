# Tóm Tắt Ý Nghĩa Và Thuật Toán Sử Dụng Trong Project

Tài liệu này giúp giải thích project theo hướng dễ trình bày lại khi present. Nội dung được tóm tắt từ pipeline tổng thể, kiến trúc mô hình và các notebook chạy thực nghiệm.

---

## 1. Project Này Đang Giải Quyết Bài Toán Gì?

Project xây dựng một hệ thống hỗ trợ phân tầng nguy cơ xơ vữa động mạch cảnh bằng cách kết hợp hai nguồn dữ liệu:

- Ảnh siêu âm động mạch cảnh.
- Dữ liệu lâm sàng dạng bảng như tuổi, giới, LDL-C, Lp(a), ApoB, triglyceride, cholesterol, IMT.

Bài toán chính là dự đoán bệnh nhân có mảng bám động mạch cảnh hay không (`Plaque_present`). Ngoài ra, mô hình còn học thêm hai nhiệm vụ phụ:

- Phân loại tính chất hồi âm của mảng bám (`Plaque_echogenicity`): `None`, `Low`, `Intermediate`, `High`.
- Dự đoán bệnh nhân có cần được phân tầng lại nguy cơ theo ESC/EAS 2025 hay không (`needs_reclassify`).

Ý tưởng cốt lõi: chỉ nhìn lipid panel có thể chưa đủ. Có bệnh nhân LDL-C nhìn có vẻ đạt mục tiêu, nhưng vẫn có Lp(a) cao hoặc có mảng bám trên siêu âm. Nhóm này gọi là nhóm **discordant** và là điểm nhấn lâm sàng của project.

---

## 2. Ý Nghĩa Lâm Sàng Của Project

Theo ESC/EAS 2025, nguy cơ tim mạch không chỉ dựa vào LDL-C. Một số yếu tố có thể làm bệnh nhân cần được đánh giá nguy cơ cao hơn, gọi là **risk modifiers**.

Trong project này, hai risk modifiers quan trọng là:

- `Lp(a) > 50 mg/dL`: dấu ấn sinh học làm tăng nguy cơ xơ vữa.
- Có mảng bám động mạch cảnh trên siêu âm: bằng chứng hình ảnh của xơ vữa.

Project định nghĩa nhóm **discordant** như sau:

```text
Bệnh nhân đạt LDL-C goal theo nhóm nguy cơ hiện tại
NHƯNG có ít nhất một risk modifier:
  - Lp(a) > 50 mg/dL
  - hoặc có mảng bám động mạch cảnh
```

Ý nghĩa khi present:

> Nếu chỉ nhìn LDL-C, ta có thể nghĩ bệnh nhân đã ổn. Nhưng nếu bệnh nhân có Lp(a) cao hoặc siêu âm cho thấy mảng bám, nguy cơ thực tế có thể cao hơn. Project này cố gắng dùng mô hình đa phương thức để phát hiện nhóm bệnh nhân dễ bị bỏ sót đó.

---

## 3. Dataset Sử Dụng Trong Project

Dataset có 300 bệnh nhân:

| Thành phần | Số lượng |
|---|---:|
| Tổng bệnh nhân | 300 |
| Không có mảng bám, `Plaque_present = 0` | 205 |
| Có mảng bám, `Plaque_present = 1` | 95 |
| Tỷ lệ lệch lớp | khoảng 2.16 : 1 |
| Discordant cases | 33 |

Dữ liệu ảnh:

- Bệnh nhân control, không có plaque: thường có 1 ảnh IMT.
- Bệnh nhân case, có plaque: có nhiều ảnh hơn, gồm IMT và các ảnh cross-section.

Dữ liệu bảng gồm 9 feature chính:

```text
Age
Sex
Lp(a)_mg_dL
ApoB_mg_dL
LDL_C_mg_dL
Triglyceride_mg_dL
Total_Cholesterol_mg_dL
Non_HDL_mg_dL
IMT_mm
```

Không dùng `Baseline_Risk_Score` làm input vì đây là biến đã được tính/derived, có thể gây data leakage.

---

## 4. Pipeline Tổng Thể

Project được chia thành 7 phase.

| Phase | Nội dung | Ý nghĩa |
|---|---|---|
| Phase 1 | EDA và hiểu dữ liệu | Kiểm tra phân bố, class imbalance, discordance |
| Phase 2 | Data pipeline và Dataset class | Chuẩn bị dữ liệu ảnh + tabular cho model |
| Phase 3 | Baseline đơn phương thức | Tạo điểm so sánh trước khi fusion |
| Phase 4 | Xây dựng Multimodal Fusion Model | Kết hợp CNN và MLP |
| Phase 5 | Training | Huấn luyện end-to-end với multi-task loss |
| Phase 6 | Evaluation và analysis | So sánh model, phân tích discordant group |
| Phase 7 | Nâng cao và báo cáo | Cross-attention, Grad-CAM, trả lời research questions |

Ba notebook chính tương ứng với pipeline:

```text
01_eda.ipynb       -> EDA, tạo carotid_annotated.csv
02_baseline.ipynb  -> ESC/EAS rule, RF, XGBoost, MLP, CNN-only
03_fusion.ipynb    -> Fusion Concat, Cross-Attention, Ablation, Analysis
```

---

## 5. Phase 1 - EDA Và Clinical Rules

Mục tiêu của EDA là hiểu dữ liệu trước khi huấn luyện.

Các việc chính:

- Đếm số case/control.
- Kiểm tra phân bố tuổi, giới, lipid markers.
- Phân tầng Lp(a) theo ESC/EAS 2025.
- Kiểm tra phân bố echogenicity.
- Tính số discordant cases.
- Xuất file `results/carotid_annotated.csv`.

Các rule lâm sàng chính:

| Risk Category | LDL-C goal |
|---|---:|
| Low | < 116 mg/dL |
| Moderate | < 100 mg/dL |
| High | < 70 mg/dL |
| Very High | < 55 mg/dL |

Lp(a):

| Nhóm | Ngưỡng |
|---|---|
| Bình thường | < 30 mg/dL |
| Tăng nhẹ | 30-50 mg/dL |
| Elevated | > 50 mg/dL |
| Markedly elevated | > 180 mg/dL |

Câu nói khi present:

> EDA không chỉ để vẽ biểu đồ mà còn để biến guideline ESC/EAS thành các nhãn có thể huấn luyện được, ví dụ như `is_discordant` và `needs_reclassify`.

---

## 6. Phase 2 - Data Pipeline

Project có hai loại dữ liệu nên cần pipeline riêng cho từng nhánh.

### 6.1 Nhánh ảnh

Ảnh siêu âm được xử lý bằng các bước:

- Resize về `224 x 224`.
- Augmentation cho train set:
  - lật ngang,
  - xoay nhẹ,
  - affine transform,
  - Gaussian blur,
  - chỉnh brightness/contrast.
- Normalize theo chuẩn ImageNet.

Vì một bệnh nhân có thể có nhiều ảnh, pipeline không dự đoán từng ảnh độc lập. Thay vào đó:

```text
Nhiều ảnh của cùng bệnh nhân
-> ResNet trích embedding từng ảnh
-> mean-pooling embedding
-> embedding đại diện cho bệnh nhân
```

Ý nghĩa:

> Đơn vị dự đoán là bệnh nhân, không phải từng ảnh. Điều này tránh data leakage và phù hợp với bài toán lâm sàng.

### 6.2 Nhánh tabular

Dữ liệu bảng được xử lý như sau:

- Encode `Sex`: Male = 1, Female = 0.
- Dùng `StandardScaler`.
- Fit scaler trên train set.
- Transform validation/test bằng scaler đã fit.

Lý do dùng scaler:

> Các feature như tuổi, LDL-C, Lp(a), IMT có thang đo khác nhau. MLP học ổn định hơn khi các feature được chuẩn hóa.

---

## 7. Phase 3 - Các Baseline Sử Dụng

Baseline là các mô hình dùng để so sánh xem Fusion có thật sự tốt hơn không.

### 7.1 ESC/EAS Rule-Based Baseline

Đây là baseline lâm sàng, không dùng deep learning.

Rule đơn giản:

```text
Predict positive nếu:
  LDL-C vượt goal theo risk category
  hoặc Lp(a) > 50 mg/dL
```

Không dùng `Plaque_present` trong rule vì plaque là nhãn cần dự đoán.

Ý nghĩa:

> Baseline này đại diện cho cách tiếp cận dựa trên guideline/lipid panel. Nếu Fusion tốt hơn rule này, ta có thêm bằng chứng rằng hình ảnh siêu âm và mô hình học máy cung cấp thông tin bổ sung.

### 7.2 Random Forest

Random Forest là mô hình ensemble gồm nhiều decision tree.

Cách hiểu đơn giản:

```text
Nhiều cây quyết định cùng dự đoán
-> bỏ phiếu hoặc lấy trung bình
-> kết quả ổn định hơn một cây đơn lẻ
```

Trong project, Random Forest dùng dữ liệu tabular để dự đoán plaque.

Ưu điểm:

- Chạy nhanh.
- Tốt với dữ liệu bảng nhỏ.
- Không cần scale dữ liệu quá nghiêm ngặt.
- Là baseline mạnh cho tabular data.

### 7.3 XGBoost

XGBoost cũng là mô hình cây quyết định, nhưng theo hướng boosting.

Cách hiểu:

```text
Cây sau học để sửa lỗi của cây trước
-> nhiều cây yếu kết hợp thành mô hình mạnh
```

Ưu điểm:

- Mạnh với dữ liệu bảng.
- Thường cho kết quả tốt trên dataset vừa và nhỏ.
- Có khả năng học quan hệ phi tuyến giữa biomarker và plaque.

### 7.4 MLP-Only

MLP là neural network cho dữ liệu bảng.

Input là 9 feature lâm sàng:

```text
[Age, Sex, Lp(a), ApoB, LDL-C, Triglyceride, Total Cholesterol, Non-HDL, IMT]
```

MLP học embedding lâm sàng, sau đó dự đoán plaque.

Ý nghĩa:

> MLP-only trả lời câu hỏi: nếu chỉ dùng thông tin lâm sàng, không dùng ảnh siêu âm, mô hình làm được đến đâu?

### 7.5 CNN-Only

CNN-only dùng ResNet-50 pretrained trên ImageNet để xử lý ảnh siêu âm.

Pipeline:

```text
Ảnh siêu âm
-> ResNet-50
-> Global Average Pooling
-> Fully Connected layer
-> Dự đoán Plaque_present
```

Ý nghĩa:

> CNN-only trả lời câu hỏi: nếu chỉ nhìn ảnh siêu âm, không dùng biomarker, mô hình làm được đến đâu?

---

## 8. Phase 4 - Mô Hình Multimodal Fusion

Đây là phần chính của project.

Mô hình có hai nhánh:

```text
Ảnh siêu âm -> CNN Branch -> Image Embedding
Dữ liệu bảng -> MLP Branch -> Clinical Embedding
Image Embedding + Clinical Embedding -> Fusion Layer -> Output Heads
```

### 8.1 CNN Branch - ResNet-50

ResNet-50 dùng để trích xuất đặc trưng từ ảnh siêu âm.

Kiến trúc:

```text
Input image
-> ResNet-50 pretrained
-> Global Average Pooling
-> vector 2048 chiều
-> projection 2048 -> 128
-> Image Embedding 128 chiều
```

Lý do dùng pretrained ResNet-50:

- Dataset chỉ có 300 bệnh nhân, quá nhỏ để train CNN từ đầu.
- ResNet-50 đã học được nhiều đặc trưng thị giác cơ bản.
- Fine-tune giúp mô hình thích nghi với ảnh siêu âm.

Lý do thêm projection `2048 -> 128`:

> Nếu giữ vector ảnh 2048 chiều rồi concat với clinical embedding 64 chiều, nhánh ảnh có thể áp đảo nhánh lâm sàng. Projection xuống 128 chiều giúp hai nguồn thông tin cân bằng hơn.

### 8.2 MLP Branch - Clinical Data

MLP nhận 9 feature lâm sàng và tạo embedding 64 chiều.

Kiến trúc tổng quát:

```text
Input 9 features
-> Linear 9 -> 64 + BatchNorm + ReLU
-> Linear 64 -> 128 + BatchNorm + ReLU + Dropout
-> Linear 128 -> 64 + BatchNorm + ReLU
-> Clinical Embedding 64 chiều
```

Ý nghĩa:

> Nhánh MLP học thông tin nguy cơ từ biomarker và thông số lâm sàng, ví dụ Lp(a), ApoB, LDL-C, IMT.

### 8.3 Fusion Concat

Fusion Concat là cách kết hợp đơn giản nhất.

```text
Image Embedding:    128 chiều
Clinical Embedding:  64 chiều
Concat -> 192 chiều
```

Sau đó đi qua các fully connected layers để học tương tác giữa ảnh và dữ liệu bảng.

Ý nghĩa:

> Concat cho phép mô hình nhìn cùng lúc cả bằng chứng hình ảnh và bằng chứng lâm sàng trước khi đưa ra dự đoán.

### 8.4 Cross-Attention Fusion

Cross-Attention là cách fusion nâng cao hơn.

Ý tưởng:

```text
Image embedding hỏi: thông tin clinical nào quan trọng?
Clinical embedding cung cấp key/value
Attention học mức độ liên quan giữa hai nguồn dữ liệu
```

So với concat, cross-attention linh hoạt hơn vì mô hình có thể học trọng số tương tác giữa hai modality.

Câu nói khi present:

> Concat là ghép hai nguồn thông tin lại. Cross-attention thì tiến thêm một bước: mô hình học xem thông tin lâm sàng nào nên được chú ý khi diễn giải đặc trưng ảnh.

---

## 9. Multi-Task Learning

Mô hình Fusion không chỉ học một task mà học ba task cùng lúc.

| Head | Task | Output |
|---|---|---|
| Head 1 | Plaque Detection | Có/không có mảng bám |
| Head 2 | Echogenicity Classification | None/Low/Intermediate/High |
| Head 3 | Reclassification | Có cần reclassify nguy cơ không |

Loss tổng:

```text
total_loss =
  1.0 * L_plaque
  + 0.5 * L_echogenicity
  + 0.3 * L_reclassify
```

Ý nghĩa:

> Multi-task learning giúp mô hình học biểu diễn giàu hơn. Khi học phát hiện plaque, mô hình đồng thời học đặc điểm của plaque và ý nghĩa lâm sàng của nó theo guideline.

---

## 10. Training Strategy

Project dùng 5-fold cross-validation thay vì chỉ train/test một lần.

Lý do:

- Dataset nhỏ, chỉ 300 bệnh nhân.
- Chia một lần có thể làm kết quả dao động mạnh.
- 5-fold giúp mỗi bệnh nhân đều được dùng làm validation đúng một lần.
- Kết quả cuối là Out-of-Fold predictions trên toàn bộ 300 bệnh nhân.

Training ResNet-50 dùng fine-tuning 3 giai đoạn:

| Giai đoạn | Cách train | Mục đích |
|---|---|---|
| Stage 1 | Freeze ResNet-50, train layer mới | Ổn định phần fusion và output |
| Stage 2 | Unfreeze layer4 | Cho CNN thích nghi với ảnh siêu âm |
| Stage 3 | Unfreeze toàn bộ với learning rate nhỏ | Fine-tune nhẹ toàn mô hình |

Optimizer:

```text
AdamW
```

Dùng learning rate khác nhau:

- Backbone ResNet-50: learning rate nhỏ.
- Projection, MLP, Fusion, Heads: learning rate lớn hơn.

Lý do:

> ResNet-50 đã pretrained nên không nên cập nhật quá mạnh. Các layer mới cần học từ đầu nên dùng learning rate cao hơn.

---

## 11. Xử Lý Class Imbalance

Dataset bị lệch lớp:

```text
Control: 205
Case:     95
```

Nếu không xử lý, model có thể thiên về dự đoán control.

Project dùng `compute_class_weight` để tạo trọng số cho loss:

```text
Lớp ít hơn -> weight cao hơn
Lớp nhiều hơn -> weight thấp hơn
```

Với echogenicity, imbalance còn mạnh hơn:

```text
None: 205
Low: 28
Intermediate: 40
High: 27
```

Vì vậy Head 2 cũng dùng weighted cross-entropy.

---

## 12. Evaluation Metrics

Các metric chính:

| Metric | Ý nghĩa |
|---|---|
| AUC-ROC | Khả năng phân biệt case/control ở nhiều threshold |
| F1-score | Cân bằng precision và recall, hữu ích khi lệch lớp |
| Sensitivity/Recall | Bắt đúng bệnh nhân có bệnh |
| Specificity | Tránh dự đoán dương tính quá nhiều |
| Sens@discordant | Độ nhạy trên nhóm discordant |
| Echogenicity F1-macro | Đánh giá Head 2 công bằng giữa các lớp |
| Cohen's Kappa | Đánh giá agreement cho reclassification |

Metric quan trọng nhất về mặt novelty:

```text
Sensitivity trên nhóm discordant
```

Vì metric này trả lời:

> Trong nhóm bệnh nhân dễ bị bỏ sót nếu chỉ nhìn LDL-C/lipid panel, mô hình Fusion bắt được bao nhiêu người thật sự có plaque?

---

## 13. Ablation Study

Ablation là thí nghiệm bỏ từng thành phần để xem thành phần đó có đóng góp không.

| Experiment | Ý nghĩa |
|---|---|
| Full model | Dùng tất cả thông tin |
| w/o Lp(a) | Kiểm tra vai trò của Lp(a) |
| w/o ApoB | Kiểm tra vai trò của ApoB |
| w/o IMT_mm | Kiểm tra vai trò của IMT dạng tabular |
| w/o CNN projection | Kiểm tra projection 2048 -> 128 có cần thiết không |
| CNN-only | Chỉ dùng ảnh |
| Tabular-only | Chỉ dùng dữ liệu bảng |

Cách giải thích khi present:

> Nếu bỏ Lp(a) mà kết quả giảm, điều đó ủng hộ giả thuyết Lp(a) là thông tin quan trọng. Nếu Fusion tốt hơn CNN-only và MLP-only, điều đó cho thấy hai nguồn dữ liệu bổ sung cho nhau.

---

## 14. Research Questions Của Project

Project trả lời 4 câu hỏi nghiên cứu:

| RQ | Câu hỏi | Cách trả lời |
|---|---|---|
| RQ1 | Fusion có tốt hơn CNN-only và MLP-only không? | So sánh AUC/F1/Sensitivity |
| RQ2 | Lp(a)/ApoB có giúp phân loại echogenicity không? | Ablation w/o Lp(a), w/o ApoB |
| RQ3 | Fusion có giải quyết discordance tốt hơn không? | Sens@discordant trên 33 cases |
| RQ4 | Concat hay Cross-Attention tốt hơn? | So sánh hai chiến lược fusion |

---

## 15. Cách Kể Lại Pipeline Khi Present

Có thể trình bày theo flow sau:

### Bước 1: Bối cảnh

> Xơ vữa động mạch cảnh liên quan đến nguy cơ tim mạch và đột quỵ. Đánh giá nguy cơ không nên chỉ dựa vào LDL-C mà cần xét thêm risk modifiers như Lp(a) và bằng chứng hình ảnh từ siêu âm.

### Bước 2: Vấn đề

> Một số bệnh nhân đạt LDL-C goal nhưng vẫn có Lp(a) cao hoặc plaque. Đây là nhóm discordant, có nguy cơ bị bỏ sót nếu chỉ dùng lipid panel.

### Bước 3: Dữ liệu

> Dataset gồm 300 bệnh nhân, 205 control và 95 case. Mỗi bệnh nhân có dữ liệu tabular và ảnh siêu âm. Project chuẩn hóa dữ liệu theo ESC/EAS 2025 để tạo các nhãn `is_discordant` và `needs_reclassify`.

### Bước 4: Baseline

> Trước tiên, project xây dựng các baseline: rule ESC/EAS, Random Forest, XGBoost, MLP-only và CNN-only. Các baseline này giúp biết từng nguồn dữ liệu riêng lẻ mạnh đến đâu.

### Bước 5: Model chính

> Model chính là Multimodal Fusion. Ảnh đi qua ResNet-50 để tạo image embedding. Dữ liệu lâm sàng đi qua MLP để tạo clinical embedding. Hai embedding được kết hợp bằng concat hoặc cross-attention rồi đưa vào các output heads.

### Bước 6: Training

> Vì dataset nhỏ, project dùng 5-fold cross-validation. Vì class bị lệch, loss được gán class weight. Vì ResNet-50 pretrained, quá trình fine-tune được chia thành 3 giai đoạn để ổn định.

### Bước 7: Đánh giá

> Mô hình được đánh giá bằng AUC, F1, sensitivity, specificity, và đặc biệt là sensitivity trên nhóm discordant. Đây là metric thể hiện giá trị lâm sàng của Fusion.

### Bước 8: Kết luận

> Nếu Fusion vượt CNN-only và MLP-only, đặc biệt trên nhóm discordant, project chứng minh rằng kết hợp ảnh siêu âm và biomarker lâm sàng giúp phát hiện nguy cơ tốt hơn từng nguồn dữ liệu riêng lẻ.

---

## 16. Một Đoạn Script Ngắn Để Nói Khi Thuyết Trình

> Đề tài của nhóm em tập trung vào bài toán phân tầng nguy cơ xơ vữa động mạch cảnh bằng mô hình Multimodal Fusion. Điểm đặc biệt là mô hình không chỉ dùng ảnh siêu âm mà còn kết hợp dữ liệu lâm sàng như Lp(a), ApoB, LDL-C và IMT.
>
> Theo ESC/EAS 2025, có những bệnh nhân tuy đạt mục tiêu LDL-C nhưng vẫn có risk modifier như Lp(a) cao hoặc mảng bám động mạch cảnh. Nhóm này gọi là discordant và có thể bị bỏ sót nếu chỉ nhìn lipid panel. Vì vậy project xây dựng pipeline để phát hiện plaque và đánh giá lại nguy cơ cho nhóm này.
>
> Về thuật toán, nhóm em xây dựng các baseline gồm rule-based ESC/EAS, Random Forest, XGBoost, MLP-only và CNN-only. Mô hình chính gồm hai nhánh: ResNet-50 xử lý ảnh siêu âm và MLP xử lý dữ liệu bảng. Hai embedding được kết hợp bằng Fusion Concat hoặc Cross-Attention, sau đó mô hình học ba nhiệm vụ: phát hiện plaque, phân loại echogenicity và dự đoán reclassification.
>
> Vì dataset chỉ có 300 bệnh nhân và bị lệch lớp, nhóm em dùng 5-fold cross-validation, weighted loss và fine-tuning ResNet-50 theo 3 giai đoạn. Kết quả được đánh giá bằng AUC, F1, sensitivity, specificity và đặc biệt là sensitivity trên nhóm discordant để chứng minh giá trị lâm sàng của mô hình Fusion.

---

## 17. Từ Khóa Cần Nhớ

```text
Multimodal Fusion
Carotid Atherosclerosis
Ultrasound Image
Tabular Clinical Data
ResNet-50
MLP
Random Forest
XGBoost
ESC/EAS 2025
Lp(a)
ApoB
LDL-C goal
Discordance
5-fold Cross-Validation
Multi-task Learning
Class Imbalance
Ablation Study
Cross-Attention
Sensitivity on Discordant Cases
```

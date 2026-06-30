# Hướng dẫn chạy mô hình trên Google Colab

**Đề tài:** Phân Tầng Nguy Cơ Xơ Vữa Động Mạch Cảnh — Multimodal Fusion (ResNet-50 + MLP)  
**Repo:** https://github.com/HiimDManh/it2039-xulytinhieuhinhanhykhoa  
**GPU mục tiêu:** Google Colab T4 (miễn phí, ~15 GB VRAM)

---

## Tổng quan pipeline — 3 notebook chạy tuần tự

| Notebook | Nội dung | Yêu cầu | Thời gian ước tính |
|---|---|---|---|
| `01_eda.ipynb` | EDA, sanity check clinical rules, export annotated CSV | CPU | ~5 phút |
| `02_baseline.ipynb` | ESC/EAS rule + RF/XGB/MLP + CNN-only 5-fold | GPU T4 | ~2–3h |
| `03_fusion.ipynb` | Full Fusion Concat + CrossAttn + Ablations + Analysis | GPU T4 | ~10–13h |

> **Quan trọng:** Chạy đúng thứ tự `01 → 02 → 03`. Notebook 03 load OOF predictions từ 02 để so sánh — chạy sai thứ tự sẽ báo `FileNotFoundError`.

---

## Phần A — Người có code (thực hiện trên máy local trước khi share)

### A1. Commit và push `project/` lên GitHub

Toàn bộ source code trong `project/` hiện chưa được commit. Phải push trước khi share cho teammate.

> **Nếu bỏ qua bước này:** Teammate clone repo sẽ không thấy bất kỳ file nào trong `project/` — repo chỉ có docs và assets cũ.

```bash
# Từ thư mục gốc của repo
# Đảm bảo .venv/ đã được ignore
echo ".venv/" >> project/.gitignore

git add project/
git status  # kiểm tra — phải thấy rất nhiều file mới trong project/

git commit -m "feat: add full multimodal fusion pipeline (Phase 0-7)

- project/src/: clinical_rules, dataset, models (CNN+MLP+Fusion), train, evaluate
- project/notebooks/: 01_eda, 02_baseline, 03_fusion (fully filled)
- project/configs/default.yaml
- project/requirements.txt
- project/results/: baseline predictions + figures"

git push origin main
```

### A2. Upload data lên Google Drive và share với teammate

Data gốc không được commit lên GitHub (quá nặng). Upload thủ công lên Drive:

1. Vào [drive.google.com](https://drive.google.com)
2. Tạo cấu trúc folder: `My Drive/it2039/project/data/`
3. Upload `carotid_clinical_dataset_300cases.csv` vào `data/`
4. Upload toàn bộ thư mục `CAROTID_IMAGES/` (300+ ảnh PNG) vào `data/`
5. Right-click folder `it2039` → **Share → Anyone with the link (Viewer)**
6. Copy link và gửi cho teammate

Cấu trúc Drive sau khi upload:
```
My Drive/it2039/project/data/
├── carotid_clinical_dataset_300cases.csv
└── CAROTID_IMAGES/
    ├── P001_IMT.png
    ├── P003_CCA_L1.png
    ├── P003_CCA_L2.png
    ├── P003_CCA_R1.png
    ├── P003_CCA_R2.png
    ├── P003_IMT.png
    └── ... (300 bệnh nhân × 1–5 ảnh)
```

---

## Phần B — Teammate chạy trên Google Colab

### B1. Chuẩn bị Colab: Runtime GPU + Mount Drive

**Chọn GPU runtime (bắt buộc):**
- Menu: **Runtime → Change runtime type**
- Hardware accelerator: **T4 GPU**
- Click **Save**

**Copy data từ Drive shared folder về Drive cá nhân:**
1. Vào link Drive được share → right-click folder `it2039`
2. Chọn "Add shortcut to My Drive" hoặc "Make a copy" vào My Drive
3. Đảm bảo path là: `My Drive/it2039/project/data/...`

**Mount Drive trong Colab:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

Kiểm tra thành công:
```bash
!ls /content/drive/MyDrive/it2039/project/data/
# phải thấy: carotid_clinical_dataset_300cases.csv  CAROTID_IMAGES/
```

### B2. Clone repo và cài dependencies

```bash
# Clone repo
!git clone https://github.com/HiimDManh/it2039-xulytinhieuhinhanhykhoa.git \
  /content/it2039

# Verify structure
!ls /content/it2039/project/
# phải thấy: configs/ data/ notebooks/ requirements.txt results/ src/
```

Tạo symlink data từ Drive vào project (tránh copy 300+ ảnh vào session):
```bash
!ln -s /content/drive/MyDrive/it2039/project/data \
        /content/it2039/project/data

# Xác nhận symlink OK
!ls /content/it2039/project/data/CAROTID_IMAGES/ | head -5
```

Cài packages:
```bash
%%bash
cd /content/it2039/project
pip install -q \
  numpy>=1.26 pandas>=2.2 scikit-learn>=1.4 \
  matplotlib>=3.8 seaborn>=0.13 Pillow>=10.3 \
  PyYAML>=6.0 xgboost>=2.0

# PyTorch với CUDA 12.1 (T4 dùng CUDA 12.x)
pip install -q torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# → Tesla T4
```

> **Nếu thấy "CUDA not available":** Runtime chưa phải GPU. Quay lại **Runtime → Change runtime type → T4 GPU → Save → Factory reset runtime**.

Set working directory (các notebook đã có cell này, chỉ cần confirm path đúng):
```python
import sys, os
os.chdir('/content/it2039/project')
sys.path.insert(0, '/content/it2039/project')
```

### B3. Chạy `01_eda.ipynb` — EDA & sanity check (~5 phút, CPU)

Mở trong Colab: **File → Open notebook → Google Drive → it2039/project/notebooks/01_eda.ipynb**

Kết quả phải thấy sau khi chạy hết:
- DataFrame 300 × 24 (9 derived cols từ ESC/EAS clinical rules)
- `assert n_discordant == 33` — pass
- 8 figures lưu vào `results/figures/`
- `results/carotid_annotated.csv` được tạo

### B4. Chạy `02_baseline.ipynb` — Baselines (~2–3h GPU)

Notebook có 3 phần: Section 1 ESC/EAS rule (CPU, nhanh), Section 2 tabular ML RF/XGB/MLP (CPU, ~10 phút), Section 3 CNN-only 5-fold (GPU, ~2–3h).

**Tránh timeout Colab (disconnect sau 90 phút idle):** Paste đoạn này vào Browser Console (F12):
```javascript
function clickConnect() {
  const btn = document.querySelector('colab-connect-button')
    ?.shadowRoot?.querySelector('button');
  if (btn) btn.click();
}
setInterval(clickConnect, 60000);
```

Kết quả sau Section 1–2 (CPU, chạy nhanh):
- `results/predictions/pred_esceas_rule.csv`
- `results/predictions/pred_rf.csv`
- `results/predictions/pred_xgb.csv`
- `results/predictions/pred_mlp.csv`

Kết quả sau Section 3 (CNN-only, cần GPU):
- `results/predictions/pred_cnn_only.csv`
- `results/checkpoints/cnn_only_fold*.pt` (5 checkpoints)

> **Lưu kết quả xuống Drive ngay sau khi xong** — session Colab sẽ xóa `/content/` khi disconnect!

```bash
!cp -r /content/it2039/project/results/ \
  /content/drive/MyDrive/it2039/project/results/
```

### B5. Chạy `03_fusion.ipynb` — Main Fusion Training (~10–13h GPU)

Chia thành 4 block training độc lập — khuyến nghị chạy trong 3 session Colab riêng để tránh timeout:

| Session | Sections | Nội dung | Thời gian |
|---|---|---|---|
| Session 1 | 0–3 | Setup + smoke test + Full Fusion Concat 5-fold | ~3–4h |
| Session 2 | 4 | Cross-Attention Fusion 5-fold (RQ4) | ~3–4h |
| Session 3 | 5–9 | Ablation 5 variants + Analysis + Figures | ~2–3h |

Mỗi session bắt đầu: mount Drive + clone repo + pip install lại (~5 phút overhead).

**Restore predictions đã chạy từ Drive về session mới:**
```bash
!cp -r /content/drive/MyDrive/it2039/project/results/ \
        /content/it2039/project/results/
# Verify
!ls /content/it2039/project/results/predictions/
```

**Sync Drive sau mỗi section GPU nặng:**
```bash
!rsync -av /content/it2039/project/results/ \
           /content/drive/MyDrive/it2039/project/results/
```

Output files của từng section:

| Section | Output |
|---|---|
| Section 3 — Fusion Concat 5-fold | `pred_fusion_concat.csv` + `fusion_concat_fold*.pt` |
| Section 4 — CrossAttn 5-fold | `pred_fusion_crossattn.csv` + `fusion_crossattn_fold*.pt` |
| Section 5 — Ablation (5 variants) | `pred_ablation_*.csv` (5 files) |
| Section 6–9 — Analysis | figures + `discordance_case_table.csv` |

### B6. Tải kết quả về máy local

Sau khi chạy xong cả 3 notebook, `results/` chứa:

```
results/
├── carotid_annotated.csv           # 300×24, 9 derived cols
├── baseline_summary.csv            # metrics tất cả models
├── discordance_case_table.csv      # 33 discordant cases × từng model
├── predictions/
│   ├── pred_esceas_rule.csv
│   ├── pred_rf.csv
│   ├── pred_xgb.csv
│   ├── pred_mlp.csv
│   ├── pred_cnn_only.csv
│   ├── pred_fusion_concat.csv
│   ├── pred_fusion_crossattn.csv
│   └── pred_ablation_*.csv (5 files)
├── checkpoints/
│   └── *.pt (best weights per fold)
└── figures/
    └── fig_*.png (~15 PNGs)
```

Tải về máy local:
```python
# Option 1: Zip và download từ Colab
!zip -r /content/results_final.zip /content/it2039/project/results/
from google.colab import files
files.download('/content/results_final.zip')

# Option 2: Đã sync Drive — tải thẳng từ Google Drive về PC
```

---

## Kết quả baseline hiện tại & mục tiêu

| Model | AUC | F1 | Sens@discordant | Trạng thái |
|---|---|---|---|---|
| ESC/EAS 2025 Rule | 0.523 | 0.462 | 0.154 | Đã có |
| Random Forest (5-fold) | 0.698 | 0.462 | 0.423 | Đã có |
| XGBoost (5-fold) | 0.640 | 0.452 | 0.423 | Đã có |
| MLP tabular (5-fold) | 0.673 | 0.426 | 0.308 | Đã có |
| CNN-only ResNet-50 | — | — | — | Cần Colab |
| **Fusion Concat** (main) | **> 0.75 ?** | — | **> 0.50 ?** | Cần Colab |
| Cross-Attention (RQ4) | — | — | — | Cần Colab |

**Novelty metric chính:** `Sens@discordant` — Sensitivity trên 33 bệnh nhân ESC/EAS discordant (đạt LDL-C goal nhưng có Lp(a) > 50 mg/dL hoặc có mảng bám động mạch cảnh). Best baseline hiện tại: RF tại **0.423** (bắt được 14/33). Fusion cần vượt con số này để trả lời RQ3.

---

## Troubleshooting

| Lỗi | Nguyên nhân | Cách fix |
|---|---|---|
| `ModuleNotFoundError: src` | `sys.path` chưa include `/content/it2039/project` | Chạy lại cell setup đầu notebook |
| `FileNotFoundError: carotid_clinical_dataset` | Symlink data chưa tạo hoặc Drive chưa mount | Mount Drive lại → tạo lại symlink |
| `CUDA out of memory` | Batch size 16 quá lớn với 5 ảnh/patient | Trong `FUSION_CFG`, đổi `batch_size: 8` |
| `AssertionError: n_discordant != 33` | Data file khác version, hoặc clinical_rules bị sửa | Dùng đúng file CSV gốc |
| Colab disconnect giữa chừng | Idle timeout 90 phút | Dùng browser console trick ở B4 |
| `pred_fusion_concat.csv not found` trong Section 6 | Section 3 chưa chạy hoặc chưa restore từ Drive | Section 6 sẽ tự skip Fusion — đây là behavior mong muốn |

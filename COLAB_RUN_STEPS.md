# Huong Dan Chay Du An Tren Google Colab

Tai lieu nay tong hop cac buoc chay chuan tren Google Colab cho du an:

**Phan tang nguy co xo vua dong mach canh - Multimodal Fusion (ResNet-50 + MLP)**

Pipeline gom 3 notebook va can chay dung thu tu:

```text
01_eda.ipynb -> 02_baseline.ipynb -> 03_fusion.ipynb
```

Neu chay sai thu tu, notebook `03_fusion.ipynb` co the loi vi thieu prediction/output tu `02_baseline.ipynb`.

---

## 1. Chuan Bi Data Tren Google Drive

Tren Google Drive, tao dung cau truc:

```text
My Drive/it2039/project/data/
├── carotid_clinical_dataset_300cases.csv
└── CAROTID_IMAGES/
    ├── P001_IMT.png
    ├── P003_CCA_L1.png
    ├── P003_CCA_L2.png
    └── ...
```

Can upload:

- `carotid_clinical_dataset_300cases.csv`
- Thu muc anh `CAROTID_IMAGES/`

Neu data duoc share tu nguoi khac, vao link Drive va chon:

```text
Add shortcut to My Drive
```

Dam bao path tren Colab se la:

```text
/content/drive/MyDrive/it2039/project/data/
```

---

## 2. Bat GPU Tren Colab

Trong Google Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save
```

Neu gap loi CUDA hoac khong thay GPU:

```text
Runtime -> Change runtime type -> T4 GPU -> Save -> Factory reset runtime
```

---

## 3. Mount Google Drive

Chay cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Kiem tra data:

```bash
!ls /content/drive/MyDrive/it2039/project/data/
```

Ket qua can thay:

```text
carotid_clinical_dataset_300cases.csv
CAROTID_IMAGES/
```

---

## 4. Clone Repo

Chay:

```bash
!git clone https://github.com/HiimDManh/it2039-xulytinhieuhinhanhykhoa.git /content/it2039
```

Kiem tra project:

```bash
!ls /content/it2039/project/
```

Can thay cac muc nhu:

```text
configs/
data/
notebooks/
requirements.txt
results/
src/
```

Neu Colab bao folder da ton tai, reset bang:

```bash
!rm -rf /content/it2039
!git clone https://github.com/HiimDManh/it2039-xulytinhieuhinhanhykhoa.git /content/it2039
```

---

## 5. Link Data Tu Drive Vao Project

Vi data nam tren Drive, tao symlink vao project:

```bash
!rm -rf /content/it2039/project/data
!ln -s /content/drive/MyDrive/it2039/project/data /content/it2039/project/data
```

Kiem tra anh:

```bash
!ls /content/it2039/project/data/CAROTID_IMAGES/ | head -5
```

Neu thay ten file PNG thi data da duoc link dung.

---

## 6. Cai Dependencies Dung Cho Colab

Khong cai truc tiep toan bo `requirements.txt` tren Colab neu file co `jupyter` hoac `ipykernel`.

Ly do: Google Colab da quan ly san cac package Jupyter. Neu pip nang version cua `ipykernel`, `notebook`, `jupyter-server`, co the gap loi conflict nhu:

```text
google-colab 1.0.0 requires ipykernel==6.17.1
jupyter-kernel-gateway requires notebook<7.0
```

Dung cell cai dat sau:

```bash
%%bash
cd /content/it2039/project

grep -v -E '^(jupyter|ipykernel)' requirements.txt > /tmp/requirements-colab.txt
pip install -q -r /tmp/requirements-colab.txt

pip install -q torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Kiem tra GPU:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

Ket qua mong muon:

```text
CUDA available: True
GPU: Tesla T4
```

Neu truoc do da cai loi dependency, nen reset runtime:

```text
Runtime -> Disconnect and delete runtime
```

Sau do chay lai tu buoc mount Drive.

---

## 7. Set Working Directory

Chay:

```python
import os
import sys

os.chdir('/content/it2039/project')
sys.path.insert(0, '/content/it2039/project')

print(os.getcwd())
```

Ket qua phai la:

```text
/content/it2039/project
```

---

## 8. Mo Va Chay Notebook

Co 2 cach mo notebook.

### Cach 1: Mo Tu GitHub

1. Vao https://colab.research.google.com
2. Chon `File -> Open notebook`
3. Chon tab `GitHub`
4. Dan repo:

```text
https://github.com/HiimDManh/it2039-xulytinhieuhinhanhykhoa
```

5. Chon notebook trong:

```text
project/notebooks/
```

### Cach 2: Mo Sau Khi Clone Repo

Sau khi clone repo vao `/content/it2039`, mo notebook theo duong dan:

```text
/content/it2039/project/notebooks/01_eda.ipynb
/content/it2039/project/notebooks/02_baseline.ipynb
/content/it2039/project/notebooks/03_fusion.ipynb
```

---

## 9. Chay `01_eda.ipynb`

Mo:

```text
project/notebooks/01_eda.ipynb
```

Chay:

```text
Runtime -> Run all
```

Notebook nay chay khoang 5 phut, CPU cung duoc.

Ket qua can co:

```text
results/carotid_annotated.csv
results/figures/
```

Kiem tra:

```bash
!ls /content/it2039/project/results/
```

Trong notebook can pass sanity check:

```python
assert n_discordant == 33
```

---

## 10. Chay `02_baseline.ipynb`

Mo:

```text
project/notebooks/02_baseline.ipynb
```

Chay:

```text
Runtime -> Run all
```

Notebook nay gom:

- ESC/EAS rule baseline
- Random Forest
- XGBoost
- MLP tabular
- CNN-only ResNet-50 5-fold

Thoi gian uoc tinh: 2-3 gio voi GPU T4.

Ket qua can co:

```text
results/predictions/pred_esceas_rule.csv
results/predictions/pred_rf.csv
results/predictions/pred_xgb.csv
results/predictions/pred_mlp.csv
results/predictions/pred_cnn_only.csv
results/checkpoints/cnn_only_fold*.pt
```

Kiem tra:

```bash
!ls /content/it2039/project/results/predictions/
```

Sau khi chay xong, luu ket qua ve Drive:

```bash
!cp -r /content/it2039/project/results/ /content/drive/MyDrive/it2039/project/results/
```

---

## 11. Chay `03_fusion.ipynb`

Mo:

```text
project/notebooks/03_fusion.ipynb
```

Notebook nay gom:

- Fusion Concat
- Cross-Attention Fusion
- Ablation
- Analysis
- Figures

Thoi gian uoc tinh: 10-13 gio voi GPU T4.

Khuyen nghi khong chay mot lan duy nhat neu dung Colab free. Nen chia thanh 3 session:

```text
Session 1: Sections 0-3
Session 2: Section 4
Session 3: Sections 5-9
```

Truoc moi session moi, chay lai setup:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!rm -rf /content/it2039
!git clone https://github.com/HiimDManh/it2039-xulytinhieuhinhanhykhoa.git /content/it2039
```

```bash
!rm -rf /content/it2039/project/data
!ln -s /content/drive/MyDrive/it2039/project/data /content/it2039/project/data
```

```bash
%%bash
cd /content/it2039/project

grep -v -E '^(jupyter|ipykernel)' requirements.txt > /tmp/requirements-colab.txt
pip install -q -r /tmp/requirements-colab.txt

pip install -q torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Restore ket qua tu cac notebook/session truoc:

```bash
!cp -r /content/drive/MyDrive/it2039/project/results/ /content/it2039/project/results/
```

Sau moi section nang, sync ket qua ve Drive:

```bash
!rsync -av /content/it2039/project/results/ /content/drive/MyDrive/it2039/project/results/
```

---

## 12. Cach Chay Mot Phan Notebook

Neu khong muon `Runtime -> Run all`, co the chay tung cell bang:

```text
Shift + Enter
```

Hoac chay mot vung cell:

1. Click cell dau cua section can chay.
2. Giu `Shift`, click cell cuoi cua section.
3. Chon:

```text
Runtime -> Run selection
```

---

## 13. Tranh Colab Disconnect

Khi chay lau, Colab co the disconnect. Co the mo Browser Console bang `F12` va dan:

```javascript
function clickConnect() {
  const btn = document.querySelector('colab-connect-button')
    ?.shadowRoot?.querySelector('button');
  if (btn) btn.click();
}
setInterval(clickConnect, 60000);
```

---

## 14. Ket Qua Cuoi Cung

Sau khi chay xong 3 notebook, ket qua nam o:

```text
/content/it2039/project/results/
```

Va nen duoc sync ve:

```text
/content/drive/MyDrive/it2039/project/results/
```

Cau truc ket qua:

```text
results/
├── carotid_annotated.csv
├── baseline_summary.csv
├── discordance_case_table.csv
├── predictions/
│   ├── pred_esceas_rule.csv
│   ├── pred_rf.csv
│   ├── pred_xgb.csv
│   ├── pred_mlp.csv
│   ├── pred_cnn_only.csv
│   ├── pred_fusion_concat.csv
│   ├── pred_fusion_crossattn.csv
│   └── pred_ablation_*.csv
├── checkpoints/
│   └── *.pt
└── figures/
    └── fig_*.png
```

---

## 15. Tai Ket Qua Ve May

Cach 1: zip va download tu Colab:

```python
!zip -r /content/results_final.zip /content/it2039/project/results/

from google.colab import files
files.download('/content/results_final.zip')
```

Cach 2: tai truc tiep tu Google Drive:

```text
My Drive/it2039/project/results/
```

---

## 16. Checklist Chay Nhanh

```text
1. Upload data len Google Drive
2. Bat Colab T4 GPU
3. Mount Drive
4. Clone repo
5. Link data bang symlink
6. Cai dependencies, bo qua jupyter/ipykernel
7. Set working directory
8. Chay 01_eda.ipynb
9. Chay 02_baseline.ipynb
10. Sync results ve Drive
11. Chay 03_fusion.ipynb theo section
12. Sync results sau moi section nang
13. Download results cuoi cung
```

---

## 17. Loi Thuong Gap

### Loi dependency Colab

Loi:

```text
google-colab requires ipykernel==6.17.1
jupyter-kernel-gateway requires notebook<7.0
```

Cach fix:

- Reset runtime.
- Cai dependencies bang file tam da bo `jupyter` va `ipykernel`:

```bash
grep -v -E '^(jupyter|ipykernel)' requirements.txt > /tmp/requirements-colab.txt
pip install -q -r /tmp/requirements-colab.txt
```

### `ModuleNotFoundError: src`

Nguyen nhan: chua set working directory hoac `sys.path`.

Cach fix:

```python
import os
import sys

os.chdir('/content/it2039/project')
sys.path.insert(0, '/content/it2039/project')
```

### `FileNotFoundError: carotid_clinical_dataset`

Nguyen nhan: Drive chua mount hoac symlink data chua dung.

Cach fix:

```bash
!ls /content/drive/MyDrive/it2039/project/data/
!rm -rf /content/it2039/project/data
!ln -s /content/drive/MyDrive/it2039/project/data /content/it2039/project/data
```

### CUDA khong available

Nguyen nhan: runtime khong phai GPU.

Cach fix:

```text
Runtime -> Change runtime type -> T4 GPU -> Save -> Factory reset runtime
```

### Thieu prediction khi chay `03_fusion.ipynb`

Nguyen nhan: chua chay `02_baseline.ipynb` hoac chua restore results tu Drive.

Cach fix:

```bash
!cp -r /content/drive/MyDrive/it2039/project/results/ /content/it2039/project/results/
!ls /content/it2039/project/results/predictions/
```

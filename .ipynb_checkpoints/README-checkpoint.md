# IVF Embryo Morphology Grading – Deep Learning System  
A complete pipeline for automatic scoring of human blastocysts based on Gardner criteria using deep learning (EfficientNet-B0). The system includes **single-task quality classification**, **multi-task Gardner scoring**, **evaluation metrics**, and **Grad-CAM explainability**.

---

## Overview

This repository implements two models:

### 1) Single-task classifier  
Predicts an aggregated 3-class embryo quality label:
- low
- medium
- high

Labels are derived from Gardner expansion, ICM, and TE grades.

### 2) Multi-task Gardner model  
Predicts all three original clinical sub-scores:

| Task | Classes |
|------|---------|
| Expansion | 0–4 |
| ICM | 0–3 |
| TE | 0–3 |

Both models use EfficientNet-B0 as the backbone.

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── Gardner_train_silver.csv
│   │   ├── Gardner_test_gold_onlyGardnerScores.csv
│   │   └── archive.zip
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   └── ivf_embryo_grading_demo.ipynb
│
├── reports/
│   └── figures/
│       ├── confusion_matrix_efficientnet_b0.png
│       ├── confusion_multitask_expansion.png
│       ├── confusion_multitask_icm.png
│       ├── confusion_multitask_te.png
│       ├── gradcam_efficientnet_b0.png
│       ├── gradcam_multitask_expansion.png
│       ├── gradcam_multitask_icm.png
│       └── gradcam_multitask_te.png
│
├── src/
│   ├── config.py
│   ├── models/
│   │   ├── cnn_model.py
│   │   ├── multitask_model.py
│   │   └── *.pth (saved model weights)
│   ├── training/
│   │   ├── train.py
│   │   └── train_multitask.py
│   └── utils/
│       ├── annotation_utils.py
│       ├── clean_annotations.py
│       ├── dataset_split.py
│       ├── data_loader.py
│       ├── multitask_dataset.py
│       ├── multitask_data_loader.py
│       ├── plot_history.py
│       ├── confusion_matrix.py
│       ├── multitask_eval.py
│       ├── grad_cam.py
│       └── gradcam_multitask_demo.py
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/sintabh/deep-learning-embryo-morphology-grading-ivf.git
cd deep-learning-embryo-grading-ivf

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
```

Ensure raw dataset CSVs are placed in:
```
data/raw/
```

Then preprocess:
```bash
python -m src.utils.dataset_split
python -m src.utils.clean_annotations
```

---

## Training

### Single-Task Quality Model (3-class)

```bash
python -m src.training.train
```

Outputs:
- src/models/best_efficientnet_b0.pth
- reports/figures/confusion_matrix_efficientnet_b0.png
- src/models/training_history_efficientnet_b0.pt

Evaluate:
```bash
python -m src.utils.plot_history
python -m src.utils.confusion_matrix
```

### Multi-Task Gardner Model

```bash
python -m src.training.train_multitask
```

Outputs:
- src/models/best_multitask_efficientnet_b0.pth
- Confusion matrices for all heads
- Final test metrics

Evaluate:
```bash
python -m src.utils.multitask_eval
```

---

## Performance (Latest)

### Single-task (3-class quality)
- Val accuracy: ~0.84
- Balanced learning curve

### Multi-task (EXP/ICM/TE)
- Test accuracy:
  - Expansion ≈ 0.73
  - ICM ≈ 0.63
  - TE ≈ 0.60

---

## Explainability (Grad-CAM)

### Single-task:
```bash
python -m src.utils.grad_cam_demo
```

### Multi-task:
```bash
python -m src.utils.gradcam_multitask_demo
```

Saves heatmaps to `reports/figures/`.

---

## Author
- Siamak Tahmasbi.
- Deep Learning IVF Embryo Gradi
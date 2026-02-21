قدم ۲۲: نوشتن یک README اولیه برای ریپو (با عنوان نهایی GitHub)

در این قدم فقط یک کار می‌کنیم:
✅ **پر کردن فایل `README.md` با توضیح انگلیسی و عنوان نهایی پروژه.**

---

### ۱) محتوای `README.md` را کامل با این متن جایگزین کن

فایل `README.md` در ریشه پروژه را باز کن و محتوایش را این قرار بده:

````markdown
# Deep Learning for Embryo Morphology Grading (IVF Project)

This repository implements a deep learning pipeline for **embryo morphology grading** in the context of IVF, using convolutional neural networks (CNNs) and Grad-CAM–based explainability.

---

## 1. Overview

The project includes:

- A configurable project structure with a central `config.py` for paths
- Train/val/test split from a public IVF embryo dataset
- A CNN classifier (EfficientNet-B0 by default, with optional ResNet50)
- Training and evaluation scripts (accuracy, loss, confusion matrix)
- Grad-CAM visualization for model explainability
- A demo Jupyter notebook for interactive exploration

---

## 2. Dataset

The project assumes a public IVF embryo image dataset, downloaded manually into:

```text
data/raw/
    archive.zip
    <unzipped_dataset_folders>/
````

After running the splitting utility, the processed data structure is:

```text
data/processed/
    train/
        <class_1>/
        <class_2>/
        ...
    val/
        <class_1>/
        <class_2>/
        ...
    test/
        <class_1>/
        <class_2>/
        ...
```

Each `<class_x>` corresponds to an embryo morphology/quality grade.

---

## 3. Project Structure

```text
ivf-embryo-grading/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  └─ ivf_embryo_grading_demo.ipynb
├─ reports/
│  └─ figures/
│     ├─ accuracy_efficientnet_b0.png
│     ├─ loss_efficientnet_b0.png
│     └─ confusion_matrix_efficientnet_b0.png
├─ src/
│  ├─ config.py
│  ├─ models/
│  │  └─ cnn_model.py
│  ├─ training/
│  │  └─ train.py
│  └─ utils/
│     ├─ dataset_split.py
│     ├─ data_loader.py
│     ├─ plot_history.py
│     ├─ confusion_matrix.py
│     └─ grad_cam.py
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## 4. Environment and Installation

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. Data Preparation

1. Download the embryo dataset (e.g. public IVF blastocyst dataset).
2. Place the archive and/or extracted folders under:

```text
data/raw/
```

3. Run the dataset split script:

```bash
python -m src.utils.dataset_split
```

This will populate `data/processed/train`, `val`, and `test`.

---

## 6. Training

To train the model (EfficientNet-B0 by default) and save the best checkpoint and training history:

```bash
python -m src.training.train
```

Outputs:

* `src/models/best_efficientnet_b0.pth`
* `src/models/training_history_efficientnet_b0.pt`

---

## 7. Evaluation and Plots

Generate accuracy and loss curves:

```bash
python -m src.utils.plot_history
```

Generate a confusion matrix on the test set:

```bash
python -m src.utils.confusion_matrix
```

These figures are saved to `reports/figures/`.

---

## 8. Grad-CAM Explainability

A Grad-CAM utility is provided to visualize which regions of the embryo images the model focuses on:

```bash
python
>>> from src.utils.grad_cam import generate_gradcam_for_image
>>> image_path = r"path\to\test\image.png"
>>> out_path = generate_gradcam_for_image(image_path, model_name="efficientnet_b0")
>>> print(out_path)
```

The resulting heatmap overlay is saved under `reports/figures/`.

The demo notebook also includes an interactive Grad-CAM visualization.

---

## 9. Demo Notebook

Open the demo notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/ivf_embryo_grading_demo.ipynb
```

The notebook demonstrates:

* Loading a trained model
* Running inference on a test image
* Visualizing Grad-CAM over the same image

---

## 10. Reproducibility and Configuration

All important paths are centralized in `src/config.py`. If the project is moved, updating `ROOT_DIR` (or leaving it as automatic project root) keeps the rest of the code working without changing hard-coded paths.

```

---

وقتی این متن رو داخل `README.md` گذاشتی و ذخیره کردی، فقط بنویس:  
**«README نوشته شد، قدم بعد»**
::contentReference[oaicite:0]{index=0}
```

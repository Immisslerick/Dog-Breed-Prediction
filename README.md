# Dog Breed Prediction (Computer Vision)

A reproducible computer vision project for classifying dog images into breeds. This repo converts the original Colab-style workflow into a **clean, runnable Python project** (CLI scripts, saved artifacts, evaluation reports).

> **Note:** The baseline is configured for **3 breeds** by default to keep compute reasonable, but the code supports expanding to more classes.

---

## Features
- Reproducible train/val/test splits (**stratified**)
- `tf.data` input pipeline (no “load everything into RAM” requirement)
- Baseline CNN **or** transfer learning (MobileNetV2)
- Saved model + training history
- Evaluation report (accuracy + classification report + confusion matrix)
- Simple inference script (`predict.py`)

---

## Data
This project expects a Kaggle-style folder:
- `dog_dataset/labels.csv`
- `dog_dataset/train/<id>.jpg`

If you use Kaggle API, download/unzip into `dog_dataset/`.

---

## Quickstart (local)
### 1) Create venv & install deps
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2) (Optional) Kaggle download
Requires `kaggle.json` in `~/.kaggle/`:
```bash
kaggle datasets download catherinehorng/dogbreedidfromcomp -p dog_dataset --unzip
rm -f dog_dataset/sample_submission.csv
```

### 3) Train
```bash
python -m dogbreed.train --data-dir dog_dataset --classes scottish_deerhound,maltese_dog,bernese_mountain_dog --model mobilenet --epochs 20
```

Artifacts are written to `artifacts/` by default:
- `artifacts/model.keras`
- `artifacts/history.csv`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`

### 4) Predict on one image
```bash
python -m dogbreed.predict --model artifacts/model.keras --image path/to/image.jpg --top-k 3
```

---

## Why this is better than a raw Colab export
The original Colab script mixed shell commands, uploads, and training logic in one place. This version separates:
- data loading (`data.py`)
- model building (`model.py`)
- training (`train.py`)
- evaluation & plots (`evaluate.py`)
- inference (`predict.py`)

---

## Next improvements
- Expand to all breeds using transfer learning + class-balanced sampling
- Add experiment tracking (e.g., MLflow)
- Add unit tests + GitHub Actions CI
- Streamlit demo app

- Model: MobileNetV2 (transfer learning), 70 classes

Test Accuracy: 96.0%

Test Loss: 0.192

# Dog Breed Prediction (Computer Vision)

A reproducible computer vision project for classifying dog images into breeds. This repo converts the original Colab-style workflow into a **clean, runnable Python project** (CLI scripts, saved artifacts, evaluation reports).

> Default config uses **3 breeds** to keep compute reasonable, but you can expand to more classes.

## Features
- Stratified train/val/test splits
- `tf.data` input pipeline (no “load everything into RAM” requirement)
- Baseline CNN **or** transfer learning (MobileNetV2)
- Saved model + training history
- Evaluation report (accuracy + classification report + confusion matrix)
- Simple inference script (`predict.py`)

## Data
This project expects a Kaggle-style folder:
- `dog_dataset/labels.csv`
- `dog_dataset/train/<id>.jpg`

(Optional) Kaggle download (requires `kaggle.json` in `~/.kaggle/`):
```bash
kaggle datasets download catherinehorng/dogbreedidfromcomp -p dog_dataset --unzip
rm -f dog_dataset/sample_submission.csv
```

## Quickstart (local)

### 1) Create venv & install deps
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2) Train
```bash
python -m dogbreed.train --data-dir dog_dataset --classes scottish_deerhound,maltese_dog,bernese_mountain_dog --model mobilenet --epochs 20
```

Artifacts are written to `artifacts/` by default:
- `artifacts/model.keras`
- `artifacts/history.csv`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`

### 3) Predict on one image
```bash
python -m dogbreed.predict --model artifacts/model.keras --image path/to/image.jpg --top-k 3
```

## Notes for portfolio polish
- Add screenshots of `confusion_matrix.png` and a few misclassified examples
- Add a short “Results” section (Top-1 accuracy, Top-5 accuracy if you implement it)
- Keep the original notebook in `notebooks/` (optional), but make the CLI the “main” entry point

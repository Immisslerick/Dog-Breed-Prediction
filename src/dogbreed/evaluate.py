from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_report(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    out_dir: str | Path,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true: list[int] = []
    y_pred: list[int] = []

    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig_path = out_dir / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": str(fig_path),
    }

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from .utils import set_global_seed
from .data import make_splits
from .model import build_model
from .evaluate import evaluate_and_report

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a dog-breed classifier.")
    p.add_argument("--data-dir", type=str, required=True, help="Path to dataset folder (contains labels.csv and train/).")
    p.add_argument("--classes", type=str, default="scottish_deerhound,maltese_dog,bernese_mountain_dog", help="Comma-separated class names.")
    p.add_argument("--model", type=str, default="mobilenet", choices=["cnn", "mobilenet"], help="Model architecture.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts-dir", type=str, default="artifacts", help="Where to write outputs.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    splits = make_splits(
        data_dir=args.data_dir,
        class_names=class_names,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        test_size=0.10,
        val_size=0.20,
        augment_train=True,
    )

    model = build_model(args.model, (args.image_size, args.image_size, 3), len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(artifacts / "model.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(artifacts / "history.csv")),
    ]

    model.fit(
        splits.train,
        validation_data=splits.val,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = model.evaluate(splits.test, verbose=0)
    metrics_map = dict(zip(model.metrics_names, [float(m) for m in metrics]))
    extra = evaluate_and_report(model, splits.test, splits.class_names, artifacts)

    payload = {
        "test_metrics": metrics_map,
        **extra,
        "class_names": splits.class_names,
        "model": args.model,
    }
    with open(artifacts / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved artifacts to:", artifacts.resolve())
    print("Test metrics:", metrics_map)

if __name__ == "__main__":
    main()

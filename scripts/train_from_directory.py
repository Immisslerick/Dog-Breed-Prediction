import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def main() -> None:
    p = argparse.ArgumentParser(description="Train MobileNetV2 dog-breed classifier from train/valid/test folders.")
    p.add_argument("--data-dir", type=str, default=".", help="Folder containing train/ valid/ test/")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=str, default="artifacts_70breeds")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    for d in [train_dir, valid_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Missing folder: {d}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(args.image_size, args.image_size, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out / "model.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(test_ds, verbose=0)

    y_true, y_pred = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    (out / "metrics.json").write_text(
        json.dumps(
            {
                "test_accuracy": float(acc),
                "test_loss": float(loss),
                "num_classes": num_classes,
                "class_names": class_names,
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved to:", out.resolve())
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Literal, Tuple
import tensorflow as tf

def build_baseline_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """A simple CNN baseline."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (5, 5), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(16, (7, 7), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(8, (5, 5), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")

def build_mobilenet_transfer(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    weights: str | None = "imagenet",
    train_base: bool = False,
) -> tf.keras.Model:
    """MobileNetV2 transfer learning model."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling="avg",
    )
    base.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")

def build_model(kind: Literal["cnn", "mobilenet"], input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    if kind == "cnn":
        return build_baseline_cnn(input_shape, num_classes)
    if kind == "mobilenet":
        return build_mobilenet_transfer(input_shape, num_classes)
    raise ValueError(f"Unknown model kind: {kind}")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE

@dataclass(frozen=True)
class DatasetSplits:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: List[str]

def load_labels(labels_csv: str | Path, class_names: List[str]) -> pd.DataFrame:
    """Load labels.csv and filter to desired class names."""
    labels_csv = Path(labels_csv)
    df = pd.read_csv(labels_csv)
    df = df[df["breed"].isin(class_names)].copy()
    return df.reset_index(drop=True)

def stratified_splits(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    stratify_col: str = "breed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_col],
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=seed,
        stratify=train_val[stratify_col],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def _decode_and_resize(path: tf.Tensor, image_size: int) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (image_size, image_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def _make_label_lookup(class_names: List[str]) -> tf.lookup.StaticHashTable:
    keys = tf.constant(class_names)
    vals = tf.constant(list(range(len(class_names))), dtype=tf.int64)
    init = tf.lookup.KeyValueTensorInitializer(keys, vals)
    return tf.lookup.StaticHashTable(init, default_value=-1)

def build_dataset(
    df: pd.DataFrame,
    train_dir: str | Path,
    class_names: List[str],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build a tf.data dataset of (image, one_hot_label)."""
    train_dir = Path(train_dir)
    paths = (train_dir / (df["id"].astype(str) + ".jpg")).astype(str).to_numpy()
    breeds = df["breed"].astype(str).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, breeds))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    label_lookup = _make_label_lookup(class_names)

    def _map_fn(path, breed):
        img = _decode_and_resize(path, image_size)
        if augment:
            img = tf.image.random_flip_left_right(img, seed=seed)
            img = tf.image.random_brightness(img, max_delta=0.10, seed=seed)
        idx = label_lookup.lookup(breed)
        y = tf.one_hot(idx, depth=len(class_names))
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def make_splits(
    data_dir: str | Path,
    class_names: List[str],
    image_size: int = 224,
    batch_size: int = 32,
    seed: int = 42,
    test_size: float = 0.10,
    val_size: float = 0.20,
    augment_train: bool = True,
) -> DatasetSplits:
    """Create train/val/test datasets from a Kaggle-style folder."""
    data_dir = Path(data_dir)
    labels_csv = data_dir / "labels.csv"
    train_dir = data_dir / "train"

    df = load_labels(labels_csv, class_names)
    train_df, val_df, test_df = stratified_splits(df, test_size=test_size, val_size=val_size, seed=seed)

    train_ds = build_dataset(train_df, train_dir, class_names, image_size, batch_size, shuffle=True, augment=augment_train, seed=seed)
    val_ds = build_dataset(val_df, train_dir, class_names, image_size, batch_size, shuffle=False, augment=False, seed=seed)
    test_ds = build_dataset(test_df, train_dir, class_names, image_size, batch_size, shuffle=False, augment=False, seed=seed)

    return DatasetSplits(train=train_ds, val=val_ds, test=test_ds, class_names=class_names)

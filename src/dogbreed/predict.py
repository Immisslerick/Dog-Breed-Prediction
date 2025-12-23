from __future__ import annotations

import argparse
import numpy as np
import tensorflow as tf

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on a single image.")
    p.add_argument("--model", type=str, required=True, help="Path to saved Keras model (.keras).")
    p.add_argument("--image", type=str, required=True, help="Path to image file.")
    p.add_argument("--classes", type=str, default="scottish_deerhound,maltese_dog,bernese_mountain_dog", help="Comma-separated class names.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()

def load_and_preprocess(image_path: str, image_size: int) -> np.ndarray:
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (image_size, image_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()

def main() -> None:
    args = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]

    model = tf.keras.models.load_model(args.model)
    img = load_and_preprocess(args.image, args.image_size)
    probs = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

    idxs = np.argsort(probs)[::-1][: int(args.top_k)]
    print("Top predictions:")
    for i in idxs:
        print(f"- {class_names[i]}: {probs[i]:.4f}")

if __name__ == "__main__":
    main()

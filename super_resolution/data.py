"""Input pipeline helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import tensorflow as tf


SPLIT_TO_PARTITION = {"train": "0", "val": "1", "valid": "1", "test": "2"}


def discover_images(image_dir, patterns=("*.jpg", "*.jpeg", "*.png")):
    image_dir = Path(image_dir)
    paths = []
    for pattern in patterns:
        paths.extend(image_dir.rglob(pattern))
    return [str(path) for path in sorted(paths)]


def celeba_paths(image_dir, partition_csv, split: str, limit: int | None = None):
    image_dir = Path(image_dir)
    partition = SPLIT_TO_PARTITION.get(split, split)
    paths = []
    with Path(partition_csv).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["partition"] == str(partition):
                path = image_dir / row["image_id"]
                if path.exists():
                    paths.append(str(path))
                    if limit and len(paths) >= limit:
                        break
    if not paths:
        raise FileNotFoundError(
            f"No images found for split={split!r} under {image_dir}. "
            "Check --image-dir and --partition-csv."
        )
    return paths


def _decode_rgb(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(image, tf.float32)


def _center_square(image):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    size = tf.minimum(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def _augment(image):
    # Horizontal flips are safe for face SR; vertical flips/rotations create
    # unrealistic faces and can reduce validation quality on CelebA.
    image = tf.image.random_flip_left_right(image)
    return image


def make_pair(path, hr_size: int = 176, scale: int = 4, augment: bool = False):
    image = _center_square(_decode_rgb(path))
    image = tf.image.resize(image, [hr_size, hr_size], method="bicubic", antialias=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    if augment:
        image = _augment(image)
    lr_size = hr_size // scale
    lr = tf.image.resize(image, [lr_size, lr_size], method="bicubic", antialias=True)
    lr = tf.clip_by_value(lr, 0.0, 1.0)
    return lr, image


def make_dataset(
    paths,
    batch_size: int,
    hr_size: int = 176,
    scale: int = 4,
    shuffle: bool = False,
    augment: bool = False,
    cache: bool = False,
):
    if hr_size % scale != 0:
        raise ValueError("hr_size must be divisible by scale")

    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 10000), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path: make_pair(path, hr_size=hr_size, scale=scale, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if cache:
        ds = ds.cache()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

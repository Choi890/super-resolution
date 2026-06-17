from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from super_resolution.data import celeba_paths, discover_images, make_dataset
from super_resolution.models import build_model, compile_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a super-resolution model.")
    parser.add_argument("--image-dir", required=True, help="Directory containing training images.")
    parser.add_argument("--partition-csv", default="list_eval_partition.csv")
    parser.add_argument("--use-celeba-splits", action="store_true")
    parser.add_argument("--architecture", default="residual-espcn", choices=["residual-espcn", "espcn"])
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--hr-size", type=int, default=176)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--xla", action="store_true", help="Enable XLA JIT compilation.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    if args.xla:
        tf.config.optimizer.set_jit(True)
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    if args.use_celeba_splits:
        train_paths = celeba_paths(args.image_dir, args.partition_csv, "train", args.limit_train)
        val_paths = celeba_paths(args.image_dir, args.partition_csv, "val", args.limit_val)
    else:
        paths = discover_images(args.image_dir)
        if not paths:
            raise FileNotFoundError(f"No images found under {args.image_dir}")
        split_at = max(1, int(len(paths) * 0.9))
        train_paths = paths[:split_at]
        val_paths = paths[split_at:] or paths[:1]
        if args.limit_train:
            train_paths = train_paths[: args.limit_train]
        if args.limit_val:
            val_paths = val_paths[: args.limit_val]

    train_ds = make_dataset(
        train_paths,
        batch_size=args.batch_size,
        hr_size=args.hr_size,
        scale=args.scale,
        shuffle=True,
        augment=True,
        cache=args.cache,
    )
    val_ds = make_dataset(
        val_paths,
        batch_size=args.batch_size,
        hr_size=args.hr_size,
        scale=args.scale,
        shuffle=False,
        augment=False,
        cache=args.cache,
    )

    model = build_model(
        args.architecture,
        scale=args.scale,
        filters=args.filters,
        blocks=args.blocks,
    )
    compile_model(model, learning_rate=args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{args.architecture}_x{args.scale}_best.keras"
    final_path = output_dir / f"{args.architecture}_x{args.scale}_final.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_psnr_metric", mode="max", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(output_dir / "logs")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(final_path)
    print(f"Saved final model: {final_path}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

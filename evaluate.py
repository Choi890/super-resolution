from __future__ import annotations

import argparse

import tensorflow as tf

from super_resolution.data import celeba_paths, discover_images, make_dataset
from super_resolution.losses import psnr_metric, ssim_metric
from super_resolution.inference import load_super_resolution_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a super-resolution model.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--partition-csv", default="list_eval_partition.csv")
    parser.add_argument("--use-celeba-splits", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--architecture", default="residual-espcn", choices=["residual-espcn", "espcn"])
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--hr-size", type=int, default=176)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-bicubic-baseline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_celeba_splits:
        paths = celeba_paths(args.image_dir, args.partition_csv, args.split, args.limit)
    else:
        paths = discover_images(args.image_dir)
        if args.limit:
            paths = paths[: args.limit]
    ds = make_dataset(
        paths,
        batch_size=args.batch_size,
        hr_size=args.hr_size,
        scale=args.scale,
        shuffle=False,
        augment=False,
    )
    model = load_super_resolution_model(
        args.weights,
        architecture=args.architecture,
        scale=args.scale,
        filters=args.filters,
        blocks=args.blocks,
    )

    if args.include_bicubic_baseline:
        psnr_values = []
        ssim_values = []
        for lr, hr in ds:
            size = tf.shape(hr)[1:3]
            bicubic = tf.image.resize(lr, size, method="bicubic")
            bicubic = tf.clip_by_value(bicubic, 0.0, 1.0)
            psnr_values.append(psnr_metric(hr, bicubic))
            ssim_values.append(ssim_metric(hr, bicubic))
        print(f"bicubic_psnr_metric: {float(tf.reduce_mean(psnr_values).numpy()):.5f}")
        print(f"bicubic_ssim_metric: {float(tf.reduce_mean(ssim_values).numpy()):.5f}")

    results = model.evaluate(ds, return_dict=True)
    for key, value in results.items():
        print(f"{key}: {value:.5f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from super_resolution.inference import (
    load_super_resolution_model,
    read_image,
    upscale_image,
    write_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Upscale an image with a trained SR model.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--weights", required=True, help="Path to .keras, .h5, or weights checkpoint.")
    parser.add_argument("--architecture", default="residual-espcn", choices=["residual-espcn", "espcn"])
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_super_resolution_model(
        args.weights,
        architecture=args.architecture,
        scale=args.scale,
        filters=args.filters,
        blocks=args.blocks,
    )
    image = read_image(args.input)
    result = upscale_image(model, image)
    write_image(Path(args.output), result)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

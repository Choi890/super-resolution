"""Image loading, model loading, and inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from .layers import BicubicUpsample, ClipByValue, PixelShuffle, ResidualScale, Subpixel
from .losses import charbonnier_ssim_loss, psnr_metric, ssim_metric
from .models import build_model


CUSTOM_OBJECTS = {
    "PixelShuffle": PixelShuffle,
    "BicubicUpsample": BicubicUpsample,
    "ClipByValue": ClipByValue,
    "ResidualScale": ResidualScale,
    "Subpixel": Subpixel,
    "charbonnier_ssim_loss": charbonnier_ssim_loss,
    "psnr_metric": psnr_metric,
    "ssim_metric": ssim_metric,
}


def read_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def load_super_resolution_model(
    weights_path,
    architecture: str = "residual-espcn",
    scale: int = 4,
    filters: int = 64,
    blocks: int = 8,
):
    weights_path = Path(weights_path)
    if weights_path.suffix in {".keras", ".h5"}:
        try:
            return keras.models.load_model(weights_path, custom_objects=CUSTOM_OBJECTS)
        except Exception:
            pass

    model = build_model(architecture, scale=scale, filters=filters, blocks=blocks)
    model.load_weights(weights_path)
    return model


def upscale_image(model, image):
    batch = np.expand_dims(image.astype(np.float32), axis=0)
    pred = model.predict(batch, verbose=0)[0]
    return np.clip(pred, 0.0, 1.0)

"""Model builders for ESPCN-style super-resolution."""

from __future__ import annotations

from tensorflow import keras

from .layers import BicubicUpsample, ClipByValue, PixelShuffle, ResidualScale
from .losses import charbonnier_ssim_loss, psnr_metric, ssim_metric


def _residual_block(x, filters: int, residual_scale: float, name: str):
    shortcut = x
    x = keras.layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    if residual_scale != 1.0:
        x = ResidualScale(residual_scale, name=f"{name}_scale")(x)
    return keras.layers.Add(name=f"{name}_add")([shortcut, x])


def _pixel_shuffle_stack(x, scale: int, filters: int):
    if scale not in (2, 3, 4):
        raise ValueError("scale must be one of 2, 3, or 4")

    factors = [2, 2] if scale == 4 else [scale]
    for i, factor in enumerate(factors, start=1):
        x = keras.layers.Conv2D(
            filters * factor * factor,
            3,
            padding="same",
            name=f"upsample_{i}_conv",
        )(x)
        x = PixelShuffle(factor, name=f"upsample_{i}_pixel_shuffle")(x)
        x = keras.layers.PReLU(shared_axes=[1, 2], name=f"upsample_{i}_prelu")(x)
    return x


def build_espcn(
    input_shape=(None, None, 3),
    scale: int = 4,
    channels: int = 3,
    name: str = "espcn",
):
    """Small baseline ESPCN model compatible with variable input sizes."""

    inputs = keras.Input(shape=input_shape, name="lr")
    x = keras.layers.Conv2D(64, 5, padding="same", activation="relu", name="conv1")(inputs)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.Conv2D(channels * scale * scale, 3, padding="same", name="shuffle_conv")(x)
    x = PixelShuffle(scale, name="pixel_shuffle")(x)
    outputs = keras.layers.Activation("sigmoid", name="sr")(x)
    return keras.Model(inputs, outputs, name=name)


def build_residual_espcn(
    input_shape=(None, None, 3),
    scale: int = 4,
    channels: int = 3,
    filters: int = 64,
    num_blocks: int = 8,
    residual_scale: float = 0.1,
    name: str = "residual_espcn",
):
    """Residual ESPCN with LR-space residual blocks and a bicubic skip path."""

    inputs = keras.Input(shape=input_shape, name="lr")
    bicubic = BicubicUpsample(scale, name="bicubic_skip")(inputs)

    x = keras.layers.Conv2D(filters, 3, padding="same", name="head_conv")(inputs)
    x = keras.layers.PReLU(shared_axes=[1, 2], name="head_prelu")(x)
    trunk = x

    for i in range(num_blocks):
        x = _residual_block(x, filters, residual_scale, name=f"resblock_{i + 1}")

    x = keras.layers.Conv2D(filters, 3, padding="same", name="trunk_conv")(x)
    x = keras.layers.Add(name="long_skip")([trunk, x])
    x = _pixel_shuffle_stack(x, scale, filters)
    x = keras.layers.Conv2D(channels, 3, padding="same", name="residual_rgb")(x)
    x = keras.layers.Add(name="residual_plus_bicubic")([bicubic, x])
    outputs = ClipByValue(0.0, 1.0, name="clip")(x)
    return keras.Model(inputs, outputs, name=name)


def compile_model(model, learning_rate: float = 1e-4):
    """Compile with an SR-oriented loss and perceptual-quality metrics."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=charbonnier_ssim_loss,
        metrics=[psnr_metric, ssim_metric, "mae"],
    )
    return model


def build_model(
    architecture: str,
    input_shape=(None, None, 3),
    scale: int = 4,
    filters: int = 64,
    blocks: int = 8,
):
    architecture = architecture.lower()
    if architecture == "espcn":
        return build_espcn(input_shape=input_shape, scale=scale)
    if architecture in {"residual-espcn", "residual_espcn", "resespcn"}:
        return build_residual_espcn(
            input_shape=input_shape,
            scale=scale,
            filters=filters,
            num_blocks=blocks,
        )
    raise ValueError(f"Unknown architecture: {architecture}")

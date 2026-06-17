"""Losses and metrics for image super-resolution."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="SuperResolution")
def charbonnier_loss(y_true, y_pred, epsilon: float = 1e-3):
    """Robust L1 loss that is less blurry than pure MSE."""

    error = y_true - y_pred
    return tf.reduce_mean(tf.sqrt(tf.square(error) + epsilon * epsilon))


@keras.utils.register_keras_serializable(package="SuperResolution")
def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


@keras.utils.register_keras_serializable(package="SuperResolution")
def charbonnier_ssim_loss(y_true, y_pred):
    return charbonnier_loss(y_true, y_pred) + 0.1 * ssim_loss(y_true, y_pred)


@keras.utils.register_keras_serializable(package="SuperResolution")
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


@keras.utils.register_keras_serializable(package="SuperResolution")
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

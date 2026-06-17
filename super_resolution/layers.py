"""Keras layers used by the super-resolution models."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="SuperResolution")
class PixelShuffle(keras.layers.Layer):
    """Fast sub-pixel upsampling using TensorFlow's depth_to_space op."""

    def __init__(self, scale: int, **kwargs):
        super().__init__(**kwargs)
        if scale < 2:
            raise ValueError("scale must be >= 2")
        self.scale = int(scale)

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        out_height = None if height is None else height * self.scale
        out_width = None if width is None else width * self.scale
        out_channels = None if channels is None else channels // (self.scale**2)
        return batch, out_height, out_width, out_channels

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


@keras.utils.register_keras_serializable(package="SuperResolution")
class BicubicUpsample(keras.layers.Layer):
    """Resize an image tensor by an integer scale factor."""

    def __init__(self, scale: int, **kwargs):
        super().__init__(**kwargs)
        if scale < 2:
            raise ValueError("scale must be >= 2")
        self.scale = int(scale)

    def call(self, inputs):
        height = tf.shape(inputs)[1] * self.scale
        width = tf.shape(inputs)[2] * self.scale
        return tf.image.resize(inputs, [height, width], method="bicubic")

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        out_height = None if height is None else height * self.scale
        out_width = None if width is None else width * self.scale
        return batch, out_height, out_width, channels

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


@keras.utils.register_keras_serializable(package="SuperResolution")
class ClipByValue(keras.layers.Layer):
    """Clip image predictions into a valid range."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


@keras.utils.register_keras_serializable(package="SuperResolution")
class ResidualScale(keras.layers.Layer):
    """Scale residual branch activations for stable deeper training."""

    def __init__(self, scale: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


@keras.utils.register_keras_serializable(package="SuperResolution")
class Subpixel(keras.layers.Layer):
    """Backward-compatible replacement for the original custom Subpixel layer."""

    def __init__(
        self,
        filters,
        kernel_size,
        r,
        padding="valid",
        data_format=None,
        strides=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = kernel_size
        self.r = int(r)
        self.padding = padding
        self.data_format = data_format
        self.strides = strides
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.conv = keras.layers.Conv2D(
            filters=self.filters * self.r * self.r,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.shuffle = PixelShuffle(self.r)

    def call(self, inputs):
        return self.shuffle(self.conv(inputs))

    def compute_output_shape(self, input_shape):
        conv_shape = self.conv.compute_output_shape(input_shape)
        return self.shuffle.compute_output_shape(conv_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "r": self.r,
                "padding": self.padding,
                "data_format": self.data_format,
                "strides": self.strides,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
                "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            }
        )
        return config

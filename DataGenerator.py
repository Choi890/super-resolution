from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    """Loads preprocessed ``x_*`` and ``y_*`` .npy pairs for Keras.

    The previous implementation dropped the final partial batch, used float64
    arrays, and built label paths with Unix-only string splitting. This version
    is platform-safe and keeps all available samples.
    """

    def __init__(
        self,
        list_IDs,
        labels=None,
        batch_size=32,
        dim=(32, 32),
        n_channels=1,
        n_classes=None,
        shuffle=True,
        scale=4,
        dtype=np.float32,
    ):
        self.dim = tuple(dim)
        self.batch_size = int(batch_size)
        self.labels = labels
        self.list_IDs = [str(path) for path in list_IDs]
        self.n_channels = int(n_channels)
        self.n_classes = n_classes
        self.shuffle = bool(shuffle)
        self.scale = int(scale)
        self.dtype = dtype
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_ids = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(batch_ids)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _target_path(self, input_path):
        path = Path(input_path)
        parent = path.parent
        target_parent = parent.with_name("y" + parent.name[1:])
        return target_parent / path.name

    def __data_generation(self, batch_ids):
        batch_size = len(batch_ids)
        x = np.empty((batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        y = np.empty(
            (
                batch_size,
                self.dim[0] * self.scale,
                self.dim[1] * self.scale,
                self.n_channels,
            ),
            dtype=self.dtype,
        )

        for i, input_path in enumerate(batch_ids):
            x[i] = np.load(input_path).astype(self.dtype, copy=False)
            y_path = self._target_path(input_path)
            y[i] = np.load(y_path).astype(self.dtype, copy=False)

        return x, y

import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbackConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbackConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        # Use pathlib for directory construction
        tb_log_dir = self.config.tensorboard_root_log_dir / \
            f"tb_logs_at_{timestamp}"
        # Create directory if it doesn't exist
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        return tf.keras.callbacks.TensorBoard(log_dir=str(tb_log_dir))

    @property
    def _create_ckpt_callbacks(self):
        # Ensure parent directory exists and convert Path to string
        self.config.checkpoint_model_filepath.parent.mkdir(
            parents=True, exist_ok=True)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]

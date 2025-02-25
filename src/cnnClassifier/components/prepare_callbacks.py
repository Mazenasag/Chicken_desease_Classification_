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
        # Create a unique timestamp
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,  # Base log directory
            f"tb_logs_at_{timestamp}",  # Create a folder with timestamp
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,  # Save model to this path
            save_best_only=True  # Save only if it improves
        )

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]

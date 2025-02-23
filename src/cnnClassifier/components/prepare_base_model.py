import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class prepareBasedModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_based_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top

        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input,
                                           outputs=prediction)\

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_based_model(self):
        """Updates the base model by freezing layers and saves the updated model"""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,  # Fixed typo
            freeze_all=True,  # Fixed typo
            freeze_till=None,
            learning_rate=self.config.params_learning_rate  # Fixed typo
        )

        # Correctly calling the static method
        self.save_model(path=self.config.update_base_model_path,
                        model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the given Keras model to the specified path"""
        model.save(path)

from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.common import read_yaml, create_directories
import tensorflow as tf
tf.config.run_functions_eagerly(True)


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            str(self.config.updated_base_model_path))
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Train the model using the prepared data generators.

        Args:
        callback_list (list): List of Keras callbacks (e.g., EarlyStopping, ModelCheckpoint).
        """
        model.save(str(path))

    def train(self, callback_list: list):
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            validation_steps=self.valid_generator.samples // self.valid_generator.batch_size,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )
        self.save_model(self.config.trained_model_path, self.model)
# class Training:
#     def __init__(self, config: TrainingConfig):
#         """
#         Initialize the Training class with a configuration object.

#         Args:
#         config: An instance of TrainingConfig containing model paths,
#                 data paths, training parameters, etc.
#         """
#         self.config = config

#     def get_base_model(self):
#         """
#         Load the pre-trained base model from the specified path.
#         This model will be fine-tuned with new training data.
#         """
#         self.model = tf.keras.models.load_model(
#             str(self.config.updated_base_model_path))
#         self.model.compile(
#             optimizer=tf.keras.optimizers.SGD(
#                 learning_rate=self.config.params_learning_rate),
#             loss=tf.keras.losses.CategoricalCrossentropy(),
#             metrics=["accuracy"]
#         )

#     def train_valid_generator(self):
#         """
#         Create training and validation data generators using ImageDataGenerator.

#         - The validation generator always performs only rescaling.
#         - The training generator applies augmentation if enabled.
#         """
#         # Common preprocessing settings for both training and validation
#         datagenerator_kwargs = dict(rescale=1./255, validation_split=0.20)
#         dataflow_kwargs = dict(
#             # Normalize pixel values to range [0,1]
#             target_size=self.config.params_image_size[:-1],
#             # Use 20% of the dataset for validation
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear"  # Interpolation method for resizing
#         )
#         # Create validation data generator (NO augmentation, only rescaling)
#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs)
#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=str(self.config.training_data),
#             subset="validation",  # Load validation set
#             shuffle=False,
#             **dataflow_kwargs
#         )
#         if self.config.params_is_augmentation:
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=40,  # Random rotation up to 40 degrees
#                 horizontal_flip=True,  # Flip images horizontally
#                 width_shift_range=0.2,  # Shift image width by 20%
#                 height_shift_range=0.2,  # Shift image height by 20%
#                 shear_range=0.2,  # Shear transformation
#                 zoom_range=0.2,  # Zoom in/out up to 20%
#                 **datagenerator_kwargs  # Include common rescaling & split
#             )
#         else:
#          # If no augmentation, reuse validation generator settings for training
#             train_datagenerator = valid_datagenerator
#             self.train_generator = train_datagenerator.flow_from_directory(
#                 directory=str(self.config.training_data),
#                 subset="training",
#                 shuffle=True,
#                 **dataflow_kwargs
#             )

#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         """
#         Train the model using the prepared data generators.

#         Args:
#         callback_list (list): List of Keras callbacks (e.g., EarlyStopping, ModelCheckpoint).
#         """
#         model.save(str(path))

#     def train(self, callback_list: list):
#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
#             validation_steps=self.valid_generator.samples // self.valid_generator.batch_size,
#             validation_data=self.valid_generator,
#             callbacks=callback_list
#         )
#         self.save_model(self.config.trained_model_path, self.model)

from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.common import read_yaml, create_directories
import tensorflow as tf
tf.config.run_functions_eagerly(True)


class Training:
    def __init__(self, config):
        """
        Initialize the Training class with a configuration object.

        Args:
        config: An instance of TrainingConfig containing model paths,
                data paths, training parameters, etc.
        """
        self.config = config

    def get_base_model(self):
        """
        Load the pre-trained base model from the specified path.
        This model will be fine-tuned with new training data.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path)

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',  # Assuming you're doing classification
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        """
        Create training and validation data generators using ImageDataGenerator.

        - The validation generator always performs only rescaling.
        - The training generator applies augmentation if enabled.
        """

        # Common preprocessing settings for both training and validation
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,  # Normalize pixel values to range [0,1]
            validation_split=0.20  # Use 20% of the dataset for validation
        )

        # Settings for how the data should be loaded
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Resize images
            batch_size=self.config.params_batch_size,  # Batch size for training
            interpolation="bilinear"  # Interpolation method for resizing
        )

        # Create validation data generator (NO augmentation, only rescaling)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",  # Load validation set
            shuffle=False,  # No shuffling needed for validation
            **dataflow_kwargs
        )

        # If augmentation is enabled, create a separate training generator
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,  # Random rotation up to 40 degrees
                horizontal_flip=True,  # Flip images horizontally
                width_shift_range=0.2,  # Shift image width by 20%
                height_shift_range=0.2,  # Shift image height by 20%
                shear_range=0.2,  # Shear transformation
                zoom_range=0.2,  # Zoom in/out up to 20%
                **datagenerator_kwargs  # Include common rescaling & split
            )
        else:
            # If no augmentation, reuse validation generator settings for training
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",  # Load training set
            shuffle=True,  # Shuffle for better training performance
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to the specified path.

        Args:
        path (Path): File path to save the model.
        model (tf.keras.Model): The trained Keras model.
        """
        model.save(path)

    def train(self, callback_list: list):
        """
        Train the model using the prepared data generators.

        Args:
        callback_list (list): List of Keras callbacks (e.g., EarlyStopping, ModelCheckpoint).
        """
        # Define training steps per epoch based on dataset size and batch size
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,  # Number of epochs
            steps_per_epoch=self.steps_per_epoch,  # Steps per training epoch
            validation_steps=self.validation_steps,  # Validation dataset
            validation_data=self.valid_generator,  # Steps per validation epoch
            callbacks=callback_list  # Callbacks for monitoring and saving
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

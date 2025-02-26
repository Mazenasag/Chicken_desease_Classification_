from cnnClassifier.utils.common import save_json
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        # Prepare the validation data generator with image rescaling and validation split
        datagenerator_kwargs = dict(rescale=1. / 255, validation_split=0.30)

        # Additional parameters for the dataflow
        dataflow_kwargs = dict(
            # Resize images to target size (excluding channels)
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,  # Batch size from config
            interpolation="bilinear"  # Interpolation method for resizing images
        )

        # Initialize the ImageDataGenerator for validation
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs)

        # Create a validation data generator by loading images from the directory
        self.valid_generator = valid_datagenerator.flow_from_directory(
            # Directory for training data
            directory=str(self.config.training_data),
            subset="validation",  # We are only using the validation subset
            shuffle=False,  # Don't shuffle to maintain consistency during evaluation
            # Apply the dataflow arguments for resizing, batch size, etc.
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        # Load a pre-trained model from the given path
        return tf.keras.models.load_model(path)

    def evaluation(self):
        # Load the model to be evaluated
        # Corrected to use `load_model` method
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()  # Prepare the validation data generator
        # Evaluate the model on the validation data
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        # Save the evaluation scores (loss and accuracy) into a JSON file
        # Fix the reference to `score[1]` for accuracy
        scores = {'loss': self.score[0], "accuracy": self.score[1]}
        # Save the scores to a JSON file
        save_json(path=Path("scores.json"), data=scores)

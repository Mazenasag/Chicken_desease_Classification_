import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Define the PredictionPipeline class


class PredictionPipeline:
    def __init__(self, filename):
        # Initialize with the filename (path to the image to be predicted)
        self.filename = filename

    def predict(self):
        # Load the pre-trained model from the specified path
        model = load_model(os.path.join(
            "artifacts", "training", "model.keras"))

        # The image file to predict is provided via self.filename
        imagenet = self.filename

        # Load and resize the image to the required target size (244x244 pixels)
        test_image = image.load_img(imagenet, target_size=(224, 224))

        # Convert the image to a NumPy array (model expects a NumPy array)
        test_image = image.img_to_array(test_image)

        # Add an extra dimension to the image array to simulate a batch of 1 image
        test_image = np.expand_dims(test_image, axis=0)

        # Perform prediction using the model and get the class with the highest probability
        result = np.argmax(model.predict(test_image), axis=1)

        # Print the predicted class (for debugging purposes)
        print(result)
        if result[0] == 1:
            prediction = 'Healthy'
            return [{"image": prediction}]
        else:
            prediction = 'Coccidiosis'
            return [{"image": prediction}]

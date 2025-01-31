from tensorflow import keras
import numpy as np

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset."""
    # Load the data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
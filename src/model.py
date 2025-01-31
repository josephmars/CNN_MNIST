from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes):
    """Create CNN model for MNIST classification."""
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=input_shape),
        
        # First convolutional layer
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten layer
        layers.Flatten(),
        
        # Fourth fully connected layer
        layers.Dense(100, activation="relu"),
        
        # Last classifier layer
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    return model
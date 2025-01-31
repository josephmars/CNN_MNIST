import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Model parameters
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Training parameters
BATCH_SIZE = 128
EPOCHS = 15
VALIDATION_SPLIT = 0.1
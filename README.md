# MNIST CNN Classifier

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. The model architecture consists of three convolutional layers followed by dense layers for classification.

The CNN was trained using a batch size of 128 for 15 epochs. In this dataset we have 60000 
train samples and 10000 test samples. Baseline CNN Model Structure: 
• First convolutional layer: 8@3x3 + BatchNormalization + MaxPooling(2x2) + ReLU 
• Second convolutional layer: 16@3x3 + BatchNormalization + MaxPooling(2x2) + 
ReLU 
• Third convolutional layer: 32@3x3 + BatchNormalization + MaxPooling(2x2) + 
ReLU 
• Fourth fully connected layer: 100 hidden units + ReLU 
• The last classifier layer with softmax for classification (10 units) 


## Project Structure

- `src/config.py`: Configuration parameters for the model and training
- `src/data.py`: Data loading and preprocessing utilities
- `src/model.py`: CNN model architecture definition
- `src/train.py`: Training and evaluation scripts

## Model Architecture

The CNN architecture includes:
- 3 Convolutional layers (8, 16, and 32 filters) with ReLU activation
- Batch normalization after each convolutional layer
- Max pooling layers
- Dense layers (100 units and 10 output classes)

## Requirements

- TensorFlow
- NumPy
- Pandas

## Usage

Run the training script:
```
python
python src/train.py
```


## Performance

The model achieves:
- Training accuracy: ~99%
- Test accuracy: ~98%

## Credits

Model architecture based on: https://keras.io/examples/vision/mnist_convnet/
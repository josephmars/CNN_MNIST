from config import *
from data import load_and_preprocess_data
from model import create_model

def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    model = create_model(INPUT_SHAPE, NUM_CLASSES)
    
    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    # Train model
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )
    
    # Evaluate model
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print("Train loss:", train_score[0])
    print("Train accuracy:", train_score[1])

    test_score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", test_score[0])
    print("Test accuracy:", test_score[1])

if __name__ == "__main__":
    main()
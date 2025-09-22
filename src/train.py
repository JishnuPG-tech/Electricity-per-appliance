
import tensorflow as tf
from src.data_preprocessing import preprocess_data
from src.model import build_model
import os

def train_model():
    """Trains the neural network model and saves it."""
    # 1. Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

    # 2. Build the model
    # The input shape is the number of features in our processed data
    input_shape = X_train.shape[1]
    model = build_model(input_shape)

    print("\n--- Starting Model Training ---")
    # 3. Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=50, # An epoch is one full pass through the training data
        validation_split=0.2, # Use 20% of training data for validation
        batch_size=32, # Number of samples per gradient update
        verbose=1 # Show training progress
    )
    print("--- Model Training Complete")

    # 4. Evaluate the model on the test set
    print("\n--- Evaluating Model Performance ---")
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Mean Absolute Error: {mae:.2f} kWh")

    # 5. Save the trained model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_model')
    os.makedirs(model_path, exist_ok=True) # Ensure the directory exists
    model.save(model_path)
    
    # We also need to save the preprocessor
    import joblib
    preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'saved_model', 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)

    print(f"\nModel and preprocessor saved to: {model_path}")

    return history, model

if __name__ == "__main__":
    train_model()

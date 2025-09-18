
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_forecasting_model(input_shape):
    """Builds and compiles the LSTM forecasting model."""
    model = Sequential([
        # LSTM layer to capture temporal patterns
        # input_shape should be (timesteps, features)
        LSTM(50, activation='relu', input_shape=input_shape),
        
        # Output layer - a single neuron for regression
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

if __name__ == '__main__':
    # Example of how to build the model
    # The input shape will be (number of time steps, number of features)
    # e.g., (3, 1) for using the last 3 months' bills to predict the next one
    example_input_shape = (3, 1) 
    model = build_forecasting_model(example_input_shape)
    model.summary() # Print a summary of the model architecture


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(input_shape):
    """Builds and compiles the neural network model."""
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=[input_shape]),
        
        # Hidden layers
        Dense(64, activation='relu'),
        Dropout(0.2), # Dropout layer to prevent overfitting
        Dense(32, activation='relu'),
        
        # Output layer
        # A single neuron with a linear activation for regression
        Dense(1)
    ])

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    
    return model

if __name__ == '__main__':
    # Example of how to build the model
    # The input shape will depend on our preprocessed data
    # We will get this shape from our preprocessor in the training script
    example_input_shape = 10 # This is just a placeholder
    model = build_model(example_input_shape)
    model.summary() # Print a summary of the model architecture


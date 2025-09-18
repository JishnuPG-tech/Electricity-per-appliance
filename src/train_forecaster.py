
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.forecasting_model import build_forecasting_model

# --- Configuration ---
LOOK_BACK = 3 # Number of previous time steps to use as input variables

def create_dataset(dataset, look_back=LOOK_BACK):
    """Create sequences for time series forecasting."""
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_forecasting_model():
    """Trains the LSTM model and saves it."""
    # 1. Load the dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'monthly_bills.csv')
    df = pd.read_csv(data_path)
    dataset = df['total_bill'].values.reshape(-1, 1)

    # 2. Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 3. Create training sequences
    train_X, train_Y = create_dataset(dataset)

    # Reshape input to be [samples, time steps, features]
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))

    # 4. Build and train the model
    model = build_forecasting_model(input_shape=(LOOK_BACK, 1))
    
    print("\n--- Starting Forecasting Model Training ---")
    model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=1)
    print("--- Forecasting Model Training Complete ---")

    # 5. Save the model and scaler
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_model')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'forecasting_model.keras')
    model.save(model_path)
    
    scaler_path = os.path.join(model_dir, 'forecasting_scaler.joblib')
    joblib.dump(scaler, scaler_path)

    print(f"\nForecasting model and scaler saved to: {model_dir}")

if __name__ == "__main__":
    train_forecasting_model()

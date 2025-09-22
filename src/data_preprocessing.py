
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def preprocess_data():
    """Loads, preprocesses, and splits the appliance data."""

    # Load the dataset
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'appliances.csv')
    df = pd.read_csv(file_path)

    # Define features (X) and target (y)
    X = df.drop("daily_consumption_kwh", axis=1)
    y = df["daily_consumption_kwh"]

    # Define categorical and numerical features
    categorical_features = ["appliance_name"]
    numerical_features = ["power_rating_watts", "daily_usage_hours", "star_rating", "monthly_bill"]

    # Create a preprocessing pipeline
    # This will scale numerical features and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply the preprocessing pipeline to the data
    # We fit the preprocessor on the training data and transform both training and test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Data preprocessing complete.")
    print("Shape of processed training features:", X_train_processed.shape)
    print("Shape of processed testing features:", X_test_processed.shape)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    preprocess_data()


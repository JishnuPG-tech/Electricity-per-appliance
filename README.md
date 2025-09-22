
# 💡 Household Electricity Bill Disaggregation & Prediction

> A web application for disaggregating household electricity bills, predicting future energy consumption, and providing energy-saving recommendations.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Key Features

- **Appliance-Level Consumption Prediction**: Predicts the daily energy consumption (in kWh) for each household appliance.
- **Bill Disaggregation**: Allocates the total monthly electricity bill across different appliances based on their predicted consumption.
- **Bill Forecasting**: Provides an estimated electricity bill for the next month.
- **Usage Optimization Suggestions**: Offers tips and recommendations to help users reduce their electricity expenses.
- **Interactive Web Interface**: A user-friendly web interface built with Streamlit for easy data entry and visualization of results.

---

## 🛠️ Technologies Used

- **Backend**: Python
- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow (Keras), Scikit-learn
- **Data Manipulation**: Pandas, NumPy

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ Usage

1.  **Train the Model**:
    Before running the application, you need to train the model and create the preprocessor. Run the following script from the project's root directory:
    ```sh
    python src/train.py
    ```
    This will generate `model_v2.keras` and `preprocessor_v2.joblib` in the `saved_model` directory.

2.  **Run the Streamlit Application**:
    ```sh
    streamlit run app/app.py
    ```

3.  **Access the Application**:
    Open your web browser and go to `http://localhost:8501`.

---

## 📂 Project Structure

```
.
├── app/
│   └── app.py              # Streamlit web application
├── data/
│   ├── appliances.csv      # Original dataset
│   └── appliances_v2.csv   # Augmented dataset
├── saved_model/
│   ├── model.keras         # Trained model (V1)
│   ├── model_v2.keras      # Trained model (V2)
│   ├── preprocessor.joblib # Preprocessor for V1 model
│   └── preprocessor_v2.joblib# Preprocessor for V2 model
├── src/
│   ├── data_preprocessing.py # Data cleaning and feature engineering
│   ├── generate_data.py    # Script to augment the original dataset
│   ├── model.py            # Neural network model definition
│   └── train.py            # Model training script
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


import pandas as pd
import numpy as np
import os

def generate_time_series_data(years=5, output_dir=None):
    """
    Generates a synthetic time series of monthly electricity bills with seasonality.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'monthly_bills.csv')

    # Create a date range for the specified number of years
    dates = pd.date_range(start='2020-01-01', periods=years * 12, freq='M')
    
    # Base bill amount
    base_bill = 1500
    
    # Seasonal component (higher in summer, lower in winter)
    # We'll use a sine wave to simulate this
    seasonal_component = 500 * np.sin(np.linspace(0, years * 2 * np.pi, years * 12))
    
    # Trend component (gradual increase over time)
    trend_component = np.linspace(0, 300, years * 12)
    
    # Random noise
    noise = np.random.normal(0, 50, years * 12)
    
    # Combine components to get the total bill
    total_bill = base_bill + seasonal_component + trend_component + noise
    
    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'total_bill': total_bill})
    df['total_bill'] = df['total_bill'].round(2)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Synthetic time series data saved to {output_path}")

if __name__ == "__main__":
    generate_time_series_data()

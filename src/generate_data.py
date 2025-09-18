
import pandas as pd
import numpy as np
import os

# Define the appliances and their typical power ratings (in watts)
APPLIANCES = {
    "Refrigerator": 200,
    "Washing Machine": 500,
    "Air Conditioner": 1500,
    "Television": 100,
    "Microwave": 1000,
    "Fan": 80,
    "Lighting": 60,
}

# Number of data points to generate
N_SAMPLES = 1000

def generate_data():
    """Generates synthetic appliance data and saves it to a CSV file."""
    data = []
    for _ in range(N_SAMPLES):
        for appliance, base_power in APPLIANCES.items():
            # Add some variability to power rating
            power_rating = base_power + np.random.randint(-20, 20)

            # Generate random daily usage hours
            if appliance in ["Refrigerator"]:
                usage_hours = np.random.uniform(8, 12) # Always on, but compressor runs intermittently
            elif appliance in ["Air Conditioner", "Fan"]:
                usage_hours = np.random.uniform(1, 10)
            elif appliance in ["Washing Machine", "Microwave"]:
                usage_hours = np.random.uniform(0.2, 1.5)
            else:
                usage_hours = np.random.uniform(1, 6)

            # Generate star rating (1 to 5)
            star_rating = np.random.randint(1, 6)

            # Calculate daily consumption in kWh
            # Higher star rating reduces consumption
            efficiency_factor = 1 - (star_rating * 0.05) # 5% more efficient per star
            daily_consumption_kwh = (power_rating / 1000) * usage_hours * efficiency_factor

            data.append({
                "appliance_name": appliance,
                "power_rating_watts": power_rating,
                "daily_usage_hours": usage_hours,
                "star_rating": star_rating,
                "daily_consumption_kwh": daily_consumption_kwh,
            })

    df = pd.DataFrame(data)

    # Add a feature for the total monthly bill for the household
    # This is a bit of a simplification, but useful for the model
    # We'll calculate a household_id for each group of appliances
    df['household_id'] = df.index // len(APPLIANCES)
    household_monthly_consumption = df.groupby('household_id')['daily_consumption_kwh'].sum() * 30
    
    # Assume a price per kWh
    price_per_kwh = 8 # Example price
    household_monthly_bill = household_monthly_consumption * price_per_kwh
    
    df = df.merge(household_monthly_bill.rename('monthly_bill'), on='household_id')
    df = df.drop(columns=['household_id'])


    # Save the data to the data directory
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'appliances.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved data to {output_path}")

if __name__ == "__main__":
    generate_data()


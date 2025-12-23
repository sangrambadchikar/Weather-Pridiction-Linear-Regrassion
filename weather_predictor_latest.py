
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

# --- CONFIGURATION ---
# The number of previous days' temperatures to use as features (the 'look-back' period).
LOOK_BACK_DAYS = 7 
# Number of days to use for the primary analysis and training (approx. one month)
ANALYSIS_DAYS = 30
# ---------------------

def create_mock_weather_data(total_days=365):
    """
    Creates a mock dataset simulating daily average temperature up to yesterday.
    """
    np.random.seed(42)
    # Data ends yesterday (current date is December 1, 2025)
    dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=total_days, freq='D')

    days_in_year = 365.25
    t = np.arange(total_days)

    # 1. Base Seasonal Trend
    # Adjusting the phase to simulate being near December/winter
    seasonal_trend = 10 * np.sin(2 * np.pi * t / days_in_year + np.pi/2) 

    # 2. Base Temperature and Noise
    base_temp = 20  
    noise = np.random.normal(0, 1.5, total_days)

    temperature = base_temp + seasonal_trend + noise

    df = pd.DataFrame({'Date': dates, 'Temperature': temperature})
    df.set_index('Date', inplace=True)
    return df

def prepare_time_series_data(df, look_back):
    """
    Converts the time series data into a supervised learning problem (Features X, Target y).
    """
    X, y = [], []
    data = df['Temperature'].values

    for i in range(len(data) - look_back):
        # Features X: Temperatures from day i to day i + look_back - 1
        X.append(data[i:i + look_back])

        # Target y: Temperature on day i + look_back (the next day)
        y.append(data[i + look_back])

    return np.array(X), np.array(y)

# --- EXECUTION ---

# 1. Load Data and Filter to the Latest Month
weather_df = create_mock_weather_data()

# Filter the data to include only the last ANALYSIS_DAYS for training
latest_df = weather_df.tail(ANALYSIS_DAYS).copy()
print(f"1. Analyzing the latest {ANALYSIS_DAYS} days of data.")

# 2. Prepare Data for Regression using only the latest data
X, y = prepare_time_series_data(latest_df, LOOK_BACK_DAYS)

# Ensure enough data points exist for training and testing
if len(X) < 1:
    print("\nError: Not enough data points to create features and targets.")
    exit()

# 3. Use all prepared data for training (since the set is already small and latest)
X_train, y_train = X, y

# 4. Initialize and Train the Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)
print("2. Model Training Complete (Linear Regression on latest data).")


# 5. Predict the Next Day's Temperature
# The input for prediction is the temperature sequence of the absolute last LOOK_BACK_DAYS
last_sequence = latest_df['Temperature'].values[-LOOK_BACK_DAYS:]
X_next_day = last_sequence.reshape(1, -1)

next_day_temp_predicted = model.predict(X_next_day)[0]
next_day_date = latest_df.index[-1] + timedelta(days=1)

print("-" * 50)
print(f"☀️ Predicted Temperature for {next_day_date.strftime('%Y-%m-%d')}: {next_day_temp_predicted:.2f} °C")
print("-" * 50)

# 6. Visualization (Showing the last 30 days and the prediction)
plt.figure(figsize=(10, 5))

# Plot the historical temperature used for training
plt.plot(latest_df.index, latest_df['Temperature'], label='Historical Temperature', color='blue')

# Plot the single prediction point
plt.scatter(next_day_date, next_day_temp_predicted, 
            label=f'Prediction ({next_day_temp_predicted:.2f}°C)', 
            color='red', marker='o', s=100, zorder=5)

plt.title(f'Temperature Trend and Prediction (Last {ANALYSIS_DAYS} Days)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

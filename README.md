🌤️ Weather Data Analysis and Prediction
This project implements a simple time series forecasting model to analyze historical temperature data and predict future temperature trends.

🎯 Methodology: Linear Regression
The model uses Linear Regression on a small, recent subset of the historical data (the last 30 days) to predict the temperature for the next day. This method is effective for identifying local, short-term temperature trends.

⚙️ How it Works
Time Series Transformation: Historical daily temperatures are converted into a supervised learning problem. The temperatures of the last 7 days are used as features (input X) to predict the temperature of the next day (output Y).
Training: The model is trained exclusively on the most recent 30 days of this transformed data ("latest to latest").
Prediction: The model forecasts the temperature for the day immediately following the last date in the dataset.
🛠️ Setup and Execution
Requirements
Install the necessary libraries:

pip install pandas scikit-learn matplotlib numpy

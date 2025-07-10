from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from datetime import timedelta
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # <-- Explicitly allow all origins

@app.route('/forecast', methods=['GET'])
def forecast():
    # === Fetch Data ===
    url = "https://opticoolweb-backend.onrender.com/api/v1/powerconsumptions"
    response = requests.get(url)
    data = response.json()

    # === Convert to DataFrame ===
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'consumption']].dropna()
    df = df.set_index('timestamp').sort_index()

    # === Resample Hourly ===
    df_hourly = df.resample('1H').mean().dropna()
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['dayofweek'] = df_hourly.index.dayofweek
    df_hourly['is_weekend'] = df_hourly['dayofweek'].isin([5, 6]).astype(int)
    df_hourly['lag1'] = df_hourly['consumption'].shift(1)
    df_hourly['rolling3'] = df_hourly['consumption'].rolling(3).mean()
    df_hourly.dropna(inplace=True)

    # === Train/Test Split ===
    features = ['hour', 'dayofweek', 'is_weekend', 'lag1', 'rolling3']
    X = df_hourly[features]
    y = df_hourly['consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # === Train XGBoost Model ===
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    # === Predict for Next 3 Months (Hourly) ===
    last_timestamp = df_hourly.index[-1]
    future_hours = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=24*31*3, freq='H')
    future_df = pd.DataFrame({'timestamp': future_hours})
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['dayofweek'] = future_df['timestamp'].dt.dayofweek
    future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
    latest_val = df_hourly.iloc[-1]['consumption']
    future_df['lag1'] = latest_val
    future_df['rolling3'] = latest_val
    future_df['predicted'] = model.predict(future_df[features])

    # === Resample Daily, Exclude Sundays ===
    future_df['date'] = future_df['timestamp'].dt.date
    future_df['weekday'] = future_df['timestamp'].dt.weekday
    daily_pred = future_df.groupby(['date', 'weekday'])['predicted'].mean().reset_index()
    daily_pred = daily_pred[daily_pred['weekday'] != 6]  # Exclude Sundays (weekday==6)

    # === Daily Forecast: Next 3 Days (Exclude Sundays) ===
    next_days = daily_pred[daily_pred['date'] > last_timestamp.date()].head(3)
    daily_forecast = next_days[['date', 'predicted']]
    daily_forecast = daily_forecast.rename(columns={'date': 'timestamp'})

    # === Monthly Forecast: Next 3 Months (Exclude Sundays) ===
    daily_pred['month'] = pd.to_datetime(daily_pred['date']).dt.to_period('M')
    monthly = daily_pred.groupby('month').agg(
        total_predicted=('predicted', 'sum'),
        days=('predicted', 'count')
    ).reset_index()
    monthly = monthly.head(3)  # Next 3 months only
    monthly['predicted_average'] = monthly['total_predicted'] / monthly['days']
    monthly_forecast = monthly[['month', 'predicted_average']]
    monthly_forecast['month'] = monthly_forecast['month'].dt.strftime('%b %Y')
    # === Return results as JSON ===
    return jsonify({
        "daily_forecast": daily_forecast.to_dict(orient='records'),
        "monthly_forecast": monthly_forecast.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(port=5001)

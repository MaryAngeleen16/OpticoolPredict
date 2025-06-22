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

    # === Predict for Next 3 Days ===
    last_timestamp = df_hourly.index[-1]
    future_hours = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=24*3, freq='H')
    future_df = pd.DataFrame({'timestamp': future_hours})
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['dayofweek'] = future_df['timestamp'].dt.dayofweek
    future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
    latest_val = df_hourly.iloc[-1]['consumption']
    future_df['lag1'] = latest_val
    future_df['rolling3'] = latest_val
    future_df['predicted'] = model.predict(future_df[features])

    # === Resample Daily ===
    daily_forecast = future_df.set_index('timestamp').resample('D').mean().reset_index()
    daily_forecast['timestamp'] = daily_forecast['timestamp'].dt.strftime('%Y-%m-%d')

    # === Predict 3-Month Average ===
    avg_daily = daily_forecast['predicted'].mean()
    monthly_forecast = pd.DataFrame({
        'month': pd.date_range(start=pd.to_datetime(daily_forecast['timestamp'].iloc[-1]) + timedelta(days=1), periods=3, freq='MS').strftime('%Y-%m'),
        'predicted_average': [avg_daily * 30] * 3
    })

    # Return results as JSON
    return jsonify({
        "daily_forecast": daily_forecast[['timestamp', 'predicted']].to_dict(orient='records'),
        "monthly_forecast": monthly_forecast.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(port=5001)

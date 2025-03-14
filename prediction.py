# prediction.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import json

class PricePredictor:
    """Predict prices with unique hype-adjusted LSTM."""
    def __init__(self, processed_dir="processed_data"):
        self.processed_dir = processed_dir
        self.data = {}
        self.metadata = {}
        self.predictions = {}
    
    def load_processed_data(self):
        """Load processed data and metadata."""
        try:
            for file in os.listdir(self.processed_dir):
                if file.endswith(".csv"):
                    ticker = file.split("_")[0]
                    filepath = os.path.join(self.processed_dir, file)
                    self.data[ticker] = pd.read_csv(filepath, index_col="Date", parse_dates=True)
                elif file.startswith("metadata") and file.endswith(".json"):
                    with open(os.path.join(self.processed_dir, file), 'r') as f:
                        self.metadata = json.load(f)
            if not self.data:
                raise ValueError("No processed data found.")
        except Exception as e:
            print(f"Error loading processed data: {e}")
    
    def predict(self, forecast_days=10):
        """Generate LSTM predictions adjusted by hype score."""
        self.load_processed_data()
        for ticker, df in self.data.items():
            if not df.empty and len(df) >= 60:
                try:
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
                    time_step = 60
                    X = [scaled_data[i-time_step:i, 0] for i in range(time_step, len(scaled_data))]
                    X = np.array(X).reshape((len(X), time_step, 1))
                    
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                    model.add(LSTM(50))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X, scaled_data[time_step:], epochs=5, batch_size=32, verbose=0)
                    
                    last_sequence = scaled_data[-time_step:].reshape((1, time_step, 1))
                    forecast = []
                    for _ in range(forecast_days):
                        pred = model.predict(last_sequence, verbose=0)
                        forecast.append(pred[0, 0])
                        last_sequence = np.roll(last_sequence, -1, axis=1)
                        last_sequence[0, -1, 0] = pred[0, 0]
                    
                    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
                    hype_adjust = self.metadata.get(ticker, {}).get("hype_score", 0) / 100
                    self.predictions[ticker] = forecast.flatten() * (1 + hype_adjust)
                except Exception as e:
                    print(f"Error predicting for {ticker}: {e}")
        return self.predictions
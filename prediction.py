# prediction.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os
import joblib

class PricePredictor:
    """Predict prices with unique hype-adjusted LSTM."""
    def __init__(self, model_dir="models"):
        """
        Initializes the predictor and ensures the model storage directory exists.
        Args:
            model_dir (str): Directory to save/load trained models and scalers.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def predict(self, data, metadata, forecast_days=10):
        """Generate LSTM predictions adjusted by hype score."""
        predictions = {}
        # NOTE: This now loads pre-trained models if available, and trains/saves them otherwise.
        # This is much faster on subsequent runs.
        for ticker, df in data.items():
            if not df.empty and len(df) >= 60:
                try:
                    model_path = os.path.join(self.model_dir, f"{ticker}_lstm_model.h5")
                    scaler_path = os.path.join(self.model_dir, f"{ticker}_scaler.joblib")
                    time_step = 60

                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        model = load_model(model_path)
                        scaler = joblib.load(scaler_path)
                    else:
                        scaler = MinMaxScaler()
                        scaled_data_for_training = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
                        
                        X, y = [], []
                        for i in range(time_step, len(scaled_data_for_training)):
                            X.append(scaled_data_for_training[i-time_step:i, 0])
                            y.append(scaled_data_for_training[i, 0])
                        X, y = np.array(X), np.array(y)
                        X = X.reshape(X.shape[0], X.shape[1], 1)
                        
                        model = Sequential([
                            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                            LSTM(50),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                        
                        model.save(model_path)
                        joblib.dump(scaler, scaler_path)

                    scaled_data = scaler.transform(df['Close'].values.reshape(-1, 1))
                    last_sequence = scaled_data[-time_step:].reshape((1, time_step, 1))
                    forecast = []
                    for _ in range(forecast_days):
                        pred = model.predict(last_sequence, verbose=0)
                        forecast.append(pred[0, 0])
                        last_sequence = np.roll(last_sequence, -1, axis=1)
                        last_sequence[0, -1, 0] = pred[0, 0]
                    
                    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
                    hype_adjust = metadata.get(ticker, {}).get("hype_score", 0) / 100
                    predictions[ticker] = forecast.flatten() * (1 + hype_adjust)
                except Exception as e:
                    print(f"Error predicting for {ticker}: {e}")
        return predictions
# data_fetching.py
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import os
import json

def fetch_x_sentiment(ticker):
    """Fetch simulated X sentiment (proxy via price change). Replaceable with real X post analysis."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="7d")
        sentiment = np.clip(df['Close'].pct_change().mean() * 10, -1, 1)
        return sentiment
    except Exception as e:
        print(f"Error fetching sentiment for {ticker}: {e}")
        return 0

class DataFetcher:
    """Fetch stock data and sentiment with unique multi-source fusion."""
    def __init__(self, tickers, start_date, end_date, output_dir="data"):
        self.tickers = [t.upper() for t in tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.data = {}
        self.metadata = {"fetch_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    def fetch(self):
        """Fetch data, calculate metadata, and return them."""
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                if df.empty:
                    print(f"No data for {ticker}")
                    continue
                
                # Add event-driven anomaly detection
                df['Daily_Return'] = df['Close'].pct_change()
                df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
                df['Volume_Spike'] = np.where(df['Volume'] > 2 * df['Volume_MA'], True, False)
                # NOTE: This is a rough approximation for earnings dates. For higher accuracy,
                # a dedicated API or yfinance's calendar data should be used.
                earnings_dates = pd.date_range(self.start_date, self.end_date, freq='Q').strftime('%Y-%m-%d').tolist()
                df['Earnings_Event'] = df.index.isin(earnings_dates)
                self.data[ticker] = df
                
                # Multi-source fusion with X sentiment
                sentiment = fetch_x_sentiment(ticker)
                hype_score = (df['Daily_Return'].mean() * 10 + df['Volume_Spike'].mean() * 30 + sentiment * 30) / 3
                self.metadata[ticker] = {
                    "sentiment": sentiment,
                    "hype_score": float(np.clip(hype_score, -100, 100)),
                    "spike_count": int(df['Volume_Spike'].sum())
                }
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        return self.data, self.metadata

    def save_to_disk(self):
        """Save the fetched data and metadata to the output directory."""
        if not self.data:
            print("No data to save. Run fetch() first.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        for ticker, df in self.data.items():
            if not df.empty:
                filepath = os.path.join(self.output_dir, f"{ticker}_stock_data_{date_str}.csv")
                df.to_csv(filepath)
        metadata_file = os.path.join(self.output_dir, f"metadata_{date_str}.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
# data_processing.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

class StockDataProcessor:
    """A class to process and enrich stock data for analysis."""
    
    def __init__(self, input_dir: str = "data"):
        """
        Initialize with the directory containing collected data.
        Args:
            input_dir (str): Directory with stock data CSVs and metadata.
        """
        self.input_dir = input_dir
        self.data = {}
        self.metadata = {}
    
    def load_data(self):
        """Load stock data and metadata from files."""
        try:
            for file in os.listdir(self.input_dir):
                if file.endswith(".csv"):
                    ticker = file.split("_")[0]
                    filepath = os.path.join(self.input_dir, file)
                    self.data[ticker] = pd.read_csv(filepath, index_col="Date", parse_dates=True)
                elif file.startswith("metadata") and file.endswith(".json"):
                    with open(os.path.join(self.input_dir, file), 'r') as f:
                        self.metadata = json.load(f)
            if not self.data:
                raise ValueError("No CSV files found in input directory.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def clean_data(self):
        """Clean data by filling missing values and removing outliers."""
        for ticker, df in self.data.items():
            # Fill missing values with forward-fill, then backward-fill
            df = df.ffill()
            df = df.bfill()
            
            # Remove extreme outliers in returns (>5 std dev)
            return_std = df['Daily_Return'].std()
            df['Daily_Return'] = np.where(
                np.abs(df['Daily_Return']) > 5 * return_std,
                np.nan, df['Daily_Return']
            )
            df['Daily_Return'] = df['Daily_Return'].fillna(df['Daily_Return'].mean())
            self.data[ticker] = df
    
    def enrich_data(self):
        """Add detailed metrics to the data with unique enhancements."""
        for ticker, df in self.data.items():
            # Rolling volatility (20-day)
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # Sentiment-adjusted returns (unique enhancement)
            sentiment = self.metadata.get(ticker, {}).get("sentiment", 0)
            df['Sentiment_Adj_Return'] = df['Daily_Return'] * (1 + sentiment)
            
            # Cumulative hype score trend (unique enhancement)
            df['Hype_Score_Cumulative'] = df['Volume_Spike'].cumsum() * (1 + sentiment)
            self.data[ticker] = df
    
    def validate_data(self):
        """Validate data integrity."""
        for ticker, df in self.data.items():
            if df.empty or df['Close'].isnull().all():
                print(f"Warning: {ticker} data is empty or invalid.")
                self.data[ticker] = pd.DataFrame()
    
    def process_data(self, data, metadata):
        """Main method to clean, enrich, and validate data."""
        self.data = data
        self.metadata = metadata
        if not self.data:
            return {}
        self.clean_data()
        self.enrich_data()
        self.validate_data()
        return self.data
    
    def save_processed_data(self, output_dir: str = "processed_data"):
        """Save processed data to files."""
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        for ticker, df in self.data.items():
            if not df.empty:
                filename = f"{ticker}_processed_data_{date_str}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath)
                print(f"Saved processed {ticker} data to {filepath}")
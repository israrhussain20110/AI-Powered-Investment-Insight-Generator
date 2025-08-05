# main.py
from datetime import datetime, timedelta
import streamlit as st
import numpy as np
from data_fetching import DataFetcher
from data_processing import StockDataProcessor
from prediction import PricePredictor
from analysis import DataAnalyzer
from visualuzation import Visualizer
from portfolio import PortfolioSimulator

@st.cache_data
def get_data(tickers_tuple, start_date_str, end_date_str):
    """
    Fetches and caches stock data using Streamlit's caching.
    The cache is invalidated if the input arguments change.
    We use a tuple for tickers because lists are not hashable.
    """
    fetcher = DataFetcher(list(tickers_tuple), start_date_str, end_date_str)
    data, metadata = fetcher.fetch()
    return data, metadata

def main():
    st.title("AI-Powered Investment Insight Generator")
    
    # User Inputs
    st.sidebar.header("Settings")
    tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated)", "AAPL, TSLA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    weights = {}
    st.sidebar.subheader("Portfolio Weights")
    if tickers:
        # Distribute 100% among all tickers for default values
        num_tickers = len(tickers)
        default_weights = [100 // num_tickers] * num_tickers
        remainder = 100 % num_tickers
        for i in range(remainder):
            default_weights[i] += 1
        for i, ticker in enumerate(tickers):
            weight = st.sidebar.slider(f"{ticker} Weight (%)", 0, 100, default_weights[i]) / 100
            weights[ticker] = weight
    
    if st.sidebar.button("Run Analysis"):
        if not np.isclose(sum(weights.values()), 1.0):
            st.sidebar.warning("Total weights must be equal to 100%.")
            return
        
        # Data Fetching using the cached function
        st.toast(f"Fetching fresh data for {', '.join(tickers)}...")
        with st.spinner("Loading Data..."):
            # Convert list of tickers to a tuple so it's hashable for the cache
            data, metadata = get_data(tuple(tickers), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if not data:
                st.error("No data loaded. Check tickers or connection.")
                return
        
        # Data Processing
        with st.spinner("Processing Data..."):
            processor = StockDataProcessor()
            processed_data = processor.process_data(data, metadata)
            if not processed_data:
                st.error("Data processing failed.")
                return
        
        # Prediction
        with st.spinner("Predicting Prices..."):
            # Note: This retrains the model on every run, which can be slow.
            # For production, consider training and saving the model separately.
            predictor = PricePredictor() 
            predictions = predictor.predict(processed_data, metadata)
        
        # Analysis
        with st.spinner("Analyzing Data..."):
            analyzer = DataAnalyzer(processed_data, metadata)
            insights = analyzer.generate_insights()
        
        # Portfolio Simulation & Visualization
        with st.spinner("Simulating Portfolio..."):
            simulator = PortfolioSimulator(processed_data, metadata)
            portfolio = simulator.simulate(weights)
        
        # Visualization
        visualizer = Visualizer(processed_data, predictions)
        for ticker in tickers:
            if ticker in processed_data and not processed_data[ticker].empty:
                st.subheader(f"{ticker} Analysis")
                st.write(insights[ticker])
                fig = visualizer.create_stock_visualization(ticker, insights[ticker])
                st.plotly_chart(fig)
        
        if portfolio:
            st.subheader("Portfolio Simulation")
            st.write(f"Weights: {weights}")
            st.write(f"Base Return (final): {portfolio['base'].iloc[-1]:.2f}")
            st.write(f"Hype Shock Return (final): {portfolio['hype_shock'].iloc[-1]:.2f}")
            st.write(f"Crash Return (final): {portfolio['crash'].iloc[-1]:.2f}")
            portfolio_fig = visualizer.create_portfolio_visualization(portfolio)
            st.plotly_chart(portfolio_fig)

if __name__ == "__main__":
    main()
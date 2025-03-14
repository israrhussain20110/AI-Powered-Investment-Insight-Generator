# main.py
from datetime import datetime, timedelta
import streamlit as st
from data_fetching import DataFetcher
from data_processing import StockDataProcessor
from prediction import PricePredictor
from analysis import DataAnalyzer
from visualuzation import Visualizer
from portfolio import PortfolioSimulator

def main():
    st.title("AI-Powered Investment Insight Generator")
    
    # User Inputs
    st.sidebar.header("Settings")
    tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated)", "AAPL, TSLA")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    weights = {}
    st.sidebar.subheader("Portfolio Weights")
    for ticker in tickers:
        weight = st.sidebar.slider(f"{ticker} Weight (%)", 0, 100, 50 if ticker == tickers[0] else 0) / 100
        weights[ticker] = weight
    
    if st.sidebar.button("Run Analysis"):
        if sum(weights.values()) != 1:
            st.sidebar.warning("Weights must sum to 100%")
            return
        
        # Data Fetching
        with st.spinner("Fetching Data..."):
            fetcher = DataFetcher(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            data, metadata = fetcher.fetch()
            if not data:
                st.error("No data fetched. Check tickers or connection.")
                return
        
        # Data Preprocessing (using your version)
        with st.spinner("Processing Data..."):
            processor = StockDataProcessor(input_dir="data")
            processor.data = data  # Pass fetched data directly
            processor.metadata = metadata
            processed_data = processor.process_data()
            processor.save_processed_data()
            if not processed_data:
                st.error("Data processing failed.")
                return
        
        # Prediction
        with st.spinner("Predicting Prices..."):
            predictor = PricePredictor()
            predictions = predictor.predict()
        
        # Analysis
        with st.spinner("Analyzing Data..."):
            analyzer = DataAnalyzer(processed_data, metadata)
            insights = analyzer.generate_insights()
        
        # Portfolio Simulation
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
        
        st.subheader("Portfolio Simulation")
        st.write(f"Weights: {weights}")
        st.write(f"Base Return (last): {portfolio['base'][-1]:.2f}")
        st.write(f"Hype Shock Return (last): {portfolio['hype_shock'][-1]:.2f}")
        st.write(f"Crash Return (last): {portfolio['crash'][-1]:.2f}")
        portfolio_fig = visualizer.create_portfolio_visualization(portfolio)
        st.plotly_chart(portfolio_fig)

if __name__ == "__main__":
    main()
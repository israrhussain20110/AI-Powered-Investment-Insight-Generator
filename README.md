# AI-Powered-Investment-Insight-Generator

This project is a Streamlit-based web application that automates financial analysis by fetching stock data, processing it to generate unique metrics, predicting future prices, and simulating portfolio performance.

## Features

  **Data Fetching:** Fetches historical stock data from Yahoo Finance.
  **Custom Metrics:**
      **Hype Score:** A unique metric calculated from price momentum, volume spikes, and simulated social media sentiment.
      **Sentiment-Adjusted Returns:** Daily returns adjusted based on the sentiment score.
      **Event Detection:** Identifies volume spikes and approximate earnings dates.
  **Price Prediction:** Uses an LSTM neural network to forecast future stock prices, adjusted by the Hype Score.   **AI-Driven Insights:** Generates narrative text insights based on the calculated metrics.
  **Portfolio Simulation:** Simulates portfolio returns under different scenarios (Base, Hype Shock, Crash).   **Interactive Visualizations:** Uses Plotly to create interactive charts for stock analysis and portfolio performance.

## Project Structure

- `main.py`: The main Streamlit application file.
- `data_fetching.py`: Handles fetching data from APIs.
- `data_processing.py`: Cleans, validates, and enriches the raw data.
- `prediction.py`: Contains the price prediction logic using an LSTM model.
- `analysis.py`: Generates qualitative insights from quantitative data.
- `portfolio.py`: Simulates portfolio performance.
- `visualization.py`: Creates all the Plotly visualizations.

## Model Caching

To improve performance, the application caches the trained LSTM models. The first time you run an analysis for a specific stock, the model is trained and saved to the `models/` directory. On subsequent runs, the pre-trained model is loaded directly, which significantly speeds up the "Predicting Prices" step.

If you wish to force a retraining of the models, simply delete the `models/` directory.

## How to Run

1. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

3. Open your web browser to the local URL provided by Streamlit. Use the sidebar to configure tickers, dates, and portfolio weights, then click "Run Analysis".
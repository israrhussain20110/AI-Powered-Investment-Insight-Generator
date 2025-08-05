# visualization.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    """Create unique interactive visualizations."""
    def __init__(self, data, predictions):
        self.data = data
        self.predictions = predictions
    
    def create_stock_visualization(self, ticker, insight):
        """Visualize stock with event markers and hype trends."""
        df = self.data[ticker]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Price & Predictions", "Volatility", "Hype Score"))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", line=dict(color="blue")), row=1, col=1)
        future_dates = pd.date_range(df.index[-1], periods=11, freq='B')[1:]
        if ticker in self.predictions:
            fig.add_trace(go.Scatter(x=future_dates, y=self.predictions[ticker], name="Predicted", line=dict(color="red", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[df['Volume_Spike']], y=df['Close'][df['Volume_Spike']], mode="markers", name="Volume Spike", marker=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[df['Earnings_Event']], y=df['Close'][df['Earnings_Event']], mode="markers", name="Earnings", marker=dict(color="green")), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name="Volatility", line=dict(color="purple")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Hype_Score_Cumulative'], name="Hype Score", line=dict(color="teal")), row=3, col=1)
        
        fig.update_layout(title=f"{ticker}: {insight}", height=800)
        return fig
    
    def create_portfolio_visualization(self, portfolio):
        """Visualize portfolio with multi-scenario view."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio["base"].index, y=portfolio["base"], name="Base", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=portfolio["hype_shock"].index, y=portfolio["hype_shock"], name="Hype Shock", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=portfolio["crash"].index, y=portfolio["crash"], name="Crash", line=dict(color="orange")))
        fig.update_layout(title="Portfolio Scenarios", yaxis_title="Cumulative Return")
        return fig
# portfolio.py
import pandas as pd

class PortfolioSimulator:
    """Simulate portfolio with unique hype shock scenario."""
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.portfolio = {}
    
    def simulate(self, weights):
        """Simulate portfolio across base, hype shock, and crash scenarios."""
        try:
            portfolio_return = pd.Series(0, index=self.data[list(self.data.keys())[0]].index)
            for ticker, weight in weights.items():
                if ticker in self.data and not self.data[ticker].empty:
                    portfolio_return += self.data[ticker]['Daily_Return'] * weight
            
            self.portfolio["base"] = (1 + portfolio_return).cumprod()
            self.portfolio["hype_shock"] = (1 + portfolio_return * (1 + sum(w * self.metadata.get(t, {}).get("sentiment", 0) * 2 for t, w in weights.items()))).cumprod()
            self.portfolio["crash"] = (1 + portfolio_return * 0.9).cumprod()
            return self.portfolio
        except Exception as e:
            print(f"Error simulating portfolio: {e}")
            return {}